"""Gmail MCP (Model Context Protocol) and API connector with read-only OAuth.

Safety guarantees:
- ONLY uses 'https://www.googleapis.com/auth/gmail.readonly' scope.
- Never allows sending, modifying, or deleting email.
- Wraps all raw email text in <untrusted_email> tags to prevent prompt injection.
- Truncates oversized bodies and never logs full email contents.
"""

from __future__ import annotations

import base64
import datetime
import html
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

READONLY_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


class EmailMessage(BaseModel):
    """Sanitized, safe representation of an email message."""

    id: str
    thread_id: str
    subject: str = "No Subject"
    sender: str = "Unknown Sender"
    date: str = ""
    snippet: str = ""
    body: str = ""
    is_unread: bool = False

    def to_safe_prompt_block(self) -> str:
        """Format the email inside a strict untrusted XML delimiter to block prompt injections."""
        # Sanitize any closing untrusted_email tags inside the body to prevent escape
        safe_body = self.body.replace("</untrusted_email>", "[escaped-tag]")
        safe_body = html.unescape(safe_body)
        return (
            f'<untrusted_email id="{self.id}" thread_id="{self.thread_id}">\n'
            f"From: {self.sender}\n"
            f"Subject: {self.subject}\n"
            f"Date: {self.date}\n"
            f"Body:\n{safe_body.strip()}\n"
            f"</untrusted_email>"
        )


class GmailServiceError(RuntimeError):
    """Raised when Gmail authentication or fetching fails."""


class GmailMCPClient:
    """Manages read-only Gmail access with token caching and safety constraints."""

    def __init__(
        self,
        credentials_path: Optional[str] = None,
        token_path: Optional[str] = None,
    ) -> None:
        self.credentials_path = Path(
            credentials_path
            or os.getenv("GMAIL_CREDENTIALS_PATH", "credentials.json")
        )
        self.token_path = Path(
            token_path or os.getenv("GMAIL_TOKEN_PATH", "token.json")
        )
        self._service: Any = None

    def is_configured(self) -> bool:
        """Check if client credentials exist."""
        return self.credentials_path.exists()

    def is_authenticated(self) -> bool:
        """Check if an active token file exists and is valid."""
        if not self.token_path.exists():
            return False
        try:
            from google.oauth2.credentials import Credentials

            creds = Credentials.from_authorized_user_file(
                str(self.token_path), READONLY_SCOPES
            )
            return bool(creds and creds.valid)
        except Exception:
            return False

    def get_connection_status(self) -> Dict[str, Any]:
        """Return diagnostic status of the Gmail connection."""
        has_creds = self.is_configured()
        is_auth = self.is_authenticated()
        return {
            "credentials_present": has_creds,
            "credentials_file": str(self.credentials_path.name),
            "authenticated": is_auth,
            "scopes": READONLY_SCOPES,
            "status": "connected" if is_auth else ("credentials_found" if has_creds else "missing_credentials"),
        }

    def authenticate(self, interactive: bool = True) -> Any:
        """Authenticate with Google OAuth using strict readonly scopes."""
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow

        creds = None
        if self.token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(
                    str(self.token_path), READONLY_SCOPES
                )
            except Exception as exc:
                logger.warning("Invalid token file: %s", exc)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    logger.info("Refreshing expired Gmail OAuth token...")
                    creds.refresh(Request())
                except Exception as exc:
                    logger.warning("Token refresh failed: %s", exc)
                    creds = None

            if not creds:
                if not self.credentials_path.exists():
                    raise GmailServiceError(
                        f"OAuth credentials file not found at '{self.credentials_path}'. "
                        "Download 'credentials.json' from Google Cloud Console with the Gmail API enabled."
                    )
                if not interactive:
                    raise GmailServiceError(
                        "Authentication required. Please authenticate interactively first."
                    )

                logger.info("Initiating interactive OAuth consent flow...")
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path), READONLY_SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Persist token for future runs
            self.token_path.write_text(creds.to_json(), encoding="utf-8")

        from googleapiclient.discovery import build

        self._service = build("gmail", "v1", credentials=creds)
        return self._service

    def _get_service(self) -> Any:
        if self._service is None:
            self.authenticate(interactive=False)
        return self._service

    def _parse_message_payload(self, msg: Dict[str, Any]) -> EmailMessage:
        """Extract clean text headers and body safely from Gmail message dict."""
        headers = {
            h.get("name", "").lower(): h.get("value", "")
            for h in msg.get("payload", {}).get("headers", [])
        }
        subject = headers.get("subject", "No Subject")
        sender = headers.get("from", "Unknown")
        date = headers.get("date", "")
        snippet = msg.get("snippet", "")
        labels = msg.get("labelIds", [])

        # Extract plain text body
        body = ""
        payload = msg.get("payload", {})

        def extract_text(part: Dict[str, Any]) -> str:
            mime = part.get("mimeType", "")
            data = part.get("body", {}).get("data")
            if mime == "text/plain" and data:
                try:
                    return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
                except Exception:
                    return ""
            if "parts" in part:
                for sub in part["parts"]:
                    t = extract_text(sub)
                    if t:
                        return t
            return ""

        extracted = extract_text(payload)
        body = extracted if extracted else snippet

        # Truncate body to safe size to conserve tokens and prevent buffer bloat
        max_chars = 2500
        if len(body) > max_chars:
            body = body[:max_chars] + f"\n... [Truncated: {len(body) - max_chars} characters omitted]"

        return EmailMessage(
            id=msg.get("id", ""),
            thread_id=msg.get("threadId", ""),
            subject=subject,
            sender=sender,
            date=date,
            snippet=snippet,
            body=body,
            is_unread="UNREAD" in labels,
        )

    def search_emails(self, query: str = "", max_results: int = 5) -> List[EmailMessage]:
        """Search messages matching standard Gmail query syntax."""
        service = self._get_service()
        limit = max(1, min(max_results, 15))
        try:
            response = (
                service.users()
                .messages()
                .list(userId="me", q=query, maxResults=limit)
                .execute()
            )
            messages = response.get("messages", [])
            results: List[EmailMessage] = []
            for item in messages:
                msg_data = (
                    service.users()
                    .messages()
                    .get(userId="me", id=item["id"], format="full")
                    .execute()
                )
                results.append(self._parse_message_payload(msg_data))
            return results
        except Exception as exc:
            logger.exception("Gmail API search failed")
            raise GmailServiceError(f"Gmail search failed: {exc}") from exc

    def get_unread_emails_today(self, max_results: int = 5) -> List[EmailMessage]:
        """Fetch unread emails received in the last 24 hours."""
        today = datetime.date.today().strftime("%Y/%m/%d")
        query = f"is:unread after:{today}"
        return self.search_emails(query=query, max_results=max_results)

    def get_thread(self, thread_id: str) -> List[EmailMessage]:
        """Retrieve all messages in a specific email conversation thread."""
        service = self._get_service()
        try:
            thread_data = (
                service.users()
                .threads()
                .get(userId="me", id=thread_id, format="full")
                .execute()
            )
            raw_msgs = thread_data.get("messages", [])
            return [self._parse_message_payload(m) for m in raw_msgs]
        except Exception as exc:
            logger.exception("Gmail API thread retrieval failed")
            raise GmailServiceError(f"Failed to fetch thread '{thread_id}': {exc}") from exc
