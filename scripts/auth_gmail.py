#!/usr/bin/env python3
"""One-time interactive Gmail OAuth login script.

Run this in your PowerShell terminal:
    py scripts/auth_gmail.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.mcp_gmail import GmailMCPClient

print("\n" + "=" * 60)
print("  STUDYMATRIX RAG — GMAIL OAUTH LOGIN")
print("=" * 60)
print("\nOpening your default browser for Google Login...")
print("If the browser does not open automatically, check the URL printed below.\n")

client = GmailMCPClient()
try:
    client.authenticate(interactive=True)
    print("\n" + "=" * 60)
    print("  SUCCESS! Gmail authenticated successfully.")
    print("  Saved token to: token.json")
    print("  You can now return to http://localhost:8501 and use the Inbox Agent!")
    print("=" * 60 + "\n")
except Exception as e:
    print(f"\n[ERROR] Authentication failed: {e}")
    sys.exit(1)
