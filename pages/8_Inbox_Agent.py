"""Streamlit page for the Autonomous Inbox & Academic Agent with Gmail MCP."""

from __future__ import annotations

import os
import streamlit as st
from core.agent import AcademicAgent
from core.mcp_gmail import GmailMCPClient
from core.ui import configure_page, page_header

configure_page("Inbox Agent", "✉️")
page_header(
    "08 / AGENT",
    "Inbox Agent & MCP",
    "Autonomous agent with Gmail MCP email tools, lecture retrieval, and study material synthesis.",
)

gmail_client = GmailMCPClient()
agent = AcademicAgent(gmail_client=gmail_client)

# Status indicators
hf_token = os.getenv("HF_TOKEN", "")
gmail_status = gmail_client.get_connection_status()

c1, c2, c3 = st.columns(3)
with c1:
    if hf_token:
        st.success(f"Hugging Face: Connected ({os.getenv('HF_CHAT_MODEL', 'Qwen2.5-7B')})")
    else:
        st.error("Hugging Face: Missing HF_TOKEN in .env")

with c2:
    if gmail_status["authenticated"]:
        st.success("Gmail MCP: Connected (Read-Only)")
    elif gmail_status["credentials_present"]:
        st.warning("Gmail MCP: Credentials Found (Needs Login)")
    else:
        st.info("Gmail MCP: No credentials.json")

with c3:
    st.info(f"Max Agent Steps: {os.getenv('MCP_MAX_AGENT_STEPS', '8')}")

if not gmail_status["authenticated"] and gmail_status["credentials_present"]:
    st.info("💡 **Recommended:** To complete Google Login cleanly without browser popup blockers, open PowerShell in this folder and run:")
    st.code("py scripts/auth_gmail.py", language="powershell")
    st.markdown("Or try connecting in this browser:")
    if st.button("Connect Gmail Account (OAuth)"):
        with st.spinner("Opening Google login in your default browser..."):
            try:
                gmail_client.authenticate(interactive=True)
                st.success("Gmail authenticated successfully! Please reload the page.")
                st.rerun()
            except Exception as e:
                st.error(f"Gmail authentication failed: {e}")

st.markdown("---")

st.markdown("### Choose an Academic Task or Type Below")
col_p1, col_p2, col_p3 = st.columns(3)

prompt_input = ""
with col_p1:
    if st.button("📬 Unread Emails Today", use_container_width=True):
        prompt_input = "Summarize my unread emails from today, highlighting any urgent announcements or notices."

with col_p2:
    if st.button("📝 Teacher Assignments & Deadlines", use_container_width=True):
        prompt_input = "Find emails from my teacher about assignments and list all upcoming deadlines with dates."

with col_p3:
    if st.button("🃏 Key Points to Flashcards", use_container_width=True):
        prompt_input = "Search for recent emails discussing course projects or homework, and turn the key points into study flashcards."

goal = st.text_area(
    "Agent Goal:",
    value=prompt_input,
    placeholder="e.g. Find emails about our biology assignment, compare with lecture notes, and list the key deadlines...",
    height=100,
)

if st.button("Run Agent", type="primary", use_container_width=True):
    if not goal.strip():
        st.warning("Please enter a goal for the agent.")
    elif not hf_token:
        st.error("Please add your HF_TOKEN to your .env file before running the agent.")
    else:
        st.markdown("### Execution Trace")
        trace_container = st.container()

        with trace_container:
            for step in agent.run_stream(goal):
                if step.step_type == "tool_call":
                    with st.expander(f"Step {step.step_number}: Calling `{step.tool_name}`", expanded=True):
                        st.json(step.tool_args or {})
                elif step.step_type == "tool_result":
                    with st.expander(f"Step {step.step_number}: Tool Result (`{step.tool_name}`)", expanded=False):
                        st.code(step.tool_output or "", language="markdown")
                elif step.step_type == "final_answer":
                    st.markdown("### Final Answer")
                    st.markdown(step.content or "")
                elif step.step_type == "error":
                    st.error(step.content)
