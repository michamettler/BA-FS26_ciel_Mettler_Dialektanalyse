"""Shared-password gate for the visualization (anti-crawl, not real authentication)."""
import hmac

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


def require_password() -> None:
    """Block the page until the shared password is entered. Call at the top of every page."""
    if st.session_state.get("authed"):
        return
    try:
        expected = st.secrets["password"]
    except (KeyError, StreamlitSecretNotFoundError):
        expected = None
    if not isinstance(expected, str) or not expected:
        st.error("Access is not configured: set `password` in `.streamlit/secrets.toml` (see deploy/DEPLOY.md).")
        st.stop()

    password = st.text_input("Password", type="password")
    if password and hmac.compare_digest(password, expected):
        st.session_state["authed"] = True
        st.rerun()
    elif password:
        st.error("Wrong password")
    st.stop()
