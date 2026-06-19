"""Shared-password gate for the visualization (anti-crawl, not real authentication)."""
import hmac

import streamlit as st


def require_password() -> None:
    """Block the page until the shared password is entered. Call at the top of every page:
    Streamlit allows deep-linking straight to a page, so gating only Home leaves the others open.
    Once entered, `authed` persists in session_state across pages."""
    if st.session_state.get("authed"):
        return
    password = st.text_input("Password", type="password")
    if password and hmac.compare_digest(password, st.secrets["password"]):
        st.session_state["authed"] = True
        st.rerun()
    elif password:
        st.error("Wrong password")
    st.stop()
