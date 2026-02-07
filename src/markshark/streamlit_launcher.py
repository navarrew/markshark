#!/usr/bin/env python3
"""
MarkShark
streamlit_launcher.py  —  Streamlit web interface launcher

Launches the Streamlit-based web GUI (app_streamlit.py).
For the native desktop GUI, use: python -m markshark.gui
"""

def main():
    from importlib import resources
    from streamlit.web import cli as stcli
    import sys
    script = str(resources.files("markshark") / "app_streamlit.py")
    sys.argv = ["streamlit", "run", script]
    raise SystemExit(stcli.main())
