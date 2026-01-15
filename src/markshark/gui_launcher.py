#!/usr/bin/env python3
"""
MarkShark
gui_launcher.py  —  Streamlit GUI launching engine

Features:
 - CSV includes: correct, incorrect, blank, multi, percent
 - KEY row written below header (when a key is provided)
 - Annotated PNGs:
     * Names/ID: blue circles, optional white % text
     * Answers: green=correct, red=incorrect, grey=blank, orange=multi
 - Optional % fill text via --label-density
 - Columns limited to len(key) when a key is provided
"""

def main():
    from importlib import resources
    from streamlit.web import cli as stcli
    import sys
    script = str(resources.files("markshark") / "app_streamlit.py")
    sys.argv = ["streamlit", "run", script]
    raise SystemExit(stcli.main())
