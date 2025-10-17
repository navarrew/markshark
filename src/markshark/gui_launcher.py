
def main():
    from importlib import resources
    from streamlit.web import cli as stcli
    import sys
    script = str(resources.files("markshark") / "app_streamlit.py")
    sys.argv = ["streamlit", "run", script]
    raise SystemExit(stcli.main())
