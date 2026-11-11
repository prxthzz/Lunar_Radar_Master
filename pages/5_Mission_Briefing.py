import streamlit as st

from luna_core.ui_theme import inject_theme, require_result

st.set_page_config(page_title="Mission Briefing", page_icon="\U0001F4E1", layout="wide")
inject_theme()
result = require_result()

st.markdown('<div class="page-enter"><h2>Mission Briefing</h2>'
            '<p class="twin-sub">Auto-generated report \u2014 read this aloud in a demo.</p></div>',
            unsafe_allow_html=True)
st.write("")

st.markdown(f'<div class="briefing-box">{result["briefing"]}</div>', unsafe_allow_html=True)

st.write("")
st.download_button(
    "\U0001F4C4 Download briefing (.txt)",
    data=result["briefing"],
    file_name="lunatwin_mission_briefing.txt",
    use_container_width=True,
)
