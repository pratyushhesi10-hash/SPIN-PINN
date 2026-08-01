"""
style.py
Drop this next to your app.py, then call load_css() once at the top
of your script, right after st.set_page_config(...).
"""

import streamlit as st

# Palette reference — reuse these anywhere you build custom
# charts (Plotly/Matplotlib) so everything matches exactly.
CHARCOAL = "#4A4A4A"
LIGHT_GRAY = "#CBCBCB"
CREAM = "#FFFFE3"
SLATE = "#6D8196"


def load_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        h1, h2, h3 {{
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            color: {CHARCOAL};
            letter-spacing: -0.02em;
        }}

        /* Numbers / equations / code look sharper in mono */
        code, .stMarkdown code {{
            font-family: 'IBM Plex Mono', monospace;
            background-color: {LIGHT_GRAY}55;
            border-radius: 4px;
        }}

        /* Hide default Streamlit chrome for a custom look */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}

        /* Tighten default padding */
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1150px;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: {LIGHT_GRAY}44;
            border-right: 1px solid {LIGHT_GRAY};
        }}

        /* Slider spacing + accent color */
        div[data-testid="stSlider"] {{
            margin-bottom: 1.4rem;
        }}
        div[data-testid="stSlider"] > div > div > div > div {{
            background-color: {SLATE};
        }}

        /* Buttons */
        .stButton > button {{
            border-radius: 8px;
            border: 1px solid {SLATE};
            background-color: transparent;
            color: {SLATE};
            font-weight: 500;
            padding: 0.5rem 1.4rem;
            transition: all 0.15s ease;
        }}
        .stButton > button:hover {{
            background-color: {SLATE};
            color: {CREAM};
        }}

        /* Metric cards */
        div[data-testid="stMetric"] {{
            background-color: #FFFFFF;
            border: 1px solid {LIGHT_GRAY};
            border-radius: 10px;
            padding: 1rem;
        }}

        /* Tabs */
        button[data-baseweb="tab"] {{
            font-weight: 500;
            color: {CHARCOAL}99;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {SLATE};
        }}
        div[data-baseweb="tab-highlight"] {{
            background-color: {SLATE};
        }}

        /* Custom badge/chip helper — use with st.markdown(..., unsafe_allow_html=True) */
        .chip {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 6px;
            border: 1px solid {SLATE}55;
            background: {SLATE}11;
            color: {SLATE};
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.85rem;
            margin-right: 8px;
        }}
        .chip.accent {{
            border-color: {SLATE};
            background: {SLATE};
            color: {CREAM};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def chip(label: str, accent: bool = False) -> str:
    """Return HTML for a small pill-style badge, e.g. for equations or status text."""
    cls = "chip accent" if accent else "chip"
    return f'<span class="{cls}">{label}</span>'
