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
            font-family: 'Inter', sans-serif !important;
        }}

        /* Force the whole app surface — kills any leftover dark background */
        .stApp {{
            background-color: {CREAM} !important;
            color: {CHARCOAL} !important;
        }}

        [data-testid="stHeader"] {{
            background-color: {CREAM} !important;
        }}

        h1, h2, h3, h4, p, span, label, li {{
            color: {CHARCOAL} !important;
            font-family: 'Inter', sans-serif;
        }}
        h1, h2, h3 {{
            font-weight: 600;
            letter-spacing: -0.02em;
        }}

        /* Numbers / equations / code look sharper in mono */
        code, .stMarkdown code {{
            font-family: 'IBM Plex Mono', monospace;
            background-color: {LIGHT_GRAY}55 !important;
            color: {CHARCOAL} !important;
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
            background-color: {LIGHT_GRAY}55 !important;
            border-right: 1px solid {LIGHT_GRAY};
        }}
        section[data-testid="stSidebar"] * {{
            color: {CHARCOAL} !important;
        }}

        /* Slider spacing */
        div[data-testid="stSlider"] {{
            margin-bottom: 1.4rem;
        }}
        /* Catch-all: force every bit of text inside a slider widget to be readable */
        div[data-testid="stSlider"] * {{
            color: {CHARCOAL} !important;
        }}
        /* Slider track fill */
        div[data-testid="stSlider"] [role="slider"] {{
            background-color: {SLATE} !important;
            border-color: {SLATE} !important;
        }}
        /* The floating current-value label above the handle — bump contrast */
        div[data-testid="stSliderThumbValue"],
        div[data-testid="stThumbValue"] {{
            color: {CHARCOAL} !important;
            font-weight: 700 !important;
            background-color: {LIGHT_GRAY}aa !important;
            border-radius: 4px;
            padding: 0 4px;
        }}
        /* Slider widget label above the whole control */
        div[data-testid="stSlider"] label p {{
            color: {CHARCOAL} !important;
            font-weight: 500;
        }}

        /* Buttons */
        .stButton > button {{
            border-radius: 8px;
            border: 1px solid {SLATE};
            background-color: transparent;
            color: {SLATE} !important;
            font-weight: 500;
            padding: 0.5rem 1.4rem;
            transition: all 0.15s ease;
        }}
        .stButton > button:hover {{
            background-color: {SLATE};
            color: {CREAM} !important;
        }}

        /* Metric cards */
        div[data-testid="stMetric"] {{
            background-color: #FFFFFF !important;
            border: 1px solid {LIGHT_GRAY};
            border-radius: 10px;
            padding: 1rem;
        }}

        /* Tabs */
        button[data-baseweb="tab"] p {{
            font-weight: 500;
            color: {CHARCOAL}99 !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] p {{
            color: {SLATE} !important;
        }}
        div[data-baseweb="tab-highlight"] {{
            background-color: {SLATE} !important;
        }}
        div[data-baseweb="tab-border"] {{
            background-color: {LIGHT_GRAY} !important;
        }}

        /* Custom badge/chip helper — use with st.markdown(..., unsafe_allow_html=True) */
        .chip {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 6px;
            border: 1px solid {SLATE}55;
            background: {SLATE}11;
            color: {SLATE} !important;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.85rem;
            margin-right: 8px;
        }}
        .chip.accent {{
            border-color: {SLATE};
            background: {SLATE};
            color: {CREAM} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def chip(label: str, accent: bool = False) -> str:
    """Return HTML for a small pill-style badge, e.g. for equations or status text."""
    cls = "chip accent" if accent else "chip"
    return f'<span class="{cls}">{label}</span>'


def style_matplotlib():
    """
    Call this once near the top of your script (after importing matplotlib.pyplot as plt)
    to make all matplotlib charts match the app theme with readable text.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": CREAM,
        "axes.facecolor": CREAM,
        "savefig.facecolor": CREAM,

        "text.color": CHARCOAL,
        "axes.labelcolor": CHARCOAL,
        "axes.titlecolor": CHARCOAL,
        "xtick.color": CHARCOAL,
        "ytick.color": CHARCOAL,

        "axes.edgecolor": CHARCOAL,
        "axes.grid": True,
        "grid.color": LIGHT_GRAY,
        "grid.alpha": 0.6,

        "legend.facecolor": CREAM,
        "legend.edgecolor": LIGHT_GRAY,
        "legend.labelcolor": CHARCOAL,

        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
    })
