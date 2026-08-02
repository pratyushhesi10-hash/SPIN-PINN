"""
style.py
Drop this next to your app.py, then call load_css() once at the top
of your script, right after st.set_page_config(...).
"""

import streamlit as st

# Palette reference — reuse these anywhere you build custom
# charts (Plotly/Matplotlib) so everything matches exactly.
CHARCOAL = "#1A202C"
LIGHT_GRAY = "#E2E8F0"
CREAM = "#FFFFFF"
SLATE = "#4A5568"
BLUE = "#3182CE"


def load_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif !important;
        }}

        /* Clean Light Background */
        .stApp {{
            background: linear-gradient(180deg, #FAFBFC 0%, #F5F7FA 100%) !important;
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
            font-family: 'JetBrains Mono', monospace;
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
            max-width: 1200px;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: {CREAM} !important;
            border-right: 1px solid {LIGHT_GRAY};
        }}
        section[data-testid="stSidebar"] * {{
            color: {CHARCOAL} !important;
        }}

        /* Slider spacing */
        div[data-testid="stSlider"] {{
            margin-bottom: 1.4rem;
        }}
        div[data-testid="stSlider"] * {{
            color: {CHARCOAL} !important;
        }}
        div[data-testid="stSlider"] [role="slider"] {{
            background-color: {BLUE} !important;
            border-color: {BLUE} !important;
        }}
        div[data-testid="stSliderThumbValue"],
        div[data-testid="stThumbValue"] {{
            color: {CHARCOAL} !important;
            font-weight: 700 !important;
            background-color: {LIGHT_GRAY}aa !important;
            border-radius: 4px;
            padding: 0 4px;
        }}
        div[data-testid="stSlider"] label p {{
            color: {CHARCOAL} !important;
            font-weight: 600;
            font-size: 0.9rem;
        }}

        /* Buttons */
        .stButton > button {{
            border-radius: 8px;
            border: 1px solid {LIGHT_GRAY};
            background-color: {CREAM};
            color: {SLATE} !important;
            font-weight: 600;
            font-size: 0.9rem;
            padding: 10px 20px;
            transition: all 0.2s ease;
        }}
        .stButton > button:hover {{
            background-color: {BLUE};
            border-color: {BLUE};
            color: {CREAM} !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(49, 130, 206, 0.2);
        }}

        /* Metric cards */
        div[data-testid="stMetric"] {{
            background-color: {CREAM} !important;
            border: 1px solid {LIGHT_GRAY};
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            transition: all 0.2s ease;
        }}
        div[data-testid="stMetric"]:hover {{
            border-color: {SLATE};
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            transform: translateY(-2px);
        }}

        /* Tabs */
        button[data-baseweb="tab"] p {{
            font-weight: 600;
            color: {SLATE} !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] p {{
            color: {BLUE} !important;
        }}
        div[data-baseweb="tab-highlight"] {{
            background-color: {BLUE} !important;
        }}
        div[data-baseweb="tab-border"] {{
            background-color: {LIGHT_GRAY} !important;
        }}

        /* Custom badge/chip helper */
        .chip {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            border: 1px solid {LIGHT_GRAY};
            background: {CREAM};
            color: {SLATE} !important;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            margin-right: 8px;
            font-weight: 500;
            transition: all 0.2s ease;
        }}
        .chip.accent {{
            border-color: {BLUE};
            background: #EBF8FF;
            color: {BLUE} !important;
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

        "axes.edgecolor": LIGHT_GRAY,
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
