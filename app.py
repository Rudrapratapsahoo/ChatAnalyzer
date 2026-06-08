import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import numpy as np
from collections import Counter
import random, datetime
import chat_helper as helper
from preprocessor import preprocess

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChatAnalyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session State for Theme ───────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state["theme"] = "light"

def toggle_theme():
    st.session_state["theme"] = "dark" if st.session_state["theme"] == "light" else "light"

# ── Theme Variables ───────────────────────────────────────────────────────────
if st.session_state["theme"] == "light":
    bg_main = "#ffffff"
    text_main = "#0f172a"
    text_sec = "#334155"
    border_main = "#bae6fd"
    border_hover = "#0ea5e9"
    card_bg = "rgba(0, 0, 0, 0.05)"
    card_border = "rgba(0, 0, 0, 0.1)"
    card_hover = "rgba(0, 0, 0, 0.3)"
    icon_1, icon_2, icon_3, icon_4 = "#1e293b", "#334155", "#0f172a", "#475569"
else:
    bg_main = "#0f172a"
    text_main = "#f8fafc"
    text_sec = "#cbd5e1"
    border_main = "#334155"
    border_hover = "#38bdf8"
    card_bg = "rgba(255, 255, 255, 0.05)"
    card_border = "rgba(255, 255, 255, 0.1)"
    card_hover = "rgba(255, 255, 255, 0.2)"
    icon_1, icon_2, icon_3, icon_4 = "#cbd5e1", "#94a3b8", "#f8fafc", "#64748b"

# ── Chart Palette (visible bar / gradient colours for both themes) ────────────
if st.session_state["theme"] == "light":
    bar_muted = "#93c5fd"
    bar_accent = "#0284c7"
    gradient_start, gradient_end = "#dbeafe", "#0ea5e9"
    sentiment_pos, sentiment_neg, sentiment_neu = "#16a34a", "#dc2626", "#6b7280"
else:
    bar_muted = "#1e3a5f"
    bar_accent = "#38bdf8"
    gradient_start, gradient_end = "#0c2d48", "#38bdf8"
    sentiment_pos, sentiment_neg, sentiment_neu = "#4ade80", "#f87171", "#64748b"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* Dynamic Theme */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

/* Hide sidebar completely */
section[data-testid="stSidebar"] {{ display: none !important; }}
button[kind="header"] {{ display: none !important; }}

/* Main Background */
.stApp {{
    background-color: {bg_main};
    color: {text_main};
    min-height: 100vh;
}}

/* Floating doodle overlay */
.doodle-bg {{
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}}
.doodle-bg .icon {{
    position: absolute;
    opacity: 0.15;
    animation-timing-function: linear;
    animation-iteration-count: infinite;
}}
@keyframes floatUp {{
    0% {{ transform: translateY(110vh) rotate(0deg); }}
    100% {{ transform: translateY(-15vh) rotate(360deg); }}
}}
@keyframes floatDiag {{
    0% {{ transform: translate(0, 110vh) rotate(0deg); }}
    100% {{ transform: translate(-80px, -15vh) rotate(-360deg); }}
}}

/* Text colors */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p,
.stApp label, .stApp span, .stApp div {{
    color: {text_main};
}}

/* Elements */
.stMarkdown, .stText {{ color: {text_main}; }}
[data-testid="stFileUploader"] label {{ color: {text_main} !important; }}
[data-testid="stSelectbox"] label {{ color: {text_main} !important; }}

/* File uploader styling */
[data-testid="stFileUploader"] {{
    background: {card_bg};
    border-radius: 16px;
    padding: 1.5rem;
    border: 2px dashed {border_main};
    transition: all 0.3s ease;
}}
[data-testid="stFileUploader"]:hover {{
    border-color: {border_hover};
    background: {card_bg};
}}
[data-testid="stFileUploader"] button {{
    background: {text_main} !important;
    color: {bg_main} !important;
    border-radius: 50px !important;
    border: none !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}}
[data-testid="stFileUploader"] button *, [data-testid="stFileUploader"] button p {{
    color: {bg_main} !important;
}}
[data-testid="stFileUploader"] button:hover {{
    background: {border_hover} !important;
    color: #ffffff !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3) !important;
}}
[data-testid="stFileUploader"] button:hover *, [data-testid="stFileUploader"] button:hover p {{
    color: #ffffff !important;
}}
[data-testid="stFileUploader"] small, [data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzoneInstructions"] div {{
    color: {text_sec} !important;
}}

/* Select box styling */
[data-testid="stSelectbox"] > div > div {{
    background: {card_bg} !important;
    border-color: {border_main} !important;
    color: {text_main} !important;
    border-radius: 8px !important;
}}

/* Text input styling */
input[type="text"], input[type="password"] {{
    background: {card_bg} !important;
    border: 2px solid {border_main} !important;
    color: {text_main} !important;
    border-radius: 12px !important;
    padding: 0.8rem 1rem !important;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
    font-size: 1rem !important;
    transition: all 0.2s ease;
}}
input[type="text"]:focus, input[type="password"]:focus {{
    border-color: {border_hover} !important;
    background: {bg_main} !important;
    box-shadow: 0 4px 12px rgba(14, 165, 233, 0.15) !important;
}}

/* Buttons */
div[data-testid="stButton"] > button,
div[data-testid="stFormSubmitButton"] > button {{
    background: {text_main} !important;
    color: {bg_main} !important;
    border: 2px solid {border_main} !important;
    border-radius: 50px !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1.05rem !important;
    padding: 0.8rem 2rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.15) !important;
}}
div[data-testid="stButton"] > button p,
div[data-testid="stFormSubmitButton"] > button p,
div[data-testid="stDownloadButton"] > button p {{
    color: {bg_main} !important;
}}
div[data-testid="stButton"] > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(14, 165, 233, 0.3) !important;
    background: {border_hover} !important;
}}
div[data-testid="stButton"] > button:hover p,
div[data-testid="stFormSubmitButton"] > button:hover p {{
    color: #ffffff !important;
}}

/* Download button */
div[data-testid="stDownloadButton"] > button {{
    background: {text_main};
    color: {bg_main} !important;
    border: none;
    border-radius: 30px;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}}
div[data-testid="stDownloadButton"] > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(14, 165, 233, 0.3);
    background: {border_hover};
}}
div[data-testid="stDownloadButton"] > button:hover p {{
    color: #ffffff !important;
}}

/* Glassmorphism Cards */
.glass-card {{
    background: {card_bg};
    backdrop-filter: blur(12px);
    border: 1px solid {card_border};
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    animation: fadeInUp 0.6s ease-out forwards;
}}
.glass-card:hover {{ transform: translateY(-3px); border-color: {card_hover}; }}

/* KPI Cards */
.kpi-card {{
    background: {card_bg};
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    border: 1px solid {card_border};
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    animation: fadeInUp 0.6s ease-out forwards;
    transition: all 0.3s ease;
}}
.kpi-card:hover {{ transform: translateY(-3px); border-color: {card_hover}; }}
.kpi-icon {{ font-size: 1.8rem; margin-bottom: 0.4rem; }}
.kpi-label {{ font-weight: 500; font-size: 0.75rem; color: {text_sec}; margin-bottom: 0.3rem; }}
.kpi-value {{ font-size: 1.8rem; font-weight: 700; color: {text_main}; }}

/* Feature Cards */
.feature-card {{
    background: {card_bg};
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid {card_border};
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    transition: all 0.3s ease;
    width: 100%;
    max-width: 260px;
    margin: auto;
}}
.feature-card:hover {{ transform: translateY(-4px); box-shadow: 0 4px 12px rgba(14, 165, 233, 0.15); border-color: {border_hover}; }}
.feature-card h4 {{ color: {text_main}; margin-top: 0.5rem; }}
.feature-card p {{ color: {text_sec}; font-size: 0.85rem; }}

/* Animations */
@keyframes fadeInUp {{ from {{ opacity:0; transform:translateY(20px); }} to {{ opacity:1; transform:translateY(0); }} }}
@keyframes float {{ 0%,100% {{ transform:translateY(0); }} 50% {{ transform:translateY(-8px); }} }}

/* Header */
.dash-title, .hero-title {{
    color: {text_main};
    font-weight: 900;
    letter-spacing: -0.03em;
}}

/* Misc utility */
.gradient-divider {{ height:2px; background: rgba(14, 165, 233, 0.6); margin:2rem 0; border-radius: 2px; }}
.stCaption {{ color: {text_sec} !important; }}

/* Login / Hero */
.login-container, .hero-title {{ background: transparent; }}
.login-title {{ color: {text_main}; font-weight: 800; }}

/* Alert overrides */
[data-testid="stAlert"] {{
    background: {card_bg} !important;
    border-radius: 12px !important;
    border-left-width: 4px !important;
    border-color: {border_hover} !important;
    color: {text_main} !important;
}}

/* Caption styling */
.stCaption {{ color: {text_sec} !important; }}
</style>

<!-- Floating Chat Doodle Icons Background -->
<div class="doodle-bg">
    <svg class="icon" style="left:3%;  width:38px; animation: floatUp 28s 0s infinite;" viewBox="0 0 24 24" fill="none" stroke="{icon_1}" stroke-width="1.5"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
    <svg class="icon" style="left:12%; width:28px; animation: floatDiag 35s 3s infinite;" viewBox="0 0 24 24" fill="none" stroke="{icon_2}" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>
    <svg class="icon" style="left:22%; width:32px; animation: floatUp 32s 6s infinite;" viewBox="0 0 24 24" fill="none" stroke="{icon_3}" stroke-width="1.5"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>
    <svg class="icon" style="left:32%; width:26px; animation: floatDiag 26s 1s infinite;" viewBox="0 0 24 24" fill="none" stroke="{icon_4}" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
    <svg class="icon" style="left:42%; width:34px; animation: floatUp 30s 8s infinite;" viewBox="0 0 24 24" fill="none" stroke="{icon_1}" stroke-width="1.5"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>
    <svg class="icon" style="left:53%; width:30px; animation: floatDiag 33s 5s infinite;" viewBox="0 0 24 24" fill="none" stroke="{icon_3}" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
    <svg class="icon" style="left:63%; width:36px; animation: floatUp 27s 2s infinite;" viewBox="0 0 24 24" fill="none" stroke="{icon_2}" stroke-width="1.5"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/></svg>
    <svg class="icon" style="left:73%; width:24px; animation: floatDiag 36s 7s infinite;" viewBox="0 0 24 24" fill="none" stroke="{icon_1}" stroke-width="1.5"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
    <svg class="icon" style="left:83%; width:32px; animation: floatUp 31s 4s infinite;" viewBox="0 0 24 24" fill="none" stroke="{icon_3}" stroke-width="1.5"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/><polyline points="22,6 12,13 2,6"/></svg>
    <svg class="icon" style="left:93%; width:28px; animation: floatDiag 29s 9s infinite;" viewBox="0 0 24 24" fill="none" stroke="{icon_4}" stroke-width="1.5"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/></svg>
    <svg class="icon" style="left:7%;  width:22px; animation: floatUp 34s 11s infinite;" viewBox="0 0 24 24" fill="none" stroke="{icon_1}" stroke-width="1.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
    <svg class="icon" style="left:48%; width:26px; animation: floatDiag 25s 13s infinite;" viewBox="0 0 24 24" fill="none" stroke="{icon_3}" stroke-width="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
    <svg class="icon" style="left:58%; width:30px; animation: floatUp 37s 10s infinite;" viewBox="0 0 24 24" fill="none" stroke="{icon_2}" stroke-width="1.5"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>
    <svg class="icon" style="left:88%; width:34px; animation: floatDiag 32s 15s infinite;" viewBox="0 0 24 24" fill="none" stroke="{icon_1}" stroke-width="1.5"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
    <svg class="icon" style="left:18%; width:28px; animation: floatUp 29s 14s infinite;" viewBox="0 0 24 24" fill="none" stroke="{icon_3}" stroke-width="1.5"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
</div>
""", unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# FOOTER HTML (rendered via components.html for proper SVG support)
# ══════════════════════════════════════════════════════════════════════════════





def render_footer():
    """Render footer using components.html for proper SVG support."""
    FOOTER_HTML = f"""
    <style>
        body {{
            background-color: transparent;
            margin: 0;
            color: {text_main};
        }}
    </style>
    <div style="
        background-color: transparent;
        padding: 3rem 2rem 1rem;
        font-family: 'Poppins', sans-serif;
        margin-top: 3rem;
        border-top: 1px solid {card_border};
    ">
        <div style="
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            max-width: 1100px;
            margin: 0 auto;
            gap: 2rem;
        ">
            <div style="flex: 1.5; min-width: 220px;">
                <div style="font-size: 1.4rem; font-weight: 700; color: {text_main}; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00d2ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
                    Chat<span style="color: #00d2ff;">Analyzer</span>
                </div>
                <p style="font-size: 0.85rem; line-height: 1.6; color: {text_sec};">Empowering users with smart chat analysis tools, helping you discover insights and understand your conversations better.</p>
            </div>
            <div style="flex: 1; min-width: 160px;">
                <div style="color: {text_main}; font-weight: 600; margin-bottom: 1rem; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 0.08em;">Quick Links</div>
                <ul style="list-style: none; padding: 0; margin: 0;">
                    <li style="margin-bottom: 0.7rem;"><a href="#" style="color: {text_sec}; text-decoration: none; font-size: 0.85rem; transition: color 0.3s;">Upload Chat</a></li>
                    <li style="margin-bottom: 0.7rem;"><a href="#" style="color: {text_sec}; text-decoration: none; font-size: 0.85rem;">View Analytics</a></li>
                    <li style="margin-bottom: 0.7rem;"><a href="#" style="color: {text_sec}; text-decoration: none; font-size: 0.85rem;">Sentiment Dashboard</a></li>
                </ul>
            </div>
            <div style="flex: 1; min-width: 160px;">
                <div style="color: {text_main}; font-weight: 600; margin-bottom: 1rem; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 0.08em;">Platform Values</div>
                <ul style="list-style: none; padding: 0; margin: 0;">
                    <li style="margin-bottom: 0.7rem; display: flex; align-items: center;"><span style="color: #00d2ff; margin-right: 0.5rem;">&#8226;</span><span style="font-size: 0.85rem;">100% Private</span></li>
                    <li style="margin-bottom: 0.7rem; display: flex; align-items: center;"><span style="color: #00d2ff; margin-right: 0.5rem;">&#8226;</span><span style="font-size: 0.85rem;">No Data Storage</span></li>
                    <li style="margin-bottom: 0.7rem; display: flex; align-items: center;"><span style="color: #00d2ff; margin-right: 0.5rem;">&#8226;</span><span style="font-size: 0.85rem;">Deep Insights</span></li>
                    <li style="margin-bottom: 0.7rem; display: flex; align-items: center;"><span style="color: #00d2ff; margin-right: 0.5rem;">&#8226;</span><span style="font-size: 0.85rem;">Smart Analytics</span></li>
                </ul>
            </div>
            <div style="flex: 1; min-width: 180px;">
                <div style="color: {text_main}; font-weight: 600; margin-bottom: 1rem; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 0.08em;">Get In Touch</div>
                <ul style="list-style: none; padding: 0; margin: 0;">
                    <li style="margin-bottom: 0.7rem; font-size: 0.85rem;">&#128205; India</li>
                    <li style="margin-bottom: 0.7rem; font-size: 0.85rem;">&#128222; +91 98765 43210</li>
                    <li style="margin-bottom: 0.7rem; font-size: 0.85rem;">&#9993; support@chatanalyzer.com</li>
                </ul>
            </div>
        </div>

        <div style="text-align: center; padding-top: 2rem; margin-top: 2rem; border-top: 1px solid {card_border}; font-size: 0.85rem;">
            Made with <span style="color: #e74c3c;">&#10084;</span> by <span style="color: #00d2ff;">Rudra</span> &amp; <span style="color: #00d2ff;">Keshav</span>
            <div style="display: flex; justify-content: center; gap: 1.5rem; margin: 1rem 0;">
                <a href="https://github.com/Rudrapratapsahoo" target="_blank" title="Rudra's GitHub" style="color: {text_sec}; transition: color 0.3s;">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
                </a>
                <a href="https://www.linkedin.com/in/rudra-pratap-sahoo-aba483291" target="_blank" title="Rudra's LinkedIn" style="color: {text_sec}; transition: color 0.3s;">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg>
                </a>
                <a href="https://github.com/Keshav3105" target="_blank" title="Keshav's GitHub" style="color: {text_sec}; transition: color 0.3s;">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
                </a>
                <a href="https://www.linkedin.com/in/keshav-gupta-a16236322/" target="_blank" title="Keshav's LinkedIn" style="color: {text_sec}; transition: color 0.3s;">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg>
                </a>
            </div>
            <span style="color: {text_sec};">&copy; 2026 ChatAnalyzer. All rights reserved.</span>
        </div>
    </div>
    """
    components.html(FOOTER_HTML, height=500, scrolling=True)


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 1 — LOGIN
# ══════════════════════════════════════════════════════════════════════════════
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["username"] = ""

if not st.session_state["logged_in"]:
    # ── Header ──
    col_th_empty, col_th = st.columns([12, 1])
    with col_th:
        theme_icon = "☀️" if st.session_state["theme"] == "dark" else "🌙"
        st.button(theme_icon, on_click=toggle_theme, key="theme_toggle_login", help="Toggle Theme")
        
    st.markdown(f"""
    <div style="text-align:center; padding-top: 4rem; padding-bottom: 2rem; animation: fadeInUp 0.8s ease-out;">
        <div style="display: inline-block; margin-bottom: 1rem; padding: 0.5rem 1.5rem; background: rgba(14, 165, 233, 0.1); color: #0ea5e9; border-radius: 50px; font-weight: 600; font-size: 0.9rem; letter-spacing: 0.05em; border: 1px solid rgba(14, 165, 233, 0.2);">
            ✨ THE ULTIMATE CHAT ANALYSIS TOOL
        </div>
        <h1 style="font-family: 'Inter', sans-serif; font-weight:900; font-size:5.5rem; color:{text_main}; margin-bottom: 1rem; letter-spacing: -0.04em; line-height: 1.1;">
            Unlock Insights from Your <br>
            <span style="background: linear-gradient(135deg, #0ea5e9, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Conversations</span>
        </h1>
        <p style="font-family: 'Inter', sans-serif; font-weight:500; font-size:1.25rem; color:{text_sec}; max-width: 600px; margin: 0 auto; line-height: 1.6;">
            Transform your raw chat exports into beautiful, interactive dashboards. Discover peak hours, sentiment trends, and who really talks the most.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Login Form ──
    col_empty1, col_login, col_empty2 = st.columns([1.5, 1, 1.5])
    with col_login:
        if not st.session_state.get("show_login_form", False):
            if st.button("Get Started Now ➜", use_container_width=True):
                st.session_state["show_login_form"] = True
                st.rerun()
        else:
            with st.form("login_form"):
                st.markdown("<h3 style='text-align: center; margin-bottom: 1rem;'>Login</h3>", unsafe_allow_html=True)
                email = st.text_input("Email", placeholder="Enter your email")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit = st.form_submit_button("Enter", use_container_width=True)
                if submit:
                    if email and password:
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = email.split("@")[0].capitalize() if "@" in email else email.capitalize()
                        st.rerun()
                    else:
                        st.error("Please enter email and password")

    st.markdown("<br><br><br>", unsafe_allow_html=True)

    # ── Key Features Grid ──
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 3rem; animation: fadeInUp 1s ease-out;">
        <h2 style="font-family: 'Inter', sans-serif; font-weight: 800; font-size: 2.5rem; color: {text_main};">Powerful Features</h2>
        <p style="color: {text_sec}; font-size: 1.1rem;">Everything you need to understand your digital relationships.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center; height: 100%;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🔒</div>
            <h3 style="color: {text_main}; font-size: 1.3rem; font-weight: 700; margin-bottom: 0.5rem;">Privacy First</h3>
            <p style="color: {text_sec}; font-size: 0.95rem; line-height: 1.5;">Your data never leaves your browser. 100% local processing for absolute peace of mind.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-card" style="text-align: center; height: 100%;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">☁️</div>
            <h3 style="color: {text_main}; font-size: 1.3rem; font-weight: 700; margin-bottom: 0.5rem;">Word Clouds</h3>
            <p style="color: {text_sec}; font-size: 0.95rem; line-height: 1.5;">Instantly visualize your most frequently used vocabulary and favorite emojis.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center; height: 100%;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📈</div>
            <h3 style="color: {text_main}; font-size: 1.3rem; font-weight: 700; margin-bottom: 0.5rem;">Timeline Analytics</h3>
            <p style="color: {text_sec}; font-size: 0.95rem; line-height: 1.5;">Track your conversation activity over months and days with interactive charts.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-card" style="text-align: center; height: 100%;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">😊</div>
            <h3 style="color: {text_main}; font-size: 1.3rem; font-weight: 700; margin-bottom: 0.5rem;">Sentiment Analysis</h3>
            <p style="color: {text_sec}; font-size: 0.95rem; line-height: 1.5;">Evaluate the emotional tone of messages to see who brings the most positivity.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center; height: 100%;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🔥</div>
            <h3 style="color: {text_main}; font-size: 1.3rem; font-weight: 700; margin-bottom: 0.5rem;">Activity Heatmap</h3>
            <p style="color: {text_sec}; font-size: 0.95rem; line-height: 1.5;">Discover exactly which days and hours you and your friends are most active.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-card" style="text-align: center; height: 100%;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🏆</div>
            <h3 style="color: {text_main}; font-size: 1.3rem; font-weight: 700; margin-bottom: 0.5rem;">User Leaderboards</h3>
            <p style="color: {text_sec}; font-size: 0.95rem; line-height: 1.5;">Find out who sends the most messages, links, and media in your group chats.</p>
        </div>
        """, unsafe_allow_html=True)

    # Spacer to push footer to bottom of viewport
    st.markdown("<div style='min-height: 50vh;'></div>", unsafe_allow_html=True)
    render_footer()
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 2 & 3 — UPLOAD / DASHBOARD (logged in)
# ══════════════════════════════════════════════════════════════════════════════

# ── Top bar ───────────────────────────────────────────────────────────────────
top_l, top_t, top_r = st.columns([10, 1, 1])
with top_l:
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 1rem; padding: 0.5rem 0;">
        <span style="font-size: 1.5rem; font-weight: 700;">
            <span style="background: linear-gradient(135deg, #00d2ff, #7b2ff7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                💬 ChatAnalyzer
            </span>
        </span>
        <span class="user-badge" style="color: {text_main};">👋 {st.session_state['username']}</span>
    </div>
    """, unsafe_allow_html=True)
with top_t:
    theme_icon = "☀️" if st.session_state["theme"] == "dark" else "🌙"
    st.button(theme_icon, on_click=toggle_theme, key="theme_toggle_main", help="Toggle Theme")
with top_r:
    if st.button("Logout", help="Logout and return to home"):
        st.session_state["logged_in"] = False
        st.session_state["show_login_form"] = False
        st.session_state["username"] = ""
        st.session_state.pop("df", None)
        st.session_state.pop("last_file", None)
        st.session_state.pop("analyzed", None)
        st.rerun()

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ── File upload (centered) ────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state["df"] = None

col_u1, col_u2, col_u3 = st.columns([1, 3, 1])
with col_u2:
    uploaded_file = st.file_uploader(
        "📁 Upload your WhatsApp chat export (.txt)",
        type=["txt"],
        help="WhatsApp → Chat → Export Chat → Without Media"
    )

    if uploaded_file:
        fname = uploaded_file.name
        if st.session_state.get("last_file") != fname:
            with st.spinner("Parsing chat…"):
                try:
                    data = uploaded_file.read().decode("utf-8")
                    df_raw = preprocess(data)
                    st.session_state["df"] = df_raw
                    st.session_state["last_file"] = fname
                    st.success(f"✅ {len(df_raw):,} messages loaded from **{fname}**")
                except Exception as e:
                    st.error(f"Parse error: {e}")
                    st.session_state["df"] = None
                    st.session_state["last_file"] = fname

df_all = st.session_state.get("df")

# ── User selector + Analyze button (centered) ────────────────────────────────
if df_all is not None and not df_all.empty:
    col_s1, col_s2, col_s3 = st.columns([1, 3, 1])
    with col_s2:
        user_list = ["Overall"] + sorted(df_all["user"].unique().tolist())
        selected_user = st.selectbox("🎯 Analyze for", user_list)
        analyze = st.button("🔍 Analyze Chat")
else:
    selected_user = "Overall"
    analyze = False


# ── Gate: show landing page until Analyze is clicked ──────────────────────────
if not analyze and "analyzed" not in st.session_state:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; padding: 2rem 0; animation: fadeInUp 0.5s ease-out forwards;">
      <div class="hero-title">Unlock the secrets of your chats 🔓</div>
      <div class="hero-subtitle">
        Discover who talks the most, what words you use, your emotional tone, and when you are most active.
      </div>

      <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; margin-top: 1rem;">
        <div class="feature-card">
          <div style="font-size: 2.5rem; margin-bottom: 1rem; animation: float 3s ease-in-out infinite;">🔒</div>
          <h4>100% Private</h4>
          <p>We don't store your data. Everything is processed directly in your browser.</p>
        </div>
        <div class="feature-card">
          <div style="font-size: 2.5rem; margin-bottom: 1rem; animation: float 3s ease-in-out 0.5s infinite;">📊</div>
          <h4>Deep Insights</h4>
          <p>Timelines, heatmaps, and word clouds to visualize your conversation history.</p>
        </div>
        <div class="feature-card">
          <div style="font-size: 2.5rem; margin-bottom: 1rem; animation: float 3s ease-in-out 1s infinite;">😊</div>
          <h4>Sentiment & Emojis</h4>
          <p>Understand the mood of the chat and see your favorite emojis at a glance.</p>
        </div>
      </div>

      <div class="steps-card">
        <h4>📱 How to export your WhatsApp chat:</h4>
        <ol>
          <li>Open WhatsApp on your phone.</li>
          <li>Go to the chat you want to analyze.</li>
          <li>Tap the <b>three dots</b> (menu) → <b>More</b> → <b>Export Chat</b>.</li>
          <li>Choose <b>Without Media</b>.</li>
          <li>Upload the <code>.txt</code> file above!</li>
        </ol>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Spacer to push footer to bottom of viewport
    st.markdown("<div style='min-height: 35vh;'></div>", unsafe_allow_html=True)
    render_footer()
    st.stop()

st.session_state["analyzed"] = True

# ── Filter by user ─────────────────────────────────────────────────────────────
df = df_all if selected_user == "Overall" else df_all[df_all["user"] == selected_user]

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="dash-header">
    <div class="dash-title">📊 Dashboard</div>
    <div class="dash-subtitle">Analyzing: <strong style="color:#00d2ff;">{selected_user}</strong></div>
</div>
<div class="gradient-divider"></div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# KPI CARDS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">Overview</div>', unsafe_allow_html=True)

total_msgs, total_words, media_cnt, links_cnt = helper.calc_stats(df)

c1, c2, c3, c4 = st.columns(4)
for col, icon, label, val in [
    (c1, "💬", "Messages",     f"{total_msgs:,}"),
    (c2, "📝", "Words",        f"{total_words:,}"),
    (c3, "🖼️", "Media Shared", f"{media_cnt:,}"),
    (c4, "🔗", "Links Shared", f"{links_cnt:,}"),
]:
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{val}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ══════════════════════════════════════════════════════════════════════════════
grid_col = "rgba(0,0,0,0.1)" if st.session_state["theme"] == "light" else "rgba(255,255,255,0.1)"
PLOT_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color=text_main, family="Inter"),
    xaxis=dict(gridcolor=grid_col),
    yaxis=dict(gridcolor=grid_col),
)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
t1, t2, t3, t4, t5, t6 = st.tabs([
    "📅 Timeline", "⏰ Activity", "👥 Users", "💬 Words", "😂 Emoji", "😊 Sentiment"
])

# ─── TAB 1 — Timeline ────────────────────────────────────────────────────────
with t1:
    L, R = st.columns(2)

    with L:
        st.markdown('<div class="sec-header">Monthly Timeline</div>', unsafe_allow_html=True)
        mt = helper.monthly_timeline(df)
        fig = px.area(mt, x="label", y="count", markers=True,
                      color_discrete_sequence=[bar_accent],
                      labels={"label": "Month", "count": "Messages"})
        fig.update_layout(**PLOT_LAYOUT, xaxis_title="", yaxis_title="Messages",
                          margin=dict(l=0, r=0, t=20, b=0))
        fig.update_traces(fillcolor="rgba(14, 165, 233, 0.15)",
                          line=dict(width=2.5), marker=dict(size=6),
                          hovertemplate='<b>%{x}</b><br>Messages: %{y:,}<extra></extra>')
        st.plotly_chart(fig, use_container_width=True)

    with R:
        st.markdown('<div class="sec-header">Daily Timeline</div>', unsafe_allow_html=True)
        dt = helper.daily_timeline(df)
        fig = px.line(dt, x="only_date", y="count",
                      color_discrete_sequence=[bar_accent],
                      labels={"only_date": "Date", "count": "Messages"})
        fig.update_traces(fill='tozeroy', fillcolor='rgba(14, 165, 233, 0.1)',
                          line=dict(width=1.5),
                          hovertemplate='<b>%{x}</b><br>Messages: %{y:,}<extra></extra>')
        fig.update_layout(**PLOT_LAYOUT, xaxis_title="", yaxis_title="Messages",
                          margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

# ─── TAB 2 — Activity ────────────────────────────────────────────────────────
with t2:
    L, R = st.columns(2)

    with L:
        st.markdown('<div class="sec-header">Busiest Day of Week</div>', unsafe_allow_html=True)
        wd = helper.week_activity(df)
        wd.columns = ["day","count"]
        max_idx = wd["count"].idxmax()
        colors = [bar_muted] * len(wd)
        colors[max_idx] = bar_accent

        fig = px.bar(wd, y="day", x="count", orientation='h',
                     labels={"day": "", "count": "Messages"})
        fig.update_traces(marker_color=colors, marker_line_width=0, opacity=0.9,
                          hovertemplate='<b>%{y}</b><br>Messages: %{x:,}<extra></extra>')
        fig.update_layout(**PLOT_LAYOUT, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with R:
        st.markdown('<div class="sec-header">Hour × Day Heatmap</div>', unsafe_allow_html=True)
        hm = helper.heatmap_data(df)
        fig = px.imshow(hm, color_continuous_scale=[gradient_start, gradient_end],
                        aspect="auto",
                        labels=dict(x="Hour of Day", y="Day of Week", color="Messages"))
        fig.update_layout(**PLOT_LAYOUT, margin=dict(l=0, r=0, t=20, b=0),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

# ─── TAB 3 — Users ───────────────────────────────────────────────────────────
with t3:
    if selected_user != "Overall":
        st.info(f"User breakdown is only available for **Overall**. Currently: **{selected_user}**")
    else:
        st.markdown('<div class="sec-header">Most Active Users</div>', unsafe_allow_html=True)
        tu, pct = helper.top_users(df_all)
        L, R = st.columns([3, 2])
        with L:
            fig = px.bar(tu.sort_values('count', ascending=True),
                         y="user", x="count", orientation='h',
                         color="count",
                         color_continuous_scale=[gradient_start, gradient_end],
                         labels={"user": "", "count": "Messages"})
            fig.update_traces(hovertemplate='<b>%{y}</b><br>Messages: %{x:,}<extra></extra>')
            fig.update_layout(**PLOT_LAYOUT,
                              margin=dict(l=0, r=0, t=0, b=0), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with R:
            st.dataframe(pct, width='stretch', hide_index=True)

# ─── TAB 4 — Words ───────────────────────────────────────────────────────────
with t4:
    L, R = st.columns(2)

    with L:
        st.markdown('<div class="sec-header">Word Cloud</div>', unsafe_allow_html=True)
        wc  = helper.make_wordcloud(df)
        fig = px.imshow(wc)
        fig.update_layout(coloraxis_showscale=False, margin=dict(l=0,r=0,t=0,b=0),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        fig.update_xaxes(showticklabels=False); fig.update_yaxes(showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)

    with R:
        st.markdown('<div class="sec-header">Top 20 Words</div>', unsafe_allow_html=True)
        tw = helper.top_words(df)
        fig = px.bar(tw.sort_values('count', ascending=True),
                     y="word", x="count", orientation='h',
                     color="count",
                     color_continuous_scale=[gradient_start, gradient_end],
                     labels={"word": "", "count": "Occurrences"})
        fig.update_traces(hovertemplate='<b>%{y}</b><br>Count: %{x:,}<extra></extra>')
        fig.update_layout(**PLOT_LAYOUT,
                          margin=dict(l=0, r=0, t=0, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

# ─── TAB 5 — Emoji ───────────────────────────────────────────────────────────
with t5:
    st.markdown('<div class="sec-header">Emoji Analysis</div>', unsafe_allow_html=True)
    em = helper.emoji_stats(df)
    if em.empty:
        st.info("No emojis found in this chat.")
    else:
        L, R = st.columns([1, 2])
        with L:
            st.dataframe(em.head(20), width='stretch', hide_index=True)
        with R:
            top8 = em.head(8)
            fig = px.pie(top8, values="Count", names="Emoji", hole=0.45,
                         color_discrete_sequence=[border_hover, "#38bdf8", "#0ea5e9", text_sec, border_main, grid_col, "#94a3b8", "#cbd5e1"])
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                              paper_bgcolor="rgba(0,0,0,0)", font=dict(color=text_main))
            st.plotly_chart(fig, use_container_width=True)

# ─── TAB 6 — Sentiment ───────────────────────────────────────────────────────
with t6:
    st.markdown('<div class="sec-header">💡 Sentiment Analysis</div>', unsafe_allow_html=True)
    per_user_sent, scored_df = helper.sentiment_detailed(df_all)
    per_user_sent = per_user_sent[per_user_sent["user"] != "group_notification"]
    scored_clean = scored_df[scored_df["user"] != "group_notification"]

    # Overall sentiment KPIs
    overall_avg = scored_clean["compound"].mean() if len(scored_clean) > 0 else 0
    pos_total = int((scored_clean["sentiment_label"] == "Positive").sum())
    neg_total = int((scored_clean["sentiment_label"] == "Negative").sum())
    neu_total = int((scored_clean["sentiment_label"] == "Neutral").sum())
    mood_emoji = "😊" if overall_avg >= 0.05 else ("😔" if overall_avg <= -0.05 else "😐")

    k1, k2, k3, k4 = st.columns(4)
    for col, icon, lbl, val in [
        (k1, mood_emoji, "Overall Mood", f"{overall_avg:+.3f}"),
        (k2, "😊", "Positive Msgs", f"{pos_total:,}"),
        (k3, "😐", "Neutral Msgs", f"{neu_total:,}"),
        (k4, "😔", "Negative Msgs", f"{neg_total:,}"),
    ]:
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{lbl}</div>
            <div class="kpi-value">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row
    L, R = st.columns(2)
    with L:
        st.markdown('<div class="sec-header">Average Score per User</div>', unsafe_allow_html=True)
        bar_colors = [sentiment_pos if v >= 0 else sentiment_neg for v in per_user_sent["avg_score"]]
        fig = go.Figure(go.Bar(
            x=per_user_sent["user"], y=per_user_sent["avg_score"],
            marker_color=bar_colors, opacity=0.9,
            hovertemplate='<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>'
        ))
        fig.add_hline(y=0, line_dash="dash", line_color=text_sec, opacity=0.5)
        fig.update_layout(**PLOT_LAYOUT, margin=dict(l=0, r=0, t=20, b=0),
                          xaxis_title="", yaxis_title="Avg Compound Score")
        st.plotly_chart(fig, use_container_width=True)

    with R:
        st.markdown('<div class="sec-header">Sentiment Distribution (%)</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Positive", x=per_user_sent["user"], y=per_user_sent["pos_pct"],
                             marker_color=sentiment_pos,
                             hovertemplate='%{x}<br>Positive: %{y:.1f}%<extra></extra>'))
        fig.add_trace(go.Bar(name="Neutral", x=per_user_sent["user"], y=per_user_sent["neu_pct"],
                             marker_color=sentiment_neu,
                             hovertemplate='%{x}<br>Neutral: %{y:.1f}%<extra></extra>'))
        fig.add_trace(go.Bar(name="Negative", x=per_user_sent["user"], y=per_user_sent["neg_pct"],
                             marker_color=sentiment_neg,
                             hovertemplate='%{x}<br>Negative: %{y:.1f}%<extra></extra>'))
        fig.update_layout(**PLOT_LAYOUT, barmode='stack',
                          margin=dict(l=0, r=0, t=20, b=0),
                          xaxis_title="", yaxis_title="Percentage (%)",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
        st.plotly_chart(fig, use_container_width=True)

    # Detailed per-person table
    st.markdown('<div class="sec-header">📋 Detailed Sentiment per Person</div>', unsafe_allow_html=True)
    tbl = per_user_sent[["user", "avg_score", "pos_count", "neu_count", "neg_count", "total", "pos_pct", "neg_pct"]].copy()
    tbl.columns = ["User", "Avg Score", "😊 Positive", "😐 Neutral", "😔 Negative", "Total Msgs", "Pos %", "Neg %"]
    tbl["Avg Score"] = tbl["Avg Score"].round(3)
    tbl["Mood"] = tbl["Avg Score"].apply(lambda x: "😊" if x >= 0.05 else ("😔" if x <= -0.05 else "😐"))
    tbl = tbl[["User", "Mood", "Avg Score", "😊 Positive", "😐 Neutral", "😔 Negative", "Total Msgs", "Pos %", "Neg %"]]
    st.dataframe(tbl, use_container_width=True, hide_index=True)
    st.caption("VADER Sentiment · Score ≥ 0.05 → Positive · Score ≤ −0.05 → Negative · Range: −1 to +1")


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

col_e1, col_e2, col_e3 = st.columns([1, 2, 1])
with col_e2:
    csv = df[["date","user","message"]].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Export filtered messages as CSV",
        data=csv,
        file_name=f"chat_{selected_user}.csv",
        mime="text/csv",
    )

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='min-height: 25vh;'></div>", unsafe_allow_html=True)
render_footer()