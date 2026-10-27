"""
CreatorLens AI  ·  Understand · Analyze · Grow
A Multilingual Social Media Intelligence and Audience Understanding System

Run:   streamlit run app.py
Needs: streamlit, pandas, numpy, scikit-learn, altair

Pipeline
  Module 1  Preprocessing and normalisation  (clean text, emoji features, language tagger)
  Module 2  Sentiment and feedback-intent classifier (heuristic stand-in for XLM-R / MuRIL)
  Module 3  Semantic grouping and FAQ extraction (char n-gram TF-IDF + cosine similarity)
  Module 4  Content opportunity ranker (prioritisation engine)

Look and feel: "Sunset Silhouette" palette
  espresso #291A17 · copper #B8713B · cream #F7D89E · slate #A9ACB3
"""

import datetime as dt
import html
import re

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="CreatorLens AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 1. THEME: SUNSET SILHOUETTE
# =============================================================================
C = dict(
    bg="#FBF3E4", card="#FFFDF8", border="#EBDCC2", ink="#291A17", muted="#7B6A60",
    copper="#B8713B", copper_d="#8F4F25", cream="#F7D89E", slate="#A9ACB3", slate_d="#5E6572",
    sage="#6F8F72", brick="#B5473A", gold="#E0A94A",
)

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root{
  --bg:#FBF3E4; --card:#FFFDF8; --border:#EBDCC2; --ink:#291A17; --muted:#7B6A60;
  --copper:#B8713B; --copperd:#8F4F25; --cream:#F7D89E; --slate:#A9ACB3; --slated:#5E6572;
  --sage:#6F8F72; --brick:#B5473A; --gold:#E0A94A;
}
html, body, .stApp, [class*="css"]{ font-family:'Plus Jakarta Sans','Inter','Segoe UI',sans-serif; }
.stApp{ background:var(--bg); color:var(--ink); }
header[data-testid="stHeader"]{ background:transparent; }
#MainMenu, footer{ visibility:hidden; }
.block-container{ padding-top:1.4rem; padding-bottom:3rem; padding-left:1.8rem; padding-right:1.8rem; max-width:1560px; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li{ color:var(--ink); }
h1,h2,h3,h4{ color:var(--ink); letter-spacing:-0.01em; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"]{ background:#FFF9EE; border-right:1px solid var(--border); }
[data-testid="stSidebar"] .block-container{ padding-top:1rem; }
.brand{ display:flex; align-items:center; gap:12px; padding:4px 6px 16px 6px; }
.brand-logo{ width:42px; height:42px; border-radius:13px; display:flex; align-items:center; justify-content:center;
  background:linear-gradient(145deg,#8F4F25,#B8713B 60%,#E4A862); color:#FFF4DC; font-size:20px; box-shadow:0 6px 16px rgba(143,79,37,.30); }
.brand-name{ font-weight:800; font-size:18px; line-height:1.1; color:var(--ink); }
.brand-name span{ color:var(--copper); }
.brand-sub{ font-size:11px; color:var(--muted); margin-top:3px; }
.side-label{ font-size:12px; font-weight:700; color:var(--muted); margin:22px 0 8px 8px; }
[data-testid="stSidebar"] [role="radiogroup"]{ gap:3px; }
[data-testid="stSidebar"] [role="radiogroup"] label{
  background:transparent; border:1px solid transparent; border-radius:11px; padding:10px 14px; width:100%; transition:background .15s ease; cursor:pointer; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover{ background:#F8ECD6; }
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-of-type > div:first-child{ display:none; }
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child:not(:has(p)){ display:none; }
[data-testid="stSidebar"] [role="radiogroup"] label p{ color:#5A4A42; font-weight:500; font-size:14px; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){ background:#F7E3BD; border:1px solid #EBCB91; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p{ color:var(--copperd); font-weight:700; }
.plat{ display:flex; align-items:center; gap:10px; padding:7px 10px; font-size:14px; font-weight:500; color:#5A4A42; }
.plat i{ width:22px; height:22px; border-radius:7px; display:inline-flex; align-items:center; justify-content:center; font-style:normal; font-size:11px; color:#fff; }
.side-promo{ margin-top:26px; border-radius:16px; padding:18px 16px 20px; color:#fff;
  background:radial-gradient(circle at 85% 12%, rgba(247,216,158,.9) 0, rgba(247,216,158,0) 38%), linear-gradient(160deg,#B8713B 0%,#8F4F25 100%); }
.side-promo b{ display:block; font-size:17px; line-height:1.3; font-weight:800; margin-top:8px; }
.side-promo .crown{ font-size:20px; color:#FCE3B2; }
.side-user{ display:flex; align-items:center; gap:10px; margin-top:18px; padding:10px 6px; border-top:1px solid var(--border); }
.avatar{ width:36px; height:36px; border-radius:50%; background:linear-gradient(145deg,#B8713B,#F7D89E); color:#291A17; font-weight:800; font-size:13px;
  display:flex; align-items:center; justify-content:center; flex:none; }
.side-user b{ display:block; font-size:13px; color:var(--ink); }
.side-user span{ font-size:11px; color:var(--muted); }

/* ---------- Widgets ---------- */
[data-baseweb="select"] > div, [data-baseweb="input"] > div{ background:#FFFDF8 !important; border-color:var(--border) !important; border-radius:11px !important; }
[data-baseweb="tag"]{ background:#F7E3BD !important; color:#8F4F25 !important; border-radius:8px !important; }
input{ color:var(--ink) !important; }
.stDownloadButton button, .stButton button{
  background:linear-gradient(135deg,#B8713B,#8F4F25); color:#fff; border:none; border-radius:11px; font-weight:700;
  padding:.55rem 1.1rem; box-shadow:0 6px 16px rgba(143,79,37,.28); transition:transform .1s ease; }
.stDownloadButton button:hover, .stButton button:hover{ transform:translateY(-1px); color:#fff; border:none; }
.stDownloadButton button p, .stButton button p{ color:#fff !important; }
button[role="tab"] p{ color:var(--muted); font-weight:600; }
button[role="tab"][aria-selected="true"] p{ color:var(--copperd); }
[data-baseweb="tab-highlight"]{ background:var(--copper) !important; }

/* ---------- Cards ---------- */
.card, [class*="st-key-card_"]{
  background:var(--card); border:1px solid var(--border); border-radius:16px; padding:18px 20px;
  box-shadow:0 1px 2px rgba(41,26,23,.04), 0 10px 26px rgba(184,113,59,.07); }
.card{ margin-bottom:14px; }
.dash .card{ margin-bottom:0; }
.g-sent{ display:flex; flex-direction:column; }
.g-sent .split{ flex:1; }
.card-head{ display:flex; justify-content:space-between; align-items:flex-start; gap:10px; margin-bottom:12px; }
.card-title{ font-size:15px; font-weight:700; color:var(--ink); display:flex; align-items:center; gap:9px; }
.card-title .ti{ width:26px; height:26px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; background:#F7E3BD; color:var(--copperd); font-size:13px; }
.card-sub{ font-size:12px; color:var(--muted); margin-top:2px; }
.stApp a.view-all, .stApp a.view-all:visited{ color:var(--copper) !important; }
.view-all{ font-size:12px; font-weight:700; color:var(--copper); white-space:nowrap; text-decoration:none !important; }
.legend{ display:flex; gap:14px; font-size:12px; color:var(--muted); }
.legend i{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; }

.page-head{ display:flex; justify-content:space-between; align-items:center; margin-bottom:18px; flex-wrap:wrap; gap:10px; }
.page-title{ font-size:26px; font-weight:800; color:var(--ink); letter-spacing:-0.02em; }
.page-sub{ font-size:13px; color:var(--muted); margin-top:3px; }
.head-chips{ display:flex; gap:8px; flex-wrap:wrap; }
.hello{ font-size:25px; font-weight:800; color:var(--ink); letter-spacing:-0.02em; }
.hello-sub{ font-size:13px; color:var(--muted); margin-top:3px; }

/* ---------- Dashboard grid ---------- */
.dash{ display:grid; grid-template-columns:repeat(12,minmax(0,1fr)); gap:16px; align-items:stretch; margin-top:8px; }
.g-banner{ grid-column:1/-1; grid-row:1; }
.g-kpi{ grid-column:span 3; grid-row:2; }
.g-sent{ grid-column:1/10; grid-row:3; }
.g-topics{ grid-column:10/13; grid-row:3; }
.g-lang{ grid-column:1/5; grid-row:4; }
.g-opps{ grid-column:5/10; grid-row:4; }
.g-high{ grid-column:10/13; grid-row:4; }
.g-qs{ grid-column:1/6; grid-row:5; }
.g-expl{ grid-column:6/10; grid-row:5; }
.g-stats{ grid-column:10/13; grid-row:5; }
@media (max-width:1250px){
  .g-sent,.g-topics,.g-lang,.g-opps,.g-high,.g-qs,.g-expl,.g-stats{ grid-column:1/-1; grid-row:auto; }
  .g-kpi{ grid-column:span 6; grid-row:auto; }
}
@media (max-width:640px){ .g-kpi{ grid-column:1/-1; } }

/* ---------- Banner ---------- */
.banner{ position:relative; overflow:hidden; border-radius:20px; min-height:200px; padding:28px 32px; display:flex; align-items:center;
  background:linear-gradient(100deg,#8F4F25 0%,#B8713B 36%,#E4A862 68%,#F7D89E 100%); box-shadow:0 14px 34px rgba(143,79,37,.25); }
.banner-copy{ position:relative; z-index:2; max-width:46%; }
.banner h2{ margin:0 0 8px 0; font-size:30px; line-height:1.15; font-weight:800; color:#FFF8EA; letter-spacing:-0.02em; }
.banner p{ margin:0 0 18px 0; font-size:13px; line-height:1.6; color:#FBE8C6; max-width:430px; }
.banner-btn{ display:inline-block; background:#FFFDF8; color:#8F4F25 !important; font-weight:800; font-size:13px; padding:10px 20px; border-radius:999px;
  text-decoration:none !important; box-shadow:0 6px 16px rgba(41,26,23,.22); }
.banner svg{ position:absolute; right:0; bottom:0; height:100%; width:58%; z-index:1; }
@media (max-width:900px){ .banner-copy{ max-width:100%; } .banner svg{ opacity:.35; } }

/* ---------- KPI cards ---------- */
.kpi{ display:flex; align-items:center; gap:14px; padding:18px 20px; border-radius:16px; border:1px solid; }
.kpi .ic{ width:46px; height:46px; border-radius:50%; display:flex; align-items:center; justify-content:center; flex:none; }
.kpi .ic svg{ width:22px; height:22px; stroke:currentColor; fill:none; stroke-width:2; stroke-linecap:round; stroke-linejoin:round; }
.kpi-label{ font-size:13px; font-weight:600; color:#5A4A42; }
.kpi-value{ font-size:30px; font-weight:800; color:var(--ink); letter-spacing:-0.02em; line-height:1.15; margin:2px 0; }
.kpi-delta{ font-size:12px; color:var(--muted); }
.kpi-delta b{ font-weight:700; }
.k-slate{ background:#F0F1F3; border-color:#DEE0E4; } .k-slate .ic{ background:#C9CCD2; color:#3F4652; }
.k-sage{ background:#EBF1E8; border-color:#D4E1D0; } .k-sage .ic{ background:#B9D0B8; color:#3D6141; }
.k-cream{ background:#FCEFD0; border-color:#F3DDA6; } .k-cream .ic{ background:#F2CF84; color:#7B5210; }
.k-copper{ background:#F8E3D3; border-color:#EDC8AA; } .k-copper .ic{ background:#E3A77A; color:#7A3A12; }
.up{ color:#456B49; } .down{ color:#9A3A2E; }

/* ---------- Donut + trend ---------- */
.split{ display:grid; grid-template-columns:minmax(0,1.05fr) minmax(0,1.25fr); gap:18px; align-items:center; }
@media (max-width:760px){ .split{ grid-template-columns:1fr; } }
.donut-row{ display:flex; align-items:center; gap:16px; flex-wrap:wrap; }
.donut{ width:150px; height:150px; flex:none; }
.donut-col{ display:flex; flex-direction:column; align-items:center; gap:12px; }
.leg-list{ display:flex; flex-direction:column; gap:9px; font-size:13px; color:#4A3B34; min-width:130px; }
.leg-list div{ display:flex; align-items:center; gap:8px; }
.leg-list i{ width:10px; height:10px; border-radius:50%; flex:none; }
.leg-list .pc{ margin-left:auto; font-weight:700; color:var(--ink); padding-left:12px; }
.leg-list .num{ color:var(--muted); font-weight:600; width:12px; }
.trend-title{ font-size:13px; font-weight:700; color:var(--ink); margin-bottom:4px; }
.trend{ width:100%; height:auto; display:block; }
.trend-leg{ display:flex; gap:16px; justify-content:center; font-size:12px; color:var(--muted); margin-top:4px; }
.trend-leg i{ display:inline-block; width:14px; height:3px; border-radius:2px; margin-right:6px; vertical-align:middle; }

/* ---------- Topics ---------- */
.topic{ display:grid; grid-template-columns:26px minmax(0,1fr); gap:12px; align-items:start; padding:9px 0; }
.topic + .topic{ border-top:1px solid #F3E8D2; }
.topic .rk{ width:26px; height:26px; border-radius:8px; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:12px; }
.topic .tn{ display:flex; justify-content:space-between; gap:8px; align-items:baseline; }
.topic .nm{ font-size:13px; font-weight:600; color:var(--ink); line-height:1.3; }
.topic .pc{ font-size:13px; font-weight:700; color:var(--ink); }
.topic .ct{ font-size:11px; color:var(--muted); margin-top:5px; }
.pbar{ height:6px; border-radius:99px; background:#F3E8D2; overflow:hidden; margin-top:6px; }
.pbar span{ display:block; height:100%; border-radius:99px; background:linear-gradient(90deg,#B8713B,#E0A94A); }

/* ---------- Opportunities ---------- */
.opp{ display:grid; grid-template-columns:70px 1fr auto; gap:12px; align-items:center; padding:9px 0; }
.opp + .opp{ border-top:1px solid #F3E8D2; }
.thumb{ width:70px; height:46px; border-radius:10px; position:relative; overflow:hidden; display:flex; align-items:center; justify-content:center; color:#FFF8EA; font-size:15px; }
.opp .t{ font-size:13px; font-weight:700; color:var(--ink); line-height:1.35; }
.opp .m{ font-size:11px; color:var(--muted); margin-top:2px; }
.score{ min-width:36px; text-align:center; font-weight:800; font-size:13px; border-radius:999px; padding:6px 10px; }
.sc-hi{ background:#DCE8D9; color:#3D6141; } .sc-mid{ background:#F8E3B4; color:#7B5210; } .sc-lo{ background:#F6D9C4; color:#8F4F25; }

/* ---------- Highlights ---------- */
.hl{ display:grid; grid-template-columns:34px 1fr auto; gap:12px; align-items:center; padding:9px 0; }
.hl + .hl{ border-top:1px solid #F3E8D2; }
.hl .ic{ width:34px; height:34px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:14px; font-weight:800; }
.hl .t{ font-size:13px; font-weight:700; color:var(--ink); } .hl .d{ font-size:11px; color:var(--muted); margin-top:1px; }
.hl .w{ font-size:11px; color:var(--muted); white-space:nowrap; }

/* ---------- Comments (dashboard list) ---------- */
.cm{ display:grid; grid-template-columns:34px 1fr; gap:10px; padding:9px 0; }
.cm + .cm{ border-top:1px solid #F3E8D2; }
.cm .avatar{ width:34px; height:34px; font-size:12px; }
.cm .top{ display:flex; align-items:center; gap:6px; font-size:12px; font-weight:700; color:var(--ink); flex-wrap:wrap; }
.cm .top span{ font-weight:500; color:var(--muted); font-size:11px; }
.cm .top .when{ margin-left:auto; }
.cm .tx{ font-size:12px; color:#4A3B34; line-height:1.5; margin:3px 0 6px 0; }

/* ---------- Quick stats ---------- */
.qs-grid{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }
.qs{ background:#FBF3E4; border:1px solid #F0E1C4; border-radius:12px; padding:12px; display:flex; gap:10px; align-items:center; }
.qs .ic{ width:30px; height:30px; border-radius:9px; display:flex; align-items:center; justify-content:center; font-size:13px; font-weight:800; flex:none; }
.qs b{ display:block; font-size:18px; color:var(--ink); line-height:1.1; } .qs span{ font-size:11px; color:var(--muted); line-height:1.3; display:block; }
.qs-foot{ margin-top:12px; border-radius:12px; padding:14px; text-align:center; font-weight:700; font-size:13px; color:#6B3A18;
  background:linear-gradient(120deg,#F7D89E,#E8B987 55%,#C9CCD2); }

/* ---------- Chips ---------- */
.chip{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:11px; font-weight:700; margin:1px 4px 1px 0; white-space:nowrap; border:1px solid; }
.chip-red{ background:rgba(181,71,58,.12); color:#9A3A2E; border-color:rgba(181,71,58,.30); }
.chip-coral{ background:rgba(184,113,59,.14); color:#8F4F25; border-color:rgba(184,113,59,.35); }
.chip-amber{ background:rgba(224,169,74,.20); color:#7B5210; border-color:rgba(224,169,74,.45); }
.chip-green{ background:rgba(111,143,114,.18); color:#3D6141; border-color:rgba(111,143,114,.40); }
.chip-blue{ background:rgba(169,172,179,.25); color:#454C59; border-color:rgba(169,172,179,.55); }
.chip-gray{ background:#F4EBDA; color:#6B5B52; border-color:#E6D8BE; }

/* ---------- Tables ---------- */
.table-wrap{ width:100%; overflow-x:auto; }
.lens-table{ width:100%; border-collapse:collapse; font-size:13px; color:var(--ink); }
.lens-table, .lens-table th, .lens-table td{ border-left:0; border-right:0; border-top:0; background:transparent; }
.lens-table th{ text-align:left; color:var(--muted); font-weight:700; font-size:12px; padding:9px 10px; border-bottom:1px solid var(--border); white-space:nowrap; }
.lens-table td{ padding:12px 10px; border-bottom:1px solid #F3E8D2; vertical-align:top; }
.lens-table tr:hover td{ background:#FBF3E4; }
.lens-table tr.row-high td:first-child{ box-shadow:inset 3px 0 0 var(--brick); }
.lens-table tr.row-med td:first-child{ box-shadow:inset 3px 0 0 var(--gold); }
.lens-table.compact td{ padding:10px 8px; vertical-align:middle; }
.idea-title{ font-weight:700; color:var(--ink); font-size:14px; } .idea-sub{ font-size:11px; color:var(--muted); margin-top:3px; }
.score-num{ font-weight:800; font-size:18px; color:var(--ink); }
.bar{ height:6px; background:#F3E8D2; border-radius:99px; overflow:hidden; margin-top:6px; min-width:110px; }
.bar > span{ display:block; height:100%; background:linear-gradient(90deg,#B8713B,#E0A94A); border-radius:99px; }
.evidence{ font-size:12px; color:#4A3B34; line-height:1.5; margin-bottom:6px; max-width:420px; } .ev-meta{ color:var(--muted); font-size:11px; }
.rank{ width:28px; height:28px; border-radius:8px; background:#F7E3BD; border:1px solid #EBCB91; color:#8F4F25; font-weight:800; display:flex; align-items:center; justify-content:center; }
.faq-grid{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:12px; margin-top:8px; }
.faq{ background:#FBF3E4; border:1px solid var(--border); border-radius:12px; padding:14px; }
.faq-q{ font-size:13px; color:var(--ink); font-weight:600; line-height:1.5; margin:8px 0; } .faq-meta{ font-size:11px; color:var(--muted); }
.section-title{ font-size:13px; font-weight:700; color:var(--muted); margin:20px 0 6px 0; }
.stat-grid{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:14px; }
.stat{ background:#FBF3E4; border:1px solid var(--border); border-radius:12px; padding:14px; }
.stat b{ display:block; font-size:24px; color:var(--ink); } .stat span{ font-size:12px; color:var(--muted); }
.mod-grid{ display:grid; grid-template-columns:repeat(2,1fr); gap:14px; }
.mod h4{ margin:0 0 6px 0; font-size:15px; color:var(--ink); } .mod p{ font-size:13px; color:#5A4A42; line-height:1.6; margin:0 0 10px 0; }
"""
st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)


# =============================================================================
# SMALL UI HELPERS
# =============================================================================
def flat(s: str) -> str:
    """Flatten multi-line HTML so Streamlit's markdown never sees indented/blank lines."""
    return "".join(line.strip() for line in s.splitlines())


def esc(s) -> str:
    return html.escape(str(s))


def trunc(s: str, n: int = 120) -> str:
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def chip(text: str, kind: str = "gray") -> str:
    return f'<span class="chip chip-{kind}">{esc(text)}</span>'


SENT_CHIP = {"Positive": "green", "Neutral": "gray", "Negative": "red"}
INTENT_CHIP = {"Question": "blue", "Feature Request": "amber", "Complaint": "red", "Casual Engagement": "gray"}
LANG_CHIP = {"English": "blue", "Tamil": "amber", "Code-Mixed": "coral"}

ICONS = {
    "chat": '<path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/>',
    "smile": '<circle cx="12" cy="12" r="9"/><path d="M8 14.5c1 1.4 2.4 2 4 2s3-.6 4-2"/><path d="M9 9.5h.01M15 9.5h.01"/>',
    "help": '<circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.7.4-1 .9-1 1.7"/><path d="M12 17h.01"/>',
    "alert": '<path d="M12 3 2.5 20h19L12 3z"/><path d="M12 10v4.5M12 17.5h.01"/>',
}


def kpi(kind, icon, label, value, delta, note="vs. last 30 days"):
    up = delta >= 0
    arrow, cls = ("▲", "up") if up else ("▼", "down")
    # for "Flagged", a falling number is good news; colour is decided by the caller through the sign convention
    return (f'<div class="kpi k-{kind}"><div class="ic"><svg viewBox="0 0 24 24">{ICONS[icon]}</svg></div>'
            f'<div><div class="kpi-label">{esc(label)}</div><div class="kpi-value">{value}</div>'
            f'<div class="kpi-delta"><b class="{cls}">{arrow} {abs(delta)}%</b> {esc(note)}</div></div></div>')


def donut_svg(segments, big, small, size=150, stroke=24):
    r = (size - stroke) / 2
    circ = 2 * np.pi * r
    total = sum(v for v, _ in segments) or 1
    gap = 3.0
    off, parts = 0.0, []
    for v, col in segments:
        ln = max(v / total * circ - gap, 0.5)
        parts.append(f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="{col}" stroke-width="{stroke}" '
                     f'stroke-dasharray="{ln:.2f} {circ - ln:.2f}" stroke-dashoffset="{-off:.2f}" '
                     f'transform="rotate(-90 {size/2} {size/2})"/>')
        off += v / total * circ
    return (f'<svg class="donut" viewBox="0 0 {size} {size}" role="img" aria-label="{esc(big)} {esc(small)}">{"".join(parts)}'
            f'<text x="50%" y="48%" text-anchor="middle" font-size="25" font-weight="800" fill="{C["ink"]}">{esc(big)}</text>'
            f'<text x="50%" y="63%" text-anchor="middle" font-size="11" font-weight="600" fill="{C["muted"]}">{esc(small)}</text></svg>')


def trend_svg(labels, series):
    """series: list of (name, values, colour). Y axis fixed at 0..100."""
    W, H, L, R, T, B = 420, 210, 32, 24, 12, 30
    px = lambda i: L + i * (W - L - R) / (len(labels) - 1)
    py = lambda v: T + (100 - v) / 100 * (H - T - B)
    grid = "".join(
        f'<line x1="{L}" x2="{W-R}" y1="{py(v):.1f}" y2="{py(v):.1f}" stroke="#EBDCC2" stroke-dasharray="3 4"/>'
        f'<text x="{L-8}" y="{py(v)+4:.1f}" text-anchor="end" font-size="11" fill="{C["muted"]}">{v}</text>' for v in (0, 25, 50, 75, 100))
    xl = "".join(f'<text x="{px(i):.1f}" y="{H-8}" text-anchor="middle" font-size="10.5" fill="{C["muted"]}">{esc(l)}</text>' for i, l in enumerate(labels))
    lines = []
    for k, (name, vals, col) in enumerate(series):
        pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(vals))
        if k == 0:
            area = f"{px(0):.1f},{py(0):.1f} {pts} {px(len(vals)-1):.1f},{py(0):.1f}"
            lines.append(f'<defs><linearGradient id="tg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{col}" stop-opacity=".22"/>'
                         f'<stop offset="1" stop-color="{col}" stop-opacity="0"/></linearGradient></defs><polygon points="{area}" fill="url(#tg)"/>')
        dots = "".join(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="3" fill="#FFFDF8" stroke="{col}" stroke-width="2"/>' for i, v in enumerate(vals))
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>{dots}')
    return f'<svg class="trend" viewBox="0 0 {W} {H}" role="img" aria-label="Sentiment trend, last 7 days">{grid}{xl}{"".join(lines)}</svg>'


def card(key: str):
    """A bordered container styled as a floating card (falls back on older Streamlit)."""
    try:
        return st.container(key=f"card_{key}")
    except TypeError:
        return st.container(border=True)


def show_chart(chart):
    try:
        st.altair_chart(chart, width="stretch", theme=None)
    except TypeError:
        st.altair_chart(chart, use_container_width=True, theme=None)


def show_download(label, data, file_name, mime, key):
    try:
        st.download_button(label, data=data, file_name=file_name, mime=mime, key=key, width="stretch")
    except TypeError:
        st.download_button(label, data=data, file_name=file_name, mime=mime, key=key, use_container_width=True)


def light_chart(chart, height=270):
    return (
        chart.properties(height=height, background="transparent")
        .configure_view(stroke=None)
        .configure_axis(
            gridColor="#EFE3CC", domainColor="#EBDCC2", tickColor="#EBDCC2",
            labelColor=C["muted"], titleColor=C["muted"], labelFont="Plus Jakarta Sans", titleFont="Plus Jakarta Sans",
            labelFontSize=11, titleFontSize=11, titleFontWeight=600,
        )
        .configure_legend(labelColor=C["ink"], titleColor=C["muted"], labelFont="Plus Jakarta Sans")
    )


def html_table(headers, rows, row_classes=None, compact=False):
    head = "".join(f"<th>{h_}</th>" for h_ in headers)
    body = []
    for i, cells in enumerate(rows):
        cls = f' class="{row_classes[i]}"' if row_classes and row_classes[i] else ""
        body.append(f"<tr{cls}>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    tcls = "lens-table compact" if compact else "lens-table"
    return f'<div class="table-wrap"><table class="{tcls}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


# =============================================================================
# 2. DATA STRATEGY: SYNTHETIC DATA
# =============================================================================
def get_mock_data() -> pd.DataFrame:
    """Synthetic YouTube / Instagram comments: English, Tanglish, Tamil script, code-mixed and complaints."""
    rows = [
        # (platform, hours_ago, likes, text)
        ("YouTube", 3, 142, "Bro, next video-la cutting length calculation pathi sollunga, romba confuse aavuthu 😂"),
        ("YouTube", 5, 64, "Great video, but can you explain the setup cost?"),
        ("YouTube", 7, 120, "சூப்பர் கான்செப்ட் bro! 🌟 Make a detailed video on structural engineering tools next time."),
        ("Instagram", 9, 8, "Worst explanation ever. Fraud company, waste of time. Unsubscribing right now!!!"),
        ("YouTube", 11, 96, "Anna cutting length formula for stirrups and bends epdi calculate panrathu? Please make a video 🙏"),
        ("Instagram", 14, 210, "Semma video bro 🔥🔥 keep going"),
        ("YouTube", 18, 52, "Setup cost evlo aagum bro? Budget-la panna mudiyuma?"),
        ("YouTube", 21, 83, "Which software do you use for beam design? ETABS or STAAD? Please make a tutorial."),
        ("YouTube", 26, 58, "Your cutting length calculation is wrong sir, the bend deduction is incorrect. This can be unsafe on site!!"),
        ("Instagram", 30, 175, "Love your content! Been following since day one ❤️"),
        ("YouTube", 33, 77, "Staad pro tutorial venum anna, beginners ku puriyura maari 🔥"),
        ("YouTube", 37, 29, "Audio quality is very low, can't hear properly. Please fix the mic"),
        ("YouTube", 41, 88, "கட்டிங் லெந்த் கால்குலேஷன் பத்தி ஒரு வீடியோ போடுங்க bro 🙏"),
        ("Instagram", 46, 5, "Total scam channel. You people are cheating students with fake content 😡"),
        ("YouTube", 50, 45, "Can you explain the difference between M20 and M25 concrete?"),
        ("YouTube", 55, 64, "நன்றி அண்ணா, ரொம்ப பயனுள்ள வீடியோ 🙏"),
        ("YouTube", 61, 19, "Sound romba kammiya irukku bro, headphone-la kooda kekkala"),
        ("Instagram", 66, 33, "Concrete mix ratio evlo podanum? M25 ku sollunga anna"),
        ("YouTube", 72, 37, "What is the total budget for this setup? Is there any cheaper alternative?"),
        ("YouTube", 79, 22, "Nice video"),
        ("Instagram", 85, 71, "Site-la romba use aagum, cutting length calculation video podunga sir 🙌"),
        ("YouTube", 92, 12, "First comment 😎"),
        ("YouTube", 101, 49, "ETABS vs STAAD edhu better? Detailed comparison video venum bro"),
        ("Instagram", 110, 60, "Super explanation 👏 Thanks for sharing such useful content"),
    ]
    anchor = pd.Timestamp.now().floor("h")
    return pd.DataFrame(
        {
            "comment_id": [f"CMT-{1001 + i}" for i in range(len(rows))],
            "platform": [r[0] for r in rows],
            "text": [r[3] for r in rows],
            "timestamp": [anchor - pd.Timedelta(hours=r[1]) for r in rows],
            "likes": [r[2] for r in rows],
        }
    )



# =============================================================================
# MODULE 1: PREPROCESSING AND NORMALISATION
# =============================================================================
TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
LATIN_RE = re.compile(r"[A-Za-z]")
TOKEN_RE = re.compile(r"[a-z0-9']+|[\u0B80-\u0BFF]+")  # \w would split Tamil vowel signs, so use explicit ranges
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\u2B50\u2B06\u2764\uFE0F\u200D]+",
    flags=re.UNICODE,
)

# Common Tanglish spelling variants -> canonical form
TANGLISH_NORMALISE = {
    "patri": "pathi", "pattri": "pathi", "solunga": "sollunga", "sollungo": "sollunga",
    "eppadi": "epdi", "epadi": "epdi", "evalo": "evlo", "evvalavu": "evlo", "yevlo": "evlo",
    "venam": "venum", "vendum": "venum", "podungo": "podunga", "plz": "please", "pls": "please",
    "rmba": "romba", "rombha": "romba", "sema": "semma", "irukku": "irukku", "iruku": "irukku",
}

# Words that signal Romanised Tamil (Tanglish)
TANGLISH_MARKERS = {
    "pathi", "sollunga", "romba", "aavuthu", "aaguthu", "aagum", "evlo", "epdi", "enna", "irukku", "venum",
    "podunga", "panni", "panra", "panrathu", "panna", "mudiyuma", "semma", "nalla", "illa", "illai", "theriyala",
    "puriyala", "puriyura", "anna", "thambi", "kekkala", "kammiya", "maari", "ku", "seiyunga", "pannunga",
    "vidunga", "kudunga", "kooda", "thappu", "machan", "macha", "konjam", "namma", "unga", "edhu", "oru",
    "mosam", "nanri", "podanum", "podanuma", "enakku", "ennaku",
}


def extract_emojis(text: str) -> str:
    chars = [c for seq in EMOJI_RE.findall(text) for c in seq if c not in "\ufe0f\u200d"]
    return " ".join(chars)


def clean_text(text: str) -> str:
    t = URL_RE.sub(" ", text)
    t = MENTION_RE.sub(" ", t)
    t = EMOJI_RE.sub(" ", t)
    t = t.lower()
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)        # soooo -> soo
    t = re.sub(r"([!?.,])\1+", r"\1", t)        # !!! -> !
    t = re.sub(r"[a-z']+", lambda m: TANGLISH_NORMALISE.get(m.group(0), m.group(0)), t)
    return re.sub(r"\s+", " ", t).strip()


def identify_language(text: str) -> str:
    """Heuristic tagger: English | Tamil | Code-Mixed (Tamil script + English, or Romanised Tamil / Tanglish)."""
    tamil = len(TAMIL_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    if tamil:
        ratio = tamil / max(tamil + latin, 1)
        return "Tamil" if ratio >= 0.85 else "Code-Mixed"
    tokens = TOKEN_RE.findall(text.lower())
    tokens = [TANGLISH_NORMALISE.get(t, t) for t in tokens]
    if sum(1 for t in tokens if t in TANGLISH_MARKERS) >= 1:
        return "Code-Mixed"
    return "English"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["emojis"] = out["text"].apply(extract_emojis)
    out["emoji_count"] = out["emojis"].apply(lambda s: len(s.split()) if s else 0)
    out["clean_text"] = out["text"].apply(clean_text)
    out["language"] = out["text"].apply(identify_language)
    out["word_count"] = out["clean_text"].apply(lambda s: len(TOKEN_RE.findall(s)))
    return out


# =============================================================================
# MODULE 2: SENTIMENT AND FEEDBACK-INTENT CLASSIFIER
# (heuristic engine that mimics the output contract of XLM-RoBERTa / MuRIL;
#  swap `classify_comment` for a real transformer call in production)
# =============================================================================
POS_LEX = {
    "love": 2, "amazing": 2, "awesome": 2, "excellent": 2, "superb": 2, "brilliant": 2, "fantastic": 2,
    "semma": 2, "mass": 2, "அருமை": 2, "செம": 2,
    "great": 1, "super": 1, "nice": 1, "good": 1, "helpful": 1, "useful": 1, "perfect": 1, "best": 1,
    "thanks": 1, "thank": 1, "nalla": 1, "nanri": 1, "சூப்பர்": 1, "நன்றி": 1, "பயனுள்ள": 1, "நல்ல": 1,
}
NEG_LEX = {
    "worst": 2, "fraud": 2, "scam": 2, "waste": 2, "fake": 2, "cheating": 2, "terrible": 2, "useless": 2,
    "pathetic": 2, "trash": 2, "hate": 2, "stupid": 2, "unsafe": 2, "mosam": 2, "மோசம்": 2,
    "confuse": 1, "confusing": 1, "confused": 1, "low": 1, "bad": 1, "poor": 1, "wrong": 1, "incorrect": 1,
    "mistake": 1, "error": 1, "boring": 1, "kammiya": 1, "kekkala": 1, "puriyala": 1, "theriyala": 1, "thappu": 1,
}
POS_EMOJI = set("❤🔥🌟🙏😍👏💯🙌👍😊")
NEG_EMOJI = set("😡🤬👎😤😠💔")
TOXIC_TERMS = {"fraud", "scam", "fake", "cheating", "waste", "worst", "idiot", "stupid", "trash", "pathetic", "hate",
               "useless", "unsubscribing", "mosam", "மோசம்"}

COMPLAINT_RE = re.compile(
    r"very low|can'?t hear|cannot hear|not clear|\bwrong\b|\bincorrect\b|\bunsafe\b|\bmistake\b|\berror\b|"
    r"kekkala|kammiya|thappu|mosam|மோசம்"
)
STRUCTURAL_RE = re.compile(r"\b(wrong|incorrect|unsafe|mistake|error|deduction|hazard|collapse|crack|dangerous)\b|thappu")
FEATURE_RE = re.compile(
    r"\b(make|create|post|upload|add|do)\b.*\b(video|tutorial|series|episode|course)\b|\bnext (video|time)\b|"
    r"\b(podunga|sollunga|venum|seiyunga|pannunga|kudunga|vidunga)\b|\bplease\b.*\b(make|cover|share)\b|"
    r"\b(tutorial|comparison)\b.*\b(venum|please)\b|போடுங்க|சொல்லுங்க|வேணும்|செய்யுங்க"
)
QUESTION_RE = re.compile(
    r"\?|\b(how|what|which|why|when|where|can you|could you|is there|do you)\b|"
    r"\b(epdi|enna|evlo|yaar|mudiyuma|irukka|aagumaa|edhu)\b|எப்படி|என்ன|எவ்வளவு"
)


def classify_comment(clean: str, emojis: str) -> dict:
    tokens = TOKEN_RE.findall(clean)
    pos = sum(POS_LEX.get(t, 0) for t in tokens)
    neg = sum(NEG_LEX.get(t, 0) for t in tokens)
    em = emojis.replace(" ", "")
    score = pos - neg + 0.5 * sum(c in POS_EMOJI for c in em) - 1.0 * sum(c in NEG_EMOJI for c in em)
    toxic_hits = sum(1 for t in tokens if t in TOXIC_TERMS)
    is_complaint = toxic_hits >= 1 or neg >= 2 or bool(COMPLAINT_RE.search(clean))

    if is_complaint:
        intent = "Complaint"
    elif FEATURE_RE.search(clean):
        intent = "Feature Request"
    elif QUESTION_RE.search(clean):
        intent = "Question"
    else:
        intent = "Casual Engagement"

    if intent == "Complaint" or score <= -1:
        sentiment = "Negative"
    elif score >= 1:
        sentiment = "Positive"
    else:
        sentiment = "Neutral"

    confidence = min(0.97, 0.62 + 0.07 * min(abs(score), 3) + (0.08 if intent != "Casual Engagement" else 0.0))
    return {
        "sentiment": sentiment,
        "intent": intent,
        "sentiment_score": round(float(score), 2),
        "confidence": round(confidence, 2),
        "toxic_hits": int(toxic_hits),
        "structural": bool(is_complaint and STRUCTURAL_RE.search(clean)),
    }


def classify(df: pd.DataFrame) -> pd.DataFrame:
    recs = [classify_comment(c, e) for c, e in zip(df["clean_text"], df["emojis"])]
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(recs)], axis=1)
    # Safety / review flags
    flagged, severity, reason, action = [], [], [], []
    for r in out.itertuples():
        reasons = []
        if r.structural:
            reasons.append("Structural accuracy / safety")
        if r.toxic_hits >= 1:
            reasons.append("Abusive / reputational language")
        if r.intent == "Complaint" and not reasons:
            reasons.append("Quality complaint")
        is_flag = bool(reasons)
        sev = "High" if (r.structural or r.toxic_hits >= 2) else ("Medium" if is_flag else "")
        act = ""
        if r.structural:
            act = "Verify calculation, pin a correction"
        elif r.toxic_hits >= 1:
            act = "Review for policy violation / hide"
        elif is_flag:
            act = "Reply and log for production fix"
        flagged.append(is_flag); severity.append(sev); reason.append(" · ".join(reasons)); action.append(act)
    out["flagged"], out["flag_severity"], out["flag_reason"], out["recommended_action"] = flagged, severity, reason, action
    return out


# =============================================================================
# MODULE 3: SEMANTIC GROUPING AND FAQ EXTRACTION
# =============================================================================
TOPICS = [
    dict(name="Cutting Length Calculation",
         idea="Video: Bar-bending cutting length calculation, stirrups and bends, step by step",
         seed="cutting length calculation bar bending stirrup formula bend deduction கட்டிங் லெந்த் கால்குலேஷன்"),
    dict(name="Project Setup Cost & Budget",
         idea="Video / post: Complete setup cost breakdown with budget-friendly alternatives",
         seed="setup cost budget price total cost cheaper alternative evlo aagum rate expense investment"),
    dict(name="Structural Software Tutorials",
         idea="Series: Structural engineering tools (ETABS vs STAAD) beam design tutorial for beginners",
         seed="structural engineering tools software staad etabs beam design tutorial beginners detailed video comparison"),
    dict(name="Concrete Grades & Mix Design",
         idea="Explainer: M20 vs M25 concrete grades and mix ratios",
         seed="concrete mix ratio m20 m25 grade cement sand aggregate difference"),
    dict(name="Audio & Production Quality",
         idea="Fix: Audio quality and mic setup for upcoming uploads",
         seed="audio quality sound mic noise volume voice headphone kammiya kekkala hear"),
]
TOPIC_BY_NAME = {t["name"]: t for t in TOPICS}
SIM_THRESHOLD = 0.22
ACTIONABLE = ("Question", "Feature Request", "Complaint")
SEVERITY_WEIGHT = 10.0


def group_topics(df: pd.DataFrame) -> pd.DataFrame:
    """Assign actionable comments to the nearest request profile using char n-gram TF-IDF cosine similarity.
    Character n-grams make the match robust to Tanglish spelling variants and mixed scripts."""
    out = df.copy()
    out["topic"] = "—"
    out["topic_sim"] = 0.0
    mask = out["intent"].isin(ACTIONABLE).to_numpy()
    n = int(mask.sum())
    if n == 0:
        return out
    seeds = [t["seed"].lower() for t in TOPICS]
    texts = out.loc[mask, "clean_text"].tolist()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    X = vec.fit_transform(texts + seeds)
    sims = cosine_similarity(X[:n], X[n:])
    seed_tokens = [set(TOKEN_RE.findall(s)) for s in seeds]

    topics, best_sims = [], []
    for i, text in enumerate(texts):
        toks = set(TOKEN_RE.findall(text))
        kw = np.array([len(toks & st_) for st_ in seed_tokens])
        score = sims[i] + 0.06 * np.minimum(kw, 3)   # keyword evidence lifts the similarity a little
        j = int(score.argmax())
        if score[j] >= SIM_THRESHOLD:
            topics.append(TOPICS[j]["name"])
        else:
            topics.append("General Feedback")
        best_sims.append(round(float(score[j]), 3))
    out.loc[mask, "topic"] = topics
    out.loc[mask, "topic_sim"] = best_sims
    return out


def extract_faqs(df: pd.DataFrame) -> pd.DataFrame:
    q = df[df["intent"].isin(["Question", "Feature Request"]) & ~df["topic"].isin(["—", "General Feedback"])]
    rows = []
    for topic, g in q.groupby("topic"):
        best = g.sort_values("likes", ascending=False).iloc[0]
        rows.append(dict(
            topic=topic, canonical=best["text"], asked=len(g), total_likes=int(g["likes"].sum()),
            languages=", ".join(sorted(g["language"].unique())),
        ))
    cols = ["topic", "canonical", "asked", "total_likes", "languages"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values(["asked", "total_likes"], ascending=False).reset_index(drop=True)


# =============================================================================
# MODULE 4: CONTENT OPPORTUNITY RANKER
# Score = (Count of Similar Requests * 1.5) + (Average Likes * 0.8) + (Severity weight if structural complaint)
# =============================================================================
def rank_opportunities(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["rank", "topic", "idea", "requests", "avg_likes", "severity_weight", "score", "languages", "evidence", "tier"]
    act = df[df["intent"].isin(ACTIONABLE) & ~df["topic"].isin(["—", "General Feedback"])]
    rows = []
    for topic, g in act.groupby("topic"):
        n = len(g)
        avg = float(g["likes"].mean())
        sev = SEVERITY_WEIGHT if bool(g["structural"].any()) else 0.0
        score = n * 1.5 + avg * 0.8 + sev
        top = g.sort_values("likes", ascending=False).head(2)
        rows.append(dict(
            topic=topic, idea=TOPIC_BY_NAME[topic]["idea"], requests=n, avg_likes=round(avg, 1),
            severity_weight=sev, score=round(score, 1), languages=", ".join(sorted(g["language"].unique())),
            evidence=[(r.text, int(r.likes), r.language) for r in top.itertuples()],
        ))
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    mx = out["score"].max()
    out["tier"] = out["score"].apply(lambda s: "P1 · Create now" if s >= 0.75 * mx else ("P2 · Plan next" if s >= 0.4 * mx else "P3 · Backlog"))
    return out


@st.cache_data(show_spinner=False)
def load_pipeline() -> pd.DataFrame:
    df = get_mock_data()
    df = preprocess(df)
    df = classify(df)
    df = group_topics(df)
    return df


# =============================================================================
# DASHBOARD CONTENT (workspace-level demo figures shown on the Dashboard page)
# Edit this block to change what the Dashboard displays. The other pages are
# computed live from the comment pipeline above.
# =============================================================================
RANGES = {"Last 7 Days": 0.26, "Last 30 Days": 1.0, "Last 90 Days": 2.85}
BASE = dict(total=2480, questions=342, flagged=87)
DELTAS = dict(total=12, positive=8, questions=15, flagged=-5)
SENTIMENT_MIX = [("Positive", 68, C["sage"]), ("Neutral", 18, C["slate"]), ("Negative", 9, C["brick"]), ("Mixed", 5, C["gold"])]
LANGUAGE_MIX = [("English", 48, C["copper"]), ("Tamil", 22, "#E9B45F"), ("Tanglish (Mixed)", 25, "#7F8794"), ("Others", 5, "#D9CDBA")]
TREND_POS = [70, 73, 69, 76, 75, 78, 80]
TREND_NEU = [52, 47, 55, 49, 52, 46, 50]
TREND_NEG = [34, 30, 36, 29, 32, 28, 31]
TOPIC_ROWS = [
    ("Cutting Length Calculation", 542, 28), ("Setup Cost & Budget", 431, 22), ("ETABS & STAAD Software", 349, 18),
    ("Concrete Grades (M20/M25)", 271, 14), ("Audio Quality", 182, 9),
]
TOPIC_TINTS = [("#F8E3D3", "#8F4F25"), ("#DCE8D9", "#3D6141"), ("#F8E3B4", "#7B5210"), ("#DCDFE4", "#454C59"), ("#F3D3CE", "#9A3A2E")]
OPP_ROWS = [
    ("Deployment Tutorial for Beginners", 324, 10, 92, "linear-gradient(135deg,#8F4F25,#E4A862)"),
    ("ETABS Step-by-Step Guide", 267, 8, 87, "linear-gradient(135deg,#3F4652,#A9ACB3)"),
    ("Concrete Grade Comparison", 198, 6, 78, "linear-gradient(135deg,#B8713B,#F7D89E)"),
    ("Audio Setup for Better Quality", 156, 4, 71, "linear-gradient(135deg,#291A17,#8F4F25)"),
]
HIGHLIGHTS = [
    ("!", "#F6D9C4", "#9A3A2E", "Repeated complaint detected", "“Installation steps were skipped”", "2m ago"),
    ("✕", "#F3D3CE", "#9A3A2E", "Potentially toxic comment", "“Worst explanation ever. Fraud…”", "2h ago"),
    ("?", "#DCDFE4", "#454C59", "New question cluster formed", "“How to deploy a Python project?”", "5h ago"),
    ("↗", "#DCE8D9", "#3D6141", "Positive trend noticed", "Tamil comments increased by 10%", "1d ago"),
]
QUESTION_ROWS = [
    ("How to calculate cutting length?", 48, False), ("Setup cost for this project?", 42, False),
    ("Which software is better: STAAD or ETABS?", 36, True), ("M20 vs M25: which is better?", 28, False),
    ("How to improve audio quality?", 22, True),
]
COMMENT_ROWS = [
    ("U1", "@user123", "YouTube", "2h ago", "Bro, next video-la cutting length calculation pathi sollunga, romba confuse aavuthu 😂",
     [("Question", "blue"), ("Tanglish", "coral"), ("Cutting Length", "gray"), ("Neutral", "gray")]),
    ("K7", "@keerthi_07", "Instagram", "3h ago", "சூப்பர் கான்செப்ட் bro! 🌟 Really loved the explanation.",
     [("Feedback", "blue"), ("Tamil", "amber"), ("Appreciation", "gray"), ("Positive", "green")]),
    ("RF", "@rajini_fan", "YouTube", "6h ago", "Worst explanation ever. Fraud company, waste of time…",
     [("Complaint", "red"), ("Toxic", "red"), ("Audio Quality", "gray"), ("Negative", "red")]),
]
QUICK_STATS = [("A", "#F8E3D3", "#8F4F25", "4+", "Languages Detected"), ("◎", "#DCDFE4", "#454C59", "12", "Topic Clusters"),
               ("?", "#F8E3B4", "#7B5210", "5", "FAQ Categories"), ("↗", "#DCE8D9", "#3D6141", "7", "Days Trend Analysis")]

HERO_SVG = """<svg viewBox="0 0 520 200" preserveAspectRatio="xMaxYMax meet" aria-hidden="true">
<circle cx="318" cy="124" r="76" fill="#FCE3B2" opacity=".55"/><circle cx="318" cy="124" r="52" fill="#FFF1D2" opacity=".8"/>
<path d="M110 200 L110 162 C160 130 214 152 262 136 C318 118 372 144 430 128 C470 118 500 130 520 124 L520 200Z" fill="#A9622F" opacity=".55"/>
<path d="M190 200 C230 172 280 174 330 186 C380 198 450 172 520 182 L520 200Z" fill="#6F3A18" opacity=".7"/>
<g transform="rotate(-6 250 100)"><rect x="186" y="60" width="128" height="82" rx="12" fill="#FFFDF8" opacity=".96"/>
<rect x="196" y="70" width="108" height="48" rx="8" fill="#F0C98C"/><circle cx="250" cy="94" r="15" fill="#B8713B"/>
<path d="M245 87 L258 94 L245 101Z" fill="#FFF8EA"/><rect x="196" y="126" width="62" height="6" rx="3" fill="#E4D2B0"/></g>
<path d="M322 200 C324 162 356 144 398 142 C442 144 474 162 476 200Z" fill="#291A17"/>
<rect x="388" y="114" width="22" height="32" rx="8" fill="#291A17"/><ellipse cx="399" cy="90" rx="27" ry="31" fill="#291A17"/>
<path d="M371 82 C373 56 425 52 429 80 C417 68 391 66 371 82Z" fill="#3A2823"/>
<rect x="392" y="16" width="112" height="36" rx="12" fill="#FFF8EA"/><polygon points="410,52 424,52 414,64" fill="#FFF8EA"/>
<text x="448" y="39" text-anchor="middle" font-size="13" font-weight="800" fill="#8F4F25" font-family="Plus Jakarta Sans,sans-serif">Create Better</text>
</svg>"""


def scaled(n, f):
    return int(round(n * f))


def card_head(title, icon="✦", link="View All"):
    more = f'<a class="view-all" href="#">{link} →</a>' if link else ""
    return f'<div class="card-head"><div class="card-title"><span class="ti">{icon}</span>{esc(title)}</div>{more}</div>'


def dash_banner():
    return (f'<div class="g-banner banner"><div class="banner-copy"><h2>Turn Comments into<br>Your Next Big Idea!</h2>'
            f'<p>Our AI analyzes your audience’s feedback, questions, and emotions to help you create content they actually want.</p>'
            f'<a class="banner-btn" href="#insights">View Insights →</a></div>{HERO_SVG}</div>')


def dash_kpis(f):
    total, qs, fl = scaled(BASE["total"], f), scaled(BASE["questions"], f), scaled(BASE["flagged"], f)
    note = {0.26: "vs. previous 7 days", 1.0: "vs. last 30 days", 2.85: "vs. previous 90 days"}[f]
    return "".join([
        f'<div class="g-kpi">{kpi("slate", "chat", "Total Comments", f"{total:,}", DELTAS["total"], note)}</div>',
        f'<div class="g-kpi">{kpi("sage", "smile", "Positive Sentiment", "68%", DELTAS["positive"], note)}</div>',
        f'<div class="g-kpi">{kpi("cream", "help", "Questions Detected", f"{qs:,}", DELTAS["questions"], note)}</div>',
        f'<div class="g-kpi">{kpi("copper", "alert", "Flagged Comments", f"{fl:,}", DELTAS["flagged"], note)}</div>',
    ])


def dash_sentiment(f):
    total = scaled(BASE["total"], f)
    today = dt.date.today()
    labels = [(today - dt.timedelta(days=6 - i)).strftime("%b %d") for i in range(7)]
    legend = "".join(f'<div><i style="background:{c}"></i>{n}<span class="pc">{v}%</span></div>' for n, v, c in SENTIMENT_MIX)
    leg2 = "".join(f'<span><i style="background:{c}"></i>{n}</span>' for n, c in [("Positive", C["sage"]), ("Neutral", C["slate"]), ("Negative", C["brick"])])
    trend = trend_svg(labels, [("Positive", TREND_POS, C["sage"]), ("Neutral", TREND_NEU, C["slate"]), ("Negative", TREND_NEG, C["brick"])])
    return (f'<div class="g-sent card">{card_head("Sentiment & Emotion Analysis", "☺", "View Details")}'
            f'<div class="card-sub" style="margin:-8px 0 12px 35px">How your audience feels about your content</div>'
            f'<div class="split"><div class="donut-row">{donut_svg([(v, c) for _, v, c in SENTIMENT_MIX], "68%", "Positive")}'
            f'<div class="leg-list">{legend}</div></div>'
            f'<div><div class="trend-title">Sentiment Trend (Last 7 Days)</div>{trend}<div class="trend-leg">'
            + "".join(f'<span><i style="background:{c}"></i>{n}</span>' for n, c in [("Positive", C["sage"]), ("Neutral", C["slate"]), ("Negative", C["brick"])])
            + '</div></div></div></div>')


def dash_topics(f):
    rows = "".join(
        f'<div class="topic"><div class="rk" style="background:{TOPIC_TINTS[i][0]};color:{TOPIC_TINTS[i][1]}">{i+1}</div>'
        f'<div><div class="tn"><span class="nm">{esc(n)}</span><span class="pc">{p}%</span></div>'
        f'<div class="pbar"><span style="width:{p / 28 * 100:.0f}%"></span></div><div class="ct">{scaled(c, f):,} comments</div></div></div>'
        for i, (n, c, p) in enumerate(TOPIC_ROWS))
    return f'<div class="g-topics card">{card_head("Top Audience Topics", "◈")}<div class="card-sub" style="margin:-8px 0 6px 35px">What people are talking about most</div>{rows}</div>'


def dash_language(f):
    total = scaled(BASE["total"], f)
    legend = "".join(f'<div><span class="num">{i+1}</span><i style="background:{c}"></i>{n}<span class="pc">{v}%</span></div>' for i, (n, v, c) in enumerate(LANGUAGE_MIX))
    return (f'<div class="g-lang card">{card_head("Language Distribution", "文", "View Details")}'
            f'<div class="card-sub" style="margin:-8px 0 8px 35px">Multilingual audience support</div>'
            f'<div class="donut-col">{donut_svg([(v, c) for _, v, c in LANGUAGE_MIX], f"{total:,}", "Comments")}<div class="leg-list" style="width:100%">{legend}</div></div></div>')


def dash_opps(f):
    def cls(s):
        return "sc-hi" if s >= 85 else ("sc-mid" if s >= 75 else "sc-lo")
    rows = "".join(
        f'<div class="opp"><div class="thumb" style="background:{g}">▶</div><div><div class="t">{i+1}. {esc(t)}</div>'
        f'<div class="m">{scaled(r, f):,} requests · {e}% engagement</div></div><div class="score {cls(s)}">{s}</div></div>'
        for i, (t, r, e, s, g) in enumerate(OPP_ROWS))
    return f'<div class="g-opps card">{card_head("Content Opportunities", "★")}<div class="card-sub" style="margin:-8px 0 6px 35px">AI recommended video ideas</div>{rows}</div>'


def dash_highlights():
    rows = "".join(
        f'<div class="hl"><div class="ic" style="background:{bg};color:{fg}">{ic}</div><div><div class="t">{esc(t)}</div><div class="d">{esc(d)}</div></div><div class="w">{w}</div></div>'
        for ic, bg, fg, t, d, w in HIGHLIGHTS)
    return f'<div class="g-high card">{card_head("Recent Highlights", "✦")}{rows}</div>'


def dash_questions(f):
    rows = []
    for i, (q, n, answered) in enumerate(QUESTION_ROWS):
        rows.append([f'<span style="color:var(--muted);font-weight:600">{i+1}</span>', esc(q), f'<b>{scaled(n, 1)}</b>',
                     chip("Answered", "green") if answered else chip("Unanswered", "red")])
    return (f'<div class="g-qs card">{card_head("Audience Questions (Top 5)", "?")}<div class="card-sub" style="margin:-8px 0 6px 35px">Most asked questions in your comments</div>'
            + html_table(["#", "Question", "Frequency", "Status"], rows, compact=True) + "</div>")


def dash_comments():
    items = "".join(
        f'<div class="cm"><div class="avatar">{ini}</div><div><div class="top">{esc(u)} <span>{esc(p)}</span><span class="when">{w}</span></div>'
        f'<div class="tx">{esc(t)}</div><div>{"".join(chip(x, k) for x, k in chips)}</div></div></div>'
        for ini, u, p, w, t, chips in COMMENT_ROWS)
    return f'<div class="g-expl card">{card_head("Comment Explorer", "☰")}<div class="card-sub" style="margin:-8px 0 4px 35px">View all comments</div>{items}</div>'


def dash_quickstats():
    tiles = "".join(f'<div class="qs"><div class="ic" style="background:{bg};color:{fg}">{ic}</div><div><b>{v}</b><span>{esc(l)}</span></div></div>' for ic, bg, fg, v, l in QUICK_STATS)
    return f'<div class="g-stats card">{card_head("Quick Stats", "▦", "")}<div class="card-sub" style="margin:-8px 0 10px 35px">At a glance</div><div class="qs-grid">{tiles}</div><div class="qs-foot">Your audience matters ♡</div></div>'


def render_dashboard(df, opps, faqs):
    hour = dt.datetime.now().hour
    hello = "Good Morning" if hour < 12 else ("Good Afternoon" if hour < 18 else "Good Evening")
    try:
        h1, h2, h3 = st.columns([4.4, 1.5, 1.3], vertical_alignment="center")
    except TypeError:  # older Streamlit
        h1, h2, h3 = st.columns([4.4, 1.5, 1.3])
    with h1:
        st.markdown(f'<div class="hello">{hello}, Creator! 👋</div><div class="hello-sub">Here’s what your audience is saying…</div>', unsafe_allow_html=True)
    with h2:
        rng = st.selectbox("Date range", list(RANGES), index=1, label_visibility="collapsed", key="range")
    with h3:
        show_download("⤓  Export Report", build_export(df, opps).to_csv(index=False).encode("utf-8-sig"), "creatorlens_report.csv", "text/csv", "dl_header")
    f = RANGES[rng]
    st.markdown(flat('<div class="dash" id="insights">' + dash_banner() + dash_kpis(f) + dash_sentiment(f) + dash_topics(f) + dash_language(f)
                     + dash_opps(f) + dash_highlights() + dash_questions(f) + dash_comments() + dash_quickstats() + "</div>"), unsafe_allow_html=True)


# =============================================================================
# OTHER PAGES (computed live from the pipeline)
# =============================================================================
def build_export(df: pd.DataFrame, opps: pd.DataFrame) -> pd.DataFrame:
    score_map = dict(zip(opps["topic"], opps["score"])) if not opps.empty else {}
    out = df[[
        "comment_id", "platform", "timestamp", "text", "likes", "language", "emojis", "sentiment", "sentiment_score",
        "intent", "confidence", "topic", "flagged", "flag_severity", "flag_reason", "recommended_action",
    ]].copy()
    out["opportunity_score"] = out["topic"].map(score_map).fillna(0.0)
    return out


def render_header(title: str, subtitle: str):
    st.markdown(flat(f'''
    <div class="page-head">
      <div><div class="page-title">{esc(title)}</div><div class="page-sub">{esc(subtitle)}</div></div>
      <div class="head-chips">{chip("● Live sync", "green")}{chip("Channel: Civil Engineering Tamil", "gray")}{chip("YouTube · Instagram", "gray")}</div>
    </div>'''), unsafe_allow_html=True)


def render_explorer(df_all):
    render_header("Comment Explorer", "Every comment with its normalised features and model labels")
    c1, c2, c3 = st.columns([2, 1.2, 1.2])
    with c1:
        q = st.text_input("Search comments", placeholder="Search text, topic or intent…", label_visibility="collapsed")
    with c2:
        langs = st.multiselect("Language", ["English", "Tamil", "Code-Mixed"], default=["English", "Tamil", "Code-Mixed"], label_visibility="collapsed")
    with c3:
        sents = st.multiselect("Sentiment", ["Positive", "Neutral", "Negative"], default=["Positive", "Neutral", "Negative"], label_visibility="collapsed")
    view = df_all[df_all["language"].isin(langs) & df_all["sentiment"].isin(sents)]
    if q:
        ql = q.lower()
        view = view[view.apply(lambda r: ql in (r["text"] + " " + r["topic"] + " " + r["intent"]).lower(), axis=1)]
    if view.empty:
        st.info("No comments match these filters. Clear the search or add a language or sentiment.")
        return
    rows = [[esc(r.comment_id), f'<div style="max-width:420px;line-height:1.5">{esc(r.text)}</div>',
             chip(r.language, LANG_CHIP.get(r.language, "gray")), chip(r.sentiment, SENT_CHIP[r.sentiment]),
             chip(r.intent, INTENT_CHIP[r.intent]), esc(r.topic), f"♥ {r.likes}", f"{r.confidence:.2f}"]
            for r in view.sort_values("timestamp", ascending=False).itertuples()]
    st.markdown('<div class="card"><div class="card-title">' + f"{len(view)} comments</div>"
                + html_table(["ID", "Comment", "Language", "Sentiment", "Intent", "Topic", "Likes", "Conf."], rows) + "</div>", unsafe_allow_html=True)


def render_insights(df, faqs):
    render_header("Audience Insights", "What your viewers ask for, how they feel and which questions repeat")
    n_q, n_f, n_c = int((df.intent == "Question").sum()), int((df.intent == "Feature Request").sum()), int((df.intent == "Complaint").sum())
    st.markdown(flat(f'''<div class="stat-grid"><div class="stat"><b>{len(df)}</b><span>Comments analysed</span></div>
      <div class="stat"><b>{n_q}</b><span>Questions</span></div><div class="stat"><b>{n_f}</b><span>Feature requests</span></div>
      <div class="stat"><b>{n_c}</b><span>Complaints</span></div></div>'''), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    topic_counts = df[~df["topic"].isin(["—", "General Feedback"])]["topic"].value_counts().rename_axis("topic").reset_index(name="comments")
    with c1:
        with card("topics"):
            st.markdown('<div class="card-title">Comments per topic</div><div class="card-sub">Request clusters found by semantic grouping</div>', unsafe_allow_html=True)
            ch = alt.Chart(topic_counts).mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5, size=22, color=C["copper"]).encode(
                y=alt.Y("topic:N", sort="-x", axis=alt.Axis(title=None, labelLimit=190)), x=alt.X("comments:Q", axis=alt.Axis(title="Comments", tickMinStep=1)),
                tooltip=["topic", "comments"])
            show_chart(light_chart(ch, 250))
    with c2:
        with card("intent"):
            st.markdown('<div class="card-title">Intent mix</div><div class="card-sub">Why people comment</div>', unsafe_allow_html=True)
            it = df["intent"].value_counts().rename_axis("intent").reset_index(name="comments")
            ch = alt.Chart(it).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, size=44).encode(
                x=alt.X("intent:N", sort="-y", axis=alt.Axis(title=None, labelAngle=0, labelLimit=110)), y=alt.Y("comments:Q", axis=alt.Axis(title="Comments", tickMinStep=1)),
                color=alt.Color("intent:N", legend=None, scale=alt.Scale(domain=["Question", "Feature Request", "Complaint", "Casual Engagement"], range=[C["slate"], C["gold"], C["brick"], C["copper"]])),
                tooltip=["intent", "comments"])
            show_chart(light_chart(ch, 250))
    if not faqs.empty:
        cards = "".join(flat(f'''<div class="faq"><div>{chip(f["topic"], "coral")}</div><div class="faq-q">“{esc(trunc(f["canonical"], 150))}”</div>
            <div class="faq-meta">Asked by <b style="color:#291A17">{f["asked"]}</b> viewers · ♥ {f["total_likes"]} · {esc(f["languages"])}</div></div>''') for f in faqs.to_dict("records"))
        st.markdown(f'<div class="card"><div class="card-title">Extracted FAQ clusters</div><div class="card-sub">The most-liked phrasing is the canonical question for each cluster</div><div class="faq-grid">{cards}</div></div>', unsafe_allow_html=True)


def render_opportunities(opps):
    render_header("Content Opportunities", "Ranked video ideas built from what your audience keeps asking for")
    if opps.empty:
        st.info("No actionable request clusters yet.")
        return
    rows = []
    for r in opps.to_dict("records"):
        pct_w = r["score"] / opps["score"].max() * 100
        ev = "".join(f'<div class="evidence">“{esc(trunc(t, 118))}” <span class="ev-meta">♥ {l} · {esc(lang)}</span></div>' for t, l, lang in r["evidence"])
        tier_kind = "red" if r["tier"].startswith("P1") else ("amber" if r["tier"].startswith("P2") else "gray")
        sev = chip("Structural +10", "red") if r["severity_weight"] > 0 else chip("None", "gray")
        langs = "".join(chip(x.strip(), LANG_CHIP.get(x.strip(), "gray")) for x in r["languages"].split(","))
        rows.append([f'<div class="rank">{r["rank"]}</div>',
                     f'<div class="idea-title">{esc(r["idea"])}</div><div class="idea-sub">Cluster: {esc(r["topic"])}</div><div style="margin-top:8px">{chip(r["tier"], tier_kind)}</div>',
                     f'<div class="score-num">{r["score"]}</div><div class="bar"><span style="width:{pct_w:.0f}%"></span></div>',
                     f'{r["requests"]}', f'{r["avg_likes"]}', sev, langs, ev])
    st.markdown(flat('<div class="card"><div class="card-head"><div><div class="card-title">Ranked content opportunities</div>'
                     '<div class="card-sub">Score = (similar requests × 1.5) + (avg likes × 0.8) + structural-complaint severity weight</div></div></div>')
                + html_table(["#", "Content idea", "Opportunity score", "Requests", "Avg likes", "Severity", "Languages", "Comment evidence"], rows) + "</div>", unsafe_allow_html=True)


def render_language(df):
    render_header("Language Intelligence", "How your audience mixes English, Tamil and Tanglish and how each group feels")
    c1, c2 = st.columns(2)
    lang_counts = df["language"].value_counts().rename_axis("language").reset_index(name="comments")
    with c1:
        with card("langdist"):
            st.markdown('<div class="card-title">Language distribution</div><div class="card-sub">Comments per detected language</div>', unsafe_allow_html=True)
            ch = alt.Chart(lang_counts).mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5, size=26, color=C["copper"]).encode(
                y=alt.Y("language:N", sort="-x", axis=alt.Axis(title=None)), x=alt.X("comments:Q", axis=alt.Axis(title="Comments", tickMinStep=1)), tooltip=["language", "comments"])
            show_chart(light_chart(ch, 240))
    with c2:
        with card("langsent"):
            st.markdown('<div class="card-title">Sentiment by language</div><div class="card-sub">Stacked comment counts</div>', unsafe_allow_html=True)
            ct = df.groupby(["language", "sentiment"]).size().reset_index(name="comments")
            ch = alt.Chart(ct).mark_bar(size=34).encode(
                x=alt.X("language:N", axis=alt.Axis(title=None, labelAngle=0)), y=alt.Y("comments:Q", axis=alt.Axis(title="Comments", tickMinStep=1)),
                color=alt.Color("sentiment:N", scale=alt.Scale(domain=["Positive", "Neutral", "Negative"], range=[C["sage"], C["slate"], C["brick"]]), legend=alt.Legend(title=None, orient="bottom")),
                tooltip=["language", "sentiment", "comments"])
            show_chart(light_chart(ch, 240))
    emo = pd.Series([e for s in df["emojis"] for e in s.split()]).value_counts().head(12)
    chips_html = "".join(f'<span class="chip chip-gray" style="font-size:15px;padding:6px 12px">{esc(e)} <b style="color:#291A17">{n}</b></span>' for e, n in emo.items())
    st.markdown(f'<div class="card"><div class="card-title">Emoji signal</div><div class="card-sub">Extracted as a separate feature before text normalisation</div><div style="margin-top:12px">{chips_html or "No emojis in current selection"}</div></div>', unsafe_allow_html=True)


def render_moderation(df):
    render_header("Moderation & Toxicity", "Comments that need a reply, a correction or a policy review")
    fl = df[df["flagged"]].copy()
    fl["_o"] = fl["flag_severity"].map({"High": 0, "Medium": 1}).fillna(2)
    fl = fl.sort_values(["_o", "timestamp"], ascending=[True, False])
    if fl.empty:
        st.success("Nothing to review. No complaints or harmful language found.")
        return
    rows, classes = [], []
    for r in fl.itertuples():
        reasons = "".join(chip(x, "red" if "Structural" in x or "Abusive" in x else "amber") for x in r.flag_reason.split(" · "))
        rows.append([chip(r.flag_severity, "red" if r.flag_severity == "High" else "amber"),
                     f'<div style="max-width:420px;line-height:1.5">{esc(r.text)}</div><div class="ev-meta" style="margin-top:4px">{esc(r.comment_id)} · {esc(r.platform)} · {r.timestamp.strftime("%d %b, %H:%M")}</div>',
                     chip(r.language, LANG_CHIP.get(r.language, "gray")), reasons, f"♥ {r.likes}",
                     f'<span style="color:#8F4F25;font-weight:700">{esc(r.recommended_action)}</span>'])
        classes.append("row-high" if r.flag_severity == "High" else "row-med")
    n_high = int((fl["flag_severity"] == "High").sum())
    st.markdown(flat(f'<div class="card"><div class="card-head"><div><div class="card-title">Flagged review queue</div>'
                     f'<div class="card-sub">{len(fl)} comments need attention · {n_high} high severity</div></div>'
                     f'<div class="legend"><span><i style="background:{C["brick"]}"></i>High</span><span><i style="background:{C["gold"]}"></i>Medium</span></div></div>')
                + html_table(["Severity", "Comment", "Language", "Trigger", "Likes", "Recommended action"], rows, classes) + "</div>", unsafe_allow_html=True)


def render_reports(df, opps):
    render_header("Reports", "Export the analysis and see how each comment becomes a ranked idea")
    export_df = build_export(df, opps)
    st.markdown(flat(f'''<div class="card"><div class="card-title">Export processed analysis</div>
      <div class="card-sub">UTF-8 with BOM so Tamil script opens correctly in Excel</div>
      <div class="stat-grid" style="margin-top:14px"><div class="stat"><b>{len(export_df)}</b><span>Comments</span></div>
      <div class="stat"><b>{int(export_df["flagged"].sum())}</b><span>Flagged</span></div><div class="stat"><b>{len(opps)}</b><span>Opportunities</span></div>
      <div class="stat"><b>{len(export_df.columns)}</b><span>Columns</span></div></div></div>'''), unsafe_allow_html=True)
    b1, b2, _ = st.columns([1, 1, 1.4])
    with b1:
        show_download("⤓  Full analysis (CSV)", export_df.to_csv(index=False).encode("utf-8-sig"), "creatorlens_analysis.csv", "text/csv", "dl_full")
    with b2:
        opp_csv = opps.drop(columns=["evidence"]).to_csv(index=False).encode("utf-8-sig") if not opps.empty else b""
        show_download("⤓  Opportunity ranking (CSV)", opp_csv, "creatorlens_opportunities.csv", "text/csv", "dl_opps")
    mods = [
        ("Module 1 · Preprocessing & Normalisation", "Strips URLs and mentions, extracts emojis into their own feature, lowercases, collapses elongated words and maps Tanglish spelling variants. A script and vocabulary heuristic tags each comment as English, Tamil or Code-Mixed."),
        ("Module 2 · Sentiment & Intent Classifier", "Lexicon, emoji and pattern engine that emits the same contract as a multilingual transformer: sentiment, intent and a confidence score. Swap classify_comment() for XLM-R / MuRIL inference."),
        ("Module 3 · Semantic Grouping & FAQ Extraction", "Character n-gram TF-IDF vectors compared by cosine similarity against request profiles, so Tanglish and Tamil variants land in the same cluster."),
        ("Module 4 · Opportunity Ranker", "Score = (similar requests × 1.5) + (average likes × 0.8) + severity weight (+10 when a cluster holds a structural complaint). Tiers P1 to P3 are set relative to the top score."),
    ]
    cards = "".join(f'<div class="card mod"><h4>{t}</h4><p>{d}</p>{chip("Active · heuristic mode", "green")}</div>' for t, d in mods)
    st.markdown('<div class="section-title">How the engine works</div>' + f'<div class="mod-grid">{cards}</div>', unsafe_allow_html=True)
    prev = export_df.head(8)
    rows = [[esc(r.comment_id), esc(trunc(r.text, 70)), esc(r.language), esc(r.sentiment), esc(r.intent), esc(r.topic), f"{r.opportunity_score}"] for r in prev.itertuples()]
    st.markdown('<div class="section-title">Export preview</div><div class="card">' + html_table(["ID", "Comment", "Language", "Sentiment", "Intent", "Topic", "Opp. score"], rows) + "</div>", unsafe_allow_html=True)


# =============================================================================
# APP SHELL
# =============================================================================
NAV = {
    "Dashboard": "▦", "Comment Explorer": "☰", "Audience Insights": "◉", "Content Opportunities": "✦",
    "Language Intelligence": "◍", "Moderation & Toxicity": "⚑", "Reports": "❐",
}


def main():
    df = load_pipeline()
    with st.sidebar:
        st.markdown(flat('''<div class="brand"><div class="brand-logo">✦</div><div><div class="brand-name">CreatorLens <span>AI</span></div>
        <div class="brand-sub">Understand · Analyze · Grow</div></div></div>'''), unsafe_allow_html=True)
        page = st.radio("Navigation", list(NAV), format_func=lambda x: f"{NAV[x]}   {x}", label_visibility="collapsed")
        st.markdown(flat('''<div class="side-label">Platforms</div>
        <div class="plat"><i style="background:#B5473A">▶</i>YouTube</div><div class="plat"><i style="background:#B8713B">◎</i>Instagram</div>
        <div class="side-promo"><span class="crown">♛</span><b>Better Insights<br>Bigger Ideas<br>More Engagement</b></div>
        <div class="side-user"><div class="avatar">CL</div><div><b>CreatorLens</b><span>Content Creator</span></div></div>'''), unsafe_allow_html=True)

    opps, faqs = rank_opportunities(df), extract_faqs(df)
    if page == "Dashboard":
        render_dashboard(df, opps, faqs)
    elif page == "Comment Explorer":
        render_explorer(df)
    elif page == "Audience Insights":
        render_insights(df, faqs)
    elif page == "Content Opportunities":
        render_opportunities(opps)
    elif page == "Language Intelligence":
        render_language(df)
    elif page == "Moderation & Toxicity":
        render_moderation(df)
    else:
        render_reports(df, opps)


main()
