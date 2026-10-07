"""Custom CSS for the Streamlit app. No remote fonts or network calls.

Colours are translucent greys or fixed mid-tones with white text, so every
component stays readable in both Streamlit's light and dark themes.
"""

from __future__ import annotations

ACCENT = "#0f766e"

CSS = """
:root {
  --rp-accent: #0f766e;
  --rp-card: rgba(127, 127, 127, 0.09);
  --rp-border: rgba(127, 127, 127, 0.30);
  --rp-radius: 16px;
  --rp-green: 31, 138, 76;
  --rp-amber: 217, 140, 0;
  --rp-red: 198, 40, 40;
}
.block-container { max-width: 1100px; padding-top: 2rem; padding-bottom: 1rem; }

@keyframes rp-fade-up { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes rp-pop { 0% { opacity: 0; transform: scale(.85); } 70% { transform: scale(1.04); } 100% { opacity: 1; transform: scale(1); } }
@keyframes rp-grow { from { width: 0; } }
@keyframes rp-sheen { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
@keyframes rp-pulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(15,118,110,.35); } 50% { box-shadow: 0 0 0 6px rgba(15,118,110,0); } }

.rp-hero {
  text-align: center; padding: 2.2rem 1.5rem 1.6rem; border-radius: 24px; margin-bottom: 1.2rem;
  border: 1px solid var(--rp-border);
  background: linear-gradient(120deg, rgba(15,118,110,.18), rgba(99,102,241,.14), rgba(15,118,110,.18));
  background-size: 200% 200%; animation: rp-sheen 14s ease infinite, rp-fade-up .6s ease both;
}
.rp-hero h1 { margin: 0; font-size: 2.8rem; letter-spacing: -0.02em; padding: 0; }
.rp-hero p { margin: .5rem auto 0; max-width: 40rem; font-size: 1.1rem; opacity: .85; }

.rp-steps { display: grid; grid-template-columns: repeat(4, 1fr); gap: .8rem; margin: 1.4rem 0 .4rem; text-align: left; }
.rp-step {
  display: flex; align-items: center; gap: .7rem; padding: .7rem .9rem; border-radius: 14px;
  background: var(--rp-card); border: 1px solid var(--rp-border); animation: rp-fade-up .6s ease both;
}
.rp-step:nth-child(1) { animation-delay: .10s; } .rp-step:nth-child(2) { animation-delay: .20s; }
.rp-step:nth-child(3) { animation-delay: .30s; } .rp-step:nth-child(4) { animation-delay: .40s; }
.rp-step-num {
  flex: none; width: 2rem; height: 2rem; border-radius: 50%; display: grid; place-items: center;
  background: var(--rp-accent); color: #fff; font-weight: 700;
}
.rp-step b { display: block; line-height: 1.2; }
.rp-step span { font-size: .82rem; opacity: .75; line-height: 1.2; }

.rp-card {
  background: var(--rp-card); border: 1px solid var(--rp-border); border-radius: var(--rp-radius);
  padding: 1rem 1.2rem; animation: rp-fade-up .5s ease both; overflow-wrap: anywhere;
}
.rp-card h4 { margin: 0 0 .5rem; font-size: 1rem; }
.rp-card p { margin: 0; white-space: pre-wrap; }
.rp-compare { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: .8rem 0; }
.rp-compare .rp-card { height: 100%; }
.rp-compare .rp-card:nth-child(2) { animation-delay: .12s; }
.rp-card.rp-good { background: rgba(var(--rp-green), .14); border-color: rgba(var(--rp-green), .55); }
.rp-card.rp-none { background: rgba(var(--rp-amber), .13); border-color: rgba(var(--rp-amber), .60); border-style: dashed; }

.rp-severity { display: flex; align-items: center; gap: 1rem; padding: 1rem 1.2rem; border-radius: var(--rp-radius);
  background: var(--rp-card); border: 1px solid var(--rp-border); animation: rp-fade-up .5s ease both; }
.rp-badge {
  display: inline-flex; align-items: center; gap: .5rem; color: #fff; font-weight: 700;
  font-size: 1.5rem; padding: .45rem 1.2rem; border-radius: 999px; white-space: nowrap;
  animation: rp-pop .55s cubic-bezier(.2,.9,.3,1.2) both;
}
.rp-meaning { font-size: 1rem; opacity: .9; }
.rp-badge-small { font-size: .95rem; font-weight: 600; padding: 3px 12px; border-radius: 999px; color: #fff; display: inline-block; }

.rp-bar-row { display: grid; grid-template-columns: 8.5rem 1fr 3.5rem; align-items: center; gap: .8rem; margin: .45rem 0; }
.rp-bar-track { height: 12px; border-radius: 999px; background: rgba(127,127,127,.25); overflow: hidden; }
.rp-bar-fill { height: 100%; border-radius: 999px; background: var(--rp-accent); animation: rp-grow .9s ease-out both; }
.rp-bar-fill.rp-hot { background: #c62828; }
.rp-bar-val { text-align: right; font-variant-numeric: tabular-nums; }

.rp-legend { display: flex; flex-wrap: wrap; gap: .6rem 1.2rem; align-items: center; font-size: .9rem; margin: .4rem 0 .8rem; }
.rp-swatch { display: inline-block; width: 2.2rem; height: 1rem; border-radius: 4px; vertical-align: middle; margin-right: .3rem; }
.rp-text { line-height: 2.1; font-size: 1.05rem; }

.rp-pill { display: inline-block; padding: 2px 12px; border-radius: 999px; font-size: .85rem; font-weight: 600; border: 1px solid; }
.rp-pill.rp-on { background: rgba(var(--rp-green), .16); border-color: rgba(var(--rp-green), .7); }
.rp-pill.rp-off { background: rgba(var(--rp-amber), .16); border-color: rgba(var(--rp-amber), .8); }

.rp-quote { border-left: 4px solid var(--rp-accent); padding: .3rem .9rem; margin: .5rem 0 .8rem; font-style: italic; }
.rp-example-head { display: flex; justify-content: space-between; align-items: center; gap: .8rem; flex-wrap: wrap; }
.rp-mark { font-weight: 700; padding: 3px 12px; border-radius: 999px; border: 1px solid; }
.rp-mark.rp-ok { background: rgba(var(--rp-green), .16); border-color: rgba(var(--rp-green), .8); }
.rp-mark.rp-bad { background: rgba(var(--rp-red), .16); border-color: rgba(var(--rp-red), .8); }
.rp-meta { display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; margin-top: .5rem; }
.rp-meta small { display: block; opacity: .7; text-transform: uppercase; letter-spacing: .04em; font-size: .72rem; }

.rp-footer { text-align: center; opacity: .75; font-size: .85rem; margin-top: 2.5rem; padding-top: 1rem;
  border-top: 1px solid var(--rp-border); }

[data-testid="stVerticalBlockBorderWrapper"] { border-radius: var(--rp-radius); }
div.stButton > button, div[data-testid="stBaseButton-secondary"] { border-radius: 12px; transition: transform .15s ease, box-shadow .15s ease; }
div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,.18); }
div.stButton > button[kind="primary"] { animation: rp-pulse 2.8s ease-in-out infinite; }
[data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input { border-radius: 12px; }
[data-testid="stSidebar"] .rp-card { padding: .7rem .9rem; }

@media (max-width: 760px) {
  .rp-steps, .rp-meta { grid-template-columns: 1fr 1fr; }
  .rp-compare { grid-template-columns: 1fr; }
  .rp-severity { flex-direction: column; text-align: center; }
  .rp-bar-row { grid-template-columns: 6rem 1fr 3rem; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
"""


def inject() -> None:
    """Inject the app stylesheet (call once per page, after set_page_config)."""
    import streamlit as st

    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
