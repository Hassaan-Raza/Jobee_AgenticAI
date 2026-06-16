import os
import html
import hashlib

import streamlit as st
import ollama
from dotenv import load_dotenv

from cv_parser import extract_text, extract_profile
from job_sources import (
    fetch_all_jobs,
    ALL_SOURCE_NAMES,
    guess_adzuna_country,
    guess_jobicy_geo,
)
from matcher import categorize_jobs, to_usd_per_year
from reranker import rerank_borderline

load_dotenv()

st.set_page_config(
    page_title="Jobee — AI Job & Internship Radar",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ──────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────

def _secret(key, default=""):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY") or _secret("OLLAMA_API_KEY", "")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL") or _secret("OLLAMA_MODEL", "gemma4:cloud")

ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID") or _secret("ADZUNA_APP_ID", "")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY") or _secret("ADZUNA_APP_KEY", "")
JSEARCH_API_KEY = os.getenv("JSEARCH_API_KEY") or _secret("JSEARCH_API_KEY", "")


@st.cache_resource(show_spinner=False)
def get_llm_client():
    return ollama.Client(host="https://ollama.com", headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"})


# ──────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────

DEFAULTS = {
    "cv_text": None,
    "cv_profile": None,
    "cv_hash": None,
    "results": None,
    "source_counts": None,
    "search_errors": [],
    "queries_used": [],
    "show_counts": {"matches": 25, "might_interest": 25, "consider": 25},
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ──────────────────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
    """,
    unsafe_allow_html=True,
)

CSS = """
:root {
  --bg: #07020D;
  --bg2: #0F0820;
  --cyan: #00E6FF;
  --pink: #FF2EB0;
  --violet: #8B5CF6;
  --green: #22C55E;
  --text: #ECEAF6;
  --muted: #9D97B5;
  --card-bg: rgba(255,255,255,0.035);
  --card-border: rgba(255,255,255,0.08);
}

html, body, [data-testid="stAppViewContainer"], .stApp {
  background:
    radial-gradient(circle at 12% 8%, rgba(0,230,255,0.10), transparent 38%),
    radial-gradient(circle at 88% 18%, rgba(255,46,176,0.10), transparent 40%),
    radial-gradient(circle at 50% 100%, rgba(139,92,246,0.08), transparent 45%),
    var(--bg) !important;
  color: var(--text);
}

* { font-family: 'Inter', sans-serif; }
.hero-title, .section-title, .cat-title, .profile-name { font-family: 'Sora', sans-serif; }
.score-ring span, .tag-pill, .eyebrow, .section-label, .cat-count { font-family: 'JetBrains Mono', monospace; }

[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1100px; }

/* ---------- hero ---------- */
.eyebrow {
  font-size: 0.75rem; letter-spacing: 0.28em; color: var(--cyan);
  text-transform: uppercase; margin-bottom: 0.6rem;
}
.hero-title {
  font-size: 3.6rem; font-weight: 800; line-height: 1.04; margin: 0;
  background: linear-gradient(120deg, var(--cyan) 0%, var(--pink) 55%, var(--violet) 100%);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub {
  color: var(--muted); font-size: 1.05rem; max-width: 620px;
  margin-top: 0.9rem; line-height: 1.65;
}

.radar-row { display: flex; align-items: center; gap: 1.6rem; flex-wrap: wrap; margin-top: 1.4rem; }
.radar {
  width: 104px; height: 104px; border-radius: 50%; flex-shrink: 0;
  border: 1px solid rgba(0,230,255,0.25); position: relative; overflow: hidden;
  background: radial-gradient(circle, rgba(0,230,255,0.06) 0%, transparent 70%);
}
.radar::before {
  content: ""; position: absolute; inset: 0; border-radius: 50%;
  background: conic-gradient(from 0deg, rgba(0,230,255,0.55), transparent 70deg);
  animation: spin 3.2s linear infinite;
}
.radar::after {
  content: ""; position: absolute; top: 50%; left: 50%; width: 8px; height: 8px;
  border-radius: 50%; background: var(--pink); box-shadow: 0 0 16px var(--pink);
  transform: translate(-50%, -50%);
}
@keyframes spin { to { transform: rotate(360deg); } }

.source-pills { display: flex; flex-wrap: wrap; gap: 0.4rem; max-width: 420px; }

/* ---------- section headers ---------- */
.section-label {
  font-size: 0.72rem; letter-spacing: 0.24em; color: var(--violet);
  text-transform: uppercase; margin-top: 2.4rem; margin-bottom: 0.35rem;
}
.section-title { font-size: 1.55rem; font-weight: 700; margin: 0 0 1rem 0; }

/* ---------- cards ---------- */
.profile-card, .job-card, .info-card {
  background: var(--card-bg); border: 1px solid var(--card-border);
  border-radius: 16px; padding: 1.3rem 1.5rem; margin-bottom: 1rem;
  backdrop-filter: blur(10px);
  animation: fadeUp 0.4s ease both;
}
@keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

.job-card { transition: border-color 0.2s, box-shadow 0.2s, transform 0.2s; }
.job-card:hover {
  border-color: rgba(0,230,255,0.35);
  box-shadow: 0 0 28px rgba(0,230,255,0.10), 0 0 28px rgba(255,46,176,0.08);
  transform: translateY(-2px);
}

.info-card { color: var(--muted); line-height: 1.6; }

/* ---------- profile card ---------- */
.exp-badge {
  display: inline-block; padding: 0.25rem 0.8rem; border-radius: 999px;
  background: linear-gradient(120deg, rgba(0,230,255,0.18), rgba(255,46,176,0.18));
  border: 1px solid rgba(0,230,255,0.3);
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
  margin-bottom: 0.7rem;
}
.profile-name { font-size: 1.25rem; font-weight: 700; }
.profile-summary { color: var(--muted); margin: 0.4rem 0 0.9rem 0; line-height: 1.55; }
.field-label {
  color: var(--muted); font-size: 0.72rem; text-transform: uppercase;
  letter-spacing: 0.16em; display: block; margin-bottom: 0.35rem;
}

/* ---------- pills ---------- */
.tag-pill {
  display: inline-block; padding: 0.2rem 0.65rem; margin: 0.18rem 0.25rem 0.18rem 0;
  border-radius: 999px; font-size: 0.7rem; font-weight: 500;
  background: rgba(139,92,246,0.14); border: 1px solid rgba(139,92,246,0.3); color: #D9CFFF;
}
.source-pill { background: rgba(0,230,255,0.10); border-color: rgba(0,230,255,0.3); color: #9CEEFF; }
.remote-pill { background: rgba(255,46,176,0.10); border-color: rgba(255,46,176,0.3); color: #FFB6E3; }
.salary-pill { background: rgba(34,197,94,0.10); border-color: rgba(34,197,94,0.35); color: #9CFFCB; }
.skill-pill { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.12); color: var(--text); }
.ai-pill {
  background: linear-gradient(120deg, rgba(0,230,255,0.16), rgba(255,46,176,0.16));
  border: 1px solid rgba(0,230,255,0.35); color: #E8FBFF; font-weight: 700;
}

/* ---------- job card internals ---------- */
.job-card-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }
.job-title { font-size: 1.05rem; font-weight: 700; line-height: 1.35; }
.job-sub { color: var(--muted); font-size: 0.85rem; margin-top: 0.2rem; }
.badges { margin-top: 0.75rem; }
.reasons { margin: 0.65rem 0 0 0; padding-left: 1.15rem; color: #BFFCEB; font-size: 0.8rem; }
.reasons li { margin-bottom: 0.15rem; }
.job-desc { color: var(--muted); font-size: 0.85rem; line-height: 1.55; margin-top: 0.65rem; }

.apply-btn {
  display: block; text-align: center; margin-top: 1rem; padding: 0.6rem 1.2rem; border-radius: 10px;
  background: linear-gradient(120deg, var(--cyan), var(--pink));
  color: #07020D !important; font-weight: 800; font-size: 0.85rem; letter-spacing: 0.02em;
  text-decoration: none; box-shadow: 0 0 18px rgba(0,230,255,0.12);
}
.apply-btn:hover { filter: brightness(1.1); box-shadow: 0 0 24px rgba(255,46,176,0.22); }

/* ---------- score ring ---------- */
.score-ring {
  --pct: 0; width: 54px; height: 54px; border-radius: 50%; flex-shrink: 0;
  background: conic-gradient(var(--ring-color, var(--cyan)) calc(var(--pct) * 1%), rgba(255,255,255,0.07) 0);
  display: flex; align-items: center; justify-content: center; position: relative;
}
.score-ring::before { content: ""; position: absolute; inset: 4px; border-radius: 50%; background: #0c0720; }
.score-ring span { position: relative; z-index: 1; font-weight: 700; font-size: 0.74rem; }

/* ---------- category headers ---------- */
.cat-header { display: flex; align-items: baseline; gap: 0.7rem; margin: 0.2rem 0 0.4rem 0; }
.cat-title { font-size: 1.3rem; font-weight: 800; }
.cat-count { color: var(--muted); font-size: 0.85rem; }
.cat-matches { background: linear-gradient(120deg, var(--cyan), var(--green)); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.cat-interest { background: linear-gradient(120deg, var(--pink), var(--violet)); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.cat-consider { background: linear-gradient(120deg, var(--violet), var(--cyan)); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.cat-sub { color: var(--muted); font-size: 0.85rem; margin-bottom: 1.1rem; }

/* ---------- widgets ---------- */
.stTextInput input, .stNumberInput input, .stTextArea textarea,
[data-baseweb="select"] > div, [data-baseweb="base-input"] {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  color: var(--text) !important; border-radius: 10px !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
  border-color: var(--cyan) !important; box-shadow: 0 0 0 1px var(--cyan) !important;
}
[data-baseweb="select"] svg { fill: var(--muted) !important; }
ul[data-testid="stSelectboxVirtualDropdown"], div[role="listbox"] {
  background: var(--bg2) !important; color: var(--text) !important;
}
[data-testid="stWidgetLabel"] p { color: var(--muted) !important; font-size: 0.85rem; }

.stButton > button {
  background: linear-gradient(120deg, var(--cyan), var(--pink)) !important;
  color: #07020D !important; font-weight: 800 !important;
  border: none !important; border-radius: 12px !important;
  padding: 0.6rem 1.4rem !important; letter-spacing: 0.02em;
}
.stButton > button:hover { filter: brightness(1.08); }
.stButton > button p, .stButton > button div { color: #07020D !important; }
.stButton > button:disabled { opacity: 0.35 !important; }

[data-testid="stFileUploaderDropzone"] {
  background: rgba(255,255,255,0.03) !important;
  border: 1px dashed rgba(0,230,255,0.35) !important;
  border-radius: 14px !important;
}
[data-testid="stFileUploaderDropzone"] * { color: var(--text) !important; }
[data-testid="stFileUploaderDropzone"] button {
  background: rgba(0,230,255,0.12) !important; border: 1px solid rgba(0,230,255,0.35) !important;
}

[data-testid="stMetricValue"] {
  font-family: 'Sora', sans-serif; font-weight: 800;
  background: linear-gradient(120deg, var(--cyan), var(--pink));
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; }

[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 0.4rem; border-bottom: 1px solid rgba(255,255,255,0.08); }
[data-baseweb="tab"] p { color: var(--muted) !important; font-weight: 600; }
[data-baseweb="tab"][aria-selected="true"] { border-bottom: 2px solid var(--cyan) !important; }
[data-baseweb="tab"][aria-selected="true"] p { color: var(--text) !important; }
[data-baseweb="tab-highlight"] { background-color: var(--cyan) !important; }

[data-testid="stExpander"] {
  background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 14px;
}
[data-testid="stExpander"] summary p { color: var(--text) !important; }

hr { border-color: rgba(255,255,255,0.08); }

.footer {
  margin-top: 3rem; padding-top: 1.3rem; border-top: 1px solid rgba(255,255,255,0.08);
  color: var(--muted); font-size: 0.8rem; text-align: center; line-height: 1.7;
}
.footer b { color: var(--text); }

@media (max-width: 768px) {
  .hero-title { font-size: 2.4rem; }
  .radar { display: none; }
}
"""

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def esc(s):
    return html.escape(str(s)) if s is not None else ""


JOB_TYPE_LABELS = {
    "internship": "Internship",
    "full_time": "Full time",
    "part_time": "Part time",
    "contract": "Contract",
    "other": "Job",
}

PERIOD_SUFFIX = {"year": "/yr", "month": "/mo", "hour": "/hr", "day": "/day", "week": "/wk"}


def format_salary(job):
    lo, hi = job.get("salary_min"), job.get("salary_max")
    if lo is None and hi is None:
        return None
    cur = job.get("salary_currency") or "USD"
    suffix = PERIOD_SUFFIX.get((job.get("salary_period") or "year").lower(), "/yr")

    def fmt(n):
        return f"{n:,.0f}"

    # Use explicit None checks — lo=0 is a valid salary value and must not be treated as falsy
    if lo is not None and hi is not None and lo != hi:
        return f"{cur} {fmt(lo)}\u2013{fmt(hi)}{suffix}"
    val = hi if hi is not None else lo
    return f"{cur} {fmt(val)}{suffix}"


def render_job_card(job, ring_color):
    pct = int(round(job.get("similarity", 0) * 100))
    salary = format_salary(job)
    jtype = JOB_TYPE_LABELS.get(job.get("job_type", "other"), "Job")

    badges = [
        f'<span class="tag-pill source-pill">{esc(job["source"])}</span>',
        f'<span class="tag-pill">{esc(jtype)}</span>',
    ]
    if job.get("remote"):
        badges.append('<span class="tag-pill remote-pill">Remote</span>')
    # Show salary pill whenever the API gave us salary data.
    # format_salary() already returns None when both salary_min and salary_max are None.
    if salary:
        badges.append(f'<span class="tag-pill salary-pill">{esc(salary)}</span>')
    for t in (job.get("tags") or [])[:4]:
        if t:
            badges.append(f'<span class="tag-pill">{esc(t)}</span>')
    if job.get("ai_reviewed"):
        badges.append('<span class="tag-pill ai-pill">\U0001F916 AI reviewed</span>')

    reasons_html = ""
    if job.get("match_reasons"):
        items = "".join(f"<li>{esc(r)}</li>" for r in job["match_reasons"])
        reasons_html = f'<ul class="reasons">{items}</ul>'

    desc = job.get("description", "")
    short_desc = desc[:260] + ("\u2026" if len(desc) > 260 else "")

    location = job.get("location") or "Location not specified"
    company = job.get("company") or "Unknown company"
    url = job.get("url") or "#"

    return f"""
    <div class="job-card">
      <div class="job-card-head">
        <div>
          <div class="job-title">{esc(job['title'])}</div>
          <div class="job-sub">{esc(company)} \u00b7 {esc(location)}</div>
        </div>
        <div class="score-ring" style="--pct:{pct}; --ring-color:{ring_color};"><span>{pct}%</span></div>
      </div>
      <div class="badges">{''.join(badges)}</div>
      {reasons_html}
      <div class="job-desc">{esc(short_desc)}</div>
      <a class="apply-btn" href="{esc(url)}" target="_blank" rel="noopener">Apply Now \u2192</a>
    </div>
    """


def build_queries(cv_profile, user_keywords):
    if user_keywords and user_keywords.strip():
        queries = [q.strip() for q in user_keywords.split(",") if q.strip()]
    else:
        queries = list((cv_profile or {}).get("target_roles", []))[:4]
    seen, final = set(), []
    for q in queries:
        ql = q.lower()
        if ql not in seen:
            seen.add(ql)
            final.append(q)
    return final[:4] if final else ["Software Engineer"]


# ──────────────────────────────────────────────────────────────────────────
# Hero
# ──────────────────────────────────────────────────────────────────────────

st.markdown('<div class="eyebrow">AI Job &amp; Internship Radar</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Jobee</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Upload your CV. Jobee reads it, figures out what you should be applying for, '
    'then scans seven job sources at once and sorts everything into what fits you, what might fit you, '
    'and what is just worth knowing about.</div>',
    unsafe_allow_html=True,
)

source_pill_html = "".join(f'<span class="tag-pill source-pill">{s}</span>' for s in ALL_SOURCE_NAMES)
st.markdown(
    f"""
    <div class="radar-row">
      <div class="radar"></div>
      <div>
        <div class="field-label">Scanning</div>
        <div class="source-pills">{source_pill_html}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────
# Step 1 — CV upload
# ──────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-label">Step 01 \u00b7 Upload</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Drop in your CV</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("CV", type=["pdf", "docx", "txt"], label_visibility="collapsed")

if uploaded is not None:
    file_bytes = uploaded.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    if st.session_state.cv_hash != file_hash:
        with st.spinner("Reading your CV and building your profile..."):
            try:
                text = extract_text(file_bytes, uploaded.name)
                client = get_llm_client()
                profile = extract_profile(text, client, OLLAMA_MODEL)
                st.session_state.cv_text = text
                st.session_state.cv_profile = profile
                st.session_state.cv_hash = file_hash
                st.session_state.results = None
            except Exception as e:
                st.error(f"Couldn't read that file: {e}")

if st.session_state.cv_profile:
    profile = st.session_state.cv_profile
    skills_html = "".join(f'<span class="tag-pill skill-pill">{esc(s)}</span>' for s in profile.get("skills", [])[:18])
    roles_html = "".join(f'<span class="tag-pill">{esc(r)}</span>' for r in profile.get("target_roles", []))
    name_line = esc(profile.get("name")) or "Your profile"
    st.markdown(
        f"""
        <div class="profile-card">
          <div class="exp-badge">{esc(profile.get('experience_level', 'Entry'))} level</div>
          <div class="profile-name">{name_line}</div>
          <div class="profile-summary">{esc(profile.get('summary', ''))}</div>
          <span class="field-label">Target roles</span>
          <div style="margin-bottom:0.8rem;">{roles_html}</div>
          <span class="field-label">Detected skills</span>
          <div>{skills_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="info-card">Upload a PDF, DOCX, or TXT CV to get started. '
        'Jobee reads it, works out what roles fit you, and scans every job source for matches.</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────
# Step 2 — preferences
# ──────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-label">Step 02 \u00b7 Preferences</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Tell Jobee what you want</div>', unsafe_allow_html=True)

default_keywords = ", ".join((st.session_state.cv_profile or {}).get("target_roles", []))
keywords = st.text_input(
    "Search keywords (comma separated)",
    value=default_keywords,
    placeholder="e.g. Machine Learning Engineer, AI Engineer, Python Developer",
)

c1, c2, c3 = st.columns(3)
with c1:
    city = st.text_input("City", placeholder="e.g. Lahore")
with c2:
    country = st.text_input("Country", placeholder="e.g. Pakistan")
with c3:
    job_type_choice = st.selectbox("Role type", ["Internship", "Any (jobs & internships)"])

sc1, sc2, sc3, sc4 = st.columns(4)
if job_type_choice == "Internship":
    # Salary inputs hidden for internship searches — most internships are unpaid
    # or have no listed salary, so these fields are irrelevant noise.
    st.caption("💡 Salary filters are hidden for internship searches since most internships don't list pay.")
    salary_min = 0
    salary_max = 0
    currency = "USD"
    period_choice = "Per Month"
else:
    with sc1:
        salary_min = st.number_input("Min pay (optional)", min_value=0, value=0, step=100)
    with sc2:
        salary_max = st.number_input("Max pay (optional)", min_value=0, value=0, step=100)
    with sc3:
        currency = st.selectbox("Currency", ["USD", "PKR", "GBP", "EUR", "INR", "AED", "CAD", "AUD", "SGD"])
    with sc4:
        period_choice = st.selectbox("Period", ["Per Year", "Per Month"])

adzuna_ready = bool(ADZUNA_APP_ID and ADZUNA_APP_KEY)
jsearch_ready = bool(JSEARCH_API_KEY)
adzuna_id = ADZUNA_APP_ID
adzuna_key = ADZUNA_APP_KEY
jsearch_key = JSEARCH_API_KEY

# ──────────────────────────────────────────────────────────────────────────
# Search
# ──────────────────────────────────────────────────────────────────────────

st.markdown("<div style='margin-top:1.4rem;'></div>", unsafe_allow_html=True)

search_disabled = st.session_state.cv_profile is None

btn_col, toggle_col = st.columns([3, 2])
with btn_col:
    search_clicked = st.button("\U0001F6F0\uFE0F Scan the job universe", disabled=search_disabled)
    if search_disabled:
        st.caption("Upload your CV above to enable scanning.")
with toggle_col:
    sources_label = ""
    if adzuna_ready and jsearch_ready:
        sources_label = "Adzuna + JSearch active"
    elif adzuna_ready:
        sources_label = "Adzuna active"
    elif jsearch_ready:
        sources_label = "JSearch active"

    deep_search = st.toggle(
        "🔍 Deep search" + (f" ({sources_label})" if sources_label else ""),
        value=bool(adzuna_ready or jsearch_ready),
        help="Enables Adzuna and JSearch on top of the 5 free sources. Requires API keys in your .env file.",
    )
    if not (adzuna_ready or jsearch_ready):
        st.caption("Add ADZUNA_APP_ID / JSEARCH_API_KEY to .env to enable.")

if search_clicked:
    profile = st.session_state.cv_profile
    queries = build_queries(profile, keywords)

    period_key = "year" if period_choice == "Per Year" else "month"
    salary_min_usd = to_usd_per_year(salary_min, currency, period_key) if salary_min > 0 else None
    salary_max_usd = to_usd_per_year(salary_max, currency, period_key) if salary_max > 0 else None

    location_terms = [t for t in [city.strip().lower(), country.strip().lower()] if t]
    job_type_pref = "internship" if job_type_choice == "Internship" else "any"

    prefs = {
        "location": ", ".join([p for p in [city.strip(), country.strip()] if p]),
        "adzuna_country": guess_adzuna_country(country),
        "jobicy_geo": guess_jobicy_geo(country),
        "himalayas_employment_type": "Intern" if job_type_pref == "internship" else None,
        "salary_min": salary_min_usd,
    }
    api_keys = {
        "adzuna_app_id": adzuna_id.strip() if deep_search else "",
        "adzuna_app_key": adzuna_key.strip() if deep_search else "",
        "jsearch_key": jsearch_key.strip() if deep_search else "",
    }

    with st.spinner(f"Scanning {len(queries)} search term(s) across every source..."):
        jobs, source_counts, errors = fetch_all_jobs(queries, prefs, api_keys)
        match_prefs = {
            "salary_min_usd": salary_min_usd,
            "salary_max_usd": salary_max_usd,
            "location_terms": location_terms,
            "job_type": job_type_pref,
        }
        buckets = categorize_jobs(jobs, st.session_state.cv_text or "", profile, match_prefs)

    with st.spinner("Jobee's AI is checking the borderline matches..."):
        try:
            buckets = rerank_borderline(buckets, profile, get_llm_client(), OLLAMA_MODEL)
        except Exception:
            pass  # embedding-only categorisation already in `buckets`, never block on this

    st.session_state.results = buckets
    st.session_state.source_counts = source_counts
    st.session_state.search_errors = errors
    st.session_state.queries_used = queries
    st.session_state.show_counts = {"matches": 25, "might_interest": 25, "consider": 25}


# ──────────────────────────────────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────────────────────────────────

if st.session_state.results is not None:
    buckets = st.session_state.results
    total = sum(len(v) for v in buckets.values())
    counts = st.session_state.source_counts or {}
    active_sources = sum(1 for v in counts.values() if v > 0)

    st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Here\u2019s everything Jobee found</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total jobs found", total)
    m2.metric("\U0001F3AF Matches", len(buckets["matches"]))
    m3.metric("\u2728 Might interest you", len(buckets["might_interest"]))
    m4.metric("Sources with hits", f"{active_sources}/{len(ALL_SOURCE_NAMES)}")

    pills = "".join(
        f'<span class="tag-pill source-pill">{name}: {counts.get(name, 0)}</span>' for name in ALL_SOURCE_NAMES
    )
    st.markdown(f'<div style="margin: 0.7rem 0 1.5rem 0;">{pills}</div>', unsafe_allow_html=True)

    if st.session_state.queries_used:
        searched = ", ".join(st.session_state.queries_used)
        st.caption(f"Searched for: {searched}")

    if total == 0:
        st.markdown(
            '<div class="info-card">Jobee didn\u2019t find anything for these terms. '
            'Try broader keywords, remove the salary filter, or add API keys above for wider coverage.</div>',
            unsafe_allow_html=True,
        )
    else:
        tab_labels = [
            f"\U0001F3AF Matches ({len(buckets['matches'])})",
            f"\u2728 Might Interest You ({len(buckets['might_interest'])})",
            f"\U0001F310 Worth Considering ({len(buckets['consider'])})",
        ]
        tabs = st.tabs(tab_labels)

        configs = [
            ("matches", "var(--cyan)", "cat-matches", "Your best matches",
             "These line up with your skills, location, pay range, and role type."),
            ("might_interest", "var(--pink)", "cat-interest", "You might like these",
             "Strong overlap with your CV, but something (location, pay, or role type) didn\u2019t fully line up."),
            ("consider", "var(--violet)", "cat-consider", "Worth a look anyway",
             "Everything else Jobee\u2019s search turned up. Lower match score, but still in your search space."),
        ]

        for tab, (key, ring_color, css_class, title, subtitle) in zip(tabs, configs):
            with tab:
                jobs_list = buckets[key]
                if not jobs_list:
                    st.markdown('<div class="info-card">Nothing landed here this time.</div>', unsafe_allow_html=True)
                    continue

                st.markdown(
                    f"""
                    <div class="cat-header">
                      <span class="cat-title {css_class}">{title}</span>
                      <span class="cat-count">{len(jobs_list)} jobs</span>
                    </div>
                    <div class="cat-sub">{subtitle}</div>
                    """,
                    unsafe_allow_html=True,
                )

                show_n = st.session_state.show_counts.get(key, 12)
                for job in jobs_list[:show_n]:
                    st.markdown(render_job_card(job, ring_color), unsafe_allow_html=True)

                remaining = len(jobs_list) - show_n
                if remaining > 0:
                    if st.button(f"Show {min(25, remaining)} more (of {remaining} remaining)", key=f"more_{key}"):
                        st.session_state.show_counts[key] = show_n + 25
                        st.rerun()

    if st.session_state.search_errors:
        with st.expander("Source errors (debug)"):
            for err in st.session_state.search_errors:
                st.caption(err)


# ──────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div class="footer">
      <b>Jobee</b> \u00b7 AI Job &amp; Internship Radar \u00b7 built by Hassaan Raza<br>
      Searches Remotive, Arbeitnow, RemoteOK, Jobicy, Himalayas and with your own free keys, Adzuna &amp; JSearch.<br>
      CV parsing via PyMuPDF / python-docx \u00b7 matching via sentence-transformers \u00b7 profile extraction via Ollama Cloud ({esc(OLLAMA_MODEL)})<br>
      Pay comparisons use approximate currency conversion so always confirm details with the employer.
    </div>
    """,
    unsafe_allow_html=True,
)
