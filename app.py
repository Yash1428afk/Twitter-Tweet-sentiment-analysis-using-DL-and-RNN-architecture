import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentIQ · Tweet Analyser",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MAXLEN = 166
SENTIMENT_MAP = {0: "Negative", 1: "Neutral", 2: "Positive", 3: "Irrelevant"}

SENTIMENT_META = {
    "Positive": {
        "emoji": "😊",
        "color": "#4ade80",          # green
        "bg":    "#052e16",
        "border":"#166534",
        "badge_bg": "rgba(74,222,128,0.15)",
        "badge_color": "#4ade80",
        "desc": "Expresses a positive emotion or favourable opinion",
        "bar_color": "#4ade80",
    },
    "Negative": {
        "emoji": "😞",
        "color": "#f87171",          # red
        "bg":    "#2d0a0a",
        "border":"#7f1d1d",
        "badge_bg": "rgba(248,113,113,0.15)",
        "badge_color": "#f87171",
        "desc": "Expresses a negative emotion or unfavourable opinion",
        "bar_color": "#f87171",
    },
    "Neutral": {
        "emoji": "😐",
        "color": "#fbbf24",          # amber
        "bg":    "#1c1204",
        "border":"#78350f",
        "badge_bg": "rgba(251,191,36,0.15)",
        "badge_color": "#fbbf24",
        "desc": "Neither clearly positive nor negative in tone",
        "bar_color": "#fbbf24",
    },
    "Irrelevant": {
        "emoji": "🔗",
        "color": "#a78bfa",          # violet
        "bg":    "#1e1040",
        "border":"#4c1d95",
        "badge_bg": "rgba(167,139,250,0.15)",
        "badge_color": "#a78bfa",
        "desc": "Off-topic, promotional, or non-opinionated content",
        "bar_color": "#a78bfa",
    },
}

EXAMPLES = [
    ("😊", "Just got promoted at work! Best day of my life 🎉"),
    ("😞", "This product is an absolute disaster. Worst purchase ever."),
    ("😐", "It rained a bit this morning. Took the bus instead."),
    ("🔗", "RT @TechDaily: New framework drops this Thursday — link in bio"),
]

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS  — deep-ocean dark theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    background: #020b18 !important;
    color: #e2e8f0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Layout ── */
.block-container {
    padding: 2.5rem 2rem 4rem !important;
    max-width: 860px !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #020f1f !important;
    border-right: 1px solid #0f2d4a !important;
}
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stMarkdown p { font-size: 13px !important; line-height: 1.7; }

/* ── Textarea ── */
.stTextArea textarea {
    background: #041221 !important;
    border: 1px solid #0f3460 !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
    padding: 14px 16px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    resize: vertical !important;
}
.stTextArea textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
    outline: none !important;
}
.stTextArea textarea::placeholder { color: #334d6e !important; }
.stTextArea label { color: #64748b !important; font-size: 12px !important; }

/* ── Primary button ── */
div[data-testid="stButton"] > button[kind="primary"],
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%) !important;
    color: #fff !important;
    border: 1px solid #2563eb !important;
    border-radius: 10px !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.5rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
    letter-spacing: 0.01em !important;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
    border-color: #3b82f6 !important;
    box-shadow: 0 4px 20px rgba(59,130,246,0.25) !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0) scale(0.98) !important;
}
div[data-testid="stButton"] > button:disabled {
    background: #0f2040 !important;
    color: #334d6e !important;
    border-color: #0f2040 !important;
    box-shadow: none !important;
    transform: none !important;
    cursor: not-allowed !important;
}

/* ── Radio / selectbox ── */
.stRadio > label, .stSelectbox > label { color: #64748b !important; font-size: 12px !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color: #94a3b8 !important; font-size: 13px !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #3b82f6 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #020b18; }
::-webkit-scrollbar-thumb { background: #0f3460; border-radius: 99px; }

/* ────────────────────────────────────────────
   CUSTOM COMPONENTS
──────────────────────────────────────────── */

/* Header */
.iq-header {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 0 0 1.8rem;
    border-bottom: 1px solid #0a2540;
    margin-bottom: 2rem;
}
.iq-logo {
    width: 56px; height: 56px;
    border-radius: 14px;
    background: linear-gradient(135deg, #0f3460 0%, #1e40af 100%);
    border: 1px solid #1d4ed8;
    display: flex; align-items: center; justify-content: center;
    font-size: 28px; flex-shrink: 0;
}
.iq-title   { font-size: 24px; font-weight: 700; color: #f1f5f9; letter-spacing: -0.02em; }
.iq-tagline { font-size: 13px; color: #475569; margin-top: 3px; }
.iq-pill {
    display: inline-block;
    font-size: 11px; font-family: 'JetBrains Mono', monospace;
    padding: 2px 8px; border-radius: 6px; margin-right: 5px;
    background: #041c38; color: #38bdf8; border: 1px solid #0c3a6b;
}

/* Section label */
.iq-section-label {
    font-size: 11px; font-weight: 600; color: #334d6e;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 10px;
}

/* Input card */
.iq-card {
    background: #041221;
    border: 1px solid #0a2540;
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 1rem;
}

/* Char counter */
.iq-counter {
    text-align: right;
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
    margin-top: -6px;
    margin-bottom: 6px;
}

/* Example chips */
.iq-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 1.2rem; }
.iq-chip {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 12px; color: #64748b;
    background: #041221; border: 1px solid #0a2540;
    border-radius: 20px; padding: 5px 12px;
    cursor: pointer; transition: all 0.15s;
    white-space: nowrap;
}
.iq-chip:hover { border-color: #1d4ed8; color: #93c5fd; background: #041c38; }

/* Result card */
.iq-result {
    border-radius: 16px;
    padding: 1.5rem;
    margin-top: 1.2rem;
    border: 1px solid;
    animation: fadeUp 0.35s cubic-bezier(0.22,1,0.36,1);
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(10px); }
    to   { opacity:1; transform:translateY(0); }
}

/* Orb */
.iq-orb-row { display:flex; align-items:center; gap:18px; margin-bottom:1.2rem; }
.iq-orb {
    width:64px; height:64px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:30px; flex-shrink:0; border:1px solid;
}
.iq-detected-label { font-size:11px; color:#475569; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; }
.iq-detected-value { font-size:28px; font-weight:700; letter-spacing:-0.02em; line-height:1.2; }
.iq-badge {
    display:inline-block; font-size:12px; padding:3px 10px;
    border-radius:20px; margin-top:5px; font-weight:500;
}

/* Confidence bars */
.iq-divider { border:none; border-top:1px solid #0a2540; margin:1.1rem 0; }
.iq-bar-row { display:flex; align-items:center; gap:10px; margin-bottom:9px; }
.iq-bar-name { font-size:13px; color:#475569; width:80px; flex-shrink:0; }
.iq-bar-track {
    flex:1; height:7px;
    background:#041221; border-radius:99px; overflow:hidden;
    border:1px solid #0a2540;
}
.iq-bar-fill  { height:100%; border-radius:99px; transition:width 0.5s ease; }
.iq-bar-pct {
    font-size:12px; font-weight:500;
    width:36px; text-align:right;
    font-family:'JetBrains Mono', monospace;
    color:#64748b; flex-shrink:0;
}

/* Tweet echo */
.iq-echo {
    background:#020b18; border-left:3px solid #1d4ed8;
    border-radius:8px; padding:10px 14px;
    margin-top:1rem;
    font-size:13px; color:#475569; font-style:italic; line-height:1.7;
}

/* History item */
.iq-hist-item {
    background:#041221; border:1px solid #0a2540;
    border-radius:10px; padding:10px 14px; margin-bottom:8px;
}
.iq-hist-sent  { font-size:12px; font-weight:600; }
.iq-hist-tweet { font-size:12px; color:#475569; margin-top:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.iq-hist-time  { font-size:11px; color:#1e3a5f; margin-top:4px; font-family:'JetBrains Mono',monospace; }

/* Stats grid */
.iq-stats-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:1.2rem; }
.iq-stat-card {
    background:#041221; border:1px solid #0a2540;
    border-radius:10px; padding:12px 14px; text-align:center;
}
.iq-stat-num  { font-size:22px; font-weight:700; color:#f1f5f9; }
.iq-stat-label { font-size:11px; color:#334d6e; margin-top:2px; }

/* Error */
.iq-error {
    background:#2d0a0a; border:1px solid #7f1d1d;
    border-radius:10px; padding:12px 14px; margin-top:1rem;
    color:#f87171; font-size:13px; line-height:1.6;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []          # list of dicts
if "tweet_input" not in st.session_state:
    st.session_state.tweet_input = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD MODEL  (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_assets():
    mdl = load_model("model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)
    return mdl, tok

with st.spinner("Loading model…"):
    model, tokenizer = load_assets()

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌊 SentimentIQ")
    st.markdown("---")

    # ── Stats ──────────────────────────────────────────────────────────────
    total = len(st.session_state.history)
    if total:
        counts = {"Positive": 0, "Negative": 0, "Neutral": 0, "Irrelevant": 0}
        for h in st.session_state.history:
            counts[h["sentiment"]] += 1
        top = max(counts, key=counts.get)
        top_emoji = SENTIMENT_META[top]["emoji"]

        st.markdown(f"""
        <div class="iq-stats-grid">
            <div class="iq-stat-card">
                <div class="iq-stat-num">{total}</div>
                <div class="iq-stat-label">Tweets analysed</div>
            </div>
            <div class="iq-stat-card">
                <div class="iq-stat-num">{top_emoji}</div>
                <div class="iq-stat-label">Most common</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Breakdown mini bars
        st.markdown('<div class="iq-section-label">Breakdown</div>', unsafe_allow_html=True)
        for sent, cnt in counts.items():
            if cnt == 0:
                continue
            pct = round(cnt / total * 100)
            color = SENTIMENT_META[sent]["bar_color"]
            st.markdown(f"""
            <div class="iq-bar-row" style="margin-bottom:7px;">
                <span class="iq-bar-name">{SENTIMENT_META[sent]['emoji']} {sent[:4]}</span>
                <div class="iq-bar-track">
                    <div class="iq-bar-fill" style="width:{pct}%;background:{color};"></div>
                </div>
                <span class="iq-bar-pct">{pct}%</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

    # ── History ────────────────────────────────────────────────────────────
    if st.session_state.history:
        st.markdown("### Recent analyses")
        for item in reversed(st.session_state.history[-8:]):
            m = SENTIMENT_META[item["sentiment"]]
            st.markdown(f"""
            <div class="iq-hist-item">
                <span class="iq-hist-sent" style="color:{m['color']};">{m['emoji']} {item['sentiment']}</span>
                <div class="iq-hist-tweet">{item['tweet'][:60]}{'…' if len(item['tweet'])>60 else ''}</div>
                <div class="iq-hist-time">{item['time']}</div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑 Clear history", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()
    else:
        st.markdown("""
        <p style="font-size:13px;color:#1e3a5f;">
        Your analysis history will appear here after you run your first prediction.
        </p>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── About ──────────────────────────────────────────────────────────────
    st.markdown("### About")
    st.markdown("""
    <p>
    LSTM-based multi-class text classifier trained on the
    <strong style="color:#e2e8f0;">Twitter Entity Sentiment</strong> dataset.
    </p>
    <p style="margin-top:8px;">
    <span class="iq-pill">TensorFlow 2.x</span>
    <span class="iq-pill">Keras</span><br><br>
    <span class="iq-pill">LSTM</span>
    <span class="iq-pill">Tokenizer</span>
    </p>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN — Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="iq-header">
    <div class="iq-logo">🌊</div>
    <div>
        <div class="iq-title">SentimentIQ</div>
        <div class="iq-tagline">
            <span class="iq-pill">LSTM</span>
            <span class="iq-pill">4-class NLP</span>
            <span class="iq-pill">TensorFlow</span>
            &nbsp;Tweet sentiment classifier
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN — Example chips  (use session_state to prefill textarea)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="iq-section-label">Quick examples</div>', unsafe_allow_html=True)

chip_cols = st.columns(len(EXAMPLES))
for i, (col, (emoji, text)) in enumerate(zip(chip_cols, EXAMPLES)):
    with col:
        short = text[:26] + "…"
        if st.button(f"{emoji} {short}", key=f"chip_{i}", use_container_width=True):
            st.session_state.tweet_input = text
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN — Input area
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)

tweet = st.text_area(
    label="tweet_input",
    value=st.session_state.tweet_input,
    placeholder="Paste or type any tweet here…",
    height=130,
    max_chars=280,
    label_visibility="collapsed",
    key="tweet_area",
)

# Sync session state
st.session_state.tweet_input = tweet

# Character counter
char_len = len(tweet)
if char_len > 260:
    counter_color = "#f87171"
elif char_len > 220:
    counter_color = "#fbbf24"
else:
    counter_color = "#1e3a5f"

st.markdown(
    f'<div class="iq-counter" style="color:{counter_color};">{char_len} / 280</div>',
    unsafe_allow_html=True
)

analyse_clicked = st.button(
    "🔍  Analyse sentiment",
    disabled=not tweet.strip(),
    use_container_width=True,
)

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN — Inference
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(text: str) -> dict:
    """Tokenise, pad, predict. Returns a result dict."""
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, padding="post", maxlen=MAXLEN)
    raw_probs = model.predict(seq, verbose=0)[0]          # shape (4,)
    pred_idx  = int(np.argmax(raw_probs))
    sentiment = SENTIMENT_MAP[pred_idx]
    return {
        "sentiment": sentiment,
        "tweet":     text,
        "time":      datetime.now().strftime("%H:%M:%S"),
        "scores": {
            "Negative":   float(raw_probs[0]),
            "Neutral":    float(raw_probs[1]),
            "Positive":   float(raw_probs[2]),
            "Irrelevant": float(raw_probs[3]),
        },
    }


def render_result(result: dict):
    """
    Render the result card using st.components.v1.html so Streamlit never
    escapes the markup — fixes the raw-HTML-as-text bug on reruns.
    """
    s  = result["sentiment"]
    m  = SENTIMENT_META[s]
    sc = result["scores"]

    def bar(label, key):
        pct = round(sc[key] * 100)
        color = SENTIMENT_META[key]["bar_color"]
        return f"""
        <div class="iq-bar-row">
          <span class="iq-bar-name">{label}</span>
          <div class="iq-bar-track">
            <div class="iq-bar-fill" style="width:{pct}%;background:{color};"></div>
          </div>
          <span class="iq-bar-pct">{pct}%</span>
        </div>"""

    preview = result["tweet"][:140] + ("…" if len(result["tweet"]) > 140 else "")

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
      * {{ box-sizing:border-box; margin:0; padding:0; }}
      body {{
        font-family:'Inter',sans-serif;
        background:transparent;
        padding: 4px 0 8px 0;
      }}

      .result-card {{
        background:{m['bg']};
        border:1px solid {m['border']};
        border-radius:16px;
        padding:1.4rem 1.5rem;
        animation:fadeUp 0.35s cubic-bezier(0.22,1,0.36,1);
      }}
      @keyframes fadeUp {{
        from {{ opacity:0; transform:translateY(10px); }}
        to   {{ opacity:1; transform:translateY(0); }}
      }}

      .orb-row {{ display:flex; align-items:center; gap:18px; margin-bottom:1.2rem; }}
      .orb {{
        width:64px; height:64px; border-radius:50%;
        background:{m['bg']}; border:1px solid {m['border']};
        display:flex; align-items:center; justify-content:center;
        font-size:30px; flex-shrink:0;
      }}
      .det-label {{
        font-size:11px; color:#475569; font-weight:600;
        text-transform:uppercase; letter-spacing:0.08em;
      }}
      .det-value {{
        font-size:28px; font-weight:700;
        color:{m['color']}; letter-spacing:-0.02em; line-height:1.2;
      }}
      .badge {{
        display:inline-block; font-size:12px; padding:3px 10px;
        border-radius:20px; margin-top:5px; font-weight:500;
        background:{m['badge_bg']}; color:{m['badge_color']};
      }}

      .divider {{ border:none; border-top:1px solid #0a2540; margin:1.1rem 0; }}
      .section-label {{
        font-size:11px; font-weight:600; color:#334d6e;
        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:10px;
      }}

      .iq-bar-row {{ display:flex; align-items:center; gap:10px; margin-bottom:9px; }}
      .iq-bar-name {{ font-size:13px; color:#475569; width:80px; flex-shrink:0; }}
      .iq-bar-track {{
        flex:1; height:7px; background:#041221;
        border-radius:99px; overflow:hidden; border:1px solid #0a2540;
      }}
      .iq-bar-fill  {{ height:100%; border-radius:99px; }}
      .iq-bar-pct {{
        font-size:12px; font-weight:500; width:36px; text-align:right;
        font-family:'JetBrains Mono',monospace; color:#64748b; flex-shrink:0;
      }}

      .echo {{
        background:#020b18; border-left:3px solid #1d4ed8;
        border-radius:8px; padding:10px 14px; margin-top:1rem;
        font-size:13px; color:#475569; font-style:italic; line-height:1.7;
      }}
    </style>
    </head>
    <body>
    <div class="result-card">
      <div class="orb-row">
        <div class="orb">{m['emoji']}</div>
        <div>
          <div class="det-label">Detected sentiment</div>
          <div class="det-value">{s}</div>
          <span class="badge">{m['desc']}</span>
        </div>
      </div>

      <hr class="divider">
      <div class="section-label">Confidence breakdown</div>

      {bar('Positive',   'Positive')}
      {bar('Neutral',    'Neutral')}
      {bar('Negative',   'Negative')}
      {bar('Irrelevant', 'Irrelevant')}

      <div class="echo">"{preview}"</div>
    </div>
    </body>
    </html>
    """
    # height tuned to card content — no scrollbar
    components.html(html, height=340, scrolling=False)


# ── Run inference on button click ────────────────────────────────────────────
if analyse_clicked and tweet.strip():
    with st.spinner("Running inference…"):
        try:
            result = run_inference(tweet)
            st.session_state.last_result = result
            st.session_state.history.append(result)
        except Exception as e:
            st.markdown(
                f'<div class="iq-error">⚠️ Inference failed — {e}<br>'
                f'<span style="color:#7f1d1d;">Make sure <code>model.h5</code> and '
                f'<code>tokenizer.pkl</code> are in the same directory.</span></div>',
                unsafe_allow_html=True,
            )

# Always re-render last result (survives reruns from example-chip clicks)
if st.session_state.last_result:
    render_result(st.session_state.last_result)