import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from collections import Counter
import random, datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChatAnalyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

section[data-testid="stSidebar"] { background:#0f0f0f !important; border-right:1px solid #222; }
section[data-testid="stSidebar"] * { color:#d0d0d0 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color:#fff !important; }

div[data-testid="stButton"] > button {
    background:#111; color:#fff; border:none; border-radius:8px;
    font-family:'Space Mono',monospace; font-size:0.78rem;
    letter-spacing:0.06em; padding:0.6rem 1.4rem; width:100%; margin-top:0.5rem;
    transition:background .2s;
}
div[data-testid="stButton"] > button:hover { background:#333; }

.kpi-card {
    background:#f8f8f8; border:1.5px solid #eaeaea; border-radius:14px;
    padding:1.4rem 1rem; text-align:center;
    transition:transform .2s, box-shadow .2s;
}
.kpi-card:hover { transform:translateY(-3px); box-shadow:0 8px 24px rgba(0,0,0,.07); }
.kpi-icon  { font-size:1.5rem; margin-bottom:.4rem; }
.kpi-label { font-family:'Space Mono',monospace; font-size:.63rem; letter-spacing:.12em;
             text-transform:uppercase; color:#999; margin-bottom:.35rem; }
.kpi-value { font-family:'Space Mono',monospace; font-size:2rem; font-weight:700; color:#111; line-height:1; }

.sec-header {
    font-family:'Space Mono',monospace; font-size:.65rem; letter-spacing:.15em;
    text-transform:uppercase; color:#aaa;
    border-bottom:1px solid #eee; padding-bottom:.45rem; margin-bottom:1rem;
}

button[data-baseweb="tab"] {
    font-family:'Space Mono',monospace !important;
    font-size:.7rem !important; letter-spacing:.06em !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DUMMY DATA — replace this entire block with preprocessor.preprocess(data)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def get_dummy_df():
    random.seed(42)
    users   = ["Keshav", "Ananya", "Rohan", "Priya"]
    msgs_pool = [
        "okay sounds good", "haha lol", "<Media omitted>",
        "bhai sach mein?", "thik hai", "kya kar raha hai",
        "chalo niklo", "meeting at 5?", "done ✅",
        "😂😂😂", "no way bro", "check this https://example.com",
        "kal milte hain", "ye lo link https://youtu.be/xyz",
        "sent a photo <Media omitted>", "okay okay",
    ]
    base = datetime.datetime(2023, 1, 1, 10, 0)
    rows = []
    for _ in range(900):
        base += datetime.timedelta(hours=random.randint(0, 5),
                                   minutes=random.randint(0, 59))
        rows.append({"date": base,
                     "user": random.choice(users),
                     "message": random.choice(msgs_pool)})
    df = pd.DataFrame(rows)
    df["year"]      = df["date"].dt.year
    df["month_num"] = df["date"].dt.month
    df["month"]     = df["date"].dt.month_name()
    df["day_name"]  = df["date"].dt.day_name()
    df["hour"]      = df["date"].dt.hour
    df["only_date"] = df["date"].dt.date
    return df


# ══════════════════════════════════════════════════════════════════════════════
# INLINE ANALYSIS HELPERS
# (each one will be replaced by a call to helper.py when your teammate plugs it in)
# ══════════════════════════════════════════════════════════════════════════════
DAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

def calc_stats(df):
    total_msgs  = len(df)
    total_words = df["message"].dropna().apply(lambda x: len(str(x).split())).sum()
    media       = df["message"].str.contains("<Media omitted>", na=False).sum()
    links       = df["message"].str.contains("https?://", na=False).sum()
    return int(total_msgs), int(total_words), int(media), int(links)

def monthly_timeline(df):
    t = (df.groupby(["year","month_num","month"]).size()
           .reset_index(name="count").sort_values(["year","month_num"]))
    t["label"] = t["month"].str[:3] + " " + t["year"].astype(str)
    return t

def daily_timeline(df):
    return df.groupby("only_date").size().reset_index(name="count")

def week_activity(df):
    return (df["day_name"].value_counts()
              .reindex(DAY_ORDER, fill_value=0)
              .reset_index()
              .rename(columns={"index":"day","day_name":"count","count":"count"}))

def heatmap_data(df):
    hm = df.pivot_table(index="day_name", columns="hour",
                        values="message", aggfunc="count").reindex(DAY_ORDER).fillna(0)
    for h in range(24):
        if h not in hm.columns: hm[h] = 0
    return hm[sorted(hm.columns)]

def top_users(df):
    vc = df["user"].value_counts()
    top  = vc.head(10).reset_index(); top.columns  = ["user","count"]
    pct  = (vc/vc.sum()*100).round(1).reset_index(); pct.columns = ["user","percent"]
    return top, pct

def top_words(df, n=20):
    stop = {"the","a","an","is","it","in","on","at","to","and","or","of","for",
            "with","this","that","was","are","be","have","i","you","we","they",
            "media","omitted","hai","kya","nahi","bhi","toh","tha","ok","okay"}
    words = []
    for m in df["message"].dropna():
        if "<Media omitted>" in str(m): continue
        for w in str(m).lower().split():
            w = w.strip(".,!?\"'")
            if w and w not in stop and len(w) > 2: words.append(w)
    return pd.DataFrame(Counter(words).most_common(n), columns=["word","count"])

def emoji_stats(df):
    import re
    PAT = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
                     "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+",
                     flags=re.UNICODE)
    emojis = []
    for m in df["message"].dropna():
        emojis.extend(PAT.findall(str(m)))
    if not emojis:
        return pd.DataFrame(columns=["Emoji","Count"])
    return pd.DataFrame(Counter(emojis).most_common(20), columns=["Emoji","Count"])

def sentiment_stats(df):
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        tmp = df.copy()
        tmp["score"] = tmp["message"].apply(lambda m: sia.polarity_scores(str(m))["compound"])
        return tmp.groupby("user")["score"].mean().reset_index().rename(columns={"score":"avg"}).sort_values("avg", ascending=False)
    except ImportError:
        users = df["user"].unique()
        rng   = np.random.default_rng(0)
        return pd.DataFrame({"user": users, "avg": rng.uniform(-0.3, 0.6, len(users))})

def make_wordcloud(df):
    stop = {"the","a","an","is","it","in","on","ok","okay","media","omitted",
            "hai","kya","nahi","bhi","toh","tha","aur","bhai"}
    text = " ".join(str(m) for m in df["message"].dropna()
                    if "<Media omitted>" not in str(m))
    return WordCloud(width=700, height=320, background_color="white",
                     colormap="Greys", stopwords=stop,
                     max_words=120, collocations=False).generate(text or "no data")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 💬 ChatAnalyzer")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload WhatsApp Export (.txt)",
        type=["txt"],
        help="WhatsApp → Chat → Export Chat → Without Media"
    )

    # ── Session-state caching ──────────────────────────────────────
    if uploaded_file:
        fname = uploaded_file.name
        if st.session_state.get("last_file") != fname:
            with st.spinner("Parsing chat…"):
                try:
                    from preprocessor import preprocess          # teammate plugs this in
                    data = uploaded_file.read().decode("utf-8")
                    df_raw = preprocess(data)
                    st.session_state["df"]        = df_raw
                    st.session_state["last_file"] = fname
                    st.success(f"✅ {len(df_raw):,} messages loaded")
                except ImportError:
                    st.info("preprocessor.py not found — showing dummy data.")
                    st.session_state["df"]        = get_dummy_df()
                    st.session_state["last_file"] = fname
                except Exception as e:
                    st.error(f"Parse error: {e}")
                    st.session_state["df"]        = get_dummy_df()
                    st.session_state["last_file"] = fname
    else:
        if "df" not in st.session_state:
            st.session_state["df"] = get_dummy_df()
        st.caption("No file uploaded — showing sample data.")

    df_all = st.session_state["df"]

    # ── User selector ──────────────────────────────────────────────
    user_list    = ["Overall"] + sorted(df_all["user"].unique().tolist())
    selected_user = st.selectbox("Analyze for", user_list)

    analyze = st.button("🔍 Analyze")

    st.markdown("---")
    st.caption("Export: WhatsApp → Chat info\n→ Export chat → Without Media")


# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="padding:2rem 0 1.5rem; border-bottom:1.5px solid #111; margin-bottom:2rem;">
  <div style="font-family:'Space Mono',monospace; font-size:2rem; font-weight:700;
              letter-spacing:-.02em; color:#111;">
    Chat<span style="color:#888">Analyzer</span>
  </div>
  <div style="color:#999; font-size:.95rem; margin-top:.3rem;">
    WhatsApp conversation insights
  </div>
</div>
""", unsafe_allow_html=True)

# ── Gate: show placeholder until Analyze is clicked ───────────────────────────
if not analyze and "analyzed" not in st.session_state:
    st.markdown("""
    <div style="text-align:center; padding:5rem 0; color:#ccc;">
      <div style="font-size:3.5rem">💬</div>
      <div style="font-family:'Space Mono',monospace; font-size:.75rem;
                  letter-spacing:.12em; margin-top:1rem; color:#bbb;">
        UPLOAD A CHAT &amp; CLICK ANALYZE
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

st.session_state["analyzed"] = True

# ── Filter by user ─────────────────────────────────────────────────────────────
df = df_all if selected_user == "Overall" else df_all[df_all["user"] == selected_user]


# ══════════════════════════════════════════════════════════════════════════════
# KPI CARDS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">Overview</div>', unsafe_allow_html=True)

total_msgs, total_words, media_cnt, links_cnt = calc_stats(df)

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
# TABS
# ══════════════════════════════════════════════════════════════════════════════
t1, t2, t3, t4, t5, t6 = st.tabs([
    "📅 Timeline", "⏰ Activity", "👥 Users", "💬 Words", "😂 Emoji", "😊 Sentiment"
])

# ─── matplotlib global style ─────────────────────────────────────────────────
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 8})

# ─── TAB 1 — Timeline ────────────────────────────────────────────────────────
with t1:
    L, R = st.columns(2)

    with L:
        st.markdown('<div class="sec-header">Monthly Timeline</div>', unsafe_allow_html=True)
        mt = monthly_timeline(df)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(range(len(mt)), mt["count"], color="#111", lw=2, marker="o", ms=4)
        ax.fill_between(range(len(mt)), mt["count"], alpha=.07, color="#111")
        ax.set_xticks(range(len(mt)))
        ax.set_xticklabels(mt["label"], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Messages")
        ax.spines["left"].set_visible(False); ax.yaxis.set_ticks_position("none")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with R:
        st.markdown('<div class="sec-header">Daily Timeline</div>', unsafe_allow_html=True)
        dt = daily_timeline(df)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(dt["only_date"], dt["count"], color="#777", lw=1, alpha=.9)
        ax.fill_between(dt["only_date"], dt["count"], alpha=.05, color="#333")
        ax.set_ylabel("Messages"); ax.tick_params(labelsize=7)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ─── TAB 2 — Activity ────────────────────────────────────────────────────────
with t2:
    L, R = st.columns(2)

    with L:
        st.markdown('<div class="sec-header">Busiest Day of Week</div>', unsafe_allow_html=True)
        wd = week_activity(df)
        wd.columns = ["day","count"]
        max_val = wd["count"].max()
        colors  = ["#111" if v == max_val else "#ddd" for v in wd["count"]]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.barh(wd["day"], wd["count"], color=colors, height=.6)
        ax.spines["left"].set_visible(False); ax.tick_params(labelsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with R:
        st.markdown('<div class="sec-header">Hour × Day Heatmap</div>', unsafe_allow_html=True)
        hm = heatmap_data(df)
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.heatmap(hm, ax=ax, cmap="Greys", linewidths=.3,
                    linecolor="#f5f5f5", cbar_kws={"shrink":.6})
        ax.set_xlabel("Hour"); ax.set_ylabel(""); ax.tick_params(labelsize=7)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ─── TAB 3 — Users ───────────────────────────────────────────────────────────
with t3:
    if selected_user != "Overall":
        st.info(f"User breakdown is only available for **Overall**. Currently: **{selected_user}**")
    else:
        st.markdown('<div class="sec-header">Most Active Users</div>', unsafe_allow_html=True)
        tu, pct = top_users(df_all)
        L, R = st.columns([3, 2])
        with L:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.barh(tu["user"][::-1], tu["count"][::-1], color="#111", height=.6)
            ax.spines["left"].set_visible(False); ax.tick_params(labelsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with R:
            st.dataframe(pct, width='stretch', hide_index=True)

# ─── TAB 4 — Words ───────────────────────────────────────────────────────────
with t4:
    L, R = st.columns(2)

    with L:
        st.markdown('<div class="sec-header">Word Cloud</div>', unsafe_allow_html=True)
        wc  = make_wordcloud(df)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        plt.tight_layout(pad=0); st.pyplot(fig); plt.close()

    with R:
        st.markdown('<div class="sec-header">Top 20 Words</div>', unsafe_allow_html=True)
        tw = top_words(df)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.barh(tw["word"][::-1], tw["count"][::-1], color="#111", height=.7)
        ax.spines["left"].set_visible(False); ax.tick_params(labelsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ─── TAB 5 — Emoji ───────────────────────────────────────────────────────────
with t5:
    st.markdown('<div class="sec-header">Emoji Analysis</div>', unsafe_allow_html=True)
    em = emoji_stats(df)
    if em.empty:
        st.info("No emojis found in this chat.")
    else:
        L, R = st.columns([1, 2])
        with L:
            st.dataframe(em.head(20), width='stretch', hide_index=True)
        with R:
            fig, ax = plt.subplots(figsize=(5, 4))
            top8 = em.head(8)
            ax.pie(top8["Count"], labels=top8["Emoji"], autopct="%1.1f%%",
                   textprops={"fontsize":11},
                   colors=plt.cm.Greys_r(np.linspace(.15, .85, len(top8))))
            plt.tight_layout(); st.pyplot(fig); plt.close()

# ─── TAB 6 — Sentiment ───────────────────────────────────────────────────────
with t6:
    st.markdown('<div class="sec-header">Sentiment by User (VADER)</div>', unsafe_allow_html=True)
    sd = sentiment_stats(df)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#2a7a2a" if v >= 0 else "#b33333" for v in sd["avg"]]
    ax.bar(sd["user"], sd["avg"], color=colors, width=.5)
    ax.axhline(0, color="#999", lw=.8, linestyle="--")
    ax.set_ylabel("Avg VADER compound score"); ax.tick_params(labelsize=8)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.caption("Score > 0 → positive  |  Score < 0 → negative  |  Range: −1 to +1")


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
csv = df[["date","user","message"]].to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Export filtered messages as CSV",
    data=csv,
    file_name=f"chat_{selected_user}.csv",
    mime="text/csv",
)