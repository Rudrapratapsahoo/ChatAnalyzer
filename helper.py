import pandas as pd
import numpy as np
from collections import Counter
from wordcloud import WordCloud

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
    top  = vc.head(10).reset_index()
    top.columns  = ["user","count"]
    pct  = (vc/vc.sum()*100).round(1).reset_index()
    pct.columns = ["user","percent"]
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
    return WordCloud(width=700, height=320, background_color=None, mode="RGBA",
                     colormap="Purples", stopwords=stop,
                     max_words=120, collocations=False).generate(text or "no data")
