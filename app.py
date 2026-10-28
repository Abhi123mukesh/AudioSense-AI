import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AudioSense — Find Your Frequency", page_icon="🎧", layout="wide", initial_sidebar_state="expanded")

# ---------- Cinematic visual system ----------
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,400,0,0&display=swap');
.material-symbols-rounded, .material-icons, .material-icons-round { font-family:'Material Symbols Rounded' !important; font-weight:normal; font-style:normal; font-size:24px; line-height:1; letter-spacing:normal; text-transform:none; display:inline-block; white-space:nowrap; word-wrap:normal; direction:ltr; -webkit-font-feature-settings:'liga'; -webkit-font-smoothing:antialiased; }
:root { --ink:#f4f2ff; --muted:#a8a6bd; --panel:rgba(22,20,39,.76); --line:rgba(255,255,255,.10); --violet:#a78bfa; --pink:#f0abfc; --cyan:#67e8f9; }
.stApp { background: radial-gradient(ellipse at 12% 5%, rgba(111,66,193,.28), transparent 34%), radial-gradient(ellipse at 88% 18%, rgba(31,141,164,.16), transparent 28%), linear-gradient(145deg,#090812 0%,#111020 52%,#090b14 100%); color:var(--ink); font-family:'DM Sans',sans-serif; }
[data-testid="stHeader"] { background:rgba(8,8,18,.38); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,rgba(16,14,30,.98),rgba(9,10,19,.98)); border-right:1px solid var(--line); }
[data-testid="stSidebar"] *:not(.material-symbols-rounded):not(.material-icons):not(.material-icons-round):not([data-testid="stIconMaterial"]) { font-family:'DM Sans',sans-serif; }
/* Keep Streamlit's icon glyphs on the Material Symbols font; otherwise names like keyboard_double_arrow_right appear as text. */
.material-symbols-rounded, .material-icons, .material-icons-round, [data-testid="stIconMaterial"] { font-family:'Material Symbols Rounded','Material Icons',sans-serif !important; font-weight:normal !important; font-style:normal !important; font-feature-settings:'liga' !important; -webkit-font-feature-settings:'liga' !important; }
[data-testid="stSidebarCollapseButton"] span, [data-testid="stSidebarNavCollapseButton"] span { font-family:'Material Symbols Rounded','Material Icons',sans-serif !important; font-feature-settings:'liga' !important; }
/* Hide Streamlit hover tooltip that exposes the raw sidebar icon name. */
div[data-testid="stTooltipContent"] { display:none !important; }
.block-container { max-width:1320px; padding-top:2rem; padding-bottom:3rem; }
h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.035em; color:#faf8ff; }
@keyframes logoPulse { 0%,100% { box-shadow:0 0 18px rgba(167,139,250,.28), inset 0 0 18px rgba(103,232,249,.08); transform:translateY(0) rotate(-3deg); } 50% { box-shadow:0 0 34px rgba(103,232,249,.42), inset 0 0 24px rgba(167,139,250,.22); transform:translateY(-3px) rotate(3deg); } }
@keyframes equalize { 0%,100% { height:7px; } 25% { height:19px; } 50% { height:12px; } 75% { height:25px; } }
@keyframes shimmer { 0% { background-position:0% 50%; } 100% { background-position:200% 50%; } }
@keyframes floatIn { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
.brand-masthead { display:flex; align-items:center; gap:1rem; padding:.45rem 0 1.35rem; animation:floatIn .75s ease-out both; }
.brand-logo { width:66px; height:66px; flex:0 0 66px; border-radius:21px; display:flex; align-items:center; justify-content:center; gap:4px; background:linear-gradient(135deg,rgba(139,92,246,.28),rgba(8,145,178,.22)); border:1px solid rgba(255,255,255,.22); animation:logoPulse 4s ease-in-out infinite; }
.eq-bar { width:5px; border-radius:8px; background:linear-gradient(180deg,#67e8f9,#c4b5fd 65%,#f0abfc); height:12px; box-shadow:0 0 10px rgba(103,232,249,.45); }
.eq-bar:nth-child(1) { animation:equalize .9s ease-in-out infinite; }
.eq-bar:nth-child(2) { animation:equalize 1.2s ease-in-out infinite .15s; }
.eq-bar:nth-child(3) { animation:equalize .75s ease-in-out infinite .3s; }
.eq-bar:nth-child(4) { animation:equalize 1.05s ease-in-out infinite .1s; }
.brand-wordmark { font-family:'Space Grotesk',sans-serif; font-size:clamp(2.15rem,4.5vw,3.55rem); line-height:.98; font-weight:700; letter-spacing:-.065em; color:#fff; }
.brand-wordmark span { background:linear-gradient(90deg,#c4b5fd,#f0abfc,#67e8f9,#c4b5fd); background-size:200% auto; -webkit-background-clip:text; background-clip:text; color:transparent; animation:shimmer 6s linear infinite; }
.brand-tagline { color:#aaa8c2; letter-spacing:.17em; text-transform:uppercase; font-size:.7rem; font-weight:700; margin-top:.48rem; }
.hero { animation:floatIn .65s ease-out both; }
.song-card { transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease; }
.song-card:hover { transform:translateY(-3px); border-color:rgba(167,139,250,.48); box-shadow:0 12px 30px rgba(0,0,0,.22); }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation:none !important; transition:none !important; } }
p, label, .stMarkdown { color:#d8d5e8; }
.eyebrow { color:var(--cyan); text-transform:uppercase; letter-spacing:.24em; font-size:.72rem; font-weight:700; margin-bottom:.65rem; }
.hero { position:relative; overflow:hidden; padding:2.4rem 2.5rem; border-radius:28px; border:1px solid rgba(255,255,255,.14); background:linear-gradient(115deg,rgba(69,43,119,.72),rgba(23,28,56,.84) 52%,rgba(14,64,77,.58)); box-shadow:0 22px 70px rgba(0,0,0,.28); margin-bottom:1.4rem; }
.hero:after { content:'♫'; position:absolute; right:5%; top:-52px; font-size:17rem; line-height:1; color:rgba(255,255,255,.055); font-family:serif; transform:rotate(10deg); pointer-events:none; }
.hero h1 { font-size:clamp(2.4rem,5vw,4.5rem); line-height:.98; margin:.25rem 0 1rem; max-width:780px; }
.hero p { color:#d5d1e8; max-width:660px; font-size:1.03rem; line-height:1.7; }
.gradient-text { background:linear-gradient(90deg,#fff 5%,#d8b4fe 46%,#67e8f9 100%); -webkit-background-clip:text; background-clip:text; color:transparent; }
.pill { display:inline-block; border:1px solid rgba(255,255,255,.18); color:#e9ddff; border-radius:99px; padding:.38rem .72rem; font-size:.76rem; background:rgba(255,255,255,.06); margin-right:.35rem; }
.section-title { font-family:'Space Grotesk',sans-serif; font-size:1.5rem; font-weight:700; margin:1.3rem 0 .2rem; }
.section-sub { color:var(--muted); margin-bottom:1rem; }
.glass { background:var(--panel); border:1px solid var(--line); border-radius:22px; padding:1.2rem 1.3rem; box-shadow:0 14px 42px rgba(0,0,0,.16); }
.song-card { background:linear-gradient(135deg,rgba(39,34,65,.88),rgba(17,22,39,.9)); border:1px solid rgba(255,255,255,.10); border-radius:18px; padding:1rem 1.15rem; margin:.55rem 0; }
.song-icon { display:flex; align-items:center; justify-content:center; width:48px; height:48px; border-radius:15px; background:linear-gradient(135deg,rgba(167,139,250,.35),rgba(103,232,249,.18)); border:1px solid rgba(255,255,255,.12); font-size:1.4rem; }
.song-name { font-family:'Space Grotesk',sans-serif; color:#fff; font-size:1.05rem; font-weight:700; }
.song-meta { color:#aaa8c2; font-size:.84rem; margin-top:.2rem; }
.score { font-family:'Space Grotesk',sans-serif; font-size:1.35rem; font-weight:700; color:#c4b5fd; text-align:right; }
.score-label { color:#9895b0; font-size:.7rem; text-align:right; text-transform:uppercase; letter-spacing:.12em; }
div[data-testid="stMetric"] { background:rgba(255,255,255,.045); border:1px solid var(--line); padding:1rem 1.1rem; border-radius:18px; }
div[data-testid="stMetricLabel"] { color:#aaa8c2; }
div[data-testid="stMetricValue"] { color:#faf8ff; font-family:'Space Grotesk',sans-serif; }
.stButton > button, .stDownloadButton > button { border-radius:12px; border:1px solid rgba(255,255,255,.15); font-weight:700; min-height:2.8rem; transition:all .2s ease; }
.stButton > button[kind="primary"] { background:linear-gradient(100deg,#8b5cf6,#6366f1 55%,#0891b2); color:white; border:0; box-shadow:0 8px 24px rgba(99,102,241,.25); }
.stButton > button:hover, .stDownloadButton > button:hover { transform:translateY(-1px); border-color:rgba(196,181,253,.7); }
.stSelectbox > div > div, .stTextInput > div > div, .stNumberInput > div > div { background:rgba(255,255,255,.045); border-color:var(--line); border-radius:12px; }
.stSlider [data-baseweb="slider"] { padding-top:.4rem; }
hr { border-color:rgba(255,255,255,.1); }
.small-note { color:#9996b1; font-size:.78rem; line-height:1.6; }
.sidebar-brand { display:flex; align-items:center; gap:.55rem; font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.3rem; letter-spacing:-.05em; color:white; }
.sidebar-brand span { color:#c4b5fd; }
.sidebar-mini-logo { display:inline-flex; align-items:center; justify-content:center; width:35px; height:35px; border-radius:11px; background:linear-gradient(135deg,#8b5cf6,#0891b2); box-shadow:0 0 18px rgba(139,92,246,.25); }
.sidebar-tagline { color:#9996b1; font-size:.82rem; margin:.2rem 0 1.5rem; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:16px; overflow:hidden; }
@media (max-width:700px) { .hero { padding:1.5rem; } .hero:after { right:-8%; font-size:10rem; } .block-container { padding-top:1rem; } .brand-logo { width:52px; height:52px; flex-basis:52px; border-radius:17px; } .brand-masthead { gap:.75rem; } .brand-tagline { letter-spacing:.08em; font-size:.6rem; } }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_songs():
    return pd.read_csv("songs.csv")

songs = load_songs()
features = ["tempo_bpm", "energy", "danceability", "acousticness", "instrumentalness"]
feature_labels = {"tempo_bpm":"Tempo (BPM)", "energy":"Energy", "danceability":"Danceability", "acousticness":"Acousticness", "instrumentalness":"Instrumentalness"}

with st.sidebar:
    st.markdown('<div class="sidebar-brand"><span class="sidebar-mini-logo">♫</span><div>Audio<span>Sense</span></div></div><div class="sidebar-tagline">Find your frequency.</div>', unsafe_allow_html=True)
    page = st.radio("YOUR SPACE", ["Discover", "Music Library", "Behind the Sound"], label_visibility="visible")
    st.divider()
    st.markdown("<div class='small-note'>MADE FOR MUSIC LOVERS<br>Powered by machine learning</div>", unsafe_allow_html=True)
    st.write("")
    st.metric("Tracks in this demo", f"{len(songs):02d}")
    st.markdown("<div class='small-note'>Demo library uses synthetic sample values, not real audio analysis.</div>", unsafe_allow_html=True)

st.markdown("""<div class="brand-masthead"><div class="brand-logo" aria-label="AudioSense animated equalizer logo"><span class="eq-bar"></span><span class="eq-bar"></span><span class="eq-bar"></span><span class="eq-bar"></span></div><div><div class="brand-wordmark">Audio<span>Sense</span></div><div class="brand-tagline">Feel the music. Find your next obsession.</div></div></div>""", unsafe_allow_html=True)

if page == "Discover":
    st.markdown('''<div class="hero"><div class="eyebrow">YOUR NEXT OBSESSION STARTS HERE</div><h1>Find the sound<br>that <span class="gradient-text">feels like you.</span></h1><p>One song can open a whole new world. Pick a track you love and let AudioSense discover music with a similar sonic fingerprint.</p><span class="pill">✦ AI-powered discovery</span><span class="pill">♫ Your vibe, your rules</span><span class="pill">◌ Content-based ML</span></div>''', unsafe_allow_html=True)

    top1, top2, top3 = st.columns(3)
    top1.metric("Curated demo tracks", f"{len(songs)}", "Across multiple genres")
    top2.metric("Sound dimensions", "05", "Tempo · energy · mood")
    top3.metric("Recommendation engine", "Cosine", "Similarity matching")

    st.markdown('<div class="section-title">Start with a song you love</div><div class="section-sub">Choose your starting point, then tune the sound profile to your taste.</div>', unsafe_allow_html=True)
    left, right = st.columns([1.08, .92], gap="large")
    with left:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        selected_name = st.selectbox("PICK YOUR TRACK", songs["song_name"].tolist(), label_visibility="visible")
        selected = songs[songs["song_name"] == selected_name].iloc[0]
        st.markdown(f'''<div style="display:flex;gap:1rem;align-items:center;margin:.8rem 0 1rem"><div class="song-icon">♫</div><div><div class="song-name">{selected['song_name']}</div><div class="song-meta">{selected['artist']} · {selected['genre']}</div></div></div>''', unsafe_allow_html=True)
        m1,m2,m3 = st.columns(3)
        m1.metric("Tempo", f"{selected['tempo_bpm']} BPM")
        m2.metric("Energy", f"{selected['energy']:.2f}")
        m3.metric("Danceability", f"{selected['danceability']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("#### Shape your sound")
        st.caption("What matters most in your perfect track?")
        weights = {
            "tempo_bpm": st.slider("⚡ Tempo", 0.0, 2.0, 1.0, 0.25),
            "energy": st.slider("🔥 Energy", 0.0, 2.0, 1.0, 0.25),
            "danceability": st.slider("✦ Danceability", 0.0, 2.0, 1.0, 0.25),
            "acousticness": st.slider("☾ Acoustic feel", 0.0, 2.0, 0.75, 0.25),
            "instrumentalness": st.slider("♬ Instrumental", 0.0, 2.0, 0.5, 0.25),
        }
        st.markdown('</div>', unsafe_allow_html=True)

    f1,f2,f3 = st.columns([1.2,1,1])
    with f1:
        genre_options = ["Any genre"] + sorted(songs["genre"].unique().tolist())
        genre_filter = st.selectbox("GENRE MOOD", genre_options)
    with f2:
        how_many = st.select_slider("HOW MANY TRACKS?", options=[3,4,5,6,7,8,9,10], value=5)
    with f3:
        st.write("")
        st.write("")
        recommend = st.button("✦ Find my next sound", type="primary", use_container_width=True)

    if recommend:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(songs[features])
        weight_values = [weights[f] for f in features]
        if sum(weight_values) == 0:
            weighted = scaled
            st.warning("All sound weights are set to zero. Equal importance was used for this search.")
        else:
            weighted = scaled * weight_values
        similarity = cosine_similarity(weighted)
        idx = songs.index[songs["song_name"] == selected_name][0]
        result = songs.copy()
        result["similarity"] = similarity[idx]
        result = result[result["song_name"] != selected_name]
        if genre_filter != "Any genre":
            result = result[result["genre"] == genre_filter]
        result = result.sort_values("similarity", ascending=False).head(how_many)
        st.session_state["recommendations"] = result
        st.session_state["recommendation_for"] = selected_name
        st.session_state["last_genre"] = genre_filter

    if st.session_state.get("recommendation_for") == selected_name and "recommendations" in st.session_state:
        result = st.session_state["recommendations"]
        st.markdown('<div class="section-title">Your next-on-repeat list</div><div class="section-sub">Ranked by sonic similarity to your chosen track.</div>', unsafe_allow_html=True)
        if result.empty:
            st.warning("No tracks match that genre filter. Try Any genre.")
        else:
            for rank, (_, row) in enumerate(result.iterrows(), start=1):
                score = max(0, min(100, float(row["similarity"]) * 100))
                c1,c2,c3 = st.columns([.12, .66, .22], gap="small")
                with c1:
                    st.markdown(f'<div class="song-icon">{rank:02d}</div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="song-card"><div class="song-name">♫ &nbsp;{row["song_name"]}</div><div class="song-meta">{row["artist"]} &nbsp;·&nbsp; {row["genre"]}</div><div class="song-meta">{row["tempo_bpm"]} BPM &nbsp; / &nbsp; Energy {row["energy"]:.2f} &nbsp; / &nbsp; Dance {row["danceability"]:.2f}</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div style="padding-top:1rem"><div class="score">{score:.1f}%</div><div class="score-label">similarity</div></div>', unsafe_allow_html=True)
            st.download_button("↓ Save your recommendations", result.to_csv(index=False).encode("utf-8"), "audiosense_recommendations.csv", "text/csv")
            st.markdown('<div class="small-note">Similarity is a mathematical comparison of the demo feature patterns — it is not a guarantee that you will like a track.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-title">The vibe is in the details</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass"><div class="eyebrow">HOW IT FEELS</div><h3>Less searching. More feeling.</h3><p>AudioSense compares the musical features of tracks and brings similar patterns together — from high-energy dance cuts to softer acoustic moods.</p></div>', unsafe_allow_html=True)

elif page == "Music Library":
    st.markdown('<div class="hero"><div class="eyebrow">THE SOUND SHELF</div><h1>Explore the <span class="gradient-text">library.</span></h1><p>Browse the demo tracks, explore genres, and compare the features behind each sound.</p></div>', unsafe_allow_html=True)
    c1,c2 = st.columns([1,1])
    with c1:
        genre = st.selectbox("Filter by genre", ["All genres"] + sorted(songs["genre"].unique().tolist()))
    with c2:
        search = st.text_input("Search tracks or artists", placeholder="Try a song or artist name…")
    data = songs.copy()
    if genre != "All genres":
        data = data[data["genre"] == genre]
    if search.strip():
        mask = data["song_name"].str.contains(search, case=False, na=False) | data["artist"].str.contains(search, case=False, na=False)
        data = data[mask]
    st.metric("Tracks in view", len(data))
    st.dataframe(data, use_container_width=True, hide_index=True)
    st.markdown('<div class="section-title">Read the sound</div><div class="section-sub">Compare how a musical feature varies across tracks.</div>', unsafe_allow_html=True)
    feature = st.selectbox("Feature to visualise", features, format_func=lambda f: feature_labels[f])
    chart_data = data.sort_values(feature, ascending=False)
    fig = px.bar(chart_data, x="song_name", y=feature, color="genre", hover_data=["artist", "genre"], title=f"{feature_labels[feature]} across tracks", template="plotly_dark")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#dedaf2", title_font_size=20, xaxis_title="Track", yaxis_title=feature_labels[feature], legend_title="Genre", margin=dict(l=10,r=10,t=55,b=10))
    st.plotly_chart(fig, use_container_width=True)

else:
    st.markdown('<div class="hero"><div class="eyebrow">UNDER THE HEADPHONES</div><h1>The science behind<br><span class="gradient-text">the feeling.</span></h1><p>AudioSense uses content-based filtering to compare the musical fingerprints of tracks and rank the closest matches.</p></div>', unsafe_allow_html=True)
    steps = [
        ("01", "Extract the sound profile", "Each track in this prototype has five numeric features: tempo, energy, danceability, acousticness, and instrumentalness."),
        ("02", "Bring features onto one scale", "StandardScaler normalises the feature values so BPM does not overpower features with smaller numeric ranges."),
        ("03", "Tune the listener's priorities", "Feature weights let you decide whether tempo, energy, danceability, or softer instrumental qualities matter most."),
        ("04", "Compare musical fingerprints", "Cosine similarity measures the angle between feature vectors to estimate how closely two tracks align."),
        ("05", "Rank the closest matches", "The selected track is excluded, and the remaining tracks are sorted by similarity to create a recommendation list.")
    ]
    for num, title, desc in steps:
        a,b = st.columns([.13,.87], gap="small")
        with a:
            st.markdown(f'<div class="song-icon">{num}</div>', unsafe_allow_html=True)
        with b:
            st.markdown(f'<div class="song-card"><div class="song-name">{title}</div><div class="song-meta" style="font-size:.92rem;line-height:1.7">{desc}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">The matching idea</div>', unsafe_allow_html=True)
    st.latex(r"\text{cosine similarity}(A,B)=\frac{A\cdot B}{\|A\|\|B\|}")
    st.info("Prototype limitation: song names and feature values are synthetic examples for demonstrating the ML workflow. The app does not analyse actual audio files or measure real listener preferences yet.")

st.markdown("<hr><div class='small-note' style='text-align:center'>AUDIOSENSE AI · FIND YOUR FREQUENCY · A MUSIC DISCOVERY ML PROTOTYPE</div>", unsafe_allow_html=True)
