
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

st.set_page_config(
    page_title="HateShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM DESIGN ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, #192653 0%, transparent 40%),
        radial-gradient(ellipse at 90% 15%, #301747 0%, transparent 35%),
        #090d18;
    color: #f4f6ff;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111a31, #10101d);
    border-right: 1px solid #29334e;
}

.hero {
    text-align: center;
    padding: 38px 20px 28px;
    border: 1px solid rgba(133, 150, 255, .25);
    border-radius: 25px;
    background: linear-gradient(135deg,
        rgba(63, 87, 180, .22),
        rgba(126, 54, 151, .14));
    box-shadow: 0 15px 50px rgba(0, 0, 0, .22);
    animation: appear .8s ease-out;
}

.hero-icon {
    font-size: 54px;
    filter: drop-shadow(0 0 18px rgba(112, 132, 255, .7));
}

.hero h1 {
    font-size: clamp(30px, 5vw, 52px);
    font-weight: 800;
    margin: 8px 0;
    background: linear-gradient(90deg, #8eb5ff, #d9a4ff, #8df5ed);
    background-size: 200% auto;
    color: #b9c7ff;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 5s linear infinite;
}

.hero p {
    color: #bac5df;
    font-size: 16px;
}

@keyframes shine {
    to { background-position: 200% center; }
}

@keyframes appear {
    from {opacity: 0; transform: translateY(14px);}
    to {opacity: 1; transform: translateY(0);}
}

.section-title {
    color: #e4e9ff;
    font-size: 23px;
    font-weight: 700;
    margin: 26px 0 12px;
}

.info-card {
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #293653;
    background: rgba(20, 29, 50, .78);
    margin-bottom: 15px;
    transition: all .25s ease;
}

.info-card:hover {
    border-color: #7b8fff;
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(75, 91, 200, .12);
}

.result-card {
    padding: 24px;
    border-radius: 20px;
    background: linear-gradient(135deg, #172540, #191a30);
    border: 1px solid #45577d;
    animation: appear .5s ease-out;
    margin: 15px 0;
}

div.stButton > button {
    border: 1px solid #788cff;
    border-radius: 12px;
    background: linear-gradient(100deg, #536ff0, #8a55d8);
    color: white;
    font-weight: 700;
    min-height: 48px;
    transition: all .25s ease;
}

div.stButton > button:hover {
    border-color: #b7c4ff;
    box-shadow: 0 0 22px rgba(111, 126, 255, .35);
    transform: translateY(-2px);
    color: white;
}

textarea {
    background: #10182a !important;
    color: #ffffff !important;
    border: 1px solid #3a4c72 !important;
    border-radius: 14px !important;
}

div[data-baseweb="textarea"]:focus-within {
    border-color: #8292ff !important;
    box-shadow: 0 0 0 2px rgba(110, 130, 255, .18);
}

div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #6487ff, #b278ff);
}

.footer {
    text-align: center;
    color: #8e9ab5;
    border-top: 1px solid #26304a;
    padding-top: 20px;
    margin-top: 45px;
    font-size: 13px;
}

#MainMenu, footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    model = joblib.load("hate_speech_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return model, tfidf


@st.cache_resource
def load_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


model, tfidf = load_model()
stop_words = load_stopwords()


# IMPORTANT: Match this preprocessing with your training code.
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    words = [
        word for word in text.split()
        if word not in stop_words
    ]
    return " ".join(words)


# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## 🛡️ HateShield AI")
    st.caption("Machine Learning • NLP Project")
    st.divider()

    st.markdown("### ✨ Features")
    st.markdown("""
    - 🔎 Text analysis
    - 🤖 ML-powered prediction
    - 📊 Class probability
    - ⚡ Instant results
    """)

    st.divider()
    st.markdown("### 📌 About")
    st.write(
        "This application uses Natural Language Processing "
        "to classify text as potentially hateful/offensive "
        "or non-hateful."
    )
    st.caption("Predictions are not perfect. Context matters.")


# ---------- HERO ----------
st.markdown("""
<div class="hero">
    <div class="hero-icon">🛡️</div>
    <h1>HateShield AI</h1>
    <p>Smart Text Analysis • Natural Language Processing</p>
    <p>Analyze text. Understand language. Promote safer conversations.</p>
</div>
""", unsafe_allow_html=True)


# ---------- FEATURE CARDS ----------
st.markdown('<div class="section-title">✨ Your AI Text Analyzer</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="info-card">
        <h3>🔍 Smart Analysis</h3>
        <p>Analyze the text using your trained NLP model.</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-card">
        <h3>⚡ Quick Results</h3>
        <p>Get a prediction within seconds.</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="info-card">
        <h3>📊 Probability</h3>
        <p>Explore the model's estimated class probabilities.</p>
    </div>
    """, unsafe_allow_html=True)


# ---------- TEXT INPUT ----------
st.markdown('<div class="section-title">📝 Enter Your Text</div>',
            unsafe_allow_html=True)

if "sample_text" not in st.session_state:
    st.session_state.sample_text = ""

s1, s2, s3 = st.columns(3)

with s1:
    if st.button("💚 Try a positive example", use_container_width=True):
        st.session_state.sample_text = "I appreciate your help and kindness."

with s2:
    if st.button("👋 Try a greeting", use_container_width=True):
        st.session_state.sample_text = "Hello everyone, have a wonderful day."

with s3:
    if st.button("🧪 Try a test sentence", use_container_width=True):
        st.session_state.sample_text = "I hate you and your ideas."

user_text = st.text_area(
    "Text to analyze",
    key="sample_text",
    height=160,
    placeholder="Type or paste your sentence here..."
)

st.caption("🔒 Enter only text you are comfortable analyzing.")


# ---------- ANALYZE ----------
if st.button("🚀 Analyze Text", use_container_width=True):
    if not user_text.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        cleaned = clean_text(user_text)

        if not cleaned.strip():
            st.warning("Please enter text containing usable words.")
        else:
            features = tfidf.transform([cleaned])
            prediction = int(model.predict(features)[0])

            st.markdown(
                '<div class="section-title">📊 Analysis Results</div>',
                unsafe_allow_html=True
            )

            if prediction == 1:
                st.markdown("""
                <div class="result-card">
                    <h2>⚠️ Potentially Hateful / Offensive</h2>
                    <p>The model classified this text as class 1.
                    Review the wording and context carefully.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-card">
                    <h2>✅ Classified as Non-Hateful</h2>
                    <p>The model classified this text as class 0.
                    This does not guarantee the text is harmless.</p>
                </div>
                """, unsafe_allow_html=True)

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[0]
                classes = list(model.classes_)

                p0 = float(probabilities[classes.index(0)])
                p1 = float(probabilities[classes.index(1)])

                a, b = st.columns(2)

                with a:
                    st.markdown("#### 🟢 Class 0")
                    st.progress(p0)
                    st.metric("Non-hateful probability", f"{p0 * 100:.2f}%")

                with b:
                    st.markdown("#### 🟣 Class 1")
                    st.progress(p1)
                    st.metric("Hateful/offensive probability", f"{p1 * 100:.2f}%")

            st.info(
                "Important: Probabilities represent the model's estimates, "
                "not verified truth. The model can misclassify text, "
                "especially when context, sarcasm or slang is involved."
            )


# ---------- FOOTER ----------
st.markdown("""
<div class="footer">
    🛡️ HateShield AI · NLP & Machine Learning Project<br><br>
    Built with Python and Streamlit · AI can make mistakes
</div>
""", unsafe_allow_html=True)
