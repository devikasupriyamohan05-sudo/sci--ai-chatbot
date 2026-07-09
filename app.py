import time, tempfile, json, os, subprocess
import numpy as np
import pandas as pd
import streamlit as st
from openai import AzureOpenAI

st.set_page_config(page_title="Care Assistant", page_icon="🕊️", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,600;1,500;1,600&family=PT+Serif:ital,wght@0,400;0,700;1,400&display=swap');
/* fonts on TEXT elements only (never on buttons/icons, to keep glyph fonts intact) */
.stApp, .stMarkdown, .stMarkdown p, p, li, label,
[data-testid="stChatMessage"], .stTextInput input, textarea, .stChatInput textarea {
  font-family: 'PT Serif', Georgia, serif !important;
}
h1, h2, h3, h4, .app-title, [data-baseweb="tab"] { font-family: 'Playfair Display', Georgia, serif !important; }
/* force a light, on-brand look regardless of the viewer's theme */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background:#ffffff !important; color:#21455a !important; }
[data-testid="stSidebar"] { background:#eef4f1 !important; }
[data-testid="stSidebar"] * { color:#21455a !important; }
#MainMenu, header[data-testid="stHeader"], footer { visibility:hidden; }
.topbar { background:#1b4053; padding:22px 26px; border-radius:6px; margin-bottom:10px; }
.topbar .app-title { color:#ffffff !important; font-style:italic; font-size:30px; margin:0; font-family:'Playfair Display',serif; }
.topbar .app-sub { color:#cfe0dc !important; font-size:14px; }
.tagline { color:#4a6472 !important; font-style:italic; margin:2px 0 16px; }
.stButton>button { background:#a8762f !important; color:#ffffff !important; border:none; border-radius:4px; font-weight:700; }
.stButton>button:hover { background:#8a6126 !important; }
h1, h2, h3 { color:#1b4053 !important; }
a, .stMarkdown a { color:#6f7d33 !important; }
[data-baseweb="tab"] { font-size:16px; }
</style>
""", unsafe_allow_html=True)

# ---------- config (from Streamlit secrets) ----------
def cfg(k, d=""):
    return st.secrets.get(k, d)

AOAI_ENDPOINT = cfg("AOAI_ENDPOINT")
AOAI_KEY      = cfg("AOAI_KEY")
CHAT_DEPLOY   = cfg("AOAI_CHAT_DEPLOY", "gpt-4o-mini")
EMBED_DEPLOY  = cfg("AOAI_EMBED_DEPLOY", "text-embedding-3-small")
API_VER       = cfg("AOAI_API_VERSION", "2024-10-21")
SPEECH_KEY    = cfg("SPEECH_KEY")
SPEECH_REGION = cfg("SPEECH_REGION", "centralus")

client = (AzureOpenAI(azure_endpoint=AOAI_ENDPOINT, api_key=AOAI_KEY, api_version=API_VER)
          if AOAI_ENDPOINT and AOAI_KEY else None)

EMOTIONS = ["neutral", "positive", "frustrated", "confused", "angry", "sad"]
SCORE = {"positive": 1, "neutral": 0, "confused": -0.5, "frustrated": -0.7, "angry": -1, "sad": -1}

# ---------- helpers ----------
def chunk_text(t, size=1000, overlap=150):
    out, i = [], 0
    while i < len(t):
        out.append(t[i:i+size]); i += size - overlap
    return out

def read_pdf(file):
    from pypdf import PdfReader
    return "\n".join((p.extract_text() or "") for p in PdfReader(file).pages)

def _to_wav(uploaded):
    # Write the upload to disk, then convert to 16 kHz mono WAV with ffmpeg
    # (no pydub -> avoids the removed 'audioop' module on Python 3.13+).
    ext = os.path.splitext(uploaded.name)[1] or ".bin"
    src = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    src.write(uploaded.getvalue())
    src.flush()
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(["ffmpeg", "-y", "-i", src.name, "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", out],
                   check=True, capture_output=True)
    return out

def transcribe_audio(uploaded):
    import azure.cognitiveservices.speech as speechsdk
    sc = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    sc.speech_recognition_language = "en-US"
    rec = speechsdk.SpeechRecognizer(speech_config=sc, audio_config=speechsdk.audio.AudioConfig(filename=_to_wav(uploaded)))
    parts, done = [], []
    rec.recognized.connect(lambda e: parts.append(e.result.text) if e.result.text else None)
    rec.session_stopped.connect(lambda e: done.append(True))
    rec.canceled.connect(lambda e: done.append(True))
    rec.start_continuous_recognition()
    while not done:
        time.sleep(0.3)
    rec.stop_continuous_recognition()
    return " ".join(parts)

def diarize_turns(uploaded):
    import azure.cognitiveservices.speech as speechsdk
    sc = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    sc.speech_recognition_language = "en-US"
    tr = speechsdk.transcription.ConversationTranscriber(sc, speechsdk.audio.AudioConfig(filename=_to_wav(uploaded)))
    turns, done = [], []
    tr.transcribed.connect(lambda e: turns.append({"speaker": e.result.speaker_id or "Unknown", "text": e.result.text})
                           if (e.result.reason == speechsdk.ResultReason.RecognizedSpeech and e.result.text) else None)
    tr.session_stopped.connect(lambda e: done.append(True))
    tr.canceled.connect(lambda e: done.append(True))
    tr.start_transcribing_async().get()
    while not done:
        time.sleep(0.3)
    tr.stop_transcribing_async().get()
    return turns

def label_emotions(turns, batch=40):
    for s in range(0, len(turns), batch):
        chunk = turns[s:s+batch]
        numbered = "\n".join(f"{i}: {t['text']}" for i, t in enumerate(chunk))
        sysmsg = ("Label the emotion of each numbered utterance. Use ONLY one of: "
                  f"{EMOTIONS}. Return JSON {{\"labels\":[{{\"i\":<index>,\"emotion\":<label>}}]}}.")
        r = client.chat.completions.create(model=CHAT_DEPLOY, response_format={"type": "json_object"},
                                           temperature=0,
                                           messages=[{"role": "system", "content": sysmsg},
                                                     {"role": "user", "content": numbered}])
        mp = {d["i"]: d["emotion"] for d in json.loads(r.choices[0].message.content)["labels"]}
        for i, t in enumerate(chunk):
            t["emotion"] = mp.get(i, "neutral")
    return turns

def embed(texts):
    r = client.embeddings.create(model=EMBED_DEPLOY, input=texts)
    return np.array([d.embedding for d in r.data], dtype="float32")

def retrieve(query, k=4):
    qv = embed([query])[0]
    M = st.session_state.vectors
    sims = (M @ qv) / (np.linalg.norm(M, axis=1) * np.linalg.norm(qv) + 1e-9)
    return [st.session_state.chunks[i] for i in sims.argsort()[::-1][:k]]

def answer(q):
    hits = retrieve(q)
    ctx = "\n\n".join(f"[{h['src']}] {h['text']}" for h in hits)
    r = client.chat.completions.create(
        model=CHAT_DEPLOY, temperature=0.2,
        messages=[{"role": "system", "content": "Answer ONLY from the context. Cite the source filename in [brackets]. If not present, say you don't know. Be warm, plain, concise."},
                  {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q}"}])
    return r.choices[0].message.content, sorted({h["src"] for h in hits})

# ---------- state ----------
st.session_state.setdefault("chunks", [])
st.session_state.setdefault("vectors", None)
st.session_state.setdefault("messages", [])

st.markdown('<div class="topbar"><div class="app-title">Care Assistant</div>'
            '<div class="app-sub">Supporting families with compassion and accuracy</div></div>',
            unsafe_allow_html=True)
st.markdown('<p class="tagline">Upload manuals and call recordings, ask grounded questions, '
            'and review how each conversation felt.</p>', unsafe_allow_html=True)

if client is None:
    st.error("Azure OpenAI secrets not set. Add AOAI_ENDPOINT and AOAI_KEY in the app's Secrets.")
    st.stop()

# ---------- sidebar: knowledge base for chat ----------
with st.sidebar:
    st.header("Knowledge sources (for chat)")
    docs = st.file_uploader("Documents (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True)
    auds = st.file_uploader("Audio (WAV/MP3/M4A)", type=["wav", "mp3", "m4a"], accept_multiple_files=True)
    st.caption("Use public/sample content only — this is a demo, not for real customer data.")
    if st.button("Build knowledge base", type="primary"):
        chunks = []
        for d in docs or []:
            text = read_pdf(d) if d.name.lower().endswith(".pdf") else d.read().decode("utf-8", "ignore")
            for c in chunk_text(text):
                chunks.append({"src": d.name, "text": c})
        for a in auds or []:
            with st.spinner(f"Transcribing {a.name}…"):
                text = transcribe_audio(a)
            for c in chunk_text(text):
                chunks.append({"src": a.name, "text": c})
        if not chunks:
            st.warning("Upload at least one document or audio file first.")
        else:
            with st.spinner("Embedding & indexing…"):
                st.session_state.chunks = chunks
                st.session_state.vectors = embed([c["text"] for c in chunks])
            st.success(f"Indexed {len(chunks)} chunks from {len(docs or [])} doc(s) + {len(auds or [])} audio.")

tab_chat, tab_insights = st.tabs(["💬 Ask", "📊 Call insights"])

# ---------- tab 1: chat ----------
with tab_chat:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"], unsafe_allow_html=True)
    q = st.chat_input("Ask a question about your uploaded content…")
    if q:
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        if st.session_state.vectors is None:
            a = "Please upload files in the sidebar and click **Build knowledge base** first."
        else:
            with st.spinner("Thinking…"):
                text, srcs = answer(q)
                a = text + f"\n\n<sub>Sources: {', '.join(srcs)}</sub>"
        st.session_state.messages.append({"role": "assistant", "content": a})
        with st.chat_message("assistant"):
            st.markdown(a, unsafe_allow_html=True)

# ---------- tab 2: diarization + emotion ----------
with tab_insights:
    st.subheader("Who spoke, and how they felt")
    st.caption("Diarizes a call (separates speakers) and labels the emotion of each turn.")
    if not SPEECH_KEY:
        st.info("Add SPEECH_KEY and SPEECH_REGION in Secrets to enable audio analysis.")
    ca = st.file_uploader("Upload one call recording", type=["wav", "mp3", "m4a"], key="callaudio")
    if st.button("Analyze call") and ca:
        with st.spinner("Diarizing (separating speakers)…"):
            turns = diarize_turns(ca)
        if not turns:
            st.warning("No speech recognized. Try a clearer/shorter clip.")
        else:
            with st.spinner("Scoring emotion per turn…"):
                label_emotions(turns)
            df = pd.DataFrame(turns)
            df["score"] = df["emotion"].map(SCORE).fillna(0)
            c1, c2 = st.columns(2)
            c1.metric("Turns", len(df))
            c2.metric("Overall sentiment", f"{df['score'].mean():.2f}")
            st.write("**Average sentiment by speaker**")
            st.bar_chart(df.groupby("speaker")["score"].mean())
            st.write("**Emotion counts**")
            st.bar_chart(df["emotion"].value_counts())
            st.write("**Transcript (speaker · emotion · text)**")
            st.dataframe(df[["speaker", "emotion", "text"]], use_container_width=True, hide_index=True)
