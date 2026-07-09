import time, tempfile
import numpy as np
import streamlit as st
from openai import AzureOpenAI

st.set_page_config(page_title="Care Assistant — Doc & Audio Q&A", page_icon="💬", layout="centered")

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

# ---------- helpers ----------
def chunk_text(t, size=1000, overlap=150):
    out, i = [], 0
    while i < len(t):
        out.append(t[i:i+size]); i += size - overlap
    return out

def read_pdf(file):
    from pypdf import PdfReader
    reader = PdfReader(file)
    return "\n".join((p.extract_text() or "") for p in reader.pages)

def transcribe_audio(uploaded):
    import azure.cognitiveservices.speech as speechsdk
    from pydub import AudioSegment
    seg = AudioSegment.from_file(uploaded).set_frame_rate(16000).set_channels(1)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    seg.export(tmp.name, format="wav")
    sc = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    sc.speech_recognition_language = "en-US"
    rec = speechsdk.SpeechRecognizer(speech_config=sc, audio_config=speechsdk.audio.AudioConfig(filename=tmp.name))
    parts, done = [], []
    rec.recognized.connect(lambda e: parts.append(e.result.text) if e.result.text else None)
    rec.session_stopped.connect(lambda e: done.append(True))
    rec.canceled.connect(lambda e: done.append(True))
    rec.start_continuous_recognition()
    while not done:
        time.sleep(0.3)
    rec.stop_continuous_recognition()
    return " ".join(parts)

def embed(texts):
    r = client.embeddings.create(model=EMBED_DEPLOY, input=texts)
    return np.array([d.embedding for d in r.data], dtype="float32")

def retrieve(query, k=4):
    qv = embed([query])[0]
    M = st.session_state.vectors
    sims = (M @ qv) / (np.linalg.norm(M, axis=1) * np.linalg.norm(qv) + 1e-9)
    idx = sims.argsort()[::-1][:k]
    return [st.session_state.chunks[i] for i in idx]

def answer(q):
    hits = retrieve(q)
    ctx = "\n\n".join(f"[{h['src']}] {h['text']}" for h in hits)
    r = client.chat.completions.create(
        model=CHAT_DEPLOY, temperature=0.2,
        messages=[
            {"role": "system", "content": "Answer ONLY from the context. Cite the source filename in [brackets]. If the answer isn't in the context, say you don't know. Be warm, plain, and concise."},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q}"},
        ])
    return r.choices[0].message.content, sorted({h["src"] for h in hits})

# ---------- state ----------
st.session_state.setdefault("chunks", [])
st.session_state.setdefault("vectors", None)
st.session_state.setdefault("messages", [])

# ---------- UI ----------
st.title("💬 Care Assistant — Document & Audio Q&A")
st.caption("Upload manuals/notes and call recordings; the assistant transcribes, indexes, and answers "
           "grounded questions with sources. Built on Azure OpenAI + Azure AI Speech.")

if client is None:
    st.error("Azure OpenAI secrets not set. Add AOAI_ENDPOINT and AOAI_KEY in the app's Secrets.")
    st.stop()

with st.sidebar:
    st.header("Knowledge sources")
    docs = st.file_uploader("Documents (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True)
    auds = st.file_uploader("Audio (WAV, MP3, M4A)", type=["wav", "mp3", "m4a"], accept_multiple_files=True)
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
            st.success(f"Indexed {len(chunks)} chunks from {len(docs or [])} document(s) + {len(auds or [])} audio file(s).")

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
