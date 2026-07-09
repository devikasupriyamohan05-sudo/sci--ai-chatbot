# Deploy the Streamlit doc + audio Q&A app (free public link)

A full demo app: upload documents and audio → it transcribes + indexes them → you chat over them,
grounded with sources. This is the closest thing yet to the SCI product.

## Files (in this folder)
- `app.py` — the Streamlit app
- `requirements.txt` — Python packages
- `packages.txt` — system packages (ffmpeg, for audio)

## Step 1 — Put the files in a GitHub repo
Create a new repo (e.g. `care-assistant-app`) and upload `app.py`, `requirements.txt`, and
`packages.txt` to the **root** (Add file → Upload files → Commit).

## Step 2 — Deploy on Streamlit Community Cloud (free)
1. Go to **https://share.streamlit.io** → sign in with GitHub.
2. **Create app / New app** → pick your repo, branch `main`, main file `app.py` → **Deploy**.
3. First build takes a few minutes (it installs the packages).

## Step 3 — Add your secrets (keys stay server-side)
In the app → **⋮ / Settings → Secrets**, paste this and fill in your values:
```toml
AOAI_ENDPOINT = "https://meetingbot-aoai.openai.azure.com/"
AOAI_KEY = "your-azure-openai-key"
AOAI_CHAT_DEPLOY = "gpt-4o-mini"
AOAI_EMBED_DEPLOY = "text-embedding-3-small"
AOAI_API_VERSION = "2024-10-21"
SPEECH_KEY = "your-azure-speech-key"
SPEECH_REGION = "centralus"
```
Save — the app restarts automatically.

## Step 4 — Use it
1. Open your app URL (looks like `https://your-app.streamlit.app`) — shareable, works on phone.
2. In the sidebar, upload a PDF/TXT and/or a WAV/MP3, click **Build knowledge base**.
3. Ask questions in the chat — answers come only from what you uploaded, with source filenames.

## How it maps to SCI
- **Documents** = manuals/runbooks · **Audio** = call recordings
- Transcription = **Azure AI Speech** · Understanding/answers = **Azure OpenAI (gpt-4o-mini)**
- Retrieval = embeddings (**text-embedding-3-small**) + similarity search (the RAG step)
This is the same pattern as the production design, in one shareable app.

## Notes
- **Cost:** Azure OpenAI usage is pennies (from your credit); Azure Speech F0 is free; Streamlit
  Community Cloud is free. Keep audio clips short on the free tier (limited memory).
- **Privacy:** it's a public app — upload only public/sample content, never real customer data.
- **Grounding:** uses an in-memory index that resets when the app sleeps/restarts — fine for a demo.
  The production version would use Azure AI Search for a persistent index.
