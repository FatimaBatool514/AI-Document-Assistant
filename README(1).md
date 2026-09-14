# Streamlit AI Document Assistant

A simple AI document question-answering app built with Streamlit.

It supports:

- PDF
- DOCX
- TXT
- Markdown (`.md`)
- Local file uploads
- Google Drive file/folder links
- Sentence Transformers embeddings
- FAISS semantic search
- Simple keyword search
- Hybrid semantic + keyword search
- Groq LLM answers
- Retrieved source display
- Streamlit session-state caching so embeddings are not recreated for every question

## 1. Project structure

```text
streamlit_ai_document_assistant/
├── app.py
├── requirements.txt
└── README.md
```

## 2. Install

Create a virtual environment if desired, then install:

```bash
pip install -r requirements.txt
```

## 3. Add the Groq API key

Do **not** put the API key directly inside `app.py`.

Create:

```text
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

The application reads the key using:

```python
st.secrets.get("GROQ_API_KEY")
```

For deployment, add `GROQ_API_KEY` in your Streamlit Cloud app's Secrets settings.

## 4. Run the app

```bash
streamlit run app.py
```

The first time the application runs, Sentence Transformers may download the
`all-MiniLM-L6-v2` model.

## 5. How the application works

### Step 1 — Extract

Each supported document is passed to a separate extraction function:

- `extract_pdf()`
- `extract_docx()`
- `extract_txt()`
- `extract_md()`

PDF text is extracted page by page, so PDF chunks keep their page number.

DOCX, TXT and MD files do not have a reliable standard page number in this
simple implementation, so their page value is `None`.

### Step 2 — Chunk

`chunk_text()` splits extracted text into manageable overlapping pieces.

The default values are:

```text
chunk size = 700 characters
overlap = 100 characters
```

Every chunk keeps:

```text
filename
page
text
```

The application displays the total number of created chunks.

### Step 3 — Embed

`SentenceTransformer("all-MiniLM-L6-v2")` converts every chunk into a vector.

The embeddings are stored in:

```python
st.session_state.embeddings
```

The embedding model itself is loaded with:

```python
@st.cache_resource
```

This prevents the model from being loaded repeatedly.

### Step 4 — FAISS

FAISS stores the document vectors and performs semantic similarity search.

The user's question is embedded and compared with the document chunk vectors.

### Step 5 — Keyword search

The application also extracts important words from the question and gives
each chunk a simple keyword-match score.

### Step 6 — Hybrid search

The final score combines both methods:

```text
Hybrid score =
    75% semantic similarity
  + 25% keyword matching
```

The top matching chunks are returned together with their metadata.

### Step 7 — Groq

The retrieved chunks are sent to Groq together with the question.

The prompt tells the model:

- use only the provided context
- do not use outside knowledge
- do not invent information
- say that the information is unavailable when it is not in the context

### Step 8 — Sources

After the answer, the application displays the retrieved chunks, including:

- filename
- page number when available
- retrieved text
- semantic score
- keyword score
- hybrid score

## 6. Google Drive

Paste a shareable Google Drive file or folder link into the Drive field.

The app uses `gdown` to download supported files and sends them through the
same extraction → chunking → embedding → FAISS → hybrid-search pipeline.

Supported Drive files are:

```text
.pdf
.docx
.txt
.md
```

For a Drive folder, the app searches downloaded files recursively and ignores
unsupported file types.

### Important

The Drive link must be accessible to the application. For private files,
authentication/permissions may prevent downloading.

## 7. Avoiding repeated embeddings

Documents are processed only when their content changes.

For local uploads, the app creates a hash from the filenames and file bytes.
For Drive imports, it also keeps a hash in session state.

After processing, the FAISS index and embeddings remain in:

```python
st.session_state.faiss_index
st.session_state.embeddings
```

Therefore, asking multiple questions does **not** recreate document embeddings.

Only the new question is embedded for each search.

## 8. Simple architecture

```text
Local Upload ───────┐
                    │
Google Drive ───────┤
                    ▼
              Text Extraction
                    │
                    ▼
                 Chunking
                    │
                    ▼
          Sentence Transformers
                    │
                    ▼
              FAISS Index
                    │
                    ├───────────────┐
                    ▼               ▼
             Semantic Search   Keyword Search
                    │               │
                    └───────┬───────┘
                            ▼
                      Hybrid Ranking
                            │
                            ▼
                     Retrieved Chunks
                            │
                            ▼
                          Groq
                            │
                            ▼
                         Answer
                            │
                            ▼
                    Retrieved Sources
```

## 9. Security notes

- Never hardcode the Groq API key.
- Keep `.streamlit/secrets.toml` out of Git.
- Add this to `.gitignore` if using Git:

```text
.streamlit/secrets.toml
__pycache__/
.venv/
```

## 10. Limitations of this simple version

- PDF extraction works for text-based PDFs. Scanned/image-only PDFs require
  OCR, which is intentionally not included here.
- DOCX page numbers are not extracted because Word documents do not expose
  reliable page boundaries through `python-docx`.
- TXT and Markdown files have no page numbers.
- Google Drive access depends on the sharing permissions of the supplied link.
- Embeddings are kept in Streamlit session state, so they are reused during
  the current browser session. They are not a permanent database.
- FAISS is rebuilt only when the document collection changes.

## 11. Recommended next improvements

For a production version, consider adding:

- persistent FAISS storage
- document IDs
- OCR for scanned PDFs
- better sentence/paragraph-aware chunking
- reranking
- conversation history
- authentication
- persistent Google Drive OAuth
- document deletion/update handling
- a database for metadata
