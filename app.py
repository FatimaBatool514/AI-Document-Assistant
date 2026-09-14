import hashlib
import io
import os
import re
import tempfile
from pathlib import Path

try:
    import faiss
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "FAISS is not installed. Make sure requirements.txt contains "
        "faiss-cpu==1.15.0, then redeploy the Streamlit app."
    ) from exc
import gdown
import numpy as np
import streamlit as st
from docx import Document
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AI Document Assistant")
st.caption("Upload documents or load supported files from Google Drive, then ask questions about them.")


# -----------------------------
# Session state
# -----------------------------
if "documents" not in st.session_state:
    st.session_state.documents = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

# File hashes are used to avoid processing the same file twice.
if "processed_files" not in st.session_state:
    st.session_state.processed_files = {}


# -----------------------------
# Cached model
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Text extraction
# -----------------------------
def extract_pdf(file_bytes, filename):
    """Extract PDF text page by page."""
    reader = PdfReader(io.BytesIO(file_bytes))
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                {
                    "filename": filename,
                    "page": page_number,
                    "text": text.strip(),
                }
            )

    return documents


def extract_docx(file_bytes, filename):
    """Extract DOCX paragraphs. DOCX page numbers are not reliably available."""
    document = Document(io.BytesIO(file_bytes))
    text = "\n".join(
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )

    if not text.strip():
        return []

    return [
        {
            "filename": filename,
            "page": None,
            "text": text.strip(),
        }
    ]


def extract_txt(file_bytes, filename):
    """Extract plain text."""
    text = file_bytes.decode("utf-8", errors="replace").strip()

    if not text:
        return []

    return [
        {
            "filename": filename,
            "page": None,
            "text": text,
        }
    ]


def extract_md(file_bytes, filename):
    """Extract Markdown as text. Markdown has no standard page number."""
    text = file_bytes.decode("utf-8", errors="replace").strip()

    if not text:
        return []

    return [
        {
            "filename": filename,
            "page": None,
            "text": text,
        }
    ]


def extract_document(file_bytes, filename):
    """Choose the correct extraction function based on file extension."""
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_bytes, filename)
    if extension == ".docx":
        return extract_docx(file_bytes, filename)
    if extension == ".txt":
        return extract_txt(file_bytes, filename)
    if extension == ".md":
        return extract_md(file_bytes, filename)

    return []


# -----------------------------
# Chunking
# -----------------------------
def chunk_text(documents, chunk_size=700, overlap=100):
    """Split extracted text into overlapping chunks while preserving metadata."""
    chunks = []

    for document in documents:
        text = document["text"]
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(
                    {
                        "text": chunk,
                        "filename": document["filename"],
                        "page": document["page"],
                    }
                )

            if end >= len(text):
                break

            start = max(0, end - overlap)

    return chunks


# -----------------------------
# Embeddings + FAISS
# -----------------------------
def build_vector_index(chunks):
    """Create embeddings once and build a FAISS index."""
    if not chunks:
        return None, None

    model = load_embedding_model()

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings


# -----------------------------
# Keyword search
# -----------------------------
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of",
    "in", "on", "for", "and", "or", "with", "what", "which", "who",
    "when", "where", "why", "how", "does", "do", "did", "this", "that",
    "these", "those", "from", "about", "can", "could", "would", "should",
    "i", "we", "you", "it", "its", "as", "by", "at"
}


def important_words(question):
    words = re.findall(r"\b[a-zA-Z0-9]+\b", question.lower())
    return [word for word in words if word not in STOP_WORDS and len(word) > 2]


def keyword_scores(question, chunks):
    """Score chunks using simple important-word matching."""
    words = important_words(question)

    if not words:
        return np.zeros(len(chunks), dtype="float32")

    scores = []

    for chunk in chunks:
        chunk_words = set(
            re.findall(r"\b[a-zA-Z0-9]+\b", chunk["text"].lower())
        )
        matched = sum(word in chunk_words for word in words)
        scores.append(matched / len(words))

    return np.array(scores, dtype="float32")


# -----------------------------
# Hybrid search
# -----------------------------
def hybrid_search(question, top_k=5):
    """Combine semantic similarity and keyword matching."""
    if not st.session_state.chunks or st.session_state.faiss_index is None:
        return []

    model = load_embedding_model()

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    semantic_scores, _ = st.session_state.faiss_index.search(
        question_embedding,
        len(st.session_state.chunks),
    )

    semantic_scores = semantic_scores[0]
    keyword_score = keyword_scores(question, st.session_state.chunks)

    # Semantic cosine similarity is roughly in [-1, 1].
    # Convert it to a simple 0-1 range before combining.
    semantic_normalized = np.clip((semantic_scores + 1) / 2, 0, 1)

    hybrid_scores = (0.75 * semantic_normalized) + (0.25 * keyword_score)

    ranked_indices = np.argsort(hybrid_scores)[::-1][:top_k]

    results = []
    for index in ranked_indices:
        result = dict(st.session_state.chunks[index])
        result["semantic_score"] = float(semantic_normalized[index])
        result["keyword_score"] = float(keyword_score[index])
        result["hybrid_score"] = float(hybrid_scores[index])
        results.append(result)

    return results


# -----------------------------
# Groq
# -----------------------------
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

    if not api_key:
        return None

    return Groq(api_key=api_key)


def answer_question(question, retrieved_chunks):
    """Ask Groq to answer only from retrieved context."""
    client = get_groq_client()

    if client is None:
        return (
            "GROQ_API_KEY is not configured. Add it to Streamlit secrets "
            "before asking questions."
        )

    context_parts = []

    for number, chunk in enumerate(retrieved_chunks, start=1):
        page_text = (
            f"Page {chunk['page']}"
            if chunk["page"] is not None
            else "Page not available"
        )

        context_parts.append(
            f"[Source {number}]\n"
            f"Filename: {chunk['filename']}\n"
            f"{page_text}\n"
            f"Text:\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
Answer the user's question using ONLY the provided document context.

Rules:
1. Do not use outside knowledge.
2. If the answer is not available in the context, say:
   "The information is not available in the provided documents."
3. Do not invent facts, citations, page numbers, or details.
4. Give a clear and concise answer.

User question:
{question}

Document context:
{context}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document question-answering assistant. "
                    "Use only the supplied context."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    return response.choices[0].message.content


# -----------------------------
# File helpers
# -----------------------------
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def file_hash(file_bytes, filename):
    return hashlib.sha256(
        filename.encode("utf-8") + b"|" + file_bytes
    ).hexdigest()


def read_supported_files_from_folder(folder):
    """Read supported files downloaded from Google Drive."""
    files = []

    for path in Path(folder).rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)

    return files


def load_drive_files(drive_url):
    """
    Download supported files from a public/shareable Google Drive file or folder.

    gdown handles common Drive file and folder sharing links.
    Private files require access that the environment can use.
    """
    temp_dir = tempfile.mkdtemp(prefix="drive_docs_")

    try:
        output = gdown.download_folder(
            url=drive_url,
            output=temp_dir,
            quiet=True,
            use_cookies=False,
        )

        paths = []

        if output:
            paths = [Path(item) for item in output if Path(item).is_file()]

        # For cases where download_folder is not suitable for a single file,
        # try gdown's file downloader.
        if not paths:
            downloaded = gdown.download(
                drive_url,
                output=os.path.join(temp_dir, "drive_file"),
                quiet=True,
                fuzzy=True,
            )

            if downloaded:
                paths = [Path(downloaded)]

        return [path for path in paths if path.suffix.lower() in SUPPORTED_EXTENSIONS]

    except Exception as exc:
        raise RuntimeError(
            "Could not load the Google Drive link. Make sure the link is "
            "shareable and points to a Drive file or folder."
        ) from exc


def add_new_files(file_items):
    """
    Process only files that have not been processed before.

    New chunk embeddings are appended to the existing FAISS index,
    so existing document embeddings are not recreated.
    """
    new_documents = []
    new_chunks = []
    new_embeddings = []

    for filename, file_bytes in file_items:
        key = file_hash(file_bytes, filename)

        if key in st.session_state.processed_files:
            continue

        extracted = extract_document(file_bytes, filename)
        chunks = chunk_text(extracted)

        if chunks:
            model = load_embedding_model()
            embeddings = model.encode(
                [chunk["text"] for chunk in chunks],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")

            new_documents.extend(extracted)
            new_chunks.extend(chunks)
            new_embeddings.append(embeddings)

        st.session_state.processed_files[key] = filename

    if not new_chunks:
        return 0

    new_embeddings = np.vstack(new_embeddings)

    if st.session_state.faiss_index is None:
        st.session_state.faiss_index = faiss.IndexFlatIP(
            new_embeddings.shape[1]
        )

    st.session_state.faiss_index.add(new_embeddings)

    st.session_state.documents.extend(new_documents)
    st.session_state.chunks.extend(new_chunks)

    if st.session_state.embeddings is None:
        st.session_state.embeddings = new_embeddings
    else:
        st.session_state.embeddings = np.vstack(
            [st.session_state.embeddings, new_embeddings]
        )

    return len(new_documents)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Settings")

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=10,
        value=5,
    )

    st.markdown(
        """
**Supported formats**
- PDF
- DOCX
- TXT
- MD

**Search**
- Sentence Transformers
- FAISS
- Keyword matching
- Hybrid ranking
"""
    )

    if st.button("Clear documents", use_container_width=True):
        st.session_state.documents = []
        st.session_state.chunks = []
        st.session_state.embeddings = None
        st.session_state.faiss_index = None
        st.session_state.processed_files = {}
        st.rerun()


# -----------------------------
# Local upload
# -----------------------------
st.subheader("1. Upload local documents")

uploaded_files = st.file_uploader(
    "Choose PDF, DOCX, TXT or MD files",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
)

if uploaded_files:
    local_items = [
        (uploaded.name, uploaded.getvalue())
        for uploaded in uploaded_files
    ]

    with st.spinner("Processing new local documents..."):
        added_count = add_new_files(local_items)

    if added_count:
        st.success(
            f"Added {added_count} new extracted document section(s). "
            "Embeddings are stored in session state."
        )
    else:
        st.info("These local files are already processed.")


# -----------------------------
# Google Drive
# -----------------------------
st.subheader("2. Load documents from Google Drive")

drive_url = st.text_input(
    "Paste a Google Drive file or folder link",
    placeholder="https://drive.google.com/...",
)

if st.button("Load from Google Drive", type="secondary"):
    if not drive_url.strip():
        st.warning("Please paste a Google Drive link.")
    else:
        try:
            with st.spinner("Downloading and processing Google Drive files..."):
                drive_paths = load_drive_files(drive_url.strip())

                if not drive_paths:
                    st.warning(
                        "No supported PDF, DOCX, TXT or MD files were found."
                    )
                else:
                    drive_items = [
                        (path.name, path.read_bytes())
                        for path in drive_paths
                    ]

                    added_count = add_new_files(drive_items)

                    if added_count:
                        st.success(
                            f"Loaded {len(drive_items)} Drive file(s) and added "
                            f"{added_count} new extracted section(s)."
                        )
                    else:
                        st.info("These Drive files are already processed.")

        except Exception as exc:
            st.error(str(exc))


# -----------------------------
# Document information
# -----------------------------
st.subheader("3. Document information")

if st.session_state.documents:
    st.write(f"**Documents:** {len(st.session_state.documents)} extracted sections")
    st.write(f"**Created chunks:** {len(st.session_state.chunks)}")

    rows = []
    seen = set()

    for document in st.session_state.documents:
        key = (document["filename"], document["page"])

        if key in seen:
            continue

        seen.add(key)

        rows.append(
            {
                "Filename": document["filename"],
                "Page": document["page"] if document["page"] else "N/A",
                "Extracted characters": len(document["text"]),
            }
        )

    st.dataframe(rows, use_container_width=True)

    with st.expander("Preview extracted text"):
        for document in st.session_state.documents[:20]:
            page = (
                f"Page {document['page']}"
                if document["page"] is not None
                else "Page N/A"
            )
            st.markdown(f"**{document['filename']} — {page}**")
            st.text(document["text"][:2000])
else:
    st.info("Upload a document or load files from Google Drive to begin.")


# -----------------------------
# Question answering
# -----------------------------
st.subheader("4. Ask a question")

question = st.text_input(
    "Question",
    placeholder="What does the document say about project cost?",
)

if question.strip():
    if not st.session_state.chunks:
        st.warning("Please add documents first.")
    else:
        with st.spinner("Searching documents..."):
            retrieved = hybrid_search(question, top_k=top_k)

        with st.spinner("Generating answer..."):
            answer = answer_question(question, retrieved)

        st.markdown("### Answer")
        st.write(answer)

        st.markdown("### Retrieved sources")

        for number, source in enumerate(retrieved, start=1):
            page = (
                f"Page {source['page']}"
                if source["page"] is not None
                else "Page N/A"
            )

            with st.expander(
                f"{number}. {source['filename']} — {page} "
                f"(score: {source['hybrid_score']:.3f})"
            ):
                st.caption(
                    f"Semantic: {source['semantic_score']:.3f} | "
                    f"Keyword: {source['keyword_score']:.3f}"
                )
                st.write(source["text"])
