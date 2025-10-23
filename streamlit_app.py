import streamlit as st

# ───────────────────────────────────────────────
# 0️⃣ FIX SQLITE FOR CHROMADB
# ───────────────────────────────────────────────
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

# ───────────────────────────────────────────────
# 1️⃣ IMPORTS
# ───────────────────────────────────────────────
import os, chromadb, fitz
from docx import Document
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
import vertexai
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ───────────────────────────────────────────────
# 2️⃣ VERTEX AI SETUP
# ───────────────────────────────────────────────
st.set_page_config(page_title="Vertex AI Local Docs Chatbot", page_icon="🤖")

from google.oauth2 import service_account

creds = service_account.Credentials.from_service_account_file(
    st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]
)

vertexai.init(
    project=st.secrets["PROJECT_ID"],
    location=st.secrets["REGION"],
    credentials=creds
)


embedding_model = embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

gen_model = GenerativeModel("gemini-2.0-flash")


# ───────────────────────────────────────────────
# 3️⃣ CHROMA SETUP
# ───────────────────────────────────────────────
CHROMA_PATH = "./vertex_vector_db"
COLLECTION_NAME = "LocalDocs"
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)

# ───────────────────────────────────────────────
# 4️⃣ HELPER FUNCTIONS
# ───────────────────────────────────────────────
def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    try:
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif ext == ".pdf":
            with fitz.open(file_path) as pdf:
                for page in pdf:
                    text += page.get_text()
        elif ext == ".docx":
            doc = Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs])
        else:
            st.warning(f"⚠️ Unsupported file format: {file_path}")
    except Exception as e:
        st.error(f"Error reading {file_path}: {e}")

    return text


def build_vector_db(folder="docs"):
    """Read all files, embed them, and store in ChromaDB"""
    # Safely reset the collection instead of invalid delete()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # Ignore if doesn't exist
    global collection
    collection = client.get_or_create_collection(COLLECTION_NAME)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    processed = 0

    if not os.path.exists(folder):
        st.error(f"❌ Folder '{folder}' not found.")
        return

    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".txt", ".pdf", ".docx"))
    ]

    if not files:
        st.warning(f"No readable files found in '{folder}'")
        return

    for file_path in files:
        text = extract_text_from_file(file_path)
        if not text.strip():
            continue
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            emb = embedding_model.get_embeddings([chunk])[0].values
            collection.add(
                documents=[chunk],
                ids=[f"{os.path.basename(file_path)}_{i}"],
                embeddings=[emb],
            )
        processed += 1

    st.success(f"✅ Indexed {processed} file(s) into ChromaDB successfully.")



def retrieve_context(query, top_k=4):
    """Search Chroma for semantically similar chunks"""
    q_emb = embedding_model.get_embeddings([query])[0].values
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    docs = results.get("documents", [[]])[0]
    return "\n---\n".join(docs) if docs else ""


def chat_response(question, history, context):
    """Generate final response with Vertex AI"""
    prompt = (
        "You are a helpful assistant. Use the CONTEXT below "
        "and the HISTORY of the chat to answer clearly.\n\n"
        f"CONTEXT:\n{context}\n\nHISTORY:\n{history}\n\nQUESTION:\n{question}"
    )
    resp = gen_model.generate_content(prompt)
    return resp.text

# ───────────────────────────────────────────────
# 5️⃣ STREAMLIT UI
# ───────────────────────────────────────────────
st.title("🤖 Vertex AI Chatbot for Local Documents")
st.write("Ask questions about your 52 text files stored in the 'docs' folder.")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.sidebar.header("Controls")

if st.sidebar.button("🔄 Rebuild Knowledge Base"):
    with st.spinner("Reading and embedding your 52 text files..."):
        build_vector_db("docs")

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_q := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    ctx = retrieve_context(user_q)
    hist = "\n".join(
        [f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:-1]]
    )

    with st.spinner("Thinking with Vertex AI..."):
        ans = chat_response(user_q, hist, ctx)

    full_ans = f"**Answer:**\n\n{ans}"
    st.session_state.messages.append({"role": "assistant", "content": full_ans})

    with st.chat_message("assistant"):
        st.markdown(full_ans)
