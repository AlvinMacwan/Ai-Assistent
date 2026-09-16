import os
import chromadb
import requests
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()  # loads variables from a local .env file into the environment

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==========================================
# RAG pipeline
# ==========================================
def load_text(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".pdf":
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]

embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_webapp_db")
collection = client.get_or_create_collection(
    name="webapp_documents",
    metadata={"hnsw:space": "cosine"}
)

def index_file(filepath):
    raw_text = load_text(filepath)
    chunks = chunk_text(raw_text, chunk_size=500, overlap=50)
    embeddings = embedder.encode(chunks).tolist()
    filename = os.path.basename(filepath)
    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename} for _ in chunks]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    return len(chunks)

def retrieve(query, top_k=3, source_filter=None):
    query_embedding = embedder.encode(query).tolist()

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
    }
    if source_filter:
        query_kwargs["where"] = {"source": source_filter}

    results = collection.query(**query_kwargs)
    matched_chunks = results["documents"][0]
    distances = results["distances"][0]
    return list(zip(distances, matched_chunks))

def generate_answer(query, retrieved_chunks):
    context = "\n\n".join([chunk for distance, chunk in retrieved_chunks])
    prompt = f"""Use the following context to answer the question. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {query}"""

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "openrouter/free",
            "messages": [{"role": "user", "content": prompt}]
        }
    )
    data = response.json()

    if "choices" not in data:
        # Surface the API's own error message instead of crashing with a KeyError
        error_msg = data.get("error", {}).get("message", "Unknown error from OpenRouter")
        raise RuntimeError(error_msg)

    return data["choices"][0]["message"]["content"]


# ==========================================
# Routes
# ==========================================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        num_chunks = index_file(filepath)
    except Exception as e:
        return jsonify({"error": f"Failed to index file: {str(e)}"}), 500

    return jsonify({
        "message": f"Indexed '{file.filename}' into {num_chunks} chunks."
    })

@app.route("/documents", methods=["GET"])
def list_documents():
    # Chroma has no built-in "distinct values" query, so pull all metadata
    # and de-duplicate in Python. Fine at small/demo scale.
    all_items = collection.get()
    sources = set()
    for meta in all_items["metadatas"]:
        if meta and "source" in meta:
            sources.add(meta["source"])
    return jsonify({"documents": sorted(sources)})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("question", "")
    source_filter = data.get("source") or None  # empty string -> None (no filter)

    if not query:
        return jsonify({"error": "No question provided"}), 400

    try:
        results = retrieve(query, top_k=3, source_filter=source_filter)
        if not results:
            return jsonify({"error": "No indexed documents match that filter."}), 400

        answer = generate_answer(query, results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Return retrieved chunks alongside the answer, for transparency in the UI
    chunks_payload = [
        {"distance": round(distance, 4), "text": chunk}
        for distance, chunk in results
    ]

    return jsonify({
        "answer": answer,
        "retrieved_chunks": chunks_payload,
    })


if __name__ == "__main__":
    app.run(debug=True)