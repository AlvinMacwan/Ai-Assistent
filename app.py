import os
import hashlib
import chromadb
import requests
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from flask_login import (
    LoginManager, login_user, logout_user, login_required, current_user
)

from models import db, User, Message

load_dotenv()  # loads variables from a local .env file into the environment

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

app = Flask(__name__)

# --- Database config (Step 1 of V2) ---
# SECRET_KEY is required by Flask for session signing (Flask-Login will
# need this once auth is wired in next). Set a real value in your .env —
# the fallback here is only for local dev convenience.
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-this")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
db.init_app(app)

with app.app_context():
    db.create_all()

# --- Login setup (Step 2 of V2) ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"  # redirect target when @login_required fails


@login_manager.user_loader
def load_user(user_id):
    # Flask-Login calls this on every request to reload the logged-in
    # user from the session. Must return None (not raise) if not found.
    return User.query.get(int(user_id))

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {".txt", ".pdf"}

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
    # Splits on word boundaries (not raw characters) so chunks never cut
    # a word in half. chunk_size/overlap are treated as approximate
    # character counts and converted to a word count using a rough
    # average of ~6 characters per word (including spaces).
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size // 6
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words).strip())
        overlap_words = max(1, overlap // 6)
        start = end - overlap_words
    return [c for c in chunks if c]

embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_webapp_db")
collection = client.get_or_create_collection(
    name="webapp_documents",
    metadata={"hnsw:space": "cosine"}
)

def index_file(filepath):
    """
    Indexes a file into Chroma. Returns the number of chunks indexed.
    Returns 0 if the file's content is an exact duplicate of content
    already indexed under a different filename (nothing new indexed).
    """
    raw_text = load_text(filepath)
    filename = os.path.basename(filepath)
    content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

    # Skip indexing if this exact content already exists under a
    # *different* source filename (cheap exact-duplicate guard; does
    # not catch near-duplicates or paraphrased content — that's a
    # later, smarter-RAG improvement).
    existing = collection.get(where={"content_hash": content_hash})
    if existing["ids"] and not all(
        meta.get("source") == filename for meta in existing["metadatas"]
    ):
        return 0

    chunks = chunk_text(raw_text, chunk_size=500, overlap=50)
    embeddings = embedder.encode(chunks).tolist()

    # Remove any existing chunks for this filename before re-indexing,
    # so a shorter re-upload doesn't leave orphaned old chunks behind.
    collection.delete(where={"source": filename})

    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "content_hash": content_hash} for _ in chunks]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    return len(chunks)

def retrieve(query, top_k=3, source_filter=None, max_distance=1.0):
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

    # Drop chunks that are too semantically distant to be useful context.
    # max_distance is a loose starting cutoff for cosine distance — tune
    # it once you've seen how it behaves on real queries.
    filtered = [(d, c) for d, c in zip(distances, matched_chunks) if d <= max_distance]
    return filtered

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
@login_required
def home():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            flash("Username and password are required.")
            return redirect(url_for("register"))

        if User.query.filter_by(username=username).first():
            flash("That username is already taken.")
            return redirect(url_for("register"))

        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        login_user(user)
        return redirect(url_for("home"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = User.query.filter_by(username=username).first()
        if user is None or not user.check_password(password):
            flash("Invalid username or password.")
            return redirect(url_for("login"))

        login_user(user)
        return redirect(url_for("home"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/upload", methods=["POST"])
@login_required
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        num_chunks = index_file(filepath)
    except Exception as e:
        return jsonify({"error": f"Failed to index file: {str(e)}"}), 500

    if num_chunks == 0:
        return jsonify({
            "message": f"'{filename}' matches content already indexed under another file — skipped."
        })

    return jsonify({
        "message": f"Indexed '{filename}' into {num_chunks} chunks."
    })

@app.route("/documents", methods=["GET"])
@login_required
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
@login_required
def ask():
    data = request.get_json()
    query = data.get("question", "")
    source_filter = data.get("source") or None  # empty string -> None (no filter)

    if not query:
        return jsonify({"error": "No question provided"}), 400

    try:
        results = retrieve(query, top_k=3, source_filter=source_filter)
        if not results:
            return jsonify({"error": "No indexed documents match that filter, or no results were relevant enough."}), 400

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