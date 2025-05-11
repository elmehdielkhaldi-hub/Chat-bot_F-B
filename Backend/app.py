from flask import Flask, request, jsonify, render_template
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
import os

app = Flask(__name__)

# Chargement des PDF
pdf_folder = "content/"
os.makedirs(pdf_folder, exist_ok=True)
pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
all_loaded_docs = []

for pdf_file in pdf_files:
    file_path = os.path.join(pdf_folder, pdf_file)
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    for page in pages:
        page.metadata["source_file"] = pdf_file
    all_loaded_docs.extend(pages)

# Split des documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(all_loaded_docs)

# Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    model_kwargs={'device': 'cpu'}
)

# Base vectorielle
db_vector = Chroma(
    persist_directory="./dbVector",
    embedding_function=embedding_model,
)
db_vector.add_documents(chunks)

# Modèle Groq
llm = ChatGroq(
    base_url="https://api.groq.com",
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    api_key="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"
)

# Recherche de documents similaires
def search_similarity(question: str, k: int = 4) -> str:
    similar_docs = db_vector.similarity_search(question, k=k)
    if not similar_docs:
        return ""
    return "\n\n---\n\n".join([doc.page_content for doc in similar_docs])

# Génération de réponse
def generate_response_based_on_context(question: str, llm_instance, k_context: int = 4) -> str:
    context = search_similarity(question, k_context)
    if not context:
        return "Aucune information trouvée dans les documents pour répondre à la question."

    prompt_text = f"""INSTRUCTIONS :
1. Répondez à la question UNIQUEMENT avec les informations du CONTEXTE.
2. Si ce n'est pas dans le CONTEXTE, dites-le explicitement.

CONTEXTE :
{context}

QUESTION : {question}

RÉPONSE :"""
    response = llm_instance.invoke(prompt_text)
    return getattr(response, "content", str(response)).strip()

# Route HTML
@app.route("/")
def index():
    return render_template("index.html")

# Endpoint API
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Aucune question fournie."}), 400

    response = generate_response_based_on_context(question, llm)
    return jsonify({"response": response})

# Lancement de l'app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
