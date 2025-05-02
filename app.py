from flask import Flask, render_template, request
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from transformers import AutoTokenizer, AutoModel
import os

# --- Initialisation Flask ---
app = Flask(__name__)

# --- Préparation LangChain Pipeline ---
# Utilisons un nouveau chemin pour la base de données
db_path = "./db_vector_new"

# 1. Charger le modèle d'embeddings Hugging Face
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Initialiser la base vectorielle avec le nouveau chemin
dbVector = Chroma(
    persist_directory=db_path, 
    embedding_function=embedding_model,
)

# 3. Vérifier si la base existe déjà et contient des documents
if not os.path.exists(db_path) or len(os.listdir(db_path)) == 0:
    print("Création d'une nouvelle base vectorielle...")
    
    # Charger le PDF
    reader = PyPDFLoader("./content/Bpifrance Creation_GUIDE PRATIQUE DU CREATEUR_2019.pdf")
    doc = reader.load()
    
    # Découper le texte
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(doc)
    
    # Ajouter les documents dans la base vectorielle
    dbVector.add_documents(chunks)
    print("Base vectorielle créée avec succès!")
else:
    print("Utilisation de la base vectorielle existante...")

# 4. Préparer le modèle LLM (ChatGroq API)
llm = ChatGroq(
    base_url="https://api.groq.com",
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    api_key="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",
)

# --- Fonctions utiles ---
def search_similarity(question):
    contexte_similaire = dbVector.similarity_search(question)
    texts = ""
    for mor in contexte_similaire:
        texts += mor.page_content
    return texts

def ask_model(question):
    contexte = search_similarity(question)
    prompt = f"Répondez à cette question : {question} en utilisant ce contexte : {contexte}"
    response = llm.invoke(prompt)
    return response.content

# --- Routes Flask ---
@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    if request.method == "POST":
        user_question = request.form["question"]
        response = ask_model(user_question)
    return render_template("index.html", response=response)

# --- Lancement ---
if __name__ == "__main__":
    app.run(debug=True)