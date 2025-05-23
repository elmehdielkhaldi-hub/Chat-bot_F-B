# Required libraries can be installed with:
# pip install flask flask-cors langchain-community langchain-huggingface langchain-chroma langchain-groq python-dotenv

import os
import re # Pour le nettoyage de texte
import shutil # Pour la suppression de dossier
import time # Pour les délais de tentatives
import gc # Pour le garbage collector
from getpass import getpass # Bien que conditionné, il reste
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- LangChain Specific Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# --- dotenv pour charger les variables d'environnement ---
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)
app.secret_key = os.urandom(24)

# --- Global variables for RAG components ---
all_loaded_docs_global = []
chunks_global = []
embeddingModel_global = None
dbVector_global = None
llm_global = None
rag_pipeline_ready_global = False
initialization_error_message = None
loaded_pdf_filenames_global = []

# --- Configuration ---
PDF_FOLDER = "content/"
CHROMA_PERSIST_DIR = "dbVector"
HF_EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
GROQ_MODEL_NAME = "llama3-8b-8192"
DEFAULT_K_RETRIEVAL = 5 # Nombre de chunks à récupérer initialement pour le contexte
# SIMILARITY_SCORE_THRESHOLD = 0.7 # Optionnel, pour filtrer par score. Distance L2, plus bas est mieux.

# --- Helper Functions ---

def normalize_text(text: str) -> str:
    """Nettoie le texte en normalisant les espaces et en supprimant quelques artefacts."""
    if not isinstance(text, str): # S'assurer que l'entrée est une chaîne
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'-\n', '', text)
    return text

def search_similarity_global(question: str, k: int = DEFAULT_K_RETRIEVAL) -> tuple[str, list[dict]]:
    if dbVector_global is None:
        app.logger.error("dbVector_global is not initialized for search_similarity_global.")
        return "", []
    app.logger.info(f"[*] Recherche de similarité pour '{question}' (top {k} documents)...")
    try:
        similar_docs_with_score = dbVector_global.similarity_search_with_score(question, k=k)

        if not similar_docs_with_score:
            app.logger.info("[!] Aucun document similaire trouvé initialement.")
            return "", []
        
        # Log des scores (distances L2, plus petit est mieux)
        for i, (doc, score) in enumerate(similar_docs_with_score):
            app.logger.debug(f"  Doc {i+1} - Source: {doc.metadata.get('source_file', 'N/A')}, Page: {doc.metadata.get('page','N/A')}, Score (Distance): {score:.4f}")

        # Ici, vous pourriez ajouter un filtrage basé sur un seuil de distance si nécessaire.
        # Par exemple: filtered_docs = [(doc, score) for doc, score in similar_docs_with_score if score < MAX_DISTANCE_THRESHOLD]
        # Pour l'instant, on prend les k premiers récupérés.

        content_list = [normalize_text(doc.page_content) for doc, score in similar_docs_with_score]
        sources_structured = [
            {
                "source_file": doc.metadata.get('source_file', 'N/A'),
                "page": str(doc.metadata.get('page', 'N/A')),
                "score": float(f"{score:.4f}")
            }
            for doc, score in similar_docs_with_score
        ]
        concatenated_texts = "\n\n---\n\n".join(content_list)
        app.logger.info(f"[*]   -> {len(similar_docs_with_score)} documents pertinents trouvés et leur contenu concaténé.")
        return concatenated_texts, sources_structured
    except Exception as e:
        app.logger.error(f"[!] ERREUR pendant la recherche de similarité : {e}")
        app.logger.exception("Traceback de la recherche de similarité:")
        return "", []

def generate_response_based_on_context_global(question: str, k_for_context: int = DEFAULT_K_RETRIEVAL) -> tuple[str, str, list[dict]]:
    global initialization_error_message
    if llm_global is None:
        app.logger.error("llm_global is not initialized for generate_response_based_on_context_global.")
        return "L'instance du modèle de langage n'est pas disponible.", "", []
    if dbVector_global is None:
        app.logger.error("dbVector_global is not initialized for RAG.")
        err_msg_db = "La base de données vectorielle n'est pas disponible."
        if initialization_error_message and "PDF" in initialization_error_message.upper():
             err_msg_db += f" Erreur initiale: {initialization_error_message}"
        return err_msg_db, "", []

    app.logger.info(f"\n[*] Préparation de la réponse RAG pour : '{question}' (k_for_context={k_for_context})")
    similar_context_text, sources_structured = search_similarity_global(question, k=k_for_context)

    if not similar_context_text:
        app.logger.info("[!] Aucun contexte pertinent trouvé pour cette question.")
        return "Je n'ai trouvé aucune information pertinente dans les documents fournis pour répondre à cette question.", "Aucun contexte trouvé.", []

    prompt_text = f"""INSTRUCTIONS STRICTES:
1. Répondez à la question ci-dessous en utilisant EXCLUSIVEMENT les informations présentes dans le 'CONTEXTE' fourni.
2. Ne faites AUCUNE supposition. N'ajoutez AUCUNE information externe non présente dans le CONTEXTE.
3. Si la réponse ne se trouve PAS dans le CONTEXTE, répondez littéralement : "L'information demandée n'est pas disponible dans le contexte fourni."
4. Soyez concis et factuel.
5. Si possible, indiquez les sources des informations que vous utilisez, en vous basant sur les métadonnées 'source_file' et 'page' fournies avec chaque segment de contexte. Par exemple: (Source: nom_fichier.pdf, Page: X).

CONTEXTE:
--- début du contexte ---
{similar_context_text}
--- fin du contexte ---

QUESTION:
{question}

RÉPONSE FACTUELLE (basée uniquement sur le contexte et citant les sources si possible):"""

    app.logger.info("[*] Invocation du LLM avec le contexte et les instructions...")
    try:
        response_llm = llm_global.invoke(prompt_text)
        final_answer = response_llm.content if hasattr(response_llm, 'content') else str(response_llm)
        app.logger.info("[*] Réponse LLM reçue.")
        return final_answer.strip(), similar_context_text, sources_structured
    except Exception as e:
        app.logger.error(f"[!] ERREUR lors de l'invocation du LLM ({type(e).__name__}): {e}")
        app.logger.exception("Traceback de l'invocation LLM:")
        return "Désolé, une erreur technique est survenue lors de la génération de la réponse.", similar_context_text, sources_structured


# --- RAG Pipeline Initialization Function ---
def initialize_rag_pipeline():
    global all_loaded_docs_global, chunks_global, embeddingModel_global
    global dbVector_global, llm_global, rag_pipeline_ready_global, initialization_error_message
    global loaded_pdf_filenames_global

    # Réinitialisation
    all_loaded_docs_global = []
    chunks_global = []
    embeddingModel_global = None
    dbVector_global = None
    llm_global = None
    rag_pipeline_ready_global = False
    initialization_error_message = None
    loaded_pdf_filenames_global = []

    app.logger.info("--- Initialisation du pipeline RAG ---")

    # 0. API Key
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        err_msg_key = "Clé API Groq (GROQ_API_KEY) non trouvée/définie dans les variables d'environnement. LLM non initialisé."
        # Conditionner getpass pour ne pas bloquer en mode API non interactif
        if os.isatty(0) and os.isatty(1): # Seulement si en terminal interactif
            app.logger.warning(f"{err_msg_key} Tentative de saisie interactive.")
            try:
                groq_api_key = getpass("🔑 Veuillez entrer votre clé API Groq (ne sera pas affichée) : ")
                if not groq_api_key:
                    err_msg_key_interactive = "Clé API Groq non fournie interactivement."
                    initialization_error_message = (initialization_error_message + "; " if initialization_error_message else "") + err_msg_key_interactive
                    app.logger.error(err_msg_key_interactive)
                else:
                    app.logger.info("✅ Clé API Groq fournie interactivement.")
            except (RuntimeError, EOFError) as e_getpass:
                err_msg_key_interactive_fail = f"Impossible de demander la clé API Groq interactivement ({e_getpass})."
                initialization_error_message = (initialization_error_message + "; " if initialization_error_message else "") + err_msg_key_interactive_fail
                app.logger.error(err_msg_key_interactive_fail)
        else: # Non interactif et clé non trouvée dans l'env
            initialization_error_message = (initialization_error_message + "; " if initialization_error_message else "") + err_msg_key
            app.logger.error(err_msg_key)
    else:
        app.logger.info("✅ Clé API Groq trouvée dans les variables d'environnement.")

    # 1. PDF Loading
    app.logger.info(f"[*] Étape 1: Chargement des PDF depuis '{PDF_FOLDER}'")
    os.makedirs(PDF_FOLDER, exist_ok=True)
    try:
        pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    except FileNotFoundError:
        err_msg = f"[!] ERREUR : Le dossier '{PDF_FOLDER}' n'a pas été trouvé."
        initialization_error_message = (initialization_error_message + "; " if initialization_error_message else "") + err_msg
        app.logger.error(err_msg)
        pdf_files = []
    
    if not pdf_files:
        app.logger.warning(f"[!] Aucun fichier PDF trouvé dans '{PDF_FOLDER}'.")
    else:
        app.logger.info(f"[*] Fichiers PDF trouvés à traiter : {pdf_files}")
        successful_loads = 0
        for pdf_file in pdf_files:
            file_path = os.path.join(PDF_FOLDER, pdf_file)
            app.logger.info(f"[*] Chargement du document : {pdf_file} ...")
            try:
                reader = PyPDFLoader(file_path)
                pages_du_document = reader.load()
                if not pages_du_document:
                    app.logger.warning(f"[!] Aucune page extraite de '{pdf_file}'. Le fichier est peut-être vide ou non supporté.")
                    continue
                
                valid_pages_for_file = []
                for page_doc in pages_du_document:
                    page_doc.page_content = normalize_text(page_doc.page_content)
                    if not page_doc.page_content.strip():
                        app.logger.debug(f"[*] Page {page_doc.metadata.get('page', 'N/A')} de '{pdf_file}' est vide après normalisation, ignorée.")
                        continue
                    page_doc.metadata["source_file"] = pdf_file
                    valid_pages_for_file.append(page_doc)
                
                if valid_pages_for_file:
                    all_loaded_docs_global.extend(valid_pages_for_file)
                    successful_loads += 1
                    if pdf_file not in loaded_pdf_filenames_global:
                        loaded_pdf_filenames_global.append(pdf_file)
                    app.logger.info(f"[*]   -> Chargé {len(valid_pages_for_file)} page(s) valides depuis '{pdf_file}'.")
                else:
                    app.logger.warning(f"[!] Aucune page avec contenu valide trouvée dans '{pdf_file}' après normalisation.")

            except Exception as e:
                app.logger.error(f"[!] ERREUR critique lors du chargement ou traitement de '{pdf_file}': {e}")
                app.logger.exception(f"Traceback du chargement de {pdf_file}:")
        
        if successful_loads == 0 and pdf_files:
            err_msg = "Des fichiers PDF étaient présents mais aucune page avec contenu valide n'a pu être chargée."
            initialization_error_message = (initialization_error_message + "; " if initialization_error_message else "") + err_msg
            app.logger.error(err_msg)
        elif all_loaded_docs_global:
            app.logger.info(f"\n[*] Chargement PDF terminé. Total de {len(all_loaded_docs_global)} pages valides chargées depuis {len(loaded_pdf_filenames_global)} fichier(s) unique(s).")
        else:
             app.logger.warning("[!] Aucun document PDF valide n'a été chargé à l'issue du processus.")

    # 2. Text Splitting
    app.logger.info(f"[*] Étape 2: Découpage du texte")
    if all_loaded_docs_global:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len, add_start_index=True,
        )
        app.logger.info(f"[*] Découpage des {len(all_loaded_docs_global)} pages en chunks...")
        try:
            chunks_global = text_splitter.split_documents(all_loaded_docs_global)
            app.logger.info(f"[*]   -> {len(chunks_global)} chunks créés.")
            if not chunks_global and all_loaded_docs_global:
                 err_msg = "Documents chargés mais aucun chunk n'a pu être créé (vérifiez le contenu et la taille des chunks)."
                 initialization_error_message = (initialization_error_message + "; " if initialization_error_message else "") + err_msg
                 app.logger.warning(f"[!] {err_msg}")
        except Exception as e:
            err_msg = f"[!] ERREUR critique lors du découpage : {e}"
            initialization_error_message = (initialization_error_message + "; " if initialization_error_message else "") + err_msg
            app.logger.error(err_msg)
            app.logger.exception("Traceback du text splitting:")
    else:
        app.logger.info("[!] Aucun document valide chargé, donc pas de découpage de texte.")

    # 3. Embedding Model
    app.logger.info(f"[*] Étape 3: Initialisation du modèle d'embedding")
    if chunks_global:
        app.logger.info(f"[*] Initialisation de HuggingFaceEmbeddings : {HF_EMBEDDING_MODEL_NAME}")
        try:
            embeddingModel_global = HuggingFaceEmbeddings(
                model_name=HF_EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'},
            )
            _ = embeddingModel_global.embed_query("Test d'embedding.")
            app.logger.info(f"✅ Modèle HuggingFace '{HF_EMBEDDING_MODEL_NAME}' initialisé.")
        except Exception as e:
            err_msg_embed = f"[!] ERREUR CRITIQUE Embedding Model : {e}. Le RAG ne sera pas fonctionnel."
            initialization_error_message = (initialization_error_message + "; " if initialization_error_message else "") + err_msg_embed
            app.logger.error(err_msg_embed)
            app.logger.exception("Traceback de l'init Embedding Model:")
    elif all_loaded_docs_global: # Chunks non créés mais documents chargés
        app.logger.warning("[!] Documents chargés mais aucun chunk créé. Initialisation Embedding/Chroma sautée.")
    else: # Pas de documents chargés
        app.logger.info("[!] Pas de chunks (car pas de documents valides), init embedding model sautée.")

    # 4. Vector Store (Chroma)
    app.logger.info(f"[*] Étape 4: Initialisation du Vector Store (Chroma)")
    if chunks_global and embeddingModel_global:
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        app.logger.info(f"[*] Dossier de persistance Chroma DB : {os.path.abspath(CHROMA_PERSIST_DIR)}")
        try:
            needs_rebuild = False
            temp_db_to_check_consistency = None

            if os.path.exists(CHROMA_PERSIST_DIR) and any(True for _ in os.scandir(CHROMA_PERSIST_DIR)):
                app.logger.info(f"[*] Base Chroma existante trouvée. Tentative de chargement pour vérification...")
                try:
                    temp_db_to_check_consistency = Chroma(
                        persist_directory=CHROMA_PERSIST_DIR,
                        embedding_function=embeddingModel_global
                    )
                    if temp_db_to_check_consistency._collection.count() == 0:
                        app.logger.warning("[!] Base Chroma chargée mais collection vide. Reconstruction nécessaire.")
                        needs_rebuild = True
                    else:
                        current_db_count = temp_db_to_check_consistency._collection.count()
                        app.logger.info(f"[*] Base existante chargée (temporairement). Contient {current_db_count} documents.")
                        if current_db_count != len(chunks_global):
                            app.logger.warning(f"[*] DISCREPANCE: DB ({current_db_count} chunks) vs Chunks actuels ({len(chunks_global)}). Reconstruction.")
                            needs_rebuild = True
                        else:
                            # Vérification des sources (simplifiée)
                            db_sources_metadata = temp_db_to_check_consistency._collection.get(limit=max(1, len(chunks_global) // 10), include=["metadatas"])['metadatas']
                            if db_sources_metadata is not None:
                                db_source_files = set(meta.get('source_file') for meta in db_sources_metadata if meta and meta.get('source_file'))
                                current_source_files = set(chunk.metadata.get('source_file') for chunk in chunks_global if chunk.metadata.get('source_file'))
                                if not db_source_files.issubset(current_source_files) or not current_source_files.issubset(db_source_files) : # check if sets are different
                                     app.logger.warning(f"[*] DISCREPANCE (approximative) des fichiers sources détectée. Reconstruction.")
                                     needs_rebuild = True
                                else:
                                    app.logger.info("✅ Cohérence de base vérifiée (nombre de chunks et échantillon de sources).")
                                    dbVector_global = temp_db_to_check_consistency
                                    temp_db_to_check_consistency = None
                            else:
                                app.logger.warning("[!] Impossible de vérifier les métadonnées de la DB existante. Reconstruction.")
                                needs_rebuild = True
                except Exception as e_load:
                    app.logger.warning(f"[!] Erreur lors du chargement/vérification de la base Chroma existante: {e_load}. Reconstruction forcée.")
                    needs_rebuild = True
                    if temp_db_to_check_consistency: del temp_db_to_check_consistency
            else:
                app.logger.info(f"[*] Aucune base Chroma existante ou dossier vide. Création nécessaire.")
                needs_rebuild = True

            if needs_rebuild:
                app.logger.info(f"[*] Reconstruction/Création de la base Chroma...")
                if dbVector_global is not None: # Si dbVector_global a été assigné mais qu'on reconstruit quand même
                    app.logger.info("[*]   Libération de l'ancienne instance dbVector_global avant reconstruction.")
                    del dbVector_global 
                    dbVector_global = None
                    gc.collect()

                if os.path.exists(CHROMA_PERSIST_DIR) and any(True for _ in os.scandir(CHROMA_PERSIST_DIR)):
                    app.logger.info(f"[*]   Tentative de suppression du contenu du dossier '{CHROMA_PERSIST_DIR}'.")
                    MAX_ATTEMPTS = 3; RETRY_DELAY = 0.5
                    for attempt in range(MAX_ATTEMPTS):
                        try:
                            for item_name in os.listdir(CHROMA_PERSIST_DIR):
                                item_path = os.path.join(CHROMA_PERSIST_DIR, item_name)
                                if os.path.isdir(item_path): shutil.rmtree(item_path)
                                else: os.unlink(item_path)
                            app.logger.info(f"[*]     Contenu de '{CHROMA_PERSIST_DIR}' supprimé.")
                            break
                        except PermissionError as e_perm:
                            app.logger.warning(f"Attempt {attempt + 1} to delete '{CHROMA_PERSIST_DIR}' failed: {e_perm}")
                            if attempt < MAX_ATTEMPTS - 1: time.sleep(RETRY_DELAY)
                            else: app.logger.error(f"Failed to delete '{CHROMA_PERSIST_DIR}' after {MAX_ATTEMPTS} attempts.")
                        except Exception as e_del_other:
                            app.logger.error(f"Unexpected error deleting '{CHROMA_PERSIST_DIR}': {e_del_other}")
                            break

                app.logger.info(f"[*]   Création d'une nouvelle base Chroma dans '{CHROMA_PERSIST_DIR}' avec {len(chunks_global)} chunks...")
                dbVector_global = Chroma.from_documents(
                    documents=chunks_global,
                    embedding=embeddingModel_global,
                    persist_directory=CHROMA_PERSIST_DIR
                )
                app.logger.info(f"✅ Nouvelle base Chroma créée. Contient {dbVector_global._collection.count()} documents.")
            
            # S'assurer que dbVector_global est bien assigné s'il n'y avait pas besoin de reconstruire
            # et que temp_db_to_check_consistency avait été assigné
            elif temp_db_to_check_consistency is not None and dbVector_global is None :
                dbVector_global = temp_db_to_check_consistency
                app.logger.info("✅ Base Chroma existante utilisée (cohérence vérifiée).")


        except Exception as e:
            err_msg_chroma = f"[!] ERREUR CRITIQUE Chroma DB : {e}. Le RAG ne sera pas fonctionnel."
            initialization_error_message = (initialization_error_message + "; " if initialization_error_message else "") + err_msg_chroma
            app.logger.error(err_msg_chroma)
            app.logger.exception("Traceback complet de l'erreur Chroma DB:")
            dbVector_global = None
    elif chunks_global and not embeddingModel_global:
        err_msg = "Chunks présents mais Embedding Model non initialisé. Chroma DB non créée."
        initialization_error_message = (initialization_error_message + "; " if initialization_error_message else "") + err_msg
        app.logger.warning(f"[!] {err_msg}")
    elif not chunks_global and all_loaded_docs_global:
         app.logger.warning("[!] Documents chargés mais aucun chunk créé. Vector Store (Chroma) sauté.")
    else: # Pas de chunks car pas de documents valides, ou embedding model manquant.
        app.logger.info("[!] Pas de chunks valides ou embedding model manquant, init Chroma DB sautée.")

    # 5. LLM (ChatGroq)
    app.logger.info(f"[*] Étape 5: Initialisation du LLM (ChatGroq)")
    if groq_api_key:
        app.logger.info(f"[*] Initialisation LLM ChatGroq ({GROQ_MODEL_NAME})...")
        try:
            llm_global = ChatGroq(model_name=GROQ_MODEL_NAME, api_key=groq_api_key, temperature=0.1)
            _ = llm_global.invoke("Bonjour")
            app.logger.info(f"✅ LLM ChatGroq initialisé.")
        except Exception as e:
            err_msg_llm = f"[!] ERREUR CRITIQUE LLM ChatGroq : {e}."
            initialization_error_message = (initialization_error_message + "; " if initialization_error_message else "") + err_msg_llm
            app.logger.error(err_msg_llm)
            app.logger.exception("Traceback de l'init LLM:")
            llm_global = None
    elif not (initialization_error_message and "GROQ_API_KEY" in initialization_error_message): # Eviter message dupliqué
        err_msg = "[!] Clé API Groq non obtenue/valide. LLM non initialisé."
        initialization_error_message = (initialization_error_message or "") + err_msg
        app.logger.error(err_msg)

    # Statut final du pipeline
    if llm_global:
        rag_pipeline_ready_global = True
        if dbVector_global and chunks_global and embeddingModel_global:
            app.logger.info("✅ --- Pipeline RAG initialisé avec succès (LLM + Documents) ! ---")
        else:
            app.logger.warning("✅ --- Pipeline initialisé (LLM seul, car composants RAG non tous prêts). ---")
            rag_component_errors = []
            if not all_loaded_docs_global and pdf_files: rag_component_errors.append("chargement PDF")
            if not chunks_global and all_loaded_docs_global : rag_component_errors.append("découpage en chunks")
            if not embeddingModel_global and chunks_global: rag_component_errors.append("modèle d'embedding")
            if not dbVector_global and chunks_global and embeddingModel_global: rag_component_errors.append("base vectorielle Chroma")
            if rag_component_errors:
                specific_error = f"Problèmes potentiels avec : {', '.join(rag_component_errors)}."
                initialization_error_message = (initialization_error_message + "; " if initialization_error_message else "") + specific_error
                app.logger.warning(specific_error)
            elif not pdf_files and not all_loaded_docs_global:
                app.logger.info("Aucun PDF trouvé ou chargé, fonctionnement en mode LLM seul.")
    else:
        rag_pipeline_ready_global = False
        app.logger.error("--- Pipeline RAG N'A PAS PU ÊTRE INITIALISÉ (LLM manquant). ---")
        if not initialization_error_message:
             initialization_error_message = "Erreur critique inconnue lors de l'initialisation du LLM."
        app.logger.error(f"Message d'erreur final pour l'initialisation: {initialization_error_message}")

# --- Flask API Routes ---
@app.route('/api/status', methods=['GET'])
def api_status():
    status_msg_text = "Système non initialisé ou erreur."
    if rag_pipeline_ready_global:
        if dbVector_global and chunks_global and embeddingModel_global:
            status_msg_text = "Système RAG prêt (LLM + Documents)."
        else:
            status_msg_text = "LLM prêt (documents RAG non disponibles ou erreur lors de leur traitement)."
    
    error_to_display = initialization_error_message
    if not rag_pipeline_ready_global and not error_to_display:
        error_to_display = "Le système n'a pas pu démarrer correctement. Vérifiez les logs du serveur."

    return jsonify({
        "rag_pipeline_ready": rag_pipeline_ready_global,
        "llm_ready": bool(llm_global),
        "db_ready": bool(dbVector_global and chunks_global and embeddingModel_global),
        "loaded_pdfs": loaded_pdf_filenames_global,
        "initialization_error": error_to_display,
        "message": status_msg_text
    })

@app.route('/api/chat', methods=['POST'])
def api_chat():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    question = data.get('question', '').strip()
    # Optionnel: récupérer k_context depuis la requête si on veut le contrôler dynamiquement
    # k_for_llm_context = data.get('k_context', 3) # Default à 3 si non fourni
    k_for_llm_context = DEFAULT_K_RETRIEVAL # Ou utiliser la constante globale

    if not question:
        return jsonify({"error": "No question provided"}), 400

    answer_text = ""
    context_for_llm = "N/A"
    sources_data = []
    mode = "UNKNOWN"
    error_detail_chat = None

    if not rag_pipeline_ready_global or not llm_global:
        current_error = initialization_error_message or "Le système n'est pas prêt (LLM non initialisé)."
        return jsonify({
            "question": question,
            "answer": "Désolé, le système n'est pas opérationnel pour le moment.",
            "context_provided_to_llm": "N/A",
            "sources": [],
            "mode": "UNAVAILABLE",
            "error": current_error
        }), 503

    if not (dbVector_global and chunks_global and embeddingModel_global):
        mode = "LLM_ONLY"
        app.logger.info(f"[*] Mode LLM seul (composants RAG non prêts). Question: {question}")
        try:
            response_llm = llm_global.invoke(question)
            answer_text = response_llm.content if hasattr(response_llm, 'content') else str(response_llm)
            context_for_llm = "Aucun contexte RAG utilisé (mode LLM seul)."
        except Exception as e:
            app.logger.error(f"Erreur LLM seul lors de la question: {e}")
            app.logger.exception("Traceback de l'erreur LLM seul:")
            answer_text = "Désolé, erreur lors de la communication avec le LLM."
            error_detail_chat = str(e)
    else:
        mode = "RAG"
        answer_text, context_for_llm, sources_data = generate_response_based_on_context_global(question, k_for_context=k_for_llm_context)

    response_payload = {
        "question": question,
        "answer": answer_text,
        "context_provided_to_llm": context_for_llm,
        "sources": sources_data,
        "mode": mode
    }
    if error_detail_chat:
        response_payload["error"] = error_detail_chat
        # Ne pas forcément retourner 500 ici, sauf si c'est une erreur fatale non gérée.
        # Une réponse d'erreur du LLM peut être une réponse 200 avec un champ 'error'.

    return jsonify(response_payload)


@app.route('/api/reinitialize', methods=['POST'])
def api_reinitialize():
    app.logger.info("Requête de réinitialisation reçue via API.")
    # Pas besoin de réinitialiser les globales ici, initialize_rag_pipeline le fait déjà.
    try:
        initialize_rag_pipeline()
        app.logger.info("Réinitialisation terminée.")
        return api_status()
    except Exception as e:
        app.logger.error(f"Erreur majeure pendant la réinitialisation via API: {e}")
        app.logger.exception("Traceback reinitialisation API:")
        return jsonify({"error": f"Erreur lors de la réinitialisation: {str(e)}", "status": "failed"}), 500

# --- Application Startup ---
if __name__ == '__main__':
    # Configurer le logging de Flask pour plus de détails si besoin
    # import logging
    # app.logger.setLevel(logging.DEBUG) # Ou INFO

    initialize_rag_pipeline()
    app.run(debug=True, host='0.0.0.0', port=5001) # Ou 5000