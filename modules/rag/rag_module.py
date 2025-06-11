# modules/rag/rag_module.py

import os
import json
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import PyPDF2
import docx
from unstructured.partition.auto import partition


@dataclass
class DocumentChunk:
    """Représente un chunk de document avec ses métadonnées"""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    chunk_id: str = ""


class DocumentProcessor:
    """Traite différents types de documents"""
    
    @staticmethod
    def load_text_file(file_path: str) -> str:
        """Charge un fichier texte"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def load_pdf_file(file_path: str) -> str:
        """Charge un fichier PDF"""
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logging.error(f"Erreur lors du chargement du PDF {file_path}: {e}")
            return ""
    
    @staticmethod
    def load_docx_file(file_path: str) -> str:
        """Charge un fichier Word"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logging.error(f"Erreur lors du chargement du DOCX {file_path}: {e}")
            return ""
    
    @staticmethod
    def load_document(file_path: str) -> str:
        """Charge un document selon son extension"""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension == '.txt':
            return DocumentProcessor.load_text_file(str(file_path))
        elif extension == '.pdf':
            return DocumentProcessor.load_pdf_file(str(file_path))
        elif extension in ['.docx', '.doc']:
            return DocumentProcessor.load_docx_file(str(file_path))
        else:
            # Utilise unstructured pour les autres formats
            try:
                elements = partition(filename=str(file_path))
                return " ".join([elem.text for elem in elements if hasattr(elem, 'text')])
            except Exception as e:
                logging.error(f"Impossible de charger le document {file_path}: {e}")
                return ""


class FAISSVectorStore:
    """Gestionnaire de la base de données vectorielle FAISS"""
    
    def __init__(self, dimension: int = 768, index_type: str = "IndexFlatIP"):
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.documents: List[DocumentChunk] = []
        self.metadata_store: Dict[int, Dict[str, Any]] = {}
        
        self._create_index()
    
    def _create_index(self):
        """Crée l'index FAISS"""
        if self.index_type == "IndexFlatIP":
            # Index exact avec produit scalaire (cosine similarity après normalisation)
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IndexIVFFlat":
            # Index avec quantification pour de gros volumes
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        elif self.index_type == "IndexHNSW":
            # Index HNSW pour recherche approximative rapide
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.hnsw.efConstruction = 40
            self.index.hnsw.efSearch = 16
        else:
            raise ValueError(f"Type d'index non supporté: {self.index_type}")
    
    def add_vectors(self, embeddings: np.ndarray, documents: List[DocumentChunk]):
        """Ajoute des vecteurs à l'index"""
        # Normalise les embeddings pour utiliser cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Entraîne l'index si nécessaire (pour IVF)
        if self.index_type == "IndexIVFFlat" and not self.index.is_trained:
            self.index.train(embeddings)
        
        # Ajoute les vecteurs
        start_id = len(self.documents)
        self.index.add(embeddings)
        
        # Met à jour les métadonnées
        for i, doc in enumerate(documents):
            doc_id = start_id + i
            self.documents.append(doc)
            self.metadata_store[doc_id] = doc.metadata
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Recherche les k documents les plus similaires"""
        # Normalise le vecteur de requête
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Effectue la recherche
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def save(self, save_path: str):
        """Sauvegarde l'index et les métadonnées"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarde l'index FAISS
        faiss.write_index(self.index, str(save_path / "index.faiss"))
        
        # Sauvegarde les documents et métadonnées
        with open(save_path / "documents.pkl", 'wb') as f:
            pickle.dump(self.documents, f)
        
        with open(save_path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(self.metadata_store, f, ensure_ascii=False, indent=2)
    
    def load(self, load_path: str):
        """Charge l'index et les métadonnées"""
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Le chemin {load_path} n'existe pas")
        
        # Charge l'index FAISS
        self.index = faiss.read_index(str(load_path / "index.faiss"))
        
        # Charge les documents
        with open(load_path / "documents.pkl", 'rb') as f:
            self.documents = pickle.load(f)
        
        # Charge les métadonnées
        with open(load_path / "metadata.json", 'r', encoding='utf-8') as f:
            # Convertit les clés string en int
            metadata_str = json.load(f)
            self.metadata_store = {int(k): v for k, v in metadata_str.items()}


class RAGModule:
    """Module RAG principal utilisant FAISS"""
    
    def __init__(self, config: Dict[str, Any]):
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        self.config = config
        self.embedding_model_name = config.get('embedding_model', 'distiluse-base-multilingual-cased')
        self.chunk_size = config.get('chunk_size', 512)
        self.chunk_overlap = config.get('chunk_overlap', 50)
        self.top_k_results = config.get('top_k_results', 5)
        self.vector_store_path = config.get('vector_store_path', './data/vector_db')
        # Seuil de similarité minimum (entre -1 et 1 pour cosine similarity)
        self.similarity_threshold = config.get('similarity_threshold', 0.3)
        
        # Initialise le modèle d'embeddings
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialise le text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        )
        
        # Initialise le vector store
        index_type = config.get('index_type', 'IndexFlatIP')
        self.vector_store = FAISSVectorStore(
            dimension=self.embedding_dimension,
            index_type=index_type
        )
        
        # Charge l'index existant s'il existe
        self._load_existing_index()
    
    def _load_existing_index(self):
        """Charge l'index existant s'il existe"""
        try:
            if Path(self.vector_store_path).exists():
                self.vector_store.load(self.vector_store_path)
                self.logger.info(f"Index chargé avec {len(self.vector_store.documents)} documents")
        except Exception as e:
            self.logger.warning(f"Impossible de charger l'index existant: {e}")
    
    def _create_chunks(self, text: str, source: str) -> List[DocumentChunk]:
        """Divise un texte en chunks"""
        chunks = self.text_splitter.split_text(text)
        
        document_chunks = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Ignore les chunks vides
                metadata = {
                    'source': source,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk)
                }
                
                chunk_id = f"{source}_{i}"
                document_chunks.append(DocumentChunk(
                    content=chunk.strip(),
                    metadata=metadata,
                    chunk_id=chunk_id
                ))
        
        return document_chunks
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Génère les embeddings pour une liste de textes"""
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def ingest_documents(self, docs_path: str) -> int:
        """Ingère tous les documents d'un répertoire"""
        docs_path = Path(docs_path)
        if not docs_path.exists():
            raise FileNotFoundError(f"Le chemin {docs_path} n'existe pas")
        
        total_chunks = 0
        
        # Traite tous les fichiers du répertoire
        for file_path in docs_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.pdf', '.docx', '.doc']:
                try:
                    chunks_added = self.ingest_document(str(file_path))
                    total_chunks += chunks_added
                    self.logger.info(f"Document {file_path.name}: {chunks_added} chunks ajoutés")
                except Exception as e:
                    self.logger.error(f"Erreur lors de l'ingestion de {file_path}: {e}")
        
        # Sauvegarde l'index
        self.save_index()
        self.logger.info(f"Ingestion terminée: {total_chunks} chunks au total")
        
        return total_chunks
    
    def ingest_document(self, file_path: str) -> int:
        """Ingère un seul document"""
        # Charge le document
        content = DocumentProcessor.load_document(file_path)
        if not content.strip():
            self.logger.warning(f"Document vide ou non lisible: {file_path}")
            return 0
        
        # Crée les chunks
        chunks = self._create_chunks(content, file_path)
        if not chunks:
            return 0
        
        # Génère les embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self._generate_embeddings(texts)
        
        # Ajoute au vector store
        self.vector_store.add_vectors(embeddings, chunks)
        
        return len(chunks)
    
    def add_text_document(self, content: str, metadata: Dict[str, Any]) -> int:
        """Ajoute un document texte directement"""
        source = metadata.get('source', 'manual_input')
        
        # Crée les chunks
        chunks = self._create_chunks(content, source)
        if not chunks:
            return 0
        
        # Met à jour les métadonnées
        for chunk in chunks:
            chunk.metadata.update(metadata)
        
        # Génère les embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self._generate_embeddings(texts)
        
        # Ajoute au vector store
        self.vector_store.add_vectors(embeddings, chunks)
        
        return len(chunks)
    
    def search_relevant_docs(self, query: str, k: int = None) -> List[Tuple[str, Dict[str, Any], float]]:
        """Recherche les documents pertinents pour une requête"""
        if k is None:
            k = self.top_k_results
        
        # Génère l'embedding de la requête
        query_embedding = self._generate_embeddings([query])[0]
        
        # Recherche les documents similaires
        # Augmenter k pour avoir plus de chances de trouver des documents au-dessus du seuil
        search_k = min(k * 5, len(self.vector_store.documents))  # Recherche 5x plus de documents
        results = self.vector_store.search(query_embedding, k=search_k)
        
        # Filtre les résultats par le seuil de similarité
        filtered_results = []
        total_docs = len(results)
        docs_above_threshold = 0
        
        self.logger.info(f"Seuil de similarité configuré: {self.similarity_threshold:.2%}")
        
        for doc, score in results:
            # Le score est une similarité cosinus (entre -1 et 1)
            # Un score > 0 indique une similarité positive
            if score >= self.similarity_threshold:
                filtered_results.append((doc.content, doc.metadata, score))
                docs_above_threshold += 1
                self.logger.debug(f"Document accepté - Score: {score:.2%}, Source: {doc.metadata.get('source', 'unknown')}")
            else:
                self.logger.debug(f"Document rejeté - Score trop faible: {score:.2%} < {self.similarity_threshold}")
        
        self.logger.info(f"Documents trouvés: {total_docs}, Documents au-dessus du seuil: {docs_above_threshold}")
        
        # Trie par score et limite au nombre demandé
        filtered_results.sort(key=lambda x: x[2], reverse=True)
        return filtered_results[:k]
    
    def build_context(self, query: str, max_context_length: int = 4000) -> tuple[Optional[str], float]:
        """
        Construit le contexte pour le LLM en utilisant les documents les plus pertinents.
        
        Args:
            query: La requête de l'utilisateur
            max_context_length: Longueur maximale du contexte en caractères
            
        Returns:
            Un tuple contenant le contexte construit et le score moyen de pertinence
        """
        try:
            # Recherche des documents pertinents
            relevant_docs = self.search_relevant_docs(query)
            
            if not relevant_docs:
                self.logger.info("Aucun document pertinent trouvé.")
                return None, 0.0
            
            # Calcul du score moyen
            scores = [doc[2] for doc in relevant_docs]  # doc[2] est le score
            avg_score = sum(scores) / len(scores)
            
            # Construction du contexte
            context_parts = []
            total_length = 0
            
            for doc in relevant_docs:
                content = doc[0]  # Le contenu est le premier élément
                metadata = doc[1]  # Les métadonnées sont le deuxième élément
                score = doc[2]    # Le score est le troisième élément
                
                # Ajoute des informations sur la source et le score
                source_info = f"[Source: {metadata.get('source', 'unknown')} - Pertinence: {score:.2%}]\n"
                chunk_text = f"{source_info}{content}\n"
                
                # Vérifie si l'ajout du document dépasserait la limite
                if total_length + len(chunk_text) > max_context_length:
                    break
                
                context_parts.append(chunk_text)
                total_length += len(chunk_text)
            
            if not context_parts:
                self.logger.warning("Le contexte dépasse la longueur maximale.")
                return None, avg_score
            
            # Assemblage du contexte final
            context = "\n---\n".join(context_parts)
        
            # Log des statistiques
            self.logger.info(f"Contexte construit avec {len(context_parts)} documents pertinents")
            
            return context, avg_score
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la construction du contexte: {e}", exc_info=True)
            return None, 0.0
    
    def save_index(self):
        """Sauvegarde l'index vectoriel"""
        try:
            self.vector_store.save(self.vector_store_path)
            self.logger.info(f"Index sauvegardé dans {self.vector_store_path}")
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du module RAG"""
        return {
            'total_documents': len(self.vector_store.documents),
            'embedding_model': self.embedding_model_name,
            'embedding_dimension': self.embedding_dimension,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'index_type': self.vector_store.index_type
        }


# Configuration d'exemple
DEFAULT_RAG_CONFIG = {
    'embedding_model': 'distiluse-base-multilingual-cased',
    'chunk_size': 512,
    'chunk_overlap': 50,
    'top_k_results': 5,
    'vector_store_path': './data/vector_db',
    'index_type': 'IndexFlatIP',  # ou 'IndexHNSW' pour de gros volumes
    'similarity_threshold': 0.3  # Réduit à 0.3 pour être moins strict
}

def test_rag_module():
    """Fonction de test du module RAG"""
    print("\n=== Test du module RAG ===")
    print("=" * 50)

    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Créer un répertoire temporaire pour les tests
    import tempfile
    import shutil
    from datetime import datetime
    
    test_dir = Path(tempfile.mkdtemp())
    vector_db_path = test_dir / "vector_db"
    docs_dir = test_dir / "docs"
    docs_dir.mkdir(parents=True)

    try:
        # Créer quelques documents de test
        test_docs = {
            "doc1.txt": "Ceci est un premier document de test.\nIl parle d'intelligence artificielle.",
            "doc2.txt": "Voici un second document.\nCelui-ci concerne le traitement du langage naturel.",
            "doc3.txt": "Un troisième document sur les bases de données vectorielles et FAISS."
        }

        # Écrire les documents
        for filename, content in test_docs.items():
            with open(docs_dir / filename, 'w', encoding='utf-8') as f:
                f.write(content)

        print("\n📁 Documents de test créés dans:", docs_dir)

        # Initialiser le module RAG avec la configuration de test
        config = DEFAULT_RAG_CONFIG.copy()
        config['vector_store_path'] = str(vector_db_path)
        rag = RAGModule(config)

        print("\n📊 Configuration du module:")
        for key, value in rag.get_stats().items():
            print(f"  {key}: {value}")

        # Test d'ingestion
        print("\n🔄 Test d'ingestion des documents...")
        total_chunks = rag.ingest_documents(str(docs_dir))
        print(f"✅ {total_chunks} chunks créés et indexés")

        # Test de recherche
        print("\n🔍 Test de recherche...")
        queries = [
            "intelligence artificielle",
            "base de données",
            "document inexistant"
        ]

        for query in queries:
            print(f"\nRecherche: '{query}'")
            results = rag.search_relevant_docs(query, k=2)
            
            if results:
                print(f"Résultats trouvés: {len(results)}")
                for content, metadata, score in results:
                    source = Path(metadata['source']).name
                    print(f"- [{source}] (score: {score:.3f})")
                    print(f"  {content[:100]}...")
            else:
                print("Aucun résultat trouvé")

        # Test de construction de contexte
        print("\n🔄 Test de construction de contexte...")
        query = "Parlez-moi de l'intelligence artificielle"
        context, avg_score = rag.build_context(query, max_context_length=1000)
        print(f"Contexte généré ({len(context)} caractères):")
        print("-" * 50)
        print(context[:500] + "..." if len(context) > 500 else context)
        print("-" * 50)
        print(f"Score moyen de pertinence: {avg_score:.2%}")

        print("\n✅ Tests terminés avec succès!")

    except Exception as e:
        print(f"\n❌ Erreur lors des tests: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Nettoyage
        try:
            shutil.rmtree(test_dir)
            print("\n🧹 Nettoyage des fichiers de test effectué")
        except Exception as e:
            print(f"\n⚠️ Erreur lors du nettoyage: {e}")

if __name__ == "__main__":
    test_rag_module()