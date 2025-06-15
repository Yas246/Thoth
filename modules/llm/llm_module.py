# modules/llm/llm_module.py

import logging
import time
from typing import Dict, Any, Optional, List, Iterator
from dataclasses import dataclass
from datetime import datetime
from llama_cpp import Llama


@dataclass
class ChatMessage:
    """Représente un message dans une conversation"""
    role: str  # "system", "user", "assistant"
    content: str
    timestamp: Optional[datetime] = None


@dataclass
class LLMResponse:
    """Représente une réponse du LLM"""
    content: str
    model_name: str
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None
    finish_reason: Optional[str] = None


class LlamaModel:
    """Client pour le modèle Llama.cpp"""
    
    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: Optional[int] = None):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.logger = logging.getLogger(__name__)
        self.llm = None
        self.initialize_model()
    
    def initialize_model(self):
        """Initialise le modèle Llama"""
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads
            )
            self.logger.info(f"Modèle Llama initialisé avec succès: {self.model_path}")
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation du modèle Llama: {e}")
            raise
    
    def check_health(self) -> bool:
        """Vérifie si le modèle est accessible"""
        return self.llm is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Récupère les informations sur le modèle"""
        return {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads
        }
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       temperature: float = 0.7,
                       max_tokens: int = 1000,
                       stream: bool = False,
                       **kwargs) -> Dict[str, Any]:
        """Effectue une completion de chat"""
        if not self.llm:
            raise RuntimeError("Le modèle n'est pas initialisé")
        
        start_time = time.time()
        
        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs
            )
            
            response_time = time.time() - start_time
            
            if stream:
                return response
            else:
                return {
                    "choices": [{
                        "message": {
                            "content": response["choices"][0]["message"]["content"]
                        }
                    }],
                    "usage": response.get("usage", {}),
                    "response_time": response_time
                }
        except Exception as e:
            self.logger.error(f"Erreur lors de la completion: {e}")
            raise
    
    def chat_completion_stream(self, 
                              messages: List[Dict[str, str]], 
                              temperature: float = 0.7,
                              max_tokens: int = 1000,
                              **kwargs) -> Iterator[str]:
        """Effectue une completion de chat en streaming"""
        try:
            response_stream = self.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            **kwargs
            )
            
            for chunk in response_stream:
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        yield delta["content"]
                                
        except Exception as e:
            self.logger.error(f"Erreur lors du streaming: {e}")
            yield ""


class PromptTemplate:
    """Gestionnaire de templates de prompts"""
    
    # Templates prédéfinis
    TEMPLATES = {
        "thoth_system": """Vous êtes Thoth, un assistant IA français spécialisé dans l'aide et l'information. 
Vous répondez de manière claire, précise et utile. Vous basez vos réponses sur les informations fournies 
dans le contexte lorsqu'elles sont disponibles, et vous indiquez clairement quand vous utilisez vos 
connaissances générales. Répondez toujours en français.""",
        
        "rag_context": """Voici le contexte pertinent trouvé dans la base de connaissances :

{context}

Question de l'utilisateur : {query}

Instructions :
- Utilisez principalement les informations du contexte pour répondre
- Si le contexte ne contient pas d'information pertinente, indiquez-le clairement
- Citez vos sources quand c'est approprié
- Répondez de manière concise et structurée""",
        
        "conversation": """Vous êtes Thoth, un assistant IA conversationnel en français.
Voici l'historique de notre conversation :

{history}

Utilisateur : {query}

Répondez de manière naturelle et cohérente avec le contexte de la conversation.""",
        
        "simple_query": """Question : {query}

Répondez de manière claire et concise en français."""
    }
    
    @classmethod
    def get_template(cls, template_name: str) -> str:
        """Récupère un template par son nom"""
        return cls.TEMPLATES.get(template_name, cls.TEMPLATES["simple_query"])
    
    @classmethod
    def format_template(cls, template_name: str, **kwargs) -> str:
        """Formate un template avec les variables fournies"""
        template = cls.get_template(template_name)
        return template.format(**kwargs)


class ConversationManager:
    """Gestionnaire de l'historique des conversations"""
    
    def __init__(self, max_history: int = 10):
        self.messages: List[ChatMessage] = []
        self.max_history = max_history
        self.system_message: Optional[str] = None
    
    def set_system_message(self, message: str):
        """Définit le message système"""
        self.system_message = message
    
    def add_message(self, role: str, content: str):
        """Ajoute un message à l'historique"""
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now()
        )
        self.messages.append(message)
        
        # Limite la taille de l'historique
        if len(self.messages) > self.max_history:
            if self.messages[0].role == "system":
                self.messages = [self.messages[0]] + self.messages[-(self.max_history-1):]
            else:
                self.messages = self.messages[-self.max_history:]
    
    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """Retourne les messages formatés pour l'API"""
        messages = []
        
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        
        for msg in self.messages:
            if msg.role != "system":
                messages.append({"role": msg.role, "content": msg.content})
        
        return messages
    
    def get_conversation_history(self) -> str:
        """Retourne l'historique formaté pour le prompt"""
        history = []
        for msg in self.messages:
            if msg.role != "system":
                speaker = "Assistant" if msg.role == "assistant" else "Utilisateur"
                history.append(f"{speaker}: {msg.content}")
        return "\n".join(history)
    
    def clear_history(self):
        """Efface l'historique de conversation"""
        system_message = self.system_message
        self.messages = []
        if system_message:
            self.set_system_message(system_message)


class LLMModule:
    """Module principal pour l'interaction avec le LLM"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Configuration du modèle
        model_config = config.get("model", {})
        self.model = LlamaModel(
            model_path=model_config.get("path"),
            n_ctx=model_config.get("n_ctx", 2048),
            n_threads=model_config.get("n_threads")
        )
        
        # Gestionnaire de conversation
        self.conversation = ConversationManager(
            max_history=config.get("max_history", 10)
        )
        
        # Configuration des prompts
        if "system_prompt" in config:
            self.set_system_prompt(config["system_prompt"])
    
    def set_system_prompt(self, prompt: str):
        """Définit le prompt système"""
        self.conversation.set_system_message(prompt)
    
    def check_connection(self) -> bool:
        """Vérifie si le modèle est accessible"""
        return self.model.check_health()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Récupère les informations sur le modèle"""
        return self.model.get_model_info()
    
    def generate_response(self, query: str, context: str = None, use_history: bool = True) -> LLMResponse:
        """Génère une réponse à partir d'une requête"""
        
        if context:
            prompt = PromptTemplate.format_template("rag_context", 
                    context=context,
                                                  query=query)
        elif use_history:
            history = self.conversation.get_conversation_history()
            prompt = PromptTemplate.format_template("conversation",
                                                  history=history,
                                                  query=query)
        else:
            prompt = PromptTemplate.format_template("simple_query",
                                                  query=query)
        
            messages = self.conversation.get_messages_for_api()
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.model.chat_completion(
                messages=messages,
                temperature=self.config.get("temperature", 0.7),
                max_tokens=self.config.get("max_tokens", 1000)
            )
            
            content = response["choices"][0]["message"]["content"]
            self.conversation.add_message("user", query)
            self.conversation.add_message("assistant", content)
                
            return LLMResponse(
                    content=content,
                model_name=self.model.model_path,
                tokens_used=response.get("usage", {}).get("total_tokens"),
                response_time=response.get("response_time"),
                finish_reason=response["choices"][0].get("finish_reason")
            )
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération de réponse: {e}")
            raise
    
    def generate_response_stream(self, query: str, context: str = None, use_history: bool = True) -> Iterator[str]:
        """Génère une réponse en streaming"""
        
        messages = self.conversation.get_messages_for_api()
        
        if context:
            prompt = PromptTemplate.format_template("rag_context",
                    context=context,
                                                  query=query)
        elif use_history:
            history = self.conversation.get_conversation_history()
            prompt = PromptTemplate.format_template("conversation",
                                                  history=history,
                                                  query=query)
        else:
            prompt = PromptTemplate.format_template("simple_query",
                                                  query=query)
        
        messages.append({"role": "user", "content": prompt})
        response_content = []
        
        try:
            for chunk in self.model.chat_completion_stream(
                messages=messages,
                temperature=self.config.get("temperature", 0.7),
                max_tokens=self.config.get("max_tokens", 1000)
            ):
                response_content.append(chunk)
                yield chunk
            
            # Ajoute les messages à l'historique une fois terminé
            full_response = "".join(response_content)
            self.conversation.add_message("user", query)
            self.conversation.add_message("assistant", full_response)
                
        except Exception as e:
            self.logger.error(f"Erreur lors du streaming: {e}")
            yield ""
    
    def ask_with_rag(self, query: str, context: str) -> LLMResponse:
        """Pose une question avec contexte RAG"""
        return self.generate_response(query, context=context)
    
    def chat(self, query: str) -> LLMResponse:
        """Chat avec historique"""
        return self.generate_response(query, use_history=True)
    
    def simple_query(self, query: str) -> LLMResponse:
        """Question simple sans contexte ni historique"""
        return self.generate_response(query, use_history=False)
    
    def clear_conversation(self):
        """Efface l'historique de conversation"""
        self.conversation.clear_history()
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Récupère l'historique de conversation"""
        history = []
        for msg in self.conversation.messages:
            history.append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None
            })
        return history
    
    def update_config(self, new_config: Dict[str, Any]):
        """Met à jour la configuration"""
        self.config.update(new_config)
        
        # Met à jour les paramètres qui peuvent l'être sans réinitialisation
        if "max_history" in new_config:
            self.conversation.max_history = new_config["max_history"]
        
        if "system_prompt" in new_config:
            self.set_system_prompt(new_config["system_prompt"])
    
    def get_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques du module"""
        return {
            "model_info": self.get_model_info(),
            "conversation_length": len(self.conversation.messages),
            "system_message_set": self.conversation.system_message is not None,
            "config": self.config
        }


# Utilitaires pour tester le module
class LLMTester:
    """Classe utilitaire pour tester le module LLM"""
    
    @staticmethod
    def test_connection(api_url: str = "http://localhost:1234") -> Dict[str, Any]:
        """Teste la connexion à LM Studio"""
        try:
            api = LMStudioAPI(api_url)
            is_connected = api.check_health()
            models = api.get_models() if is_connected else []
            current_model = api.get_current_model() if is_connected else None
            
            return {
                'connected': is_connected,
                'api_url': api_url,
                'current_model': current_model,
                'available_models': [m.get('id', 'unknown') for m in models],
                'model_count': len(models)
            }
        except Exception as e:
            return {
                'connected': False,
                'api_url': api_url,
                'error': str(e)
            }
    
    @staticmethod
    def test_basic_query(llm_module: LLMModule, query: str = "Bonjour, comment allez-vous ?") -> Dict[str, Any]:
        """Teste une requête basique"""
        try:
            start_time = time.time()
            response = llm_module.simple_query(query)
            end_time = time.time()
            
            return {
                'success': True,
                'query': query,
                'response': response.content,
                'model': response.model_name,
                'response_time': end_time - start_time,
                'tokens_used': response.tokens_used,
                'finish_reason': response.finish_reason
            }
        except Exception as e:
            return {
                'success': False,
                'query': query,
                'error': str(e)
            }
    
    @staticmethod
    def test_streaming(llm_module: LLMModule, query: str = "Expliquez-moi l'intelligence artificielle") -> Dict[str, Any]:
        """Teste le streaming"""
        try:
            start_time = time.time()
            chunks = []
            
            for chunk in llm_module.generate_response_stream(query, use_history=False):
                chunks.append(chunk)
            
            end_time = time.time()
            full_response = ''.join(chunks)
            
            return {
                'success': True,
                'query': query,
                'response': full_response,
                'chunk_count': len(chunks),
                'response_time': end_time - start_time,
                'streaming_works': len(chunks) > 1
            }
        except Exception as e:
            return {
                'success': False,
                'query': query,
                'error': str(e)
            }


# Configuration par défaut
DEFAULT_LLM_CONFIG = {
    'api_url': 'http://localhost:1234',
    'temperature': 0.7,
    'max_tokens': 1000,
    'model_name': 'auto',
    'max_history': 10,
    'system_prompt': None  # Utilise le prompt par défaut
}


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuration de test
    config = DEFAULT_LLM_CONFIG.copy()
    
    # Test de connexion
    print("\n=== Test de connexion ===")
    connection_test = LLMTester.test_connection()
    print(json.dumps(connection_test, indent=2))
    
    if connection_test['connected']:
        print("\n=== Initialisation du module LLM ===")
        # Initialise le module
        llm = LLMModule(config)
        
        print("\n=== Test basique ===")
        # Test basique
        basic_test = LLMTester.test_basic_query(llm)
        print(json.dumps(basic_test, indent=2))
        
        print("\n=== Test streaming ===")
        # Test streaming
        streaming_test = LLMTester.test_streaming(llm)
        print(json.dumps(streaming_test, indent=2))
        
        # Test conversation
        print("\n=== Test de conversation ===")
        llm.clear_conversation()
        
        questions = [
            "Bonjour, je m'appelle Marie.",
            "Quel est mon nom ?",
            "Pouvez-vous me parler de l'intelligence artificielle ?",
            "Merci pour cette explication."
        ]
        
        for question in questions:
            print(f"\nQ: {question}")
            response = llm.chat(question)
            print(f"R: {response.content}")
            print(f"Temps: {response.response_time:.2f}s")
            print("-" * 50)
    else:
        print("\n⚠️ LM Studio n'est pas accessible. Vérifiez qu'il est lancé sur http://localhost:1234")