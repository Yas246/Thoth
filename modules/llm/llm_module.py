# modules/llm/llm_module.py

import requests
import json
import logging
import time
from typing import Dict, Any, Optional, List, Iterator
from dataclasses import dataclass
from datetime import datetime


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


class LMStudioAPI:
    """Client pour l'API LM Studio"""
    
    def __init__(self, base_url: str = "http://localhost:1234"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        self.logger = logging.getLogger(__name__)
    
    def _make_request(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
        """Effectue une requête HTTP vers LM Studio"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "GET":
                response = self.session.get(url)
            elif method == "POST":
                response = self.session.post(url, json=data)
            else:
                raise ValueError(f"Méthode HTTP non supportée: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Impossible de se connecter à LM Studio sur {self.base_url}")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"Erreur HTTP {e.response.status_code}: {e.response.text}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Erreur de requête: {str(e)}")
    
    def check_health(self) -> bool:
        """Vérifie si LM Studio est accessible"""
        try:
            response = self._make_request("/models")
            return True
        except Exception as e:
            self.logger.error(f"LM Studio non accessible: {e}")
            return False
    
    def get_models(self) -> List[Dict[str, Any]]:
        """Récupère la liste des modèles disponibles"""
        try:
            response = self._make_request("/models")
            return response.get("data", [])
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des modèles: {e}")
            return []
    
    def get_current_model(self) -> Optional[str]:
        """Récupère le modèle actuellement chargé"""
        models = self.get_models()
        if models:
            return models[0].get("id", "unknown")
        return None
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       temperature: float = 0.7,
                       max_tokens: int = 1000,
                       stream: bool = False,
                       **kwargs) -> Dict[str, Any]:
        """Effectue une completion de chat"""
        
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs
        }
        
        start_time = time.time()
        response = self._make_request("/chat/completions", "POST", payload)
        response_time = time.time() - start_time
        
        # Ajoute le temps de réponse
        response["response_time"] = response_time
        
        return response
    
    def chat_completion_stream(self, 
                              messages: List[Dict[str, str]], 
                              temperature: float = 0.7,
                              max_tokens: int = 1000,
                              **kwargs) -> Iterator[str]:
        """Effectue une completion de chat en streaming"""
        
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            **kwargs
        }
        
        url = f"{self.base_url}/chat/completions"
        
        try:
            with self.session.post(url, json=payload, stream=True) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            line = line[6:]  # Retire 'data: '
                            
                            if line.strip() == '[DONE]':
                                break
                            
                            try:
                                data = json.loads(line)
                                if 'choices' in data and len(data['choices']) > 0:
                                    delta = data['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        yield delta['content']
                            except json.JSONDecodeError:
                                continue
                                
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
        
        # Limite la taille de l'historique (garde le système)
        if len(self.messages) > self.max_history:
            # Garde le message système s'il existe
            if self.messages[0].role == "system":
                self.messages = [self.messages[0]] + self.messages[-(self.max_history-1):]
            else:
                self.messages = self.messages[-self.max_history:]
    
    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """Retourne les messages formatés pour l'API"""
        messages = []
        
        # Ajoute le message système s'il existe
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        
        # Ajoute les messages de conversation
        for msg in self.messages:
            if msg.role != "system":  # Évite les doublons
                messages.append({"role": msg.role, "content": msg.content})
        
        return messages
    
    def get_conversation_history(self) -> str:
        """Retourne l'historique formaté pour les templates"""
        history_parts = []
        for msg in self.messages[-6:]:  # Les 6 derniers messages
            if msg.role == "user":
                history_parts.append(f"Utilisateur : {msg.content}")
            elif msg.role == "assistant":
                history_parts.append(f"Thoth : {msg.content}")
        
        return "\n".join(history_parts)
    
    def clear_history(self):
        """Efface l'historique (garde le système)"""
        if self.system_message:
            self.messages = [ChatMessage("system", self.system_message)]
        else:
            self.messages = []


class LLMModule:
    """Module LLM principal avec intégration LM Studio"""
    
    def __init__(self, config: Dict[str, Any]):
        # Initialise le logger en premier
        self.logger = logging.getLogger(__name__)
        
        self.config = config
        self.api_url = config.get('api_url', 'http://localhost:1234')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 1000)
        self.model_name = config.get('model_name', 'auto')
        
        # Initialise l'API LM Studio
        self.lm_studio = LMStudioAPI(self.api_url)
        
        # Gestionnaire de conversation
        self.conversation = ConversationManager(
            max_history=config.get('max_history', 10)
        )
        
        # Message système par défaut
        default_system = PromptTemplate.get_template("thoth_system")
        self.set_system_prompt(config.get('system_prompt', default_system))
    
    def set_system_prompt(self, prompt: str):
        """Définit le prompt système"""
        self.conversation.set_system_message(prompt)
        self.logger.info("Prompt système mis à jour")
    
    def check_connection(self) -> bool:
        """Vérifie la connexion à LM Studio"""
        return self.lm_studio.check_health()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Récupère les informations sur le modèle"""
        try:
            models = self.lm_studio.get_models()
            current_model = self.lm_studio.get_current_model()
            
            return {
                'current_model': current_model,
                'available_models': [m.get('id', 'unknown') for m in models],
                'api_url': self.api_url,
                'connection_status': self.check_connection()
            }
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des infos modèle: {e}")
            return {
                'current_model': 'unknown',
                'available_models': [],
                'api_url': self.api_url,
                'connection_status': False,
                'error': str(e)
            }
    
    def generate_response(self, query: str, context: str = None, use_history: bool = True) -> LLMResponse:
        """Génère une réponse à partir d'une requête"""
        
        try:
            # Prépare le prompt selon le contexte disponible
            if context:
                # Mode RAG : utilise le contexte fourni
                prompt = PromptTemplate.format_template(
                    "rag_context",
                    context=context,
                    query=query
                )
                messages = [{"role": "user", "content": prompt}]
                
                # Ajoute le message système
                if self.conversation.system_message:
                    messages.insert(0, {"role": "system", "content": self.conversation.system_message})
                    
            elif use_history and self.conversation.messages:
                # Mode conversation : utilise l'historique
                self.conversation.add_message("user", query)
                messages = self.conversation.get_messages_for_api()
                
            else:
                # Mode simple : requête directe
                prompt = PromptTemplate.format_template("simple_query", query=query)
                messages = [{"role": "user", "content": prompt}]
                
                if self.conversation.system_message:
                    messages.insert(0, {"role": "system", "content": self.conversation.system_message})
            
            # Effectue la requête
            start_time = time.time()
            response = self.lm_studio.chat_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Traite la réponse
            if 'choices' in response and len(response['choices']) > 0:
                content = response['choices'][0]['message']['content']
                finish_reason = response['choices'][0].get('finish_reason', 'unknown')
                
                # Ajoute à l'historique si en mode conversation
                if use_history and not context:
                    self.conversation.add_message("assistant", content)
                
                # Calcule les tokens utilisés si disponible
                tokens_used = None
                if 'usage' in response:
                    tokens_used = response['usage'].get('total_tokens')
                
                return LLMResponse(
                    content=content,
                    model_name=self.lm_studio.get_current_model() or "unknown",
                    tokens_used=tokens_used,
                    response_time=response.get('response_time', time.time() - start_time),
                    finish_reason=finish_reason
                )
            else:
                raise Exception("Réponse invalide du modèle")
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération de réponse: {e}")
            return LLMResponse(
                content=f"Désolé, une erreur s'est produite : {str(e)}",
                model_name="error",
                response_time=0.0,
                finish_reason="error"
            )
    
    def generate_response_stream(self, query: str, context: str = None, use_history: bool = True) -> Iterator[str]:
        """Génère une réponse en streaming"""
        
        try:
            # Prépare le prompt (même logique que generate_response)
            if context:
                prompt = PromptTemplate.format_template(
                    "rag_context",
                    context=context,
                    query=query
                )
                messages = [{"role": "user", "content": prompt}]
                
                if self.conversation.system_message:
                    messages.insert(0, {"role": "system", "content": self.conversation.system_message})
                    
            elif use_history and self.conversation.messages:
                self.conversation.add_message("user", query)
                messages = self.conversation.get_messages_for_api()
                
            else:
                prompt = PromptTemplate.format_template("simple_query", query=query)
                messages = [{"role": "user", "content": prompt}]
                
                if self.conversation.system_message:
                    messages.insert(0, {"role": "system", "content": self.conversation.system_message})
            
            # Streaming de la réponse
            full_response = ""
            for chunk in self.lm_studio.chat_completion_stream(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            ):
                if chunk:
                    full_response += chunk
                    yield chunk
            
            # Ajoute à l'historique si en mode conversation
            if use_history and not context and full_response:
                self.conversation.add_message("assistant", full_response)
                
        except Exception as e:
            self.logger.error(f"Erreur lors du streaming: {e}")
            yield f"Erreur: {str(e)}"
    
    def ask_with_rag(self, query: str, context: str) -> LLMResponse:
        """Pose une question avec un contexte RAG"""
        return self.generate_response(query, context=context, use_history=False)
    
    def chat(self, query: str) -> LLMResponse:
        """Mode chat conversationnel"""
        return self.generate_response(query, use_history=True)
    
    def simple_query(self, query: str) -> LLMResponse:
        """Requête simple sans historique"""
        return self.generate_response(query, use_history=False)
    
    def clear_conversation(self):
        """Efface l'historique de conversation"""
        self.conversation.clear_history()
        self.logger.info("Historique de conversation effacé")
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Retourne l'historique de conversation"""
        return [
            {
                'role': msg.role,
                'content': msg.content,
                'timestamp': msg.timestamp.isoformat() if msg.timestamp else None
            }
            for msg in self.conversation.messages
        ]
    
    def update_config(self, new_config: Dict[str, Any]):
        """Met à jour la configuration"""
        self.config.update(new_config)
        
        # Met à jour les paramètres modifiables
        if 'temperature' in new_config:
            self.temperature = new_config['temperature']
        if 'max_tokens' in new_config:
            self.max_tokens = new_config['max_tokens']
        if 'system_prompt' in new_config:
            self.set_system_prompt(new_config['system_prompt'])
        if 'max_history' in new_config:
            self.conversation.max_history = new_config['max_history']
        
        self.logger.info("Configuration mise à jour")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du module"""
        model_info = self.get_model_info()
        
        return {
            'model_info': model_info,
            'configuration': {
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'api_url': self.api_url,
                'max_history': self.conversation.max_history
            },
            'conversation': {
                'message_count': len(self.conversation.messages),
                'system_prompt_set': bool(self.conversation.system_message)
            }
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