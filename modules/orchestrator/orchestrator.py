import yaml
import logging
from pathlib import Path
from typing import Optional
import time # For sleep in __main__ speech test
import threading
import queue
from queue import Queue

# Assuming direct imports work. Adjust to relative if needed (e.g., from ..stt.stt_modules import STTModule)
from modules.stt.stt_modules import STTModule
from modules.rag.rag_module import RAGModule
from modules.llm.llm_module import LLMModule
from modules.tts.tts_module import TTSModule
from modules.wake_word.wake_word_module import WakeWordModule

class Orchestrator:
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config_path = Path(config_path)
        self.config = {}  # Initialize config as an empty dict

        # État de l'assistant
        self.is_listening = False
        self.interaction_queue = Queue()
        self.wake_word_detector = None
        self.current_interaction = None
        self.interrupt_event = threading.Event()
        self.background_detection_thread = None

        # Initialize a basic logger early. It will be reconfigured in _setup_logging.
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO) # Basic config for pre-setup logging

        try:
            self.config = self._load_config()
            self._setup_logging() # Configure logging based on the loaded file
            self.logger.info("Orchestrator configuration loaded successfully from %s.", self.config_path)
        except Exception as e:
            self.logger.critical(f"FATAL: Failed to load or parse configuration '{self.config_path}'. Orchestrator cannot start. Error: {e}", exc_info=True)
            raise # Re-raise, as config is critical for operation

        # Initialize modules
        try:
            stt_config = self.config.get('stt', {})
            # This assumes STTModule can accept a 'config' dict.
            # If STTModule's __init__ strictly expects config_path, this might need future adjustment
            # or STTModule refactoring. For this step, we proceed with passing the dict.
            self.stt_module = STTModule(config=stt_config)
            self.logger.info("STTModule initialized.")

            rag_config = self.config.get('rag', {})
            # Set the documents path only if not specified in config
            if 'documents_path' not in rag_config:
                rag_config['documents_path'] = 'data/documents'
            
            self.rag_module = RAGModule(config=rag_config)
            # Ingest documents if they haven't been ingested yet
            if not Path(rag_config.get('vector_store_path', './data/vector_db')).exists():
                self.logger.info("Ingesting documents from data/documents...")
                try:
                    chunks = self.rag_module.ingest_documents('data/documents')
                    self.logger.info(f"Successfully ingested {chunks} document chunks")
                except Exception as e:
                    self.logger.error(f"Error ingesting documents: {e}", exc_info=True)
            self.logger.info("RAGModule initialized.")

            llm_config = self.config.get('llm', {})
            self.llm_module = LLMModule(config=llm_config)
            self.logger.info("LLMModule initialized.")

            tts_config = self.config.get('tts', {})
            self.tts_module = TTSModule(
                cache_dir=tts_config.get('cache_dir', 'data/audio_cache'),
                audio_engine=tts_config.get('audio_engine', 'auto'),
                # Check for 'lang' in tts_config, then 'default_lang', then fallback to 'fr'
                default_lang=tts_config.get('lang', tts_config.get('default_lang', 'fr'))
            )
            self.logger.info("TTSModule initialized.")

            self.orchestrator_config = self.config.get('thoth', {})
            self.logger.info("Orchestrator specific 'thoth' configuration loaded.")

            # Initialisation du module de détection du mot de réveil
            wake_word_config = self.config.get('wake_word', {})
            if wake_word_config.get('enabled', False):
                try:
                    self.wake_word_detector = WakeWordModule(
                        access_key=wake_word_config['access_key'],
                        wake_word=wake_word_config.get('keyword', 'computer'),
                        sensitivity=wake_word_config.get('sensitivity', 0.5),
                        audio_device_index=wake_word_config.get('audio_device_index')
                    )
                    self.logger.info("WakeWordModule initialized.")
                except Exception as e:
                    self.logger.error(f"Failed to initialize wake word detection: {e}")
                    self.wake_word_detector = None

        except TypeError as te:
            # Catch TypeErrors that might occur if a module's __init__ gets unexpected kwargs
            self.logger.error(f"TypeError during module initialization, possibly due to an unexpected 'config' argument or misconfiguration: {te}", exc_info=True)
            # This often indicates a mismatch between Orchestrator's call and module's __init__ signature.
            self.logger.error("This might be due to a module (e.g. STTModule) not accepting a 'config' dictionary directly. This may require refactoring the module.")
            raise
        except Exception as e:
            self.logger.error(f"Error during module initialization: {e}", exc_info=True)
            raise # Re-raise as modules are critical

    def _load_config(self) -> dict:
        if not self.config_path.is_file():
            # This log might use the basic config if _setup_logging hasn't run
            self.logger.error(f"Configuration file not found at {self.config_path}")
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                if config_data is None: # Handles empty YAML file
                    self.logger.warning(f"Configuration file {self.config_path} is empty or invalid YAML. Returning empty config.")
                    return {}
                return config_data
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML configuration file {self.config_path}: {e}", exc_info=True)
            raise
        except Exception as e: # Catch any other unexpected error during file loading
            self.logger.error(f"An unexpected error occurred while loading configuration {self.config_path}: {e}", exc_info=True)
            raise

    def _setup_logging(self):
        # Configures logging based on the 'logging' section of self.config
        log_config = self.config.get('logging', {}) # Default to empty dict if 'logging' section is missing

        level_name = str(log_config.get('level', 'INFO')).upper() # Ensure level_name is a string
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        numeric_level = logging.getLevelName(level_name)
        if not isinstance(numeric_level, int):
            self.logger.warning(f"Invalid logging level '{level_name}' in config. Defaulting to INFO.")
            numeric_level = logging.INFO

        # Apply the configuration. Using force=True (Python 3.8+) to override any existing handlers.
        # For older Python, one might need to remove existing handlers from the root logger first.
        logging.basicConfig(level=numeric_level, format=log_format, force=True)

        # Re-assign self.logger to ensure it uses the new configuration
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging has been re-configured. Level: {level_name}, Format: '{log_format}'.")

    def run_interaction_cycle(self, input_mode: str = 'text', initial_text: Optional[str] = None) -> None:
        self.current_interaction = threading.current_thread()
        self.interrupt_event.clear()
        start_time = time.time()
        
        try:
            self.logger.info(f"--- Starting new interaction cycle (mode: {input_mode}) ---")
            user_input_text = None
            timings = {
                'stt': 0,
                'rag': 0,
                'llm': 0,
                'tts': 0,
                'total': 0
            }

            # 1. Get User Input
            stt_start = time.time()
            if input_mode == 'speech':
                self.logger.info("Input mode: Speech. Activating STT module.")
                try:
                    user_input_text = self.stt_module.start_listening()
                    if user_input_text:
                        self.logger.info(f"STT transcribed text: '{user_input_text}'")
                    else:
                        self.logger.warning("STT returned no text.")
                        return
                except Exception as e:
                    self.logger.error(f"Error during STT processing: {e}", exc_info=True)
                    return
            elif input_mode == 'text':
                if initial_text:
                    self.logger.info(f"Input mode: Text. Using provided text: '{initial_text}'")
                    user_input_text = initial_text
                else:
                    self.logger.error("Text input mode selected, but no initial_text provided.")
                    return
            else:
                self.logger.error(f"Invalid input_mode: '{input_mode}'. Must be 'text' or 'speech'.")
                return
            timings['stt'] = time.time() - stt_start

            if not user_input_text or not user_input_text.strip():
                self.logger.info("No valid user input received. Ending interaction cycle.")
                return

            # Check for interruption after STT
            if self.interrupt_event.is_set():
                self.logger.info("Interaction interrupted after STT.")
                return

            # 2. RAG (Optional)
            context_for_llm = None
            rag_start = time.time()
            use_rag = self.orchestrator_config.get('use_rag_by_default', False)
            if use_rag and not self.interrupt_event.is_set():
                self.logger.info("RAG is enabled. Retrieving context.")
                try:
                    max_context_len = self.orchestrator_config.get('max_context_length', 4000)
                    context_info = self.rag_module.build_context(user_input_text, max_context_length=max_context_len)
                    
                    if context_info and isinstance(context_info, tuple) and len(context_info) == 2:
                        context_for_llm, avg_score = context_info
                        if avg_score >= self.rag_module.similarity_threshold:
                            self.logger.info(f"RAG module provided relevant context with average score: {avg_score:.2%}")
                        else:
                            self.logger.info(f"Context rejected due to low average relevance score: {avg_score:.2%}")
                            context_for_llm = None
                    else:
                        self.logger.info("RAG module did not return significant context.")
                        context_for_llm = None
                except Exception as e:
                    self.logger.error(f"Error during RAG processing: {e}", exc_info=True)
                    context_for_llm = None
            timings['rag'] = time.time() - rag_start

            # Check for interruption after RAG
            if self.interrupt_event.is_set():
                self.logger.info("Interaction interrupted after RAG.")
                return

            # 3. LLM Processing with Streaming and Real-time TTS
            llm_start = time.time()
            llm_response_text = ""
            current_sentence = ""
            sentence_end_chars = {'.', '!', '?', '\n', ','}  # Ajout de la virgule
            
            self.logger.info("Sending request to LLM (streaming mode with real-time TTS).")
            try:
                print("\nThoth:", end=" ", flush=True)
                
                for chunk in self.llm_module.generate_response_stream(
                    query=user_input_text,
                    context=context_for_llm,
                    use_history=True
                ):
                    # Vérifier l'interruption à chaque chunk
                    if self.interrupt_event.is_set():
                        self.logger.info("Interaction interrupted during LLM response.")
                        self.tts_module.stop()  # Arrêter la synthèse vocale en cours
                        print("\n[Interaction interrompue]")
                        return

                    if chunk:
                        print(chunk, end="", flush=True)
                        llm_response_text += chunk
                        current_sentence += chunk

                        # Check if we have a complete sentence or clause
                        if any(current_sentence.rstrip().endswith(char) for char in sentence_end_chars):
                            # Speak the complete sentence or clause
                            if current_sentence.strip():
                                self.tts_module.speak(current_sentence.strip(), blocking=False)
                            current_sentence = ""

                # Speak any remaining text if not interrupted
                if current_sentence.strip() and not self.interrupt_event.is_set():
                    self.tts_module.speak(current_sentence.strip(), blocking=False)

                print()  # New line after response is complete
                
                # Wait for all speech to complete if not interrupted
                if not self.interrupt_event.is_set():
                    self.tts_module.wait_until_done()
                
            except Exception as e:
                self.logger.error(f"Error during LLM processing: {e}", exc_info=True)
                if not self.interrupt_event.is_set():
                    error_msg = "Je suis désolé, une erreur interne est survenue."
                    print(error_msg)
                    self.tts_module.speak(error_msg, blocking=True)

            timings['llm'] = time.time() - llm_start

            # Calculate total time and log timings only if not interrupted
            if not self.interrupt_event.is_set():
                timings['total'] = time.time() - start_time
                self.logger.info("=== Module Timing Summary ===")
                self.logger.info(f"STT Module:  {timings['stt']:.2f}s")
                self.logger.info(f"RAG Module:  {timings['rag']:.2f}s")
                self.logger.info(f"LLM Module:  {timings['llm']:.2f}s")
                self.logger.info(f"Total Time:  {timings['total']:.2f}s")
                self.logger.info("===========================")

        finally:
            self.current_interaction = None
            self.interrupt_event.clear()

    def clear_conversation_history(self) -> None:
        """Clears the conversation history in the LLM module."""
        try:
            self.llm_module.clear_conversation() # Assumes LLMModule has this method
            self.logger.info("Conversation history cleared.")
        except Exception as e:
            self.logger.error(f"Error clearing conversation history: {e}", exc_info=True)

    def get_conversation_history(self) -> list:
        """Retrieves the conversation history from the LLM module.
        Returns:
            list: A list of message dictionaries, or an empty list if error.
        """
        try:
            history = self.llm_module.get_conversation_history() # Assumes LLMModule has this method
            # self.logger.debug(f"Retrieved conversation history: {history}")
            return history
        except Exception as e:
            self.logger.error(f"Error retrieving conversation history: {e}", exc_info=True)
            return [] # Return empty list on error

    def startup(self):
        """Performs any startup routines for the orchestrator and its modules."""
        self.logger.info("Orchestrator starting up...")
        # Example: Check module connections or readiness if they have such methods
        if hasattr(self.llm_module, 'check_connection') and callable(self.llm_module.check_connection):
            if not self.llm_module.check_connection(): # type: ignore
                self.logger.warning("LLM module connection check failed on startup.")
            else:
                self.logger.info("LLM module connection check successful.")
        else:
            self.logger.info("LLM module does not have a 'check_connection' method or it's not callable.")

        # Add other module checks if they provide status/readiness methods
        # For STT, RAG, TTS, their __init__ usually handles setup.
        # We can add more detailed checks here if modules evolve to have them.
        self.logger.info("Orchestrator startup complete.")

    def shutdown(self):
        """Performs cleanup for the orchestrator and its modules."""
        self.logger.info("Orchestrator shutting down...")
        try:
            if hasattr(self.stt_module, 'cleanup') and callable(self.stt_module.cleanup):
                self.stt_module.cleanup()
                self.logger.info("STTModule cleaned up.")
        except Exception as e:
            self.logger.error(f"Error during STTModule cleanup: {e}", exc_info=True)

        try:
            if hasattr(self.tts_module, 'shutdown') and callable(self.tts_module.shutdown):
                self.tts_module.shutdown()
                self.logger.info("TTSModule shut down.")
        except Exception as e:
            self.logger.error(f"Error during TTSModule shutdown: {e}", exc_info=True)

        self.logger.info("Orchestrator shutdown complete.")

    def _handle_wake_word_detection(self):
        """Thread pour gérer la détection du mot de réveil en arrière-plan"""
        wake_word_config = self.config.get('wake_word', {})
        allow_interruption = wake_word_config.get('allow_interruption', True)
        interruption_timeout = wake_word_config.get('interruption_timeout', 0.2)

        def on_wake_word():
            if not self.is_listening:
                return

            if self.current_interaction and allow_interruption:
                self.logger.info("Wake word detected during interaction - interrupting...")
                # Arrêter immédiatement la synthèse vocale
                self.tts_module.stop()
                # Activer l'interruption
                self.interrupt_event.set()
                # Attendre un peu pour laisser le temps à l'interruption
                time.sleep(interruption_timeout)
                # Vider les files d'attente
                self.tts_module.clear_queues()
                # Démarrer une nouvelle interaction
                self.interaction_queue.put(True)
            elif not self.current_interaction:
                self.logger.info("Wake word detected - starting new interaction...")
                self.interaction_queue.put(True)

        if wake_word_config.get('background_detection', True):
            self.wake_word_detector.start(on_wake_word)

    def get_status(self) -> dict:
        """Gets the status of the orchestrator and its modules."""
        self.logger.debug("Fetching orchestrator and module status...")
        status = {
            'orchestrator': {
                'status': 'nominal',
                'config_path': str(self.config_path),
                'thoth_config': self.orchestrator_config
            },
            'stt_module': {'status': 'unknown', 'details': 'No status method available or check not implemented'},
            'rag_module': {'status': 'unknown', 'details': 'No status method available or check not implemented'},
            'llm_module': {'status': 'unknown', 'details': 'No status method available or check not implemented'},
            'tts_module': {'status': 'unknown', 'details': 'No status method available or check not implemented'}
        }

        try:
            if hasattr(self.stt_module, 'get_audio_devices') and callable(self.stt_module.get_audio_devices): # Example status check
                status['stt_module'] = {'status': 'operational', 'available_devices': len(self.stt_module.get_audio_devices())}
            else:
                status['stt_module'] = {'status': 'unknown', 'details': 'get_audio_devices method not found or not callable'}
        except Exception as e:
            self.logger.error(f"Error getting STT status: {e}", exc_info=True) # Corrected logger variable
            status['stt_module'] = {'status': 'error', 'detail': str(e)}

        try:
            if hasattr(self.rag_module, 'get_stats') and callable(self.rag_module.get_stats):
                rag_stats = self.rag_module.get_stats() # type: ignore
                status['rag_module'] = rag_stats # type: ignore
                status['rag_module']['status'] = 'operational' # type: ignore
            else:
                 status['rag_module'] = {'status': 'unknown', 'details': 'get_stats method not found or not callable'}
        except Exception as e:
            self.logger.error(f"Error getting RAG status: {e}", exc_info=True) # Corrected logger variable
            status['rag_module'] = {'status': 'error', 'detail': str(e)}

        try:
            llm_status_details = {}
            if hasattr(self.llm_module, 'get_stats') and callable(self.llm_module.get_stats):
                llm_stats = self.llm_module.get_stats()
                llm_status_details.update(llm_stats) # type: ignore
                # Check connection status from within llm_stats if possible
                if llm_stats.get('model_info', {}).get('connection_status', False): # type: ignore
                    llm_status_details['status'] = 'operational'
                else:
                    llm_status_details['status'] = 'connection_error_or_unknown'
            elif hasattr(self.llm_module, 'check_connection') and callable(self.llm_module.check_connection): # Fallback to simple check
                 llm_status_details['status'] = 'operational' if self.llm_module.check_connection() else 'connection_error' # type: ignore
            else:
                llm_status_details['status'] = 'unknown'
                llm_status_details['details'] = 'No get_stats or check_connection method found or callable'
            status['llm_module'] = llm_status_details
        except Exception as e:
            self.logger.error(f"Error getting LLM status: {e}", exc_info=True) # Corrected logger variable
            status['llm_module'] = {'status': 'error', 'detail': str(e)}

        try:
            if hasattr(self.tts_module, 'get_info') and callable(self.tts_module.get_info):
                tts_info = self.tts_module.get_info()
                status['tts_module'] = tts_info # type: ignore
                status['tts_module']['status'] = 'operational' # type: ignore
            else:
                status['tts_module'] = {'status': 'unknown', 'details': 'get_info method not found or not callable'}
        except Exception as e:
            self.logger.error(f"Error getting TTS status: {e}", exc_info=True) # Corrected logger variable
            status['tts_module'] = {'status': 'error', 'detail': str(e)}

        return status

    def start_continuous_mode(self):
        """Démarre le mode continu avec détection du mot de réveil"""
        if not self.wake_word_detector:
            self.logger.error("Wake word detection not available. Check configuration.")
            return False

        try:
            self.is_listening = True
            self._handle_wake_word_detection()

            # Thread pour traiter les interactions
            def interaction_loop():
                while self.is_listening:
                    try:
                        # Attend une détection du mot de réveil
                        self.interaction_queue.get(timeout=0.5)
                        
                        # Si une interaction est en cours, on l'interrompt
                        if self.current_interaction:
                            self.interrupt_event.set()
                            self.tts_module.stop()  # Arrêter immédiatement la synthèse vocale
                            time.sleep(0.2)  # Attendre que l'interruption soit traitée
                        
                        # Joue un son de confirmation
                        self.tts_module.speak("Je vous écoute.", blocking=True)
                        
                        # Lance un cycle d'interaction
                        self.run_interaction_cycle(input_mode='speech')
                        
                        # Attend que toute la synthèse vocale soit terminée
                        if not self.interrupt_event.is_set():
                            self.tts_module.wait_until_done()
                        
                        # Message clair indiquant que le système est prêt pour le prochain mot de réveil
                        print(f"\n>>> En attente du mot de réveil '{self.wake_word_detector.wake_word}' <<<")
                        
                    except queue.Empty:
                        # Timeout normal, continue d'attendre
                        continue
                    except Exception as e:
                        if self.is_listening:  # Ignore les erreurs si on arrête volontairement
                            self.logger.error(f"Error in interaction loop: {e}", exc_info=True)

            # Démarre le thread d'interaction
            interaction_thread = threading.Thread(target=interaction_loop, daemon=True)
            interaction_thread.start()

            print(f"\n>>> En attente du mot de réveil '{self.wake_word_detector.wake_word}' <<<")
            return True

        except Exception as e:
            self.logger.error(f"Error starting continuous mode: {e}", exc_info=True)
            return False

    def stop_continuous_mode(self):
        """Arrête le mode continu"""
        self.is_listening = False
        self.interrupt_event.set()  # Interrompt toute interaction en cours
        if self.wake_word_detector:
            self.wake_word_detector.stop()
        self.logger.info("Continuous mode stopped.")

if __name__ == '__main__':
    # Basic logging setup for this test script itself
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
    test_logger = logging.getLogger('OrchestratorMainTest')

    config_file_path = Path("config/settings.yaml")
    if not config_file_path.exists():
        test_logger.critical(f"CRITICAL: Main configuration file '{config_file_path}' not found. Orchestrator test cannot proceed.")
        # Consider creating a dummy settings.yaml for standalone testing if it's missing
    else:
        test_logger.info(f"Using configuration file: '{config_file_path}'")

    orchestrator_instance = None
    try:
        orchestrator_instance = Orchestrator(config_path=str(config_file_path))
        test_logger.info("Orchestrator initialized successfully for basic test run.")
    except Exception as e:
        test_logger.error(f"Failed to initialize Orchestrator for test: {e}", exc_info=True)

    if orchestrator_instance:
        orchestrator_instance.logger.info("Testing the Orchestrator's interaction cycle...")

        orchestrator_instance.logger.info("--- Test Cycle 1: Text Input (General Query) ---")
        orchestrator_instance.run_interaction_cycle(input_mode='text', initial_text="Bonjour Thoth, peux-tu m'expliquer ce qu'est un grand modèle de langage ?")

        # Test RAG if enabled in config and data is present
        # Assuming 'use_rag_by_default' might be true in 'thoth' section of settings.yaml
        orchestrator_instance.logger.info("--- Test Cycle 2: Text Input (Potentially RAG-enabled Query) ---")
        orchestrator_instance.run_interaction_cycle(input_mode='text', initial_text="Quelles sont les dernières découvertes en matière d'intelligence artificielle selon les documents que tu possèdes ?")

        # Test with speech input
        print("\n>>> Préparation du test d'entrée vocale <<<")
        print("Vérification des périphériques audio...")
        
        # Check available audio devices
        try:
            audio_devices = orchestrator_instance.stt_module.get_audio_devices()
            if audio_devices:
                print("\nPériphériques audio disponibles:")
                for i, device in enumerate(audio_devices):
                    print(f"{i}: {device}")
                print("\n>>> Le microphone est prêt. Vous pouvez parler après le bip. <<<")
                time.sleep(1)
                print("\n*bip*")
                
                orchestrator_instance.logger.info("--- Test Cycle 3: Speech Input Test ---")
                orchestrator_instance.run_interaction_cycle(input_mode='speech')
                print("\nTest d'entrée vocale terminé.")
            else:
                print("\n⚠️ Aucun périphérique audio détecté!")
        except Exception as e:
            print(f"\n❌ Erreur lors de l'initialisation audio: {e}")

        orchestrator_instance.logger.info("Orchestrator interaction cycle tests finished.")

        # Consider adding cleanup for modules if they have explicit shutdown methods
        # e.g., orchestrator_instance.stt_module.cleanup()
        # orchestrator_instance.tts_module.shutdown()

        orchestrator_instance.logger.info("--- Test Cycle 4: Testing Conversation History --- ")

        # Run a short interaction
        orchestrator_instance.run_interaction_cycle(input_mode='text', initial_text="Quelle est la capitale de la France ?")
        orchestrator_instance.run_interaction_cycle(input_mode='text', initial_text="Et quelle est sa population approximative ?")

        # Get and log history
        history = orchestrator_instance.get_conversation_history()
        orchestrator_instance.logger.info(f"Current conversation history ({len(history)} messages):")
        for i, msg in enumerate(history):
            orchestrator_instance.logger.info(f"  {i+1}. Role: {msg.get('role')}, Content: '{msg.get('content')[:80]}...'Timestamp: {msg.get('timestamp')}")

        # Clear history
        orchestrator_instance.logger.info("Clearing conversation history.")
        orchestrator_instance.clear_conversation_history()

        # Verify history is cleared
        history_after_clear = orchestrator_instance.get_conversation_history()
        orchestrator_instance.logger.info(f"Conversation history after clearing ({len(history_after_clear)} messages). Should be 0 or 1 (if system prompt persists).")
        if not history_after_clear or (len(history_after_clear) == 1 and history_after_clear[0].get('role') == 'system'):
            orchestrator_instance.logger.info("History successfully cleared (or only system prompt remains).")
        else:
            orchestrator_instance.logger.warning(f"History not fully cleared. Content: {history_after_clear}")

        orchestrator_instance.logger.info("Orchestrator history management tests finished.")

        orchestrator_instance.logger.info("--- Test Cycle 5: Testing Helper Methods --- ")

        # Test startup (mostly logs messages)
        orchestrator_instance.startup()

        # Test get_status
        current_status = orchestrator_instance.get_status()
        orchestrator_instance.logger.info(f"Orchestrator Status: {current_status}")
        # Basic check of status structure
        if current_status and current_status.get('orchestrator'):
            orchestrator_instance.logger.info("get_status method seems to be working and returning expected structure.")
        else:
            orchestrator_instance.logger.warning("get_status did not return expected structure.")

        # Test shutdown (mostly logs messages and calls module cleanup)
        orchestrator_instance.shutdown()
        orchestrator_instance.logger.info("Orchestrator helper methods tests finished.")
    else:
        test_logger.error("Orchestrator instance was not created. Cannot run helper method tests.")
