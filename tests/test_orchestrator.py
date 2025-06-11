# tests/test_orchestrator.py
import unittest
from unittest.mock import patch, MagicMock, mock_open
import yaml # For mocking open for config file
import logging
from pathlib import Path # Ensure Path is imported

# Add project root to sys.path or ensure PYTHONPATH is set for imports to work
# For simplicity in this subtask, we assume direct import works if tests are run from root
# or using a test runner that handles paths (e.g., python -m unittest discover)
from modules.orchestrator.orchestrator import Orchestrator
# If LLMResponse is used for type hinting or checking, it might need to be imported or mocked too
# from modules.llm.llm_module import LLMResponse

# Disable logging for tests unless specifically testing logging output
# logging.disable(logging.CRITICAL)

class TestOrchestratorInitialization(unittest.TestCase):

    @patch('modules.orchestrator.orchestrator.STTModule')
    @patch('modules.orchestrator.orchestrator.RAGModule')
    @patch('modules.orchestrator.orchestrator.LLMModule')
    @patch('modules.orchestrator.orchestrator.TTSModule')
    @patch('builtins.open') # Mock open for config file
    def test_orchestrator_initialization_success(self, mock_builtin_open, MockTTS, MockLLM, MockRAG, MockSTT):
        """Test successful initialization of the Orchestrator with mocked modules and config."""
        # Prepare a dummy YAML config string
        dummy_config_data = {
            'stt': {'model_size': 'tiny', 'language': 'en'}, # Added language for STTConfig default
            'rag': {'embedding_model': 'test_model', 'retrieval_k': 3}, # Added k for RAGConfig default
            'llm': {'api_url': 'http://localhost:1234', 'model_name': 'test-llm', 'temperature': 0.6}, # Added defaults for LLMConfig
            'tts': {'cache_dir': 'test_cache', 'audio_engine': 'pyttsx3', 'lang': 'en'}, # Added defaults for TTS
            'thoth': {'use_rag_by_default': True, 'max_context_length': 3000}, # Added defaults for Orchestrator config part
            'logging': {'level': 'DEBUG', 'format': '%(asctime)s - %(levelname)s - %(message)s'} # Added format for logging config
        }
        yaml_string = yaml.dump(dummy_config_data)

        # Configure the mock for open()
        # When open(path, 'r', encoding='utf-8') is called, return a mock file object
        # that has a read() method returning yaml_string.
        # Also needs to work as a context manager.
        mock_file_instance = MagicMock()
        mock_file_instance.read.return_value = yaml_string
        mock_file_instance.__enter__.return_value = mock_file_instance # for 'with open(...) as f:'
        mock_file_instance.__exit__.return_value = None
        mock_builtin_open.return_value = mock_file_instance


        # Create mock instances for each module to be returned by their constructors
        mock_stt_instance = MagicMock()
        MockSTT.return_value = mock_stt_instance

        mock_rag_instance = MagicMock()
        MockRAG.return_value = mock_rag_instance

        mock_llm_instance = MagicMock()
        MockLLM.return_value = mock_llm_instance

        mock_tts_instance = MagicMock()
        MockTTS.return_value = mock_tts_instance

        try:
            # Path for config doesn't matter as open is mocked, but provide a dummy one
            orchestrator = Orchestrator(config_path='dummy_config.yaml')
        except Exception as e:
            self.fail(f"Orchestrator initialization failed unexpectedly: {e}")

        # Assertions
        self.assertIsNotNone(orchestrator.stt_module, "STT module should be initialized")
        self.assertIsNotNone(orchestrator.rag_module, "RAG module should be initialized")
        self.assertIsNotNone(orchestrator.llm_module, "LLM module should be initialized")
        self.assertIsNotNone(orchestrator.tts_module, "TTS module should be initialized")

        # Check that module constructors were called with their specific config part
        MockSTT.assert_called_once_with(config=dummy_config_data['stt'])
        MockRAG.assert_called_once_with(config=dummy_config_data['rag'])
        MockLLM.assert_called_once_with(config=dummy_config_data['llm'])

        MockTTS.assert_called_once_with(
            cache_dir=dummy_config_data['tts']['cache_dir'],
            audio_engine=dummy_config_data['tts']['audio_engine'],
            default_lang=dummy_config_data['tts']['lang']
        )

        self.assertEqual(orchestrator.orchestrator_config, dummy_config_data['thoth'])
        # Orchestrator converts config_path to Path object before calling open
        mock_builtin_open.assert_called_once_with(Path('dummy_config.yaml'), 'r', encoding='utf-8')

    @patch('modules.orchestrator.orchestrator.STTModule')
    @patch('modules.orchestrator.orchestrator.RAGModule')
    @patch('modules.orchestrator.orchestrator.LLMModule')
    @patch('modules.orchestrator.orchestrator.TTSModule')
    @patch('builtins.open')
    def test_orchestrator_initialization_config_not_found(self, mock_builtin_open, MockTTS, MockLLM, MockRAG, MockSTT):
        """Test Orchestrator initialization fails if config file is not found."""
        # Simulate FileNotFoundError when open is called
        mock_builtin_open.side_effect = FileNotFoundError("Config file not found at dummy_path.yaml")

        with self.assertRaises(FileNotFoundError):
            Orchestrator(config_path='dummy_path.yaml')

    @patch('modules.orchestrator.orchestrator.STTModule')
    @patch('modules.orchestrator.orchestrator.RAGModule')
    @patch('modules.orchestrator.orchestrator.LLMModule')
    @patch('modules.orchestrator.orchestrator.TTSModule')
    @patch('builtins.open')
    def test_orchestrator_initialization_module_init_failure(self, mock_builtin_open, MockTTS, MockLLM, MockRAG, MockSTT):
        """Test Orchestrator handles error if a module fails to initialize."""
        # Prepare dummy config data for the file mock
        dummy_config_data = {'stt': {}, 'rag': {}, 'llm': {}, 'tts': {}, 'logging': {'level': 'INFO'}, 'thoth': {}}
        yaml_string = yaml.dump(dummy_config_data)

        mock_file_instance = MagicMock()
        mock_file_instance.read.return_value = yaml_string
        mock_file_instance.__enter__.return_value = mock_file_instance
        mock_file_instance.__exit__.return_value = None
        mock_builtin_open.return_value = mock_file_instance

        # Simulate one of the module initializations failing
        MockSTT.side_effect = Exception("STT Module Failed to Load")

        with self.assertRaises(Exception) as context:
            Orchestrator(config_path='dummy_config.yaml')
        self.assertIn("STT Module Failed to Load", str(context.exception))

if __name__ == '__main__':
    # This allows running the tests directly from this file
    unittest.main()
