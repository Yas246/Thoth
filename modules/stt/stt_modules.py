# modules/stt/stt_module.py

import os
import time
import threading
import wave
import queue
import logging
from typing import Optional, Callable # Ensure Optional and Callable are imported
from dataclasses import dataclass, fields # Ensure 'fields' is imported
from pathlib import Path

import pyaudio
import numpy as np
import whisper
from faster_whisper import WhisperModel
import torch
import yaml # For loading config in __main__

# Configuration du logging
# logging.basicConfig(level=logging.INFO) # Removed global basicConfig
logger = logging.getLogger(__name__)

@dataclass
class STTConfig:
    """Configuration pour le module STT"""
    model_size: str = "base"  # tiny, base, small, medium, large
    language: str = "fr"
    device: str = "auto"  # auto, cpu, cuda
    use_faster_whisper: bool = True
    
    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    audio_format = pyaudio.paInt16
    
    # Voice Activity Detection - Seuils réduits pour plus de sensibilité
    energy_threshold: int = 50  # Réduit de 300 à 50
    silence_duration: float = 2.0  # secondes de silence avant d'arrêter
    min_recording_duration: float = 0.5  # durée minimale d'enregistrement


class AudioRecorder:
    """Gestionnaire d'enregistrement audio avec détection de voix"""
    
    def __init__(self, config: STTConfig):
        self.config = config
        self.audio = pyaudio.PyAudio()
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.frames = []
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def get_audio_devices(self):
        """Liste les périphériques audio disponibles"""
        devices = []
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': info['name'],
                    'channels': info['maxInputChannels']
                })
        return devices
    
    def start_recording(self, device_index: Optional[int] = None):
        """Démarre l'enregistrement audio"""
        try:
            self.stream = self.audio.open(
                format=self.config.audio_format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.config.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.is_recording = True
            self.frames = []
            self.stream.start_stream()
            logger.info("Enregistrement démarré")
            
        except Exception as e:
            logger.error(f"Erreur lors du démarrage de l'enregistrement: {e}")
            raise
    
    def stop_recording(self) -> bytes:
        """Arrête l'enregistrement et retourne les données audio"""
        if not self.is_recording:
            return b''
        
        self.is_recording = False
        self.stream.stop_stream()
        self.stream.close()
        
        # Convertir les frames en bytes
        audio_data = b''.join(self.frames)
        logger.info(f"Enregistrement arrêté. Taille: {len(audio_data)} bytes")
        
        return audio_data
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback pour l'enregistrement audio"""
        if self.is_recording:
            self.frames.append(in_data)
        return (in_data, pyaudio.paContinue)
    
    def get_audio_level(self, audio_data: bytes) -> float:
        """Calcule le niveau audio pour la détection de voix"""
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        return np.sqrt(np.mean(audio_np**2))
    
    def record_with_vad(self, device_index: Optional[int] = None, 
                       callback: Optional[Callable[[str], None]] = None) -> bytes:
        """Enregistre avec détection automatique de fin de parole"""
        self.start_recording(device_index)
        
        silence_start = None
        recording_start = time.time()
        
        try:
            while self.is_recording:
                time.sleep(0.1)
                
                # Analyser les dernières données audio
                if len(self.frames) > 10:  # Assez de données pour analyser
                    recent_audio = b''.join(self.frames[-5:])  # Dernières 5 frames
                    audio_level = self.get_audio_level(recent_audio)
                    
                    if callback:
                        callback(f"Niveau audio: {audio_level:.0f}")
                    
                    # Détection de silence
                    if audio_level < self.config.energy_threshold:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > self.config.silence_duration:
                            # Assez de silence et enregistrement minimum atteint
                            if time.time() - recording_start > self.config.min_recording_duration:
                                logger.info("Fin de parole détectée")
                                break
                    else:
                        silence_start = None  # Reset du compteur de silence
            
            return self.stop_recording()
            
        except KeyboardInterrupt:
            logger.info("Enregistrement interrompu par l'utilisateur")
            return self.stop_recording()
    
    def save_audio(self, audio_data: bytes, filename: str):
        """Sauvegarde les données audio dans un fichier WAV"""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.config.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.config.audio_format))
            wf.setframerate(self.config.sample_rate)
            wf.writeframes(audio_data)
    
    def cleanup(self):
        """Nettoie les ressources audio"""
        try:
            if hasattr(self, 'stream'):
                if hasattr(self.stream, 'is_active') and self.stream.is_active():
                    self.stream.stop_stream()
                if hasattr(self.stream, 'is_stopped') and not self.stream.is_stopped():
                    self.stream.close()
            if hasattr(self, 'audio'):
                self.audio.terminate()
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage: {e}")


class WhisperSTT:
    """Classe principale pour la transcription avec Whisper"""
    
    def __init__(self, config: STTConfig):
        self.config = config
        self.recorder = AudioRecorder(config)
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle Whisper"""
        try:
            # Déterminer le device
            if self.config.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.config.device
            
            logger.info(f"Chargement du modèle Whisper {self.config.model_size} sur {device}")
            
            if self.config.use_faster_whisper:
                # Utiliser faster-whisper (plus rapide)
                compute_type = "float16" if device == "cuda" else "int8"
                self.model = WhisperModel(
                    self.config.model_size,
                    device=device,
                    compute_type=compute_type
                )
                logger.info("Modèle faster-whisper chargé")
            else:
                # Utiliser whisper standard
                self.model = whisper.load_model(
                    self.config.model_size,
                    device=device
                )
                logger.info("Modèle whisper standard chargé")
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            raise
    
    def transcribe_audio_data(self, audio_data: bytes) -> str:
        """Transcrit des données audio en texte"""
        try:
            # Sauvegarder temporairement l'audio
            temp_file = "temp_audio.wav"
            self.recorder.save_audio(audio_data, temp_file)
            
            # Transcrire
            result = self.transcribe_file(temp_file)
            
            # Nettoyer le fichier temporaire
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de la transcription des données audio: {e}")
            return ""
    
    def transcribe_file(self, audio_path: str) -> str:
        """Transcrit un fichier audio en texte"""
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Fichier audio non trouvé: {audio_path}")
            
            logger.info(f"Transcription de {audio_path}")
            
            if self.config.use_faster_whisper:
                # faster-whisper
                segments, info = self.model.transcribe(
                    audio_path,
                    language=self.config.language,
                    beam_size=5
                )
                
                # Combiner tous les segments
                transcription = " ".join([segment.text for segment in segments])
                
                logger.info(f"Langue détectée: {info.language} (probabilité: {info.language_probability:.2f})")
                
            else:
                # whisper standard
                result = self.model.transcribe(
                    audio_path,
                    language=self.config.language
                )
                transcription = result["text"]
            
            # Nettoyer la transcription
            transcription = transcription.strip()
            logger.info(f"Transcription: {transcription}")
            
            return transcription
            
        except Exception as e:
            logger.error(f"Erreur lors de la transcription: {e}")
            return ""
    
    def start_recording(self, device_index: Optional[int] = None):
        """Démarre l'enregistrement"""
        self.recorder.start_recording(device_index)
    
    def stop_recording(self) -> str:
        """Arrête l'enregistrement et retourne la transcription"""
        audio_data = self.recorder.stop_recording()
        if audio_data:
            return self.transcribe_audio_data(audio_data)
        return ""
    
    def record_and_transcribe(self, device_index: Optional[int] = None,
                            callback: Optional[Callable[[str], None]] = None) -> str:
        """Enregistre avec VAD et transcrit automatiquement"""
        logger.info("Démarrage de l'enregistrement avec détection automatique...")
        
        audio_data = self.recorder.record_with_vad(device_index, callback)
        
        if audio_data and len(audio_data) > 0:
            logger.info("Transcription en cours...")
            return self.transcribe_audio_data(audio_data)
        else:
            logger.warning("Aucune données audio enregistrées")
            return ""
    
    def get_available_devices(self):
        """Retourne la liste des périphériques audio disponibles"""
        return self.recorder.get_audio_devices()
    
    def test_microphone(self, device_index: Optional[int] = None, duration: float = 3.0):
        """Test le microphone en enregistrant pendant quelques secondes"""
        logger.info(f"Test du microphone pendant {duration} secondes...")
        
        self.recorder.start_recording(device_index)
        time.sleep(duration)
        audio_data = self.recorder.stop_recording()
        
        if audio_data:
            level = self.recorder.get_audio_level(audio_data)
            logger.info(f"Test terminé. Niveau audio moyen: {level:.0f}")
            return level > 10  # Seuil minimal réduit pour considérer que ça fonctionne
        return False
    
    def cleanup(self):
        """Nettoie les ressources"""
        self.recorder.cleanup()


# Classe principale du module STT
class STTModule:
    """Module STT principal - Interface simplifiée pour l'orchestrateur"""
    
    def __init__(self, config: Optional[dict] = None):
        current_module_config = config if config is not None else {}
        try:
            stt_config_field_names = {f.name for f in fields(STTConfig)}
            filtered_stt_params = {
                k: v for k, v in current_module_config.items() if k in stt_config_field_names
            }
            self.config = STTConfig(**filtered_stt_params)
            logger.debug(f"STTConfig initialized with provided params: {filtered_stt_params}")
        except TypeError as e:
            logger.error(
                f"TypeError initializing STTConfig with params {current_module_config}. Error: {e}",
                exc_info=True
            )
            logger.info("Falling back to default STTConfig due to TypeError.")
            self.config = STTConfig()
        except Exception as e: # Catch any other unexpected error during STTConfig init
            logger.error(
                f"Unexpected error initializing STTConfig with params {current_module_config}. Error: {e}",
                exc_info=True
            )
            logger.info("Falling back to default STTConfig due to unexpected error.")
            self.config = STTConfig()


        self.whisper_stt = WhisperSTT(self.config)
        self.is_listening = False
    
    def start_listening(self, device_index: Optional[int] = None) -> str:
        """Démarre l'écoute et retourne la transcription"""
        if self.is_listening:
            logger.warning("L'écoute est déjà en cours")
            return ""
        
        self.is_listening = True
        try:
            # Afficher le périphérique utilisé
            if device_index is not None:
                devices = self.get_audio_devices()
                device_name = next((d['name'] for d in devices if d['index'] == device_index), f"Device {device_index}")
                logger.info(f"Utilisation du périphérique: {device_name}")
            else:
                logger.info("Utilisation du périphérique par défaut")
            
            transcription = self.whisper_stt.record_and_transcribe(
                device_index=device_index,
                callback=lambda msg: logger.debug(msg)
            )
            return transcription
        finally:
            self.is_listening = False
    
    def transcribe_file(self, audio_path: str) -> str:
        """Transcrit un fichier audio"""
        return self.whisper_stt.transcribe_file(audio_path)
    
    def get_audio_devices(self):
        """Retourne les périphériques audio disponibles"""
        return self.whisper_stt.get_available_devices()
    
    def test_setup(self, device_index_to_test: Optional[int] = None) -> bool: # Added param for clarity
        """Test la configuration STT, y compris le microphone."""
        try:
            logger.info("Test du modèle Whisper et du microphone...")
            # Pass the device_index to test_microphone
            test_result = self.whisper_stt.test_microphone(device_index=device_index_to_test, duration=2.0)
            
            if test_result:
                logger.info("✅ Configuration STT et test microphone OK")
                return True
            else:
                logger.warning("⚠️ Problème détecté avec le microphone ou l'enregistrement.")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur lors du test STT: {e}", exc_info=True)
            return False
    
    def cleanup(self):
        """Nettoie les ressources"""
        logger.info("Nettoyage des ressources WhisperSTT...")
        self.whisper_stt.cleanup()
        logger.info("Ressources WhisperSTT nettoyées.")


# Script de test/démonstration
if __name__ == "__main__":
    import argparse
    
    # Setup basic logging FOR THIS TEST SCRIPT
    # Using force=True (Python 3.8+) to reconfigure logging for the test script
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
    logger.info("Executing STTModule directly for testing...")

    parser = argparse.ArgumentParser(description="Test du module STT")
    parser.add_argument("--test", action="store_true", help="Lance les tests de base du STTModule.")
    parser.add_argument("--devices", action="store_true", help="Liste les périphériques audio disponibles.")
    parser.add_argument("--record", action="store_true", help="Enregistre l'audio du microphone et le transcrit.")
    parser.add_argument("--device", type=int, default=None, help="Index du périphérique audio à utiliser pour l'enregistrement/test.")
    parser.add_argument("--file", type=str, default=None, help="Chemin vers un fichier audio à transcrire.")
    parser.add_argument("--interactive", action="store_true", help="Mode interactif pour choisir le périphérique avant l'enregistrement.")
    
    args = parser.parse_args()
    
    default_stt_test_config = {
        "model_size": "base", "language": "fr", "device": "cpu",
        "use_faster_whisper": True, "sample_rate": 16000, "channels": 1,
        "chunk_size": 1024, "energy_threshold": 70, # Slightly higher threshold for tests
        "silence_duration": 2.0, "min_recording_duration": 0.5
    }
    
    stt_params_for_test = default_stt_test_config.copy()

    try:
        # Path to config/settings.yaml relative to this file (modules/stt/stt_modules.py)
        # Assuming this script is in /app/modules/stt/
        config_yaml_path = Path(__file__).resolve().parent.parent.parent / "config/settings.yaml"

        if config_yaml_path.exists():
            logger.info(f"Attempting to load STT configuration for test from: {config_yaml_path}")
            with open(config_yaml_path, 'r', encoding='utf-8') as f:
                full_config_from_file = yaml.safe_load(f)
                if full_config_from_file and isinstance(full_config_from_file.get('stt'), dict):
                    stt_params_for_test.update(full_config_from_file['stt'])
                    logger.info(f"Successfully loaded and applied STT params from settings.yaml for testing.")
                else:
                    logger.warning(f"'stt' section not found or invalid in {config_yaml_path}. Using default test config.")
        else:
            logger.info(f"Global config file {config_yaml_path} not found. Using default STT test config.")
    except Exception as e:
        logger.warning(f"Could not load STT config from settings.yaml for testing, using defaults. Error: {e}", exc_info=True)

    stt_module_instance = None
    try:
        logger.debug(f"Initializing STTModule with effective config for test: {stt_params_for_test}")
        # Ensure only valid STTConfig fields are passed from stt_params_for_test
        valid_stt_config_keys = {f.name for f in fields(STTConfig)}
        init_config_dict = {k: v for k, v in stt_params_for_test.items() if k in valid_stt_config_keys}
        stt_module_instance = STTModule(config=init_config_dict)

        if args.devices:
            print("\n=== Périphériques audio disponibles ===")
            devices = stt_module_instance.get_audio_devices()
            if devices:
                for device in devices:
                    print(f"  Index {device['index']}: {device['name']} ({device['channels']} canaux)")
            else:
                print("Aucun périphérique d'entrée audio trouvé.")
        
        elif args.test:
            print("\n=== Test de configuration et microphone du module STT ===")
            device_index_to_test = args.device
            if device_index_to_test is not None:
                print(f"Test avec le périphérique spécifié (Index: {device_index_to_test})")
            else:
                print("Test avec le périphérique par défaut (si aucun n'est spécifié, le système choisira).")
            # Pass the selected device index to test_setup
            success = stt_module_instance.test_setup(device_index_to_test=device_index_to_test)
            print(f"Résultat du test de microphone: {'✅ OK' if success else '❌ Échec (Vérifiez le microphone, les permissions et les logs)'}")
        
        elif args.file:
            if args.file and Path(args.file).exists():
                print(f"\n=== Transcription du fichier: {args.file} ===")
                transcription = stt_module_instance.transcribe_file(args.file)
                print(f"  Transcription: {transcription}")
            else:
                print(f"Erreur: Fichier non trouvé à '{args.file}'")
        
        elif args.record or args.interactive:
            selected_device_index = args.device
            if args.interactive:
                print("\n=== Sélection interactive du périphérique audio ===")
                devices = stt_module_instance.get_audio_devices()
                if not devices:
                    print("Aucun périphérique d'entrée audio trouvé. Impossible de continuer.")
                    # Consider exiting if no devices found: import sys; sys.exit(1)
                else:
                    for device in devices:
                        print(f"  {device['index']}: {device['name']}")
                    try:
                        device_input = input("\nEntrez l'index du périphérique (ou Entrée pour défaut/précédemment spécifié): ").strip()
                        if device_input:
                            selected_device_index = int(device_input)
                    except (ValueError, KeyboardInterrupt):
                        print("Sélection invalide ou annulée.")
                        if selected_device_index is None:
                             print("Aucun périphérique sélectionné. Utilisation du périphérique par défaut du système.")
                        else:
                            print(f"Utilisation du périphérique précédemment spécifié/sélectionné: Index {selected_device_index}")

            print("\n=== Enregistrement et transcription depuis le microphone ===")
            if selected_device_index is not None:
                available_devices = stt_module_instance.get_audio_devices()
                if not any(d['index'] == selected_device_index for d in available_devices):
                    print(f"❌ Périphérique avec index {selected_device_index} non trouvé. Utilisation du périphérique par défaut.")
                    selected_device_index = None # Fallback to default
                else:
                     print(f"Utilisation du périphérique avec index: {selected_device_index}")
            else:
                print("Utilisation du périphérique par défaut du système.")
            
            print("Parlez maintenant... (L'enregistrement s'arrêtera automatiquement après une période de silence)")
            transcription = stt_module_instance.start_listening(selected_device_index)
            print(f"\n  Transcription: '{transcription}'")
        else:
            parser.print_help()

    except KeyboardInterrupt:
        logger.info("\n\nInterruption par l'utilisateur.")
    except Exception as e:
        logger.error(f"\n❌ Une erreur est survenue lors de l'exécution du test STT: {e}", exc_info=True)
    finally:
        if stt_module_instance and hasattr(stt_module_instance, 'cleanup'):
            logger.info("Nettoyage des ressources STTModule...")
            stt_module_instance.cleanup()
        # Ensure temp_audio_files directory is cleaned up if it was created by this script
        temp_audio_dir = Path("temp_audio_files")
        if temp_audio_dir.exists():
            try:
                for f in temp_audio_dir.iterdir():
                    if f.name.startswith("temp_audio_") and f.suffix == ".wav":
                        f.unlink()
                # temp_audio_dir.rmdir() # Only if sure it's empty and safe
                logger.info(f"Nettoyage des fichiers audio temporaires dans {temp_audio_dir} terminé.")
            except Exception as e_clean:
                logger.error(f"Erreur lors du nettoyage du répertoire audio temporaire {temp_audio_dir}: {e_clean}")
        logger.info("Script de test STTModule terminé.")