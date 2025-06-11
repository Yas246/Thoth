"""
Module TTS (Text-to-Speech) pour l'Assistant AI Thoth
Utilise gTTS pour une synthèse vocale française de haute qualité
"""

import os
import io
import tempfile
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import threading
import time
import hashlib
from gtts import gTTS
import pygame
import sounddevice as sd
import soundfile as sf
import numpy as np
from queue import Queue
from threading import Thread, Event, Lock

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSModule:
    """
    Module TTS pour l'assistant Thoth utilisant gTTS
    
    Fonctionnalités:
    - Synthèse vocale française de haute qualité avec Google TTS
    - Support multilingue
    - Lecture audio directe ou sauvegarde fichier
    - Gestion du cache audio
    - Plusieurs moteurs audio (pygame, sounddevice)
    """
    
    def __init__(self, 
                 cache_dir: str = "data/audio_cache",
                 audio_engine: str = "auto",
                 default_lang: str = "fr"):
        """
        Initialise le module TTS
        
        Args:
            cache_dir: Répertoire de cache audio
            audio_engine: Moteur audio ('pygame', 'sounddevice', 'auto')
            default_lang: Langue par défaut (fr, en, etc.)
        """
        self.logger = logging.getLogger(__name__)
        self.cache_dir = Path(cache_dir)
        self.audio_engine = audio_engine
        self.default_lang = default_lang
        
        # Files d'attente pour le traitement parallèle
        self.text_queue = Queue()  # File pour le texte à synthétiser
        self.audio_queue = Queue()  # File pour l'audio synthétisé
        self.stop_event = Event()
        self.is_speaking = Event()
        self.force_stop = Event()  # Nouvel event pour forcer l'arrêt
        
        # Verrou pour la synthèse
        self.synthesis_lock = Lock()
        
        # Threads pour le traitement parallèle
        self.synthesis_thread = Thread(target=self._synthesis_worker, daemon=True)
        self.playback_thread = Thread(target=self._playback_worker, daemon=True)
        
        # Créer le répertoire de cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialiser pygame pour l'audio si nécessaire
        if self.audio_engine in ['pygame', 'auto']:
            try:
                pygame.mixer.init()
                self.audio_engine = 'pygame'
            except Exception as e:
                self.logger.warning(f"Impossible d'initialiser pygame: {e}")
                self.audio_engine = 'sounddevice'
        
        # Démarrer les threads
        self.synthesis_thread.start()
        self.playback_thread.start()
        
        self.logger.info(f"Module TTS initialisé (moteur: {self.audio_engine})")
    
    def _synthesis_worker(self):
        """Thread de travail pour la synthèse audio"""
        while not self.stop_event.is_set():
            try:
                # Récupérer le prochain texte à synthétiser
                text, lang = self.text_queue.get(timeout=0.1)
                
                # Synthétiser l'audio
                with self.synthesis_lock:
                    audio_data = self.synthesize(text, lang)
                    if audio_data:
                        self.audio_queue.put((audio_data, text))
                
                self.text_queue.task_done()
            except Exception:
                continue

    def _playback_worker(self):
        """Thread de travail pour la lecture audio"""
        while not self.stop_event.is_set():
            try:
                # Récupérer le prochain audio à jouer
                audio_data, text = self.audio_queue.get(timeout=0.1)
                
                if audio_data and not self.force_stop.is_set():
                    self.is_speaking.set()
                    self._play_audio_data(audio_data, blocking=True)
                    self.is_speaking.clear()
                
                self.audio_queue.task_done()
            except Exception:
                continue

    def speak(self, text: str, lang: str = None, blocking: bool = True) -> bool:
        """
        Ajoute le texte à la file de synthèse
        
        Args:
            text: Texte à prononcer
            lang: Code de langue
            blocking: Attendre la fin de la lecture
            
        Returns:
            bool: True si l'ajout à la file a réussi
        """
        try:
            # Ajouter le texte à la file de synthèse
            self.text_queue.put((text, lang))
            
            if blocking:
                # Attendre que ce texte soit traité
                self.text_queue.join()
                self.audio_queue.join()
            return True
        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajout à la file: {e}")
            return False

    def wait_until_done(self):
        """Attend que toutes les phrases en cours soient prononcées"""
        self.text_queue.join()
        self.audio_queue.join()
        while self.is_speaking.is_set():
            time.sleep(0.01)

    def synthesize(self, text: str, lang: str = None, use_cache: bool = True) -> Optional[bytes]:
        """
        Synthétise du texte en audio
        
        Args:
            text: Texte à synthétiser
            lang: Code de langue (fr, en, etc.)
            use_cache: Utiliser le cache si disponible
            
        Returns:
            bytes: Données audio au format WAV ou None si échec
        """
        if not text.strip():
            return None
        
        lang = lang or self.default_lang
        
        # Vérifier le cache
        cache_key = self._get_cache_key(text, lang)
        cache_file = self.cache_dir / f"{cache_key}.wav"
        
        if use_cache and cache_file.exists():
            self.logger.debug(f"Utilisation du cache pour: {text[:50]}...")
            return cache_file.read_bytes()
        
        try:
            # Créer l'objet gTTS
            tts = gTTS(text=text, lang=lang, slow=False)
            
            # Utiliser des fichiers temporaires avec des noms uniques
            mp3_path = os.path.join(tempfile.gettempdir(), f"thoth_tts_{os.getpid()}_{time.time_ns()}.mp3")
            wav_path = os.path.join(tempfile.gettempdir(), f"thoth_tts_{os.getpid()}_{time.time_ns()}.wav")
            
            try:
                # Sauvegarder en MP3
                tts.save(mp3_path)
                
                # Convertir en WAV
                data, samplerate = sf.read(mp3_path)
                sf.write(wav_path, data, samplerate)
                
                # Lire les données WAV
                audio_data = Path(wav_path).read_bytes()
                
                # Sauvegarder dans le cache si nécessaire
                if use_cache:
                    cache_file.write_bytes(audio_data)
                    self.logger.debug(f"Audio mis en cache: {cache_file}")
                
                return audio_data
                
            finally:
                # Nettoyer les fichiers temporaires
                for temp_file in [mp3_path, wav_path]:
                    try:
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)
                    except Exception as e:
                        self.logger.warning(f"Impossible de supprimer le fichier temporaire {temp_file}: {e}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la synthèse: {e}")
            return None
    
    def _play_audio_data(self, audio_data: bytes, blocking: bool = True) -> bool:
        """Joue les données audio"""
        temp_file = None
        try:
            # Créer un fichier temporaire qui sera supprimé à la fermeture
            temp_file = tempfile.NamedTemporaryFile(
                prefix="thoth_tts_",
                suffix=".wav",
                delete=False
            )
            temp_path = temp_file.name
            temp_file.close()  # Fermer le fichier pour permettre sa réutilisation
            
            # Écrire les données audio dans le fichier temporaire
            Path(temp_path).write_bytes(audio_data)
            
            # Jouer l'audio avec le moteur approprié
            if self.audio_engine == 'pygame':
                success = self._play_with_pygame(temp_path, blocking)
            else:
                success = self._play_with_sounddevice(temp_path, blocking)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la lecture audio: {e}")
            return False
            
        finally:
            if temp_file:
                try:
                    # Attendre un peu que le fichier soit libéré
                    time.sleep(0.2)
                    if os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)
                except Exception as e:
                    self.logger.debug(f"Nettoyage du fichier temporaire reporté: {e}")
                    # Ajouter le fichier à une liste de nettoyage différé
                    self._add_to_cleanup_queue(temp_file.name)

    def _add_to_cleanup_queue(self, filepath: str):
        """Ajoute un fichier à la file d'attente de nettoyage"""
        if not hasattr(self, '_cleanup_queue'):
            self._cleanup_queue = set()
        self._cleanup_queue.add(filepath)
        
        # Tenter de nettoyer les anciens fichiers
        self._cleanup_old_files()
    
    def _cleanup_old_files(self):
        """Tente de nettoyer les fichiers en attente"""
        if not hasattr(self, '_cleanup_queue'):
            return
            
        for filepath in list(self._cleanup_queue):
            try:
                if os.path.exists(filepath):
                    os.unlink(filepath)
                self._cleanup_queue.remove(filepath)
            except Exception:
                pass  # Le fichier est peut-être encore utilisé

    def _play_with_pygame(self, file_path: str, blocking: bool) -> bool:
        """Joue l'audio avec pygame"""
        try:
            # Charger et jouer le fichier
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            
            # Attendre la fin si demandé et pas d'interruption
            if blocking:
                while pygame.mixer.music.get_busy() and not self.force_stop.is_set():
                    pygame.time.wait(50)  # Réduit le délai d'attente pour une meilleure réactivité
                    if self.force_stop.is_set():
                        pygame.mixer.music.stop()
                        pygame.mixer.music.unload()
                        break
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur pygame: {e}")
            return False
    
    def _play_with_sounddevice(self, file_path: str, blocking: bool) -> bool:
        """Joue l'audio avec sounddevice"""
        try:
            data, samplerate = sf.read(file_path)
            sd.play(data, samplerate)
            
            if blocking and not self.force_stop.is_set():
                try:
                    while sd.get_stream().active and not self.force_stop.is_set():
                        time.sleep(0.05)  # Réduit le délai d'attente pour une meilleure réactivité
                        if self.force_stop.is_set():
                            sd.stop()
                            break
                except:
                    # Si interrompu, arrêter la lecture
                    sd.stop()
            
            if self.force_stop.is_set():
                sd.stop()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur sounddevice: {e}")
            return False
    
    def save_audio(self, text: str, output_path: str, lang: str = None) -> bool:
        """
        Sauvegarde l'audio dans un fichier
        
        Args:
            text: Texte à synthétiser
            output_path: Chemin de sortie
            lang: Code de langue
            
        Returns:
            bool: True si la sauvegarde a réussi
        """
        try:
            audio_data = self.synthesize(text, lang)
            if not audio_data:
                return False
            
            Path(output_path).write_bytes(audio_data)
            self.logger.info(f"Audio sauvegardé: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde: {e}")
            return False
    
    def _get_cache_key(self, text: str, lang: str) -> str:
        """Génère une clé de cache pour le texte"""
        cache_input = f"{text}_{lang}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def clear_cache(self) -> int:
        """
        Vide le cache audio
        
        Returns:
            int: Nombre de fichiers supprimés
        """
        try:
            deleted = 0
            for cache_file in self.cache_dir.glob("*.wav"):
                cache_file.unlink()
                deleted += 1
            
            self.logger.info(f"Cache vidé: {deleted} fichiers supprimés")
            return deleted
            
        except Exception as e:
            self.logger.error(f"Erreur lors du vidage du cache: {e}")
            return 0
    
    def get_info(self) -> Dict[str, Any]:
        """Retourne les informations sur le module"""
        return {
            'audio_engine': self.audio_engine,
            'default_lang': self.default_lang,
            'cache_dir': str(self.cache_dir)
        }
    
    def shutdown(self):
        """Nettoie les ressources"""
        try:
            self.stop_event.set()
            
            # Vider les files
            self.clear_queues()
            
            # Attendre la fin des threads
            self.synthesis_thread.join(timeout=2)
            self.playback_thread.join(timeout=2)
            
            if self.audio_engine == 'pygame':
                pygame.mixer.quit()
            elif self.audio_engine == 'sounddevice':
                sd.stop()
            
            self.logger.info("Module TTS fermé proprement")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la fermeture: {e}")

    def clear_queues(self):
        """Vide toutes les files d'attente"""
        # Vider la file de texte
        while not self.text_queue.empty():
            try:
                self.text_queue.get_nowait()
                self.text_queue.task_done()
            except Exception:
                pass
        
        # Vider la file audio
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except Exception:
                pass

    def stop(self):
        """Arrête immédiatement la synthèse vocale en cours"""
        self.force_stop.set()
        self.clear_queues()
        
        # Arrêter la lecture en cours selon le moteur audio
        if self.audio_engine == 'pygame':
            try:
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
            except Exception as e:
                self.logger.error(f"Erreur lors de l'arrêt de pygame: {e}")
        elif self.audio_engine == 'sounddevice':
            try:
                sd.stop()
            except Exception as e:
                self.logger.error(f"Erreur lors de l'arrêt de sounddevice: {e}")
        
        # Attendre un peu pour s'assurer que tout est arrêté
        time.sleep(0.1)
        self.force_stop.clear()
        self.is_speaking.clear()
        
        # Nettoyer les fichiers temporaires en attente
        self._cleanup_old_files()


def test_tts_module():
    """Fonction de test du module TTS"""
    print("🎤 Test du module TTS Thoth")
    print("=" * 50)
    
    # Initialiser le module
    try:
        tts = TTSModule()
        
        # Afficher les informations
        info = tts.get_info()
        print("\n📊 Informations du module:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test de synthèse
        test_text = "Bonjour, je suis Thoth, votre assistant IA. Comment puis-je vous aider aujourd'hui ?"
        print(f"\n🔊 Test de synthèse: '{test_text[:50]}...'")
        
        audio_data = tts.synthesize(test_text)
        if audio_data:
            print(f"✅ Synthèse réussie: {len(audio_data)} bytes")
        else:
            print("❌ Échec de la synthèse")
        
        # Test de lecture
        print("\n🔊 Test de lecture directe...")
        test_speech = "Bonjour, je suis Thoth, votre assistant personnel. Ma voix est générée par Google Text-to-Speech, et je peux vous aider dans de nombreuses tâches."
        success = tts.speak(test_speech, blocking=True)
        if success:
            print("✅ Lecture réussie")
        else:
            print("❌ Échec de la lecture")
        
        # Test de sauvegarde
        print("\n💾 Test de sauvegarde...")
        output_file = "test_output.wav"
        save_text = "Cette phrase est sauvegardée dans un fichier audio pour démontrer la qualité de la synthèse vocale."
        if tts.save_audio(save_text, output_file):
            print(f"✅ Sauvegarde réussie: {output_file}")
        else:
            print("❌ Échec de la sauvegarde")
        
        # Nettoyer
        tts.shutdown()
        print("\n✅ Tests terminés avec succès!")
        
    except Exception as e:
        print(f"\n❌ Erreur lors du test: {str(e)}")


if __name__ == "__main__":
    test_tts_module()