"""
Module de détection de mot de réveil pour Thoth
Utilise Porcupine pour détecter des mots clés comme "Ok Thoth" ou "Hey Thoth"
"""

import os
import struct
import logging
import threading
import time
from typing import Optional, Callable
from queue import Queue
import pvporcupine
import pyaudio
import numpy as np

class WakeWordModule:
    def __init__(self, 
                 access_key: str,
                 wake_word: str = "jarvis",  # Mot par défaut disponible gratuitement
                 sensitivity: float = 0.5,
                 audio_device_index: Optional[int] = None):
        """
        Initialise le module de détection de mot de réveil
        
        Args:
            access_key: Clé d'accès Picovoice (gratuite pour usage personnel)
            wake_word: Mot de réveil à détecter
            sensitivity: Sensibilité de détection (0-1)
            audio_device_index: Index du périphérique audio à utiliser
        """
        self.logger = logging.getLogger(__name__)
        self.access_key = access_key
        self.wake_word = wake_word
        self.sensitivity = sensitivity
        self.audio_device_index = audio_device_index
        
        # État du module
        self.is_running = False
        self.detection_thread = None
        self.audio_stream = None
        self.porcupine = None
        self.pyaudio_instance = None
        
        # File pour les callbacks de détection
        self.callback_queue = Queue()
        
        # Initialisation de Porcupine
        try:
            self.porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=[wake_word],
                sensitivities=[sensitivity]
            )
            self.logger.info(f"Module de détection initialisé avec le mot '{wake_word}'")
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation de Porcupine: {e}")
            raise
    
    def start(self, callback: Callable[[], None]):
        """
        Démarre la détection en arrière-plan
        
        Args:
            callback: Fonction à appeler quand le mot de réveil est détecté
        """
        if self.is_running:
            self.logger.warning("La détection est déjà en cours")
            return
        
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Configuration du flux audio
            self.audio_stream = self.pyaudio_instance.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.porcupine.frame_length,
                input_device_index=self.audio_device_index
            )
            
            # Démarrage du thread de détection
            self.is_running = True
            self.detection_thread = threading.Thread(
                target=self._detection_loop,
                args=(callback,),
                daemon=True
            )
            self.detection_thread.start()
            
            self.logger.info("Détection du mot de réveil démarrée")
            
        except Exception as e:
            self.logger.error(f"Erreur lors du démarrage de la détection: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Arrête la détection"""
        self.is_running = False
        
        if self.detection_thread:
            self.detection_thread.join(timeout=2)
            self.detection_thread = None
        
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None
        
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
            self.pyaudio_instance = None
        
        if self.porcupine:
            self.porcupine.delete()
            self.porcupine = None
        
        self.logger.info("Détection du mot de réveil arrêtée")
    
    def _detection_loop(self, callback: Callable[[], None]):
        """Boucle principale de détection"""
        try:
            while self.is_running:
                # Lecture de l'audio
                pcm = self.audio_stream.read(self.porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                
                # Détection
                keyword_index = self.porcupine.process(pcm)
                
                # Si détection, appel du callback
                if keyword_index >= 0:
                    self.logger.info(f"Mot de réveil '{self.wake_word}' détecté!")
                    try:
                        callback()
                    except Exception as e:
                        self.logger.error(f"Erreur dans le callback de détection: {e}")
        
        except Exception as e:
            self.logger.error(f"Erreur dans la boucle de détection: {e}")
            self.is_running = False
    
    @staticmethod
    def list_audio_devices() -> list:
        """Liste les périphériques audio disponibles"""
        devices = []
        try:
            p = pyaudio.PyAudio()
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:  # Uniquement les périphériques d'entrée
                    devices.append({
                        'index': i,
                        'name': device_info['name'],
                        'channels': device_info['maxInputChannels'],
                        'sample_rate': device_info['defaultSampleRate']
                    })
            p.terminate()
        except Exception as e:
            logging.error(f"Erreur lors de la liste des périphériques: {e}")
        return devices


def test_wake_word_module():
    """Fonction de test du module"""
    print("\n=== Test du module de détection de mot de réveil ===")
    
    # Configuration du logging
    logging.basicConfig(level=logging.INFO)
    
    # Liste des périphériques audio
    print("\n📱 Périphériques audio disponibles:")
    devices = WakeWordModule.list_audio_devices()
    for device in devices:
        print(f"  [{device['index']}] {device['name']}")
    
    # Demande de la clé d'accès
    access_key = input("\n🔑 Entrez votre clé d'accès Picovoice: ")
    if not access_key:
        print("❌ Clé d'accès requise pour le test")
        return
    
    try:
        # Création du module
        wake_word_detector = WakeWordModule(
            access_key=access_key,
            wake_word="computer",  # Mot gratuit par défaut
            sensitivity=0.5
        )
        
        print("\n🎤 Démarrage de la détection...")
        print("Dites 'computer' pour tester (Ctrl+C pour arrêter)")
        
        # Callback de test
        def on_detection():
            print("\n✨ Mot de réveil détecté!")
            print("Dites 'computer' pour tester à nouveau...")
        
        # Démarrage de la détection
        wake_word_detector.start(on_detection)
        
        # Attente de l'interruption
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n🛑 Arrêt demandé...")
        
        # Arrêt propre
        wake_word_detector.stop()
        print("✅ Test terminé")
        
    except Exception as e:
        print(f"\n❌ Erreur lors du test: {e}")


if __name__ == "__main__":
    test_wake_word_module() 