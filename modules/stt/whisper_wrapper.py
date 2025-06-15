import os
import subprocess
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class WhisperCpp:
    def __init__(self, model_path: str, language: str = "fr", n_threads: int = 4):
        self.model_path = model_path
        self.language = language
        self.n_threads = n_threads
        self.whisper_cpp_path = str(Path(__file__).parent.parent.parent / "whisper.cpp")
        self.cli_path = str(Path(self.whisper_cpp_path) / "build/bin/Release/whisper-cli.exe")
        
        if not os.path.exists(self.cli_path):
            raise FileNotFoundError(f"whisper-cli non trouvé: {self.cli_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
    
    def transcribe_file(self, audio_path: str) -> str:
        """Transcrit un fichier audio en utilisant whisper.cpp"""
        try:
            # Créer un fichier temporaire pour la sortie texte
            output_txt = audio_path + ".txt"
            if os.path.exists(output_txt):
                os.remove(output_txt)
            
            cmd = [
                self.cli_path,
                "-m", self.model_path,
                "-f", audio_path,
                "-l", self.language,
                "-t", str(self.n_threads),
                "-otxt",  # Sortie dans un fichier texte
                "--print-progress",  # Afficher la progression
                "--print-special",  # Afficher les tokens spéciaux
                "--no-timestamps"  # Pas de timestamps dans la sortie
            ]
            
            logger.info(f"Exécution de whisper.cpp: {' '.join(cmd)}")
            
            # Exécuter whisper.cpp
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            # Log de la sortie pour debug
            logger.debug(f"Sortie stdout: {result.stdout}")
            logger.debug(f"Sortie stderr: {result.stderr}")
            
            if result.returncode != 0:
                logger.error(f"Erreur whisper.cpp: {result.stderr}")
                return ""
            
            # Lire le fichier de sortie
            output_txt = audio_path + ".txt"
            if os.path.exists(output_txt):
                with open(output_txt, 'r', encoding='utf-8') as f:
                    transcription = f.read().strip()
                os.remove(output_txt)  # Nettoyer
                return transcription
            
            # Si pas de fichier, essayer d'extraire de la sortie standard
            transcription = ""
            for line in result.stdout.split("\n"):
                if line.startswith("["):  # Ignorer les lignes de log
                    continue
                if "whisper_print_timings" in line:  # Ignorer les stats
                    continue
                transcription += line.strip() + " "
            
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"Erreur lors de la transcription: {e}")
            return ""

    @staticmethod
    def download_model(model_name: str, output_dir: str) -> str:
        """Télécharge un modèle depuis Hugging Face"""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"ggml-{model_name}.bin")
        
        if os.path.exists(output_path):
            logger.info(f"Modèle déjà téléchargé: {output_path}")
            return output_path
        
        url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{model_name}.bin"
        logger.info(f"Téléchargement du modèle depuis {url}")
        
        try:
            subprocess.run([
                "curl", "-L", url,
                "-o", output_path
            ], check=True)
            logger.info(f"Modèle téléchargé: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Erreur lors du téléchargement: {e}")
            raise
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement: {e}")
            raise 