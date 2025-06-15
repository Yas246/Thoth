# Thoth - Assistant Vocal Intelligent

Thoth est un assistant vocal intelligent en français qui combine plusieurs technologies avancées pour offrir une expérience conversationnelle quasi-naturelle et interactive.

## Fonctionnalités Principales

### 1. Interaction Vocale Naturelle

- **Reconnaissance Vocale (STT)** : Utilise Whisper pour une reconnaissance vocale précise en français
- **Synthèse Vocale (TTS)** : Synthèse vocale de avec gTTS
- **Détection du Mot de Réveil** : Activation par le mot "jarvis" (personnalisable)

### 2. Système d'Interruption Intelligent

- Interruption instantanée lors de la détection du mot de réveil
- Reprise automatique de l'écoute après interruption
- Configuration flexible des délais et comportements d'interruption

### 3. Traitement du Langage Naturel

- **LLM (Large Language Model)** : Utilisation de modèles de langage avancés
- **RAG (Retrieval-Augmented Generation)** : Enrichissement des réponses avec une base de connaissances locale
- Réponses progressives avec synthèse vocale en temps réel

## Installation

1. Cloner le repository :

   ```bash
   git clone https://github.com/Yas246/Thoth
   cd Thoth
   ```

2. Installer les dépendances :

   ```bash
   pip install -r requirements.txt
   ```

3. Configuration :

- Copier `config/settings.yaml.example` vers `config/settings.yaml`
- Ajuster les paramètres selon vos besoins

## Configuration

### Structure du fichier settings.yaml

```yaml
# Logging

logging:
level: INFO
format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
file: "logs/thoth.log"

# Speech-to-Text (STT)

stt:
model_size: "base"
language: "fr"
device: "auto"
use_faster_whisper: true

# Text-to-Speech (TTS)

tts:
cache_dir: "data/audio_cache"
audio_engine: "auto"
lang: "fr"

# Wake Word

wake_word:
allow_interruption: true
interruption_timeout: 0.2
background_detection: true

# Mode Continu

thoth:
continuous_mode_settings:
auto_listen: true
interrupt_on_wake_word: true
min_time_between_interrupts: 1.0
clear_context_on_interrupt: false
```

## Utilisation

### Mode Standard

```bash
python main.py
```

### Mode Continu avec Détection du Mot de Réveil

```bash
python main.py --continuous
```

### Options Supplémentaires

- `--config` : Spécifier un fichier de configuration alternatif
- `--text` : Mode texte uniquement
- `--debug` : Activer les logs de débogage

## Structure des Logs

Les logs sont stockés dans le dossier `logs/` :

- Fichier principal : `thoth.log`
- Rotation automatique des logs (max 10MB par fichier)
- Conservation des 5 derniers fichiers de logs

## Architecture

### Modules Principaux

- **Orchestrator** : Gestion centrale et coordination des modules
- **STT Module** : Reconnaissance vocale avec Whisper
- **TTS Module** : Synthèse vocale avec gTTS
- **LLM Module** : Interface avec le modèle de langage
- **RAG Module** : Gestion de la base de connaissances
- **Wake Word Module** : Détection du mot de réveil

## Fonctionnalités Avancées

### Système d'Interruption

- Détection continue du mot de réveil en arrière-plan
- Interruption immédiate de la synthèse vocale
- Gestion propre des ressources audio
- Configuration flexible des délais

### Synthèse Vocale Progressive

- Synthèse et lecture en temps réel des réponses
- Début de la synthèse dès la première virgule
- Cache audio pour optimiser les performances

## Dépendances Principales

- PyYAML : Gestion de la configuration
- Whisper : Reconnaissance vocale
- gTTS : Synthèse vocale
- PyAudio : Gestion audio
- Pygame/Sounddevice : Lecture audio
- Porcupine : Détection du mot de réveil

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :

- Signaler des bugs
- Proposer des améliorations
- Soumettre des pull requests

## Licence
