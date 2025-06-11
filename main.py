# main.py
import argparse
import logging
import sys
from pathlib import Path
import json # For pretty printing status/history
import time

# Assuming 'modules' is in PYTHONPATH or this script is run in an environment where it's accessible.
# If modules.orchestrator cannot be found, PYTHONPATH might need adjustment or a setup.py might be needed.
from modules.orchestrator.orchestrator import Orchestrator

# Configure basic logging for the main script
# The Orchestrator will configure its own logging based on settings.yaml
# This basicConfig is for messages from main.py itself, before Orchestrator's logging takes over for its parts.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [MainScript] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_help():
    """Affiche l'aide des commandes disponibles"""
    print("\nCommandes disponibles:")
    print("  !aide                     - Affiche cette aide.")
    print("  !historique               - Affiche l'historique de la conversation.")
    print("  !effacer                  - Efface l'historique de la conversation.")
    print("  !statut                   - Affiche le statut de l'assistant.")
    print("  !voix                     - Bascule en mode entrée vocale (microphone).")
    print("  !texte                    - Bascule en mode entrée texte.")
    print("  !continu                  - Active le mode continu avec détection du mot clé.")
    print("  !stop                     - Arrête le mode continu.")
    print("  quitter / exit            - Quitte l'assistant.")
    print("Tout autre texte sera traité comme une requête pour l'assistant.\n")

def main():
    parser = argparse.ArgumentParser(description="Thoth AI Assistant - Main Control Script")
    parser.add_argument(
        "--config", type=str, default="config/settings.yaml",
        help="Path to the configuration file (e.g., config/settings.yaml)"
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="Run a single interaction with the provided text input, then exit."
    )
    parser.add_argument(
        "--continuous", action="store_true",
        help="Start in continuous mode with wake word detection."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging for the main script and potentially for the orchestrator if config allows."
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG) # Set root logger to DEBUG
        logger.setLevel(logging.DEBUG) # Set this script's logger to DEBUG
        logger.debug("Debug logging enabled for main.py.")

    config_path = Path(args.config)
    if not config_path.is_file():
        logger.critical(f"Configuration file not found at '{config_path}'. Arrêt.")
        sys.exit(1)

    orchestrator_instance = None
    try:
        logger.info(f"Initialisation de l'Orchestrator avec la configuration: {config_path}")
        orchestrator_instance = Orchestrator(config_path=str(config_path))

        logger.info("Démarrage de l'Orchestrator...")
        orchestrator_instance.startup()

        # Mode d'entrée (texte par défaut)
        input_mode = 'text'
        continuous_mode = args.continuous

        print("Bienvenue à l'assistant Thoth interactif.\n")
        print_help()

        # Démarrer en mode continu si demandé
        if continuous_mode:
            print("\n🎤 Démarrage du mode continu...")
            if orchestrator_instance.start_continuous_mode():
                print(f"✓ Mode continu activé. Dites '{orchestrator_instance.wake_word_detector.wake_word}' pour commencer.")
            else:
                print("❌ Impossible d'activer le mode continu. Retour au mode normal.")
                continuous_mode = False

        while True:
            if continuous_mode:
                try:
                    time.sleep(0.1)  # Évite de surcharger le CPU
                    continue
                except KeyboardInterrupt:
                    print("\n\n🛑 Interruption détectée. Tapez une commande ou 'quitter' pour sortir.")
                    continuous_mode = False
                    orchestrator_instance.stop_continuous_mode()
                    continue

            if input_mode == 'text':
                print("\nVous:", end=" ")
                user_input = input().strip()
            else:  # input_mode == 'speech'
                print("\n>>> Parlez maintenant (ou tapez '!texte' pour revenir au mode texte) <<<")
                try:
                    audio_result = orchestrator_instance.stt_module.start_listening()
                    if audio_result:
                        print(f"\nVous avez dit: {audio_result}")
                        user_input = audio_result
                    else:
                        print("\n❌ Aucune parole détectée")
                        continue
                except Exception as e:
                    print(f"\n❌ Erreur lors de la capture audio: {e}")
                    continue

            # Traitement des commandes
            if user_input.lower() in ['quitter', 'exit']:
                print("Au revoir!")
                break
            elif user_input == '!aide':
                print_help()
                continue
            elif user_input == '!historique':
                history = orchestrator_instance.get_conversation_history()
                print("\nHistorique de la conversation:")
                for msg in history:
                    role = "Assistant" if msg['role'] == "assistant" else "Vous"
                    print(f"{role}: {msg['content']}")
                continue
            elif user_input == '!effacer':
                orchestrator_instance.clear_conversation_history()
                print("Historique effacé.")
                continue
            elif user_input == '!statut':
                status = orchestrator_instance.get_status()
                print("\nStatut de l'assistant:")
                print(json.dumps(status, indent=2, ensure_ascii=False))
                continue
            elif user_input == '!continu':
                if orchestrator_instance.start_continuous_mode():
                    continuous_mode = True
                    print(f"✓ Mode continu activé. Dites '{orchestrator_instance.wake_word_detector.wake_word}' pour commencer.")
                else:
                    print("❌ Impossible d'activer le mode continu.")
                continue
            elif user_input == '!stop':
                if continuous_mode:
                    orchestrator_instance.stop_continuous_mode()
                    continuous_mode = False
                    print("✓ Mode continu désactivé.")
                else:
                    print("Le mode continu n'est pas actif.")
                continue
            elif user_input == '!voix':
                # Vérifie les périphériques audio avant de changer le mode
                try:
                    audio_devices = orchestrator_instance.stt_module.get_audio_devices()
                    if audio_devices:
                        print("\nPériphériques audio disponibles:")
                        for i, device in enumerate(audio_devices):
                            print(f"{i}: {device}")
                        input_mode = 'speech'
                        print("\n✓ Mode vocal activé. Parlez après l'invite.")
                    else:
                        print("\n⚠️ Aucun périphérique audio détecté!")
                except Exception as e:
                    print(f"\n❌ Erreur lors de l'initialisation audio: {e}")
                continue
            elif user_input == '!texte':
                input_mode = 'text'
                print("✓ Mode texte activé.")
                continue
            elif not user_input:
                continue

            # Traitement de la requête
            orchestrator_instance.run_interaction_cycle(input_mode='text', initial_text=user_input)

    except KeyboardInterrupt:
        logger.info("Interruption utilisateur détectée. Arrêt en cours...")
    except Exception as e:
        logger.critical(f"Erreur inattendue: {e}", exc_info=True)
    finally:
        if orchestrator_instance:
            if continuous_mode:
                orchestrator_instance.stop_continuous_mode()
            logger.info("Arrêt de l'Orchestrator...")
            orchestrator_instance.shutdown()
        logger.info("Assistant Thoth terminé.")

if __name__ == "__main__":
    main()
