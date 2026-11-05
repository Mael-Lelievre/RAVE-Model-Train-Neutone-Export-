banner = r"""




-------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------

                      ____       _     __     __   _____       _____  __  __    ____     ___    ____    _____
                     |  _ \     / \    \ \   / /  | ____|     | ____| \ \/ /   |  _ \   / _ \  |  _ \  |_   _|
                     | |_) |   / _ \    \ \ / /   |  _|       |  _|    >  <    | |_) | | | | | | |_) |   | |
                     |  _ <   / ___ \    \ V /    | |___      | |___  / /\ \   |  __/  | |_| | |  _ <    | |
                     |_| \_\ /_/   \_\    \_/     |_____|     |_____| /_/  \_\ |_|      \___/  |_| \_\   |_|


-------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------

"""
print("\n")
print("Guitar-to-Voice Timbral Transfer : Neural Synthesis for Composition and Audio-Reactive Visuals via Latent Representation")
print("\n")
print("Auteurs: Maël Lelievre, Ivan Benabdillah, Lorenzo Pernice")
print("\n")
print("Last Modified: 08.03.2026")
print("\n")


#------------------------------------------------------------------------------
#                                    IMPORTS
#------------------------------------------------------------------------------

import subprocess
from pathlib import Path
from typing import List, Optional


#------------------------------------------------------------------------------
#                              EXPORT MODELE RAVE
#-------------------------------------------------------------------------------


def run(cmd: List[str], cwd: Optional[str] = None) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    # Chemin vers le checkpoint du modèle RAVE à exporter (à changer selon l'utilisateur)
    model_dir = Path("RUN/RAVE_IVAN_FINAL_V2_FULL_f381b579e1/version_2/checkpoints/epoch_20000.ckpt")

    # Dossier de destination pour le fichier TorchScript exporté (à changer selon l'utilisateur)
    output_dir = Path("EXPORT_TS/IVAN_V2_FULL")

    # Nom du fichier de sortie (à changer selon l'utilisateur)
    export_name = "IVAN_V2_FULL_500000"


    # Vérifie que le checkpoint existe avant de lancer l'export
    if not model_dir.exists():
        raise FileNotFoundError(f"Run introuvable : {model_dir}")


    rave_exe = "rave"
    channels = 1     # Nombre de canaux audio du modèle (1 = mono)


    # Construction de la commande d'export RAVE :
    # --streaming  : export en mode streaming (traitement en temps réel)
    # --fidelity   : niveau de fidélité de reconstruction (0.999 = très haute qualité)
    # --channels   : nombre de canaux audio du modèle
    export_cmd = [
        rave_exe, "export",
        "--run", str(model_dir),
        "--output", str(output_dir),
        "--name", export_name,
        "--streaming",
        "--fidelity", "0.999",
        "--channels", str(channels),
    ]

    print("Commande réalisée :", " ".join(export_cmd))
    run(export_cmd)

    print("\n✅ Export TorchScript terminé.")


if __name__ == "__main__":
    main()