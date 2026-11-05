banner = r"""

-------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------

                      ____       _     __     __   _____       _____   ____       _      ___   _   _
                     |  _ \     / \    \ \   / /  | ____|     |_   _| |  _ \     / \    |_ _| | \ | |
                     | |_) |   / _ \    \ \ / /   |  _|         | |   | |_) |   / _ \    | |  |  \| |
                     |  _ <   / ___ \    \ V /    | |___        | |   |  _ <   / ___ \   | |  | |\  |
                     |_| \_\ /_/   \_\    \_/     |_____|       |_|   |_| \_\ /_/   \_\ |___| |_| \_|


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


# ----------------------------------------------------------------------------
#                                 IMPORTS
# ----------------------------------------------------------------------------


import subprocess
import sys

import torch
torch.autograd.set_detect_anomaly(True)
import tensorboard
from torch.utils.checkpoint import checkpoint


from pathlib import Path
from typing import List, Optional


# ------------------------------------------------------------------------
#                          ENTRAINEMENT MODELE RAVE
# ------------------------------------------------------------------------


def run(cmd: List[str], cwd: Optional[str] = None) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():

# -------------------------------------------------------------
#                           PARAMETRES
# -------------------------------------------------------------

    # ---- Chemin des dossiers et paramètres généraux -----
    dataset_path = Path("DATASET/IVAN")             # Dossier du dataset avec les fichiers audio bruts (à changer selon l'utilisateur)
    train_folder_name = "RAVE_IVAN_FINAL_V2_FULL"   # Nom du dossier dans lequel les fichiers d'entraînement sont sauvegardés (à changer selon l'utilisateur)
    save_path = Path("RUN")                         # Dossier de sortie pour les runs et checkpoints (à changer selon l'utilisateur)
    checkpoint_path = Path("RUN/RAVE_IVAN_FINAL_V2_FULL_f381b579e1/version_1/checkpoints/epoch_360000.ckpt") # Checkpoint de reprise


    preprocessed_dataset_path = save_path / "preprocessed_dataset_ivan_final_full"  # Dossier de sortie pour la base donnée preprocéssée
    preprocessed_dataset_path.mkdir(parents=True, exist_ok=True) # Crée le dossier si inexistant

    val_every = 100            # Fréquence de validation pendant l'entraînement
    save_every = 10000         # Fréquence de sauvegarde des checkpoints
    channels = 1               # 1 = mono / 2 = stéréo / 4 = quadriphonique ...
    sample_rate = 48000        # Fréquence d'échantillonnage des fichiers audio
    num_signal = 131072        # Longueur d'un extrait audio en samples (valeur par défaut : ~2.73s à 48kHz)
                               # 131072 est la valeur par défault


    rave_exe = "rave"

    # ---- Config ----
    rave_architecture = "v2"            # "v1" / "v2" / "v2_small" / "v3" / "discrete" ...
    regularization_type = "default"     # "default" / "wasserstein" / "spherical"
    causal_activate = False             # Contraint le modèle à n'utiliser que les échantillons passés (+ de latence)
    noise_activate = False              #  # Ajoute un synthétiseur de bruit au décodeur
    hybrid_activate = False             # Remplace l'entrée encodeur par une représentation Mel (utile pour les voix)


    # ---- Augment ----
    mute_activate = True                # Désactive aléatoirement des lots → force le modèle à apprendre le silence
    compress_activate = False           # Applique une légère compression aléatoire → robustesse aux variations dynamiques
    gain_activate = False               # Applique un gain aléatoire (défaut : -6 à +3 dB) → robustesse aux amplitudes


# -------------------------------------------------------------
#                           VERIFICATIONS
# -------------------------------------------------------------

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {dataset_path}")
    save_path.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
#                           TENSORBOARD
# -------------------------------------------------------------------

    # Lance TensorBoard en arrière-plan pour visualiser les métriques d'entraînement
    tb_cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", str(save_path), "--port", "8880"]
    subprocess.Popen(tb_cmd)
    print("Commande réalisée :", " ".join(tb_cmd))


# ---------------------------------------------------------------
#                       1) COMMANDE PREPROCESS
# ---------------------------------------------------------------

    # Prépare et segmente le dataset audio en fichiers de longueur fixe (num_signal samples)
    preprocess_cmd = [
        rave_exe, "preprocess",
        "--input_path", str(dataset_path),
        "--output_path", str(preprocessed_dataset_path),
        "--channels", str(channels),
        "--sampling_rate", str(sample_rate),
        "--num_signal", str(num_signal),
    ]

    print("Commande réalisée :", " ".join(preprocess_cmd))
    run(preprocess_cmd)


# -------------------------------------------------------------
#                       2) COMMANDE TRAIN
# -------------------------------------------------------------


    # Construit la commande d'entraînement avec l'architecture et la régularisation choisies
    train_cmd = [
        rave_exe, "train",
        "--config", str(rave_architecture),
        "--config", str(regularization_type),
    ]


    # Ajout conditionnel des configs optionnelles
    if causal_activate:
       train_cmd += ["--config", "causal"]

    if noise_activate:
        train_cmd += ["--config", "noise"]

    if hybrid_activate:
        train_cmd += ["--config", "hybrid"]


    # Ajout conditionnel des augmentations
    if mute_activate:
        train_cmd += ["--augment", "mute"]

    if compress_activate:
        train_cmd += ["--augment", "compress"]

    if gain_activate:
        train_cmd += ["--augment", "gain"]


    # Paramètres principaux de l'entraînement
    train_cmd += [
        "--db_path", str(preprocessed_dataset_path),
        "--out_path", str(save_path),
        "--name", train_folder_name,
        "--channels", str(channels),
        "--val_every", str(val_every),
        "--save_every", str(save_every),
        "--override", f"SAMPLING_RATE={sample_rate}",
        "--n_signal", str(num_signal),
        "--ckpt", str(checkpoint_path),  # Reprise depuis le checkpoint spécifié (à supprimer si ce n'est pas le cas)
        "--gpu", "-1",                   # Ici on force l'entraînement sur le CPU (pas recommandé) car MPS est non compatible avec l'entraînement
    ]


    print("Commande réalisée :", " ".join(train_cmd))
    run(train_cmd)

    print("\n✅ Terminé.")
    print(f"Runs / checkpoints dans : {save_path}")
    print(f"Dataset preprocessé dans : {preprocessed_dataset_path}")




if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print("\n❌ Erreur pendant l'exécution d'une commande.")
        print("Commande:", e.cmd)
        print("Code retour:", e.returncode)
        sys.exit(e.returncode)





