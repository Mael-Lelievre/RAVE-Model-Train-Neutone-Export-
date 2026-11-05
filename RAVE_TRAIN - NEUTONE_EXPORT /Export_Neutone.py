
banner = r"""


-----------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------

        _   _   _____   _   _   _____    ___    _   _   _____       _____  __  __    ____     ___    ____    _____
        | \ | | | ____| | | | | |_   _|  / _ \  | \ | | | ____|     | ____| \ \/ /   |  _ \   / _ \  |  _ \  |_   _|
        |  \| | |  _|   | | | |   | |   | | | | |  \| | |  _|       |  _|    >  <    | |_) | | | | | | |_) |   | |
        | |\ | | |___   | |_| |   | |   | |_| | | |\ | | |___       | |___  / /\ \   |  __/  | |_| | |  _ <    | |
        |_| \_| |_____|  \___/    |_|    \___/  |_| \_| |_____|     |_____| /_/  \_\ |_|      \___/  |_| \_\   |_|


----------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------

"""
print("\n")
print("Guitar-to-Voice Timbral Transfer : Neural Synthesis for Composition and Audio-Reactive Visuals via Latent Representation")
print("\n")
print("Auteurs: Maël Lelievre, Ivan Benabdillah, Lorenzo Pernice")
print("\n")
print("Last Modified: 08.03.2026")
print("\n")


#------------------------------------------------------------------------------
#                                IMPORTS
#------------------------------------------------------------------------------

import glob
import os
import torch
import torch.jit

from pathlib import Path
from typing import Dict, List

from torch import Tensor, nn
from neutone_sdk import WaveformToWaveformBase, NeutoneParameter
from neutone_sdk.utils import load_neutone_model, save_neutone_model


try:
    from neutone_sdk.parameter import NeutoneParameterType
except ImportError:
    from neutone_sdk import NeutoneParameterType



#--------------------------------------------------------------------------------------------------------
#                                EXPORT MODELE RAVE POUR NEUTONE FX AVEC LE NEUTONE
#--------------------------------------------------------------------------------------------------------


class RAVEModelWrapper(WaveformToWaveformBase):


    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
        self.model = model.eval()

# ------------------------------------------------------------------------
#                             METADONNEE DU MODELE
# ------------------------------------------------------------------------

    def get_model_name(self) -> str:
        return "IVAN_V2_FULL_500000"  # <-EDIT THIS

    def get_model_authors(self) -> List[str]:
        return ["Maël Lelievre, Ivan Benabdillah, Lorenzo Pernice"]  # <-EDIT THIS

    def get_model_short_description(self) -> str:
        return "RAVE model trained on xxx sounds."  # <-EDIT THIS

    def get_model_long_description(self) -> str:
        return (  # <-EDIT THIS
            "RAVE timbre transfer model trained on xxx sounds. Useful for xxx sounds."
        )

    def get_technical_description(self) -> str:
        return "RAVE model proposed by Caillon, Antoine et al."

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "Paper": "https://arxiv.org/abs/2111.05011",
            "Code": "https://github.com/acids-ircam/RAVE",
        }

    def get_tags(self) -> List[str]:
        return ["timbre transfer", "RAVE"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        """
        set to True for models in experimental stage
        (status shown on the website)
        """
        return False  # <-EDIT THIS

# ----------------------------------------------------------------
#                    PARAMETRES DU PLUGIN
#-----------------------------------------------------------------

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        params = [
            NeutoneParameter(
                name="Chaos",
                description="Magnitude of latent noise",
                default_value=0.0,
                used=True,
                param_type=NeutoneParameterType.CONTINUOUS,
            ),
            NeutoneParameter(
                name="Z edit index",
                description="Index of latent dimension to edit",
                default_value=0.0,
                used=True,
                param_type=NeutoneParameterType.CONTINUOUS,
            ),
            NeutoneParameter(
                name="Z scale",
                description="Scale of latent variable",
                default_value=0.5,
                used=True,
                param_type=NeutoneParameterType.CONTINUOUS,
            ),
            NeutoneParameter(
                name="Z offset",
                description="Offset of latent variable",
                default_value=0.5,
                used=True,
                param_type=NeutoneParameterType.CONTINUOUS,
            ),
        ]


        for p in params:
            if not hasattr(p, "default_value_0to1"):
                dv = getattr(p, "default_value", 0.0)
                if isinstance(dv, torch.Tensor):
                    dv = float(dv.item())
                p.default_value_0to1 = float(dv)

        return params

#---------------------------------------------------------------------------
#                      FORMAT AUDIO DU MODELE
# --------------------------------------------------------------------------

    def is_input_mono(self) -> bool:
        return True  # <-Set to False for stereo (each channel processed separately)

    def is_output_mono(self) -> bool:
        return True  # <-Set to False for stereo (each channel processed separately)

    def get_native_sample_rates(self) -> List[int]:
        return [48000]  # <-EDIT THIS

    def get_native_buffer_sizes(self) -> List[int]:
        return [2048]

    def get_citation(self) -> str:
        return """Caillon, A., & Esling, P. (2021). RAVE: A variational autoencoder for fast and high-quality neural audio synthesis. arXiv preprint arXiv:2111.05011."""



# ------------------------------------------------------------------------------------
#               GESTION DES DONNEES CONTROLEES PAR LES PARAMETRES DU PLUGIN
# ------------------------------------------------------------------------------------

    @torch.no_grad()
    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:

        # Encode l'audio en espace latent
        z = self.model.encode(x.unsqueeze(1))

        # Ajout de bruit modulé par le paramètre Chaos
        noise_amp = params["Chaos"].item()
        z = z + z.new_empty(z.shape).normal_() * noise_amp

        # Sélection de la dimension latente à éditer (0~1 -> 0~latent_size)
        idx = params["Z edit index"].clamp(0.0, 0.99).item()
        idx_z = int(idx * self.model.latent_size)

        # Application du scale (0~1 -> 0~2) et de l'offset (0~1 -> -1~1)
        # sur la dimension latente sélectionnée
        z_scale = params["Z scale"].item() * 2.0
        z_offset = params["Z offset"].item() * 2.0 - 1.0
        z[:, idx_z] = z[:, idx_z] * z_scale + z_offset

        # Décode l'espace latent modifié en audio
        out = self.model.decode(z)
        out = out.squeeze(1)
        print("✅ Nombre de dimensions latente du modèle :", self.model.latent_size)
        return out



# --- Chargement du modèle TorchScript ---
ts_file = Path("EXPORT_TS/IVAN_V2_FULL/IVAN_V2_FULL_500000_streaming.ts")  # Chemin du fichier TorchScript .ts (à changer selon l'utilisateur)
print("✅ TorchScript trouvé :", ts_file)


model = torch.jit.load(ts_file).eval()
print("✅ Modèle chargé :", type(model))

# --- Instanciation du wrapper Neutone autour du modèle ---
wrapper = RAVEModelWrapper(model)

audio_sample_pairs = None
print("✅ Wrapper prêt.")


# --- Export final au format Neutone ---
neutone_save_dir = "EXPORT_NEUTONE/IVAN_V2_FULL"  # Chemin du dossier de saugearde (à changer selon l'utilisateur)
run_name = "IVAN_V2_FULL_ckpt_500000" # Non du dossier d'export conteant le modèle Neutone (à changer selon l'utilisateur)

# --- Sauvegarde le modèle au format Neutone ---
save_neutone_model(wrapper, Path(neutone_save_dir) / run_name, freeze=False, dump_samples=True, submission=True, audio_sample_pairs=audio_sample_pairs)
print(f"✅ Export Neutone terminé : {neutone_save_dir}")