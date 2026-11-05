banner = r"""

-------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------

           __  __   _   _   ____    ___    ____       ____        _          _      _____   _____   _   _   _____
          |  \/  | | | | | / ___|  |_ _|  / ___|     |___ \      | |        / \    |_   _| | ____| | \ | | |_   _|
          | |\/| | | | | | \___ \   | |  | |           __) |     | |       / _ \     | |   |  _|   |  \| |   | |
          | |  | | | |_| |  ___) |  | |  | |___       / __/      | |___   / ___ \    | |   | |___  | |\  |   | |
          |_|  |_|  \___/  |____/  |___|  \____|     |_____|     |_____| /_/   \_\   |_|   |_____| |_| \_|   |_|


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



# -------------------------------------------------------------------------
#                                 IMPORTS
# --------------------------------------------------------------------------

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from music2latent import EncoderDecoder
from sklearn.preprocessing import StandardScaler
import umap
from umap import plot

from scipy.ndimage import uniform_filter1d



# ----------------------------------------------------------------------------------------------------
#                  EXTRATION ET AFFICHAGE DES DONNEES DE L'ENCOODEUR DE MUSIC2LATENT
# ----------------------------------------------------------------------------------------------------


# Chargement du fichier audio en float32, , always_2d=True → shape [N_samples, N_channels]
audio, sr = sf.read("AUDIO_FILES/Compo_Lorenzo.wav", dtype="float32", always_2d=True)


# Transposition pour obtenir [N_channels, N_samples] attendu par Music2Latent
audio_cf = audio.T
encdec = EncoderDecoder()

# Extraction des features intermédiaires (sortie du premier réseau convolutif)
features = encdec.encode(audio_cf, extract_features=True)
features = features.detach().cpu().numpy()
print(f'Shape of latents: {features.shape}')

# Extraction des latents du goulot d'étranglement (bottleneck) → shape [C, 64, T]
latent = encdec.encode(audio_cf)
lat = latent.detach().cpu().numpy()
print(f'Shape of latents: {latent.shape}')


# Si stéréo, on fais la moyenne des canaux pour obtenir une représentation mono [Features, T]
features = features.mean(axis=0) if features.shape[0] > 1 else features[0]
latent = lat.mean(axis=0) if lat.shape[0] > 1 else lat[0]


# Nombre de frames temporelles pour chaque représentation
T_feat = features.shape[1]
T_lat = latent.shape[1]

n_samples = audio.shape[0]
duration_sec = n_samples / sr


# Signal mono pour l'affichage de la waveform
audio_mono = audio.mean(axis=1)

# Axes temporels alignés sur la durée réelle du fichier audio
t_sec = np.arange(n_samples) / sr
t_sec_feat = np.linspace(0, duration_sec, T_feat, endpoint=False)
t_sec_lat = np.linspace(0, duration_sec, T_lat, endpoint=False)


print(f'Secondes par frame latente : {t_sec_lat}')
print(f'Secondes par frame latente : {t_sec_feat}')


# -----------------------------------------------------------------------------------------------------
#                                           1) PARTIE FEATURES
# -----------------------------------------------------------------------------------------------------


#--------------------------------------------
#         Affichage des features brutes
#--------------------------------------------

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(16, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [1, 2]}  # la waveform plus petite, les features plus grandes
)


# --------- Graphe du haut : waveform ---------
ax1.plot(t_sec, audio, linewidth=0.8)
ax1.set_xlabel("Times (s)")
ax1.set_ylabel("Amplitude")
ax1.set_title("Waveform \n of a piano chord sample")


# --------- Graphe du bas : features ---------
# Affichage des 8192 dimensions de features en superposition
for d in range(8192):
    ax2.plot(t_sec_feat, features[d], alpha=0.6, linewidth=1)

ax2.set_xlabel("Times (s)")
ax2.set_ylabel("Latent Features")
ax2.set_title("8192 output features of Music2Latent’s first convolutional network, \n for a piano chord sample")

plt.grid(alpha=0.3)
fig.subplots_adjust(hspace=0.35)  # Espace vertical entre les deux sous-graphes
plt.show()


#------------------------------------------------------------------------
#           Normalisation et réduction des features avec un UMAP
#------------------------------------------------------------------------

# Transposition pour passer en [T, Features], format attendu par UMAP
features_T = features.T

# Standardisation : centre et réduit chaque feature
scaler = StandardScaler()
Xz = scaler.fit_transform(features_T)

# Réduction dimensionnelle : 8192 features → 3 composantes UMAP
reducer = umap.UMAP(
    n_components=3,
    random_state=42
)

Y = reducer.fit_transform(Xz)  # [T, 3]

# Normalisation min-max → toutes les composantes dans [0, 1]
lo = Y.min(axis=0, keepdims=True)
hi = Y.max(axis=0, keepdims=True)
Y_norm = (Y - lo) / (hi - lo)

time_indices = np.arange(len(Y))


#-------------------------------------------------------------------------------
#            Affichage des composantes UMAP normalisées des features
#-------------------------------------------------------------------------------
plt.figure(figsize=(16, 8))
for i in range(Y_norm.shape[1]):
    plt.plot(t_sec_feat, Y_norm[:, i], label=f"UMAP {i}", linewidth=1.5, alpha=0.85)
plt.xlabel("Temps (secondes)")
plt.ylabel("UMAP normalisé (0..1) des features")
plt.title("Courbes UMAP normalisées")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()



# --- LISSAGE (moyenne glissante) ---
# Lissage temporel des courbes UMAP pour atténuer les variations brusques
smoothing_window_size_features = 4  # taille en frames latentes (à modifier selon l'utilisateur)

Y_features_norm_smooth = uniform_filter1d(
    Y_norm,
    size=smoothing_window_size_features,
    axis=0,
    mode="nearest"  # évite les effets de bord aux extrémités
)

#-------------------------------------------------------------------------------
#            Affichage des composantes UMAP normalisées et lissées des features
#-------------------------------------------------------------------------------
plt.figure(figsize=(16, 8))
for i in range(Y_features_norm_smooth.shape[1]):
    plt.plot(t_sec_feat, Y_features_norm_smooth[:, i], label=f"UMAP {i}", linewidth=1.5, alpha=0.85)
plt.xlabel("Times (s)")
plt.ylabel("UMAP normalisé + lissé (0..1)")
plt.title("Courbes UMAP normalisées et lissées")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()



# --------------------------------------------------------------------------------------------------
#                                    2) PARTIE GOULOT D'ETRANGLEMENT
# -------------------------------------------------------------------------------------------------


#------------------------------------------------------------------
#      Affichage des 64 dimensions latentes brutes du bottleneck
#------------------------------------------------------------------
plt.figure()
for d in range(64):
    plt.plot(t_sec_lat, latent[d], alpha=0.6, linewidth=1)
plt.xlabel("Time (s)")
plt.ylabel("Latent dimensions")
plt.title("The 64 latent dimensions at the bottleneck of Music2Latent,\n for a piano chord sample")
plt.show()



#---------------------------------------------------------------------------
#    Normalisation et réduction des dimensions latentes avec un UMAP
#---------------------------------------------------------------------------

# Transposition pour passer en [T, 64], format attendu par UMAP
latent_T = latent.T  # [T, 64]

# Standardisation des 64 dimensions latentes
scaler = StandardScaler()
strd_latent = scaler.fit_transform(latent_T)  # [T, 64]

# Réduction dimensionnelle : 64 dimensions → 3 composantes UMAP
reducer = umap.UMAP(
    n_components=3,
    random_state=42
)

Y_latent= reducer.fit_transform(strd_latent)  # [T, 3]

# Normalisation min-max → toutes les composantes dans [0, 1]
lo = Y_latent.min(axis=0, keepdims=True)
hi = Y_latent.max(axis=0, keepdims=True)
Y_latent_norm = (Y_latent - lo) / (hi - lo )

time_indices = np.arange(len(Y))



#--------------------------------------------------------------------------
#        Affichage des composantes UMAP normalisées du bottleneck
#--------------------------------------------------------------------------
plt.figure(figsize=(16, 8))
for i in range(Y_latent_norm.shape[1]):
    plt.plot(t_sec_lat, Y_latent_norm[:, i], label=f"UMAP {i}", linewidth=1.5, alpha=0.85)
plt.xlabel("Temps (secondes)")
plt.ylabel("UMAP normalisé (0..1)")
plt.title("Courbes UMAP normalisées")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()



# --- LISSAGE (moyenne glissante) --
# Lissage temporel des courbes UMAP pour atténuer les variations brusques
smoothing_window_size_latent = 4  # taille en frames latentes (à modifier selon l'utilisateur)

Y_latent_norm_smooth = uniform_filter1d(
    Y_latent_norm,
    size=smoothing_window_size_latent,
    axis=0,
    mode="nearest"  # évite les effets de bord aux extrémités
)


#--------------------------------------------------------------------------
#        Affichage des composantes UMAP normalisées et lisées du bottleneck
#--------------------------------------------------------------------------
plt.figure(figsize=(16, 8))
for i in range(Y_latent_norm_smooth.shape[1]):
    plt.plot(t_sec_lat, Y_latent_norm_smooth[:, i], label=f"UMAP {i}", linewidth=1.5, alpha=0.85)
plt.xlabel("Time (s)")
plt.ylabel("Normalized and smoothed latent dimensions")
plt.title("The 3 normalized and smoothed UMAP dimensions, reducing the 64 dimensions of the bottleneck, for a rhythmic sample")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# -------------------------------------------------------------------------
#                              EXPORT FICHIER CSV
# -------------------------------------------------------------------------

OUT_CSV_NAME = "umap_norm_smooth_compo_lorenzo.csv"   # nom du fichier .csv (à modifier selon l'utilisateur)

# Concatène l'axe temporel et les 3 composantes UMAP lissées → shape [T, 4]
csv_mat = np.column_stack([t_sec_lat, Y_latent_norm_smooth])

# En-tête : time_sec, umap0, umap1, umap2
header = "time_sec," + ",".join([f"umap{i}" for i in range(Y_latent_norm_smooth.shape[1])])

np.savetxt(OUT_CSV_NAME, csv_mat, delimiter=",", header=header, comments="")
print("CSV exporté :", OUT_CSV_NAME, csv_mat.shape)