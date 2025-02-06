# Feature Extraction file

# -
# --
# --- * Zsombor Ivanyi
# --- * Emotion Recognition In Speech
# --- * University Of Milan, Italy, Milan
# --- * 04.02.2025
# --
# -

# Dataset links:
#
# * RAVDESS dataset: https://zenodo.org/records/1188976
#       Audio_Speech_Actors_01-24.zip: https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1
#       Audio_Song_Actors_01-24.zip: https://zenodo.org/records/1188976/files/Audio_Song_Actors_01-24.zip?download=1
#
# * TESS dataset: https://doi.org/10.5683/SP2/E8H2MF


# Execute for imports:
#       pip install -r imports.txt


# Imports
import os
import numpy as np
import librosa
import librosa.display



fs = 16000 # Sampling frequency


# RAVDESS mapping:
#    {1: "Neutral", 2: "Calm", 3: "Happy", 4: "Sad", 5: "Angry", 6: "Fearful", 7: "Disgust", 8: "Surprised"}

tess_mapping = {"neutral": 1, "happy": 3, "sad":4, "angry": 5, "fear": 6, "disgust": 7} # ps not in



# Feature extraction:
def extract(y, sr):
# - Prosodic Features (Time-based)
    #zcr = librosa.feature.zero_crossing_rate(y)  # 1 Zero-Crossing Rate
    #rms = librosa.feature.rms(y=y)  # 1 RMS Energy
    #pitch = librosa.yin(y, fmin=50, fmax=300)  # 1 Pitch feature (Fundamental Frequency)
    #f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=300)  # Pitch Contour (F0)
    #f0[np.isnan(f0)] = 0
    #pitch_std = np.std(f0)  # Pitch variation
    #pitch_mean = np.mean(f0) # Pitch mean

# - Spectral Features (Frequency-based)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40) # n_mfcc: number of coefiicents, amount of spectral detail captured

    #mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    #mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to dB
    #mel_spec_resized = resize(mel_spec_db, (128, 128), anti_aliasing=True)

    mel_spectogram = librosa.feature.melspectrogram(y=y, sr=sr)
    #spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)  # 7 Spectral contrast features
    stft=np.abs(librosa.stft(y))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)  # 12 Chroma features
    #centroid = librosa.feature.spectral_centroid(y=y, sr=sr)  # 1 Spectral centroid

# - Perceptual Features
    #hnr = librosa.effects.harmonic(y)  # 1 Harmonics-to-Noise Ratio (HNR)
    
# - Temporal Features
    #duration = librosa.get_duration(y=y, sr=sr)  # 1 Duration of audio
    
    
    feature_vector = np.hstack([
        np.mean(mfcc.T, axis=0),
        np.mean(mel_spectogram.T, axis=0),
        np.mean(chroma.T, axis=0),
        #np.mean(spectral_contrast.T, axis=0),
        #np.mean(librosa.power_to_db(spectrogram.T, ref=np.max), axis=0),
        #np.mean(centroid),
        #np.mean(zcr),
        #np.mean(rms),
        #np.mean(pitch),
        #np.mean(hnr),
        #[pitch_mean, pitch_std],
        #duration  # Scalar value
    ])
    

    features = feature_vector
    #features = np.mean(mfcc.T, axis=0)
    #features = mel_spec_resized

    return features

def get_emotion_label(filename, directory):
    if "RAVDESS" in directory:
        return int(filename.split("-")[2])
    elif "TESS" in directory:
        for emotion, emotion_id in tess_mapping.items():
            if emotion in filename.lower():
                return emotion_id
    return None

def get_features(directories):
    features, labels = [], []

    for directory in directories:
        for filename in os.listdir(directory):
            if filename.endswith(".wav"):
                emotion = get_emotion_label(filename, directory)
                if emotion is not None:
                    filepath = os.path.join(directory, filename)

                    print("Extracting features from:", filename)
                    
                    y, sr = librosa.load(filepath, sr=None, res_type='kaiser_fast')

                    features.append(extract(y, fs))
                    labels.append(emotion)


    return np.array(features), np.array(labels)

def run():
    directories = ["RAVDESS_16", "TESS_16"]
    x, y = get_features(directories)

    print(f"Extracted {x.shape[0]} samples with {x.shape[1]} MFCC features each.")

    print(y[:10])


    # Save features and labels
    np.save("features.npy", x)
    np.save("labels.npy", y)

    print("Features and labels saved successfully!")

#run()