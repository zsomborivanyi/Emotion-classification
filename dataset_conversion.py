# Dataset Conversion

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


import os
import librosa
import librosa.display
import soundfile as sf


fs = 16000

"""
def downsample_audio(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for actor in os.listdir(input_folder):
        actor_path = os.path.join(input_folder, actor)
        output_actor_path = os.path.join(output_folder, actor) 
        os.makedirs(output_actor_path, exist_ok=True)
        
        if os.path.isdir(actor_path):
            for filename in os.listdir(actor_path):
                if filename.endswith(".wav"):
                    filepath = os.path.join(actor_path, filename)
                    
                    y, sr = librosa.load(filepath, sr=fs)
                
                    output_path = os.path.join(output_actor_path, filename)
                    sf.write(output_path, y, fs)
                    print(f"Downsampled: {output_path}")
"""


def downsample_audio_RAVDESS(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for actor in os.listdir(input_folder):
        actor_path = os.path.join(input_folder, actor)
        
        if os.path.isdir(actor_path):
            for file in os.listdir(actor_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(actor_path, file)
                    
                    y, sr = librosa.load(file_path, sr=fs)
                    
                    output_path = os.path.join(output_folder, file)
                    sf.write(output_path, y, fs)

                    print(f"Downsampled: {output_path} (Original SR: {sr} Hz → New SR: {fs} Hz)")

def downsample_audio_TESS(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.endswith(".wav"):
            file_path = os.path.join(input_folder, file)

            y, sr = librosa.load(file_path, sr=fs)
            
            output_path = os.path.join(output_folder, file)
            sf.write(output_path, y, fs)

            print(f"Downsampled: {output_path} (Original SR: {sr} Hz → New SR: {fs} Hz)")



downsample_audio_RAVDESS("Audio_Song_Actors_01-24", "RAVDESS_16")
downsample_audio_TESS("TESS", "TESS_16")


print("Data conversion finished!")