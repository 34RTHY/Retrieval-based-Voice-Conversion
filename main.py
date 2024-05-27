from pathlib import Path
from dotenv import load_dotenv
from scipy.io import wavfile
from rvc.modules.vc.modules import VC
from bark.bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio
import sounddevice as sd
import soundfile as sf
import torch
import os
import traceback
import librosa
import numpy as np

SAMPLE_RATE = 22050

def check_gpu():
    if torch.cuda.is_available():
        print(f"GPU is available. Number of GPUs: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU is not available.")

def GenerateTTS():
    # download and load all models
    preload_models()

    # generate audio from text
    text_prompt = """
    Oh, Sensei! Blue Archive is a super fun game where you can meet cute and powerful high school girls like me! We team up to take on all sorts of adventures together!
    """
    audio_array = generate_audio(text_prompt)
    write_wav("bark_generation.wav", SAMPLE_RATE, audio_array)
    return "bark_generation.wav"

def check_audio_file_exists(filepath):
    return os.path.isfile(filepath)

def load_audio(file_path, sample_rate):
    data, sr = sf.read(file_path, dtype='float32')
    if sr != sample_rate:
        raise ValueError(f"Sample rate of {file_path} is {sr}, expected {sample_rate}")
    return data

def resample_audio(file_path, target_sample_rate):
    data, sr = sf.read(file_path, dtype='float32')
    if sr != target_sample_rate:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sample_rate)
        sf.write(file_path, data, target_sample_rate)
    return file_path

class CustomVC(VC):
    def load_audio(self, input_audio_path, target_sr):
        data = load_audio(input_audio_path, target_sr)
        return data

def main():
    # Directly set the index_root path
    index_root = "./"
    
    # Ensure the directory exists
    if not os.path.isdir(index_root):
        print(f"Error: The directory {index_root} does not exist.")
        return

    # Set the environment variable
    os.environ["index_root"] = index_root

    # Check if the audio file already exists
    audio_file = "bark_generation.wav"
    if not check_audio_file_exists(audio_file):
        # Generate TTS audio if the file does not exist
        audio_file = GenerateTTS()
    
    # Check if the audio file was created successfully
    if not os.path.isfile(audio_file):
        print(f"Error: The audio file {audio_file} was not created.")
        return

    # Resample audio to 16000 Hz if necessary
    try:
        resampled_audio_file = resample_audio(audio_file, 16000)
    except Exception as e:
        print(f"Error during resampling: {e}")
        return
    
    # Check if the audio file can be opened and read
    try:
        data = load_audio(resampled_audio_file, 16000)
        if data is None or len(data) == 0:
            print(f"Error: The audio file {resampled_audio_file} is empty.")
            return
    except Exception as e:
        print(f"Error: Unable to read the audio file {resampled_audio_file}. Exception: {e}")
        return
    
    # Perform voice conversion
    vc = CustomVC()
    vc.get_vc("Arona_President/Arona_President.pth")
    
    try:
        # Use the overridden load_audio function
        input_audio_path = Path(resampled_audio_file)
        audio = vc.load_audio(input_audio_path, 16000)
        
        if audio is None:
            print("Error: Audio processing failed, audio is None.")
            return
        
        wavfile.write("Bluearchive.wav", 16000, audio)
        print("Voice conversion successful, output saved to Bluearchive.wav")
        
    except Exception as e:
        print(f"Error during voice conversion: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
