import os
import numpy as np
import soundfile as sf
from tortoisetest import api, utils, preload_models, generate_audio  # replace with actual module

SAMPLE_RATE = 24000  # replace with the actual sample rate if different

def change_speed(audio, speed_multiplier):
    indices = np.round(np.arange(0, len(audio), speed_multiplier)).astype(int)
    indices = indices[indices < len(audio)]
    return audio[indices]

def write_wav(file_path, sample_rate, audio_data):
    sf.write(file_path, audio_data, sample_rate)

def load_audio_files_from_folder(folder_path):
    audio_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            audio_files.append(utils.audio.load_audio(file_path, 22050))  # Adjust the sample rate if needed
    return audio_files

def GenerateTTS(folder_path, speed_multiplier):
    # Load reference audio clips from folder
    reference_clips = load_audio_files_from_folder(folder_path)

    # Download and load all models
    preload_models()

    # Generate audio from text
    text_prompt = """
    ここでは先生がスケジュールを決められます！
    """
    tts = api.TextToSpeech(use_deepspeed=True, kv_cache=True, half=True)
    pcm_audio = tts.tts_with_preset(text_prompt, voice_samples=reference_clips, preset='fast')

    # Change speed
    pcm_audio = change_speed(pcm_audio, speed_multiplier)

    # Write the modified audio to file
    output_path = "BarkGeneration/tortoise.wav"
    write_wav(output_path, SAMPLE_RATE, pcm_audio)

    return output_path

# Usage
folder_path = 'C:/Users/User/Downloads/Vocal-20240529T163444Z-001/Vocal'  # Replace with the path to your folder containing .wav files
speed_multiplier = 1.5  # Adjust the speed multiplier as needed
GenerateTTS(folder_path, speed_multiplier)
