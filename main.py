from pathlib import Path
from dotenv import load_dotenv
from scipy.io import wavfile
from rvc.modules.vc.modules import VC
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio
import sounddevice as sd
import soundfile as sf
import torch
import os
import traceback
import librosa
import numpy as np
import time
import yaml

SAMPLE_RATE = 22050

def check_gpu():
    if torch.cuda.is_available():
        print(f"GPU is available. Number of GPUs: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU is not available.")

def read_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def GenerateTTS():
    # Download and load all models
    preload_models()

    # Generate audio from text
    text_prompt = """
    様々な授業やイベントが準備されているので、ご希望のスケジュールを選んでください！
    """
    audio_array = generate_audio(text_prompt, history_prompt='v2/ja_speaker_3')

    # Read configuration
    config_path = 'config.yaml'
    config = read_config(config_path)

    # Change speed
    speed_params = config['audio_speed']
    speed_multiplier = speed_params['speed']
    audio_array = change_speed(audio_array, speed_multiplier)

    # Write the modified audio to file
    output_path = "BarkGeneration/bark_generation.wav"
    write_wav(output_path, SAMPLE_RATE, audio_array)

    return output_path

def check_audio_file_exists(filepath):
    return os.path.isfile(filepath)

def load_audio(file_path, sample_rate):
    print(f"Loading audio file: {file_path}")
    data, sr = sf.read(file_path, dtype='float32')
    if sr != sample_rate:
        raise ValueError(f"Sample rate of {file_path} is {sr}, expected {sample_rate}")
    print(f"Loaded audio file: {file_path}")
    return data

def resample_audio(file_path, target_sample_rate):
    print(f"Resampling audio file: {file_path}")
    data, sr = sf.read(file_path, dtype='float32')
    if sr != target_sample_rate:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sample_rate)
        sf.write(file_path, data, target_sample_rate)
    print(f"Resampled audio file: {file_path}")
    return file_path

def CreateandCheck_Folder(folder_path):
    for each_path in folder_path:
        if not os.path.exists(each_path):
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created successfully.")
        else:
            print(f"Folder '{folder_path}' already exists.")

def find_first_x_file(folder_path,file_type):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(file_type):
                return os.path.join(root, file)
    return None

def change_speed(audio, speed_multiplier):
    indices = np.round(np.arange(0, len(audio), speed_multiplier)).astype(int)
    indices = indices[indices < len(audio)]
    return audio[indices]

def main():
    start_time = time.time()
    check_gpu()

    rmvpe_root = "./models"
    os.environ["rmvpe_root"] = rmvpe_root

    index_root = "./"
    if not os.path.isdir(index_root):
        print(f"Error: The directory {index_root} does not exist.")
        return

    folder = ['BarkGeneration','VocalOutput']
    CreateandCheck_Folder(folder)

    # Check if the audio file already exists
    audio_file = "BarkGeneration/bark_generation.wav"
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
    
    print("Starting voice conversion...")

    # Perform voice conversion
    vc = VC()
    print("Loading VC model...")
    vc.get_vc(find_first_x_file('Arona_President','.pth'))
    
    try:
        # Use the overridden load_audio function
        input_audio_path = Path(resampled_audio_file)
        hubert_path = "models/hubert_base.pt"
        # index_file = find_first_x_file('Arona_President','.index')

        print("Starting vc_inference...")
        config_path = 'config.yaml'
        config = read_config(config_path)
        vc_inference_params = config['vc_inference']
        tgt_sr, audio_opt, times, info = vc.vc_inference(
            sid=vc_inference_params['sid'],
            input_audio_path=input_audio_path,
            f0_up_key=vc_inference_params['f0_up_key'],
            f0_method=vc_inference_params['f0_method'],
            filter_radius=vc_inference_params['filter_radius'],
            resample_sr=vc_inference_params['resample_sr'],
            rms_mix_rate=vc_inference_params['rms_mix_rate'],
            protect=vc_inference_params['protect'],
            index_file=vc_inference_params['index_file'],
            index_rate=vc_inference_params['index_rate'],
            hubert_path=vc_inference_params['hubert_path']
        )
        
        print("vc_inference completed.")
        
        if info:
            print(f"Error during voice conversion: {info}")
        else:
            sf.write("VocalOutput/Bluearchive.wav", audio_opt, tgt_sr)
            print("Voice conversion successful, output saved to Bluearchive.wav")
        
    except Exception as e:
        print(f"Error during voice conversion: {e}")
        traceback.print_exc()
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()