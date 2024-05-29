import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import os
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

def load_wav_files_from_folder(folder_path):
    wav_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            waveform, sample_rate = torchaudio.load(file_path)
            wav_files.append(waveform)
    return wav_files

tts = TextToSpeech()

text = "様々な授業やイベントが準備されているので、ご希望のスケジュールを選んでください！"

reference_clips = [load_audio(p, 22050) for p in load_wav_files_from_folder('C:/Users/User/Downloads/Vocal-20240529T163444Z-001')]
tts = TextToSpeech()
pcm_audio = tts.tts_with_preset(text, voice_samples=reference_clips, preset='fast')
