import librosa
import soundfile as sf
import numpy as np
import torch
import torchaudio.functional as F

# Load your clean audio (16 kHz standard)
audio, sr = librosa.load("clean_recording.wav", sr=16000)

# Add constant background hiss
hiss_level = 0.005
hiss = hiss_level * np.random.randn(len(audio))
audio_noisy = audio + hiss

# Add vinyl crackle/pop (sporadic clicks)
num_crackles = int(len(audio) / sr) * 5  # about 5 crackles per second
for _ in range(num_crackles):
    idx = np.random.randint(0, len(audio) - 20)
    audio_noisy[idx:idx+20] += np.random.uniform(-0.5, 0.5, size=20)

# Low-pass filter (simulate reduced fidelity)
audio_noisy = F.lowpass_biquad(torch.tensor(audio_noisy), sr, cutoff_freq=4000).numpy()

# Simulate slight pitch instability ("wow and flutter")
def wow_flutter(audio, sr, rate=0.5, depth=0.002):
    t = np.arange(len(audio)) / sr
    modulation = depth * np.sin(2 * np.pi * rate * t)
    indices = np.clip(np.arange(len(audio)) + modulation * sr, 0, len(audio)-1)
    return np.interp(indices, np.arange(len(audio)), audio)

audio_noisy = wow_flutter(audio_noisy, sr)

# Optional subtle distortion (tube-like warmth)
audio_noisy = np.tanh(audio_noisy * 1.5)

# Save the "old-style" audio
sf.write("old_analog_recording.wav", audio_noisy, sr)
