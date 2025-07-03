# Analog Audio Effects Simulation - 1920s 78 RPM Record Style

import librosa
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

# Runtime adjustable parameters (default values provided)
hiss_level = 0.005
crackles_per_second = 5
lowpass_cutoff_freq = 4000
wow_rate = 0.5
wow_depth = 0.002
distortion_level = 1.5

# Load clean audio (supports MP3, WAV, etc.)
audio, sr = librosa.load("Music/AaronDunnGBVAria.mp3", sr=16000)

# Add constant background hiss
hiss = hiss_level * np.random.randn(len(audio))
audio_noisy = audio + hiss

# Add vinyl crackles and pops
num_crackles = int(len(audio) / sr) * crackles_per_second
for _ in range(num_crackles):
    idx = np.random.randint(0, len(audio) - 20)
    audio_noisy[idx:idx+20] += np.random.uniform(-0.5, 0.5, size=20)

# Apply 1920s-style frequency response (acoustic horn + mechanical limitations)
# First apply aggressive low-pass for 78 RPM characteristics
nyquist = sr / 2
low_cutoff = 3500 / nyquist
high_cutoff = 200 / nyquist
b_low, a_low = butter(5, low_cutoff, btype='low')
b_high, a_high = butter(3, high_cutoff, btype='high')
audio_noisy = filtfilt(b_low, a_low, audio_noisy)
audio_noisy = filtfilt(b_high, a_high, audio_noisy)

# Simulate wow and flutter (pitch instability)
def wow_flutter(audio, sr, rate=wow_rate, depth=wow_depth):
    t = np.arange(len(audio)) / sr
    modulation = depth * np.sin(2 * np.pi * rate * t)
    indices = np.clip(np.arange(len(audio)) + modulation * sr, 0, len(audio)-1)
    return np.interp(indices, np.arange(len(audio)), audio)

audio_noisy = wow_flutter(audio_noisy, sr)

# Apply subtle harmonic distortion (tube warmth)
audio_noisy = np.tanh(audio_noisy * distortion_level)

# Normalize and save the simulated 1920s 78 RPM audio
audio_noisy = audio_noisy / np.max(np.abs(audio_noisy)) * 0.95
# Convert to 16-bit int for WAV format
audio_int16 = (audio_noisy * 32767).astype(np.int16)
wavfile.write("Music/1920s_78rpm_recording.wav", int(sr), audio_int16)

print("Old analog-style audio generated and saved!")
