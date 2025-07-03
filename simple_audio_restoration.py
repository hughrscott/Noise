# Simplified Audio Restoration using Spectral Subtraction and Wiener Filtering
# Alternative to diffusion model for vintage audio restoration

import numpy as np
import librosa
from scipy.io import wavfile
from scipy.signal import wiener, medfilt
import matplotlib.pyplot as plt

class SimpleAudioRestoration:
    """Simple audio restoration using classical signal processing techniques"""
    
    def __init__(self, n_fft=2048, hop_length=512):
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def estimate_noise_profile(self, audio, noise_duration=1.0):
        """Estimate noise profile from the beginning of the audio"""
        sr = 16000
        noise_samples = int(noise_duration * sr)
        noise_segment = audio[:noise_samples]
        
        # Get noise spectrum
        noise_stft = librosa.stft(noise_segment, n_fft=self.n_fft, hop_length=self.hop_length)
        noise_magnitude = np.abs(noise_stft)
        
        # Average noise spectrum across time
        noise_profile = np.mean(noise_magnitude, axis=1, keepdims=True)
        
        return noise_profile
    
    def spectral_subtraction(self, audio, noise_profile, alpha=2.0, beta=0.1):
        """Apply spectral subtraction to reduce noise"""
        # Get audio spectrum
        audio_stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        audio_magnitude = np.abs(audio_stft)
        audio_phase = np.angle(audio_stft)
        
        # Spectral subtraction
        enhanced_magnitude = audio_magnitude - alpha * noise_profile
        
        # Prevent negative values
        enhanced_magnitude = np.maximum(enhanced_magnitude, beta * audio_magnitude)
        
        # Reconstruct audio
        enhanced_stft = enhanced_magnitude * np.exp(1j * audio_phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.hop_length)
        
        return enhanced_audio
    
    def wiener_filter_restoration(self, audio, window_size=5):
        """Apply Wiener filtering for noise reduction"""
        # Apply Wiener filter to audio directly
        filtered_audio = wiener(audio, window_size)
        return filtered_audio
    
    def median_filter_crackle_removal(self, audio, kernel_size=3):
        """Remove crackles using median filtering"""
        # Apply median filter to remove impulsive noise (crackles)
        filtered_audio = medfilt(audio, kernel_size)
        return filtered_audio
    
    def frequency_domain_enhancement(self, audio, sr=16000):
        """Enhance frequency content lost in vintage recordings"""
        # Get spectrogram
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Frequency bins
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        
        # Boost high frequencies that were lost in 78 RPM recordings
        # Create enhancement curve
        enhancement = np.ones_like(freqs)
        
        # Boost frequencies above 3kHz (lost in 78 RPM)
        high_freq_mask = freqs > 3000
        enhancement[high_freq_mask] = 1.5
        
        # Reduce low frequency noise
        low_freq_mask = freqs < 100
        enhancement[low_freq_mask] = 0.8
        
        # Apply enhancement
        enhanced_magnitude = magnitude * enhancement[:, np.newaxis]
        
        # Reconstruct audio
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.hop_length)
        
        return enhanced_audio
    
    def restore_audio(self, clean_audio_path, noisy_audio_path, output_path):
        """Complete audio restoration pipeline"""
        print("Loading audio files...")
        
        # Load clean audio for reference
        clean_audio, sr = librosa.load(clean_audio_path, sr=16000)
        
        # Load noisy audio
        noisy_audio, sr = librosa.load(noisy_audio_path, sr=16000)
        
        print("Estimating noise profile...")
        # Estimate noise profile from noisy audio
        noise_profile = self.estimate_noise_profile(noisy_audio)
        
        print("Applying spectral subtraction...")
        # Apply spectral subtraction
        restored_audio = self.spectral_subtraction(noisy_audio, noise_profile)
        
        print("Applying Wiener filtering...")
        # Apply Wiener filtering
        restored_audio = self.wiener_filter_restoration(restored_audio)
        
        print("Removing crackles...")
        # Remove crackles with median filtering
        restored_audio = self.median_filter_crackle_removal(restored_audio)
        
        print("Enhancing frequency content...")
        # Enhance frequency content
        restored_audio = self.frequency_domain_enhancement(restored_audio)
        
        print("Normalizing and saving...")
        # Normalize and save
        restored_audio = restored_audio / np.max(np.abs(restored_audio)) * 0.95
        restored_audio_int16 = (restored_audio * 32767).astype(np.int16)
        wavfile.write(output_path, 16000, restored_audio_int16)
        
        print(f"Restored audio saved to: {output_path}")
        
        return restored_audio, clean_audio, noisy_audio
    
    def analyze_restoration(self, clean_audio, noisy_audio, restored_audio):
        """Analyze the quality of restoration"""
        print("\nAnalyzing restoration quality...")
        
        # Calculate SNR improvement
        def calculate_snr(signal, noise):
            signal_power = np.mean(signal ** 2)
            noise_power = np.mean(noise ** 2)
            return 10 * np.log10(signal_power / noise_power)
        
        # Align audio lengths
        min_len = min(len(clean_audio), len(noisy_audio), len(restored_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]
        restored_audio = restored_audio[:min_len]
        
        # Calculate noise (difference from clean)
        noisy_noise = noisy_audio - clean_audio
        restored_noise = restored_audio - clean_audio
        
        # SNR calculations
        original_snr = calculate_snr(clean_audio, noisy_noise)
        restored_snr = calculate_snr(clean_audio, restored_noise)
        snr_improvement = restored_snr - original_snr
        
        print(f"Original SNR: {original_snr:.2f} dB")
        print(f"Restored SNR: {restored_snr:.2f} dB")
        print(f"SNR Improvement: {snr_improvement:.2f} dB")
        
        return original_snr, restored_snr, snr_improvement
    
    def plot_spectrograms(self, clean_audio, noisy_audio, restored_audio):
        """Plot spectrograms for comparison"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Clean audio spectrogram
        D_clean = librosa.amplitude_to_db(np.abs(librosa.stft(clean_audio)), ref=np.max)
        librosa.display.specshow(D_clean, y_axis='hz', x_axis='time', ax=axes[0])
        axes[0].set_title('Clean Audio Spectrogram')
        
        # Noisy audio spectrogram
        D_noisy = librosa.amplitude_to_db(np.abs(librosa.stft(noisy_audio)), ref=np.max)
        librosa.display.specshow(D_noisy, y_axis='hz', x_axis='time', ax=axes[1])
        axes[1].set_title('Noisy Audio Spectrogram')
        
        # Restored audio spectrogram
        D_restored = librosa.amplitude_to_db(np.abs(librosa.stft(restored_audio)), ref=np.max)
        librosa.display.specshow(D_restored, y_axis='hz', x_axis='time', ax=axes[2])
        axes[2].set_title('Restored Audio Spectrogram')
        
        plt.tight_layout()
        plt.savefig('spectrogram_comparison.png')
        plt.show()

def main():
    # File paths
    clean_audio_path = "Music/Kimiko IshizakaGBVAria.mp3"
    noisy_audio_path = "Music/1920s_78rpm_recording.wav"
    output_path = "Music/restored_classical_audio.wav"
    
    print("Initializing Simple Audio Restoration System...")
    
    # Initialize restoration system
    restoration = SimpleAudioRestoration()
    
    # Restore audio
    restored_audio, clean_audio, noisy_audio = restoration.restore_audio(
        clean_audio_path, noisy_audio_path, output_path
    )
    
    # Analyze restoration quality
    original_snr, restored_snr, snr_improvement = restoration.analyze_restoration(
        clean_audio, noisy_audio, restored_audio
    )
    
    # Plot spectrograms
    restoration.plot_spectrograms(clean_audio, noisy_audio, restored_audio)
    
    print("\nAudio restoration complete!")
    print(f"Check the restored audio at: {output_path}")

if __name__ == "__main__":
    main()