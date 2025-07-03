# SNR Measurement Tool for Audio Restoration Quality Assessment

import numpy as np
import librosa
from scipy.io import wavfile
import matplotlib.pyplot as plt

class SNRMeasurement:
    """Tool for measuring Signal-to-Noise Ratio in audio restoration"""
    
    def __init__(self):
        self.sr = 16000
    
    def load_and_align_audio(self, audio1_path, audio2_path):
        """Load and align two audio files for comparison"""
        # Load audio files
        audio1, sr1 = librosa.load(audio1_path, sr=self.sr)
        audio2, sr2 = librosa.load(audio2_path, sr=self.sr)
        
        # Align lengths (trim to shorter one)
        min_len = min(len(audio1), len(audio2))
        audio1 = audio1[:min_len]
        audio2 = audio2[:min_len]
        
        # Normalize for comparison
        audio1 = audio1 / np.max(np.abs(audio1))
        audio2 = audio2 / np.max(np.abs(audio2))
        
        return audio1, audio2
    
    def calculate_snr(self, clean_signal, noisy_signal):
        """Calculate Signal-to-Noise Ratio"""
        # Calculate noise as difference
        noise = noisy_signal - clean_signal
        
        # Calculate power
        signal_power = np.mean(clean_signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        # Avoid division by zero
        if noise_power == 0:
            return float('inf')
        
        # SNR in dB
        snr_db = 10 * np.log10(signal_power / noise_power)
        
        return snr_db
    
    def calculate_mse(self, clean_signal, noisy_signal):
        """Calculate Mean Squared Error"""
        return np.mean((clean_signal - noisy_signal) ** 2)
    
    def calculate_correlation(self, clean_signal, noisy_signal):
        """Calculate correlation coefficient"""
        return np.corrcoef(clean_signal, noisy_signal)[0, 1]
    
    def comprehensive_analysis(self, clean_path, noisy_path, restored_path=None):
        """Perform comprehensive audio quality analysis"""
        print("=" * 60)
        print("AUDIO QUALITY ANALYSIS")
        print("=" * 60)
        
        # Load clean and noisy audio
        clean_audio, noisy_audio = self.load_and_align_audio(clean_path, noisy_path)
        
        # Calculate metrics for noisy audio
        noisy_snr = self.calculate_snr(clean_audio, noisy_audio)
        noisy_mse = self.calculate_mse(clean_audio, noisy_audio)
        noisy_corr = self.calculate_correlation(clean_audio, noisy_audio)
        
        print(f"BASELINE (78 RPM vs Original):")
        print(f"  SNR: {noisy_snr:.2f} dB")
        print(f"  MSE: {noisy_mse:.6f}")
        print(f"  Correlation: {noisy_corr:.4f}")
        print()
        
        # If restored audio is provided, analyze it too
        if restored_path:
            clean_audio_r, restored_audio = self.load_and_align_audio(clean_path, restored_path)
            
            # Calculate metrics for restored audio
            restored_snr = self.calculate_snr(clean_audio_r, restored_audio)
            restored_mse = self.calculate_mse(clean_audio_r, restored_audio)
            restored_corr = self.calculate_correlation(clean_audio_r, restored_audio)
            
            print(f"RESTORED (Restored vs Original):")
            print(f"  SNR: {restored_snr:.2f} dB")
            print(f"  MSE: {restored_mse:.6f}")
            print(f"  Correlation: {restored_corr:.4f}")
            print()
            
            # Calculate improvements
            snr_improvement = restored_snr - noisy_snr
            mse_improvement = (noisy_mse - restored_mse) / noisy_mse * 100
            corr_improvement = (restored_corr - noisy_corr) / abs(noisy_corr) * 100
            
            print(f"IMPROVEMENTS:")
            print(f"  SNR Improvement: {snr_improvement:.2f} dB")
            print(f"  MSE Improvement: {mse_improvement:.1f}%")
            print(f"  Correlation Improvement: {corr_improvement:.1f}%")
            print()
            
            return {
                'baseline': {'snr': noisy_snr, 'mse': noisy_mse, 'corr': noisy_corr},
                'restored': {'snr': restored_snr, 'mse': restored_mse, 'corr': restored_corr},
                'improvements': {'snr': snr_improvement, 'mse': mse_improvement, 'corr': corr_improvement}
            }
        
        return {
            'baseline': {'snr': noisy_snr, 'mse': noisy_mse, 'corr': noisy_corr}
        }
    
    def plot_waveforms(self, clean_path, noisy_path, restored_path=None):
        """Plot waveforms for visual comparison"""
        fig, axes = plt.subplots(3 if restored_path else 2, 1, figsize=(12, 8))
        
        # Load audio
        clean_audio, sr = librosa.load(clean_path, sr=self.sr)
        noisy_audio, sr = librosa.load(noisy_path, sr=self.sr)
        
        # Time axis
        time = np.arange(len(clean_audio)) / sr
        
        # Plot clean audio
        axes[0].plot(time, clean_audio, 'b-', alpha=0.7)
        axes[0].set_title('Original Clean Audio')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True)
        
        # Plot noisy audio
        axes[1].plot(time[:len(noisy_audio)], noisy_audio, 'r-', alpha=0.7)
        axes[1].set_title('78 RPM Noisy Audio')
        axes[1].set_ylabel('Amplitude')
        axes[1].grid(True)
        
        # Plot restored audio if available
        if restored_path:
            restored_audio, sr = librosa.load(restored_path, sr=self.sr)
            time_restored = np.arange(len(restored_audio)) / sr
            axes[2].plot(time_restored, restored_audio, 'g-', alpha=0.7)
            axes[2].set_title('Restored Audio')
            axes[2].set_ylabel('Amplitude')
            axes[2].set_xlabel('Time (s)')
            axes[2].grid(True)
        else:
            axes[1].set_xlabel('Time (s)')
        
        plt.tight_layout()
        plt.savefig('audio_comparison.png')
        plt.show()
    
    def plot_spectrograms(self, clean_path, noisy_path, restored_path=None):
        """Plot spectrograms for frequency domain comparison"""
        fig, axes = plt.subplots(3 if restored_path else 2, 1, figsize=(12, 10))
        
        # Load audio
        clean_audio, sr = librosa.load(clean_path, sr=self.sr)
        noisy_audio, sr = librosa.load(noisy_path, sr=self.sr)
        
        # Clean spectrogram
        D_clean = librosa.amplitude_to_db(np.abs(librosa.stft(clean_audio)), ref=np.max)
        img1 = librosa.display.specshow(D_clean, y_axis='hz', x_axis='time', ax=axes[0], sr=sr)
        axes[0].set_title('Original Clean Audio Spectrogram')
        plt.colorbar(img1, ax=axes[0], format='%+2.0f dB')
        
        # Noisy spectrogram
        D_noisy = librosa.amplitude_to_db(np.abs(librosa.stft(noisy_audio)), ref=np.max)
        img2 = librosa.display.specshow(D_noisy, y_axis='hz', x_axis='time', ax=axes[1], sr=sr)
        axes[1].set_title('78 RPM Noisy Audio Spectrogram')
        plt.colorbar(img2, ax=axes[1], format='%+2.0f dB')
        
        # Restored spectrogram if available
        if restored_path:
            restored_audio, sr = librosa.load(restored_path, sr=self.sr)
            D_restored = librosa.amplitude_to_db(np.abs(librosa.stft(restored_audio)), ref=np.max)
            img3 = librosa.display.specshow(D_restored, y_axis='hz', x_axis='time', ax=axes[2], sr=sr)
            axes[2].set_title('Restored Audio Spectrogram')
            plt.colorbar(img3, ax=axes[2], format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig('spectrogram_comparison.png')
        plt.show()

def main():
    # File paths
    clean_path = "Music/Kimiko IshizakaGBVAria.mp3"
    noisy_path = "Music/1920s_78rpm_recording.wav"
    restored_path = "Music/restored_classical_audio.wav"
    
    # Initialize SNR measurement tool
    snr_tool = SNRMeasurement()
    
    # Perform comprehensive analysis
    print("Measuring baseline SNR (78 RPM vs Original)...")
    results = snr_tool.comprehensive_analysis(clean_path, noisy_path, restored_path)
    
    # Create visualizations
    print("Creating waveform comparison...")
    snr_tool.plot_waveforms(clean_path, noisy_path, restored_path)
    
    print("Creating spectrogram comparison...")
    snr_tool.plot_spectrograms(clean_path, noisy_path, restored_path)
    
    print("Analysis complete! Check the generated plots.")

if __name__ == "__main__":
    main()