# Fast Wavelet-based Audio Restoration
# Core concepts from multiresolution analysis + modern pattern learning

import numpy as np
import librosa
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pywt
import warnings
warnings.filterwarnings('ignore')

class FastWaveletRestoration:
    """Fast wavelet-based restoration inspired by your multiresolution approach"""
    
    def __init__(self, wavelet='db4', levels=4):
        self.wavelet = wavelet
        self.levels = levels
        self.sr = 16000
        self.clean_patterns = {}
        
    def wavelet_decompose(self, audio, max_length=30*16000):
        """Fast wavelet decomposition"""
        # Limit length for speed
        if len(audio) > max_length:
            audio = audio[:max_length]
        
        # Pad to power of 2 for efficiency
        padded_length = 2 ** int(np.ceil(np.log2(len(audio))))
        padded_audio = np.pad(audio, (0, padded_length - len(audio)), mode='edge')
        
        # Decompose
        coeffs = pywt.wavedec(padded_audio, self.wavelet, level=self.levels)
        return coeffs, len(audio)
    
    def wavelet_reconstruct(self, coeffs, original_length):
        """Reconstruct from coefficients"""
        reconstructed = pywt.waverec(coeffs, self.wavelet)
        return reconstructed[:original_length]
    
    def learn_clean_patterns(self, clean_audio):
        """Learn patterns from clean audio (like your adaptive filter learning)"""
        print("Learning clean audio patterns...")
        
        coeffs, _ = self.wavelet_decompose(clean_audio)
        
        # Learn statistics at each level
        for level, coeff in enumerate(coeffs):
            self.clean_patterns[level] = {
                'mean': np.mean(np.abs(coeff)),
                'std': np.std(coeff),
                'energy': np.sum(coeff ** 2),
                'sparsity_threshold': np.percentile(np.abs(coeff), 75),
                'noise_floor': np.percentile(np.abs(coeff), 10)
            }
        
        print(f"Learned patterns from {len(coeffs)} wavelet levels")
        return coeffs
    
    def adaptive_wavelet_filter(self, noisy_coeffs):
        """Adaptive filtering in wavelet domain (modern version of your approach)"""
        filtered_coeffs = []
        
        for level, coeff in enumerate(noisy_coeffs):
            if level in self.clean_patterns:
                pattern = self.clean_patterns[level]
                
                # Adaptive thresholding based on learned clean patterns
                threshold = pattern['sparsity_threshold'] * 0.3
                noise_floor = pattern['noise_floor']
                
                # Modern soft thresholding with learned parameters
                magnitude = np.abs(coeff)
                mask = magnitude > threshold
                
                # Keep strong coefficients, attenuate weak ones
                filtered_coeff = np.zeros_like(coeff)
                filtered_coeff[mask] = coeff[mask]
                
                # Gentle attenuation for weak coefficients
                weak_mask = (magnitude <= threshold) & (magnitude > noise_floor)
                filtered_coeff[weak_mask] = coeff[weak_mask] * 0.3
                
                filtered_coeffs.append(filtered_coeff)
            else:
                # Apply gentle filtering to unlearned levels
                filtered_coeffs.append(coeff * 0.8)
        
        return filtered_coeffs
    
    def spectral_enhancement(self, coeffs):
        """Spectral enhancement based on wavelet analysis"""
        enhanced_coeffs = []
        
        for level, coeff in enumerate(coeffs):
            if level in self.clean_patterns:
                pattern = self.clean_patterns[level]
                
                # Enhance coefficients that match clean patterns
                target_energy = pattern['energy'] / len(coeff)
                current_energy = np.sum(coeff ** 2) / len(coeff)
                
                if current_energy > 0:
                    enhancement_factor = np.sqrt(target_energy / current_energy)
                    enhancement_factor = np.clip(enhancement_factor, 0.5, 1.5)
                else:
                    enhancement_factor = 1.0
                
                enhanced_coeff = coeff * enhancement_factor
                enhanced_coeffs.append(enhanced_coeff)
            else:
                enhanced_coeffs.append(coeff)
        
        return enhanced_coeffs
    
    def restore_audio(self, clean_path, noisy_path, output_path):
        """Complete restoration pipeline"""
        print("=" * 50)
        print("FAST WAVELET RESTORATION")
        print("=" * 50)
        
        # Load audio
        print("Loading audio...")
        clean_audio, sr = librosa.load(clean_path, sr=self.sr)
        noisy_audio, sr = librosa.load(noisy_path, sr=self.sr)
        
        print(f"Clean: {len(clean_audio)/self.sr:.1f}s, Noisy: {len(noisy_audio)/self.sr:.1f}s")
        
        # Learn from clean audio
        self.learn_clean_patterns(clean_audio)
        
        # Process noisy audio
        print("Decomposing noisy audio...")
        noisy_coeffs, original_length = self.wavelet_decompose(noisy_audio)
        
        print("Applying adaptive filtering...")
        filtered_coeffs = self.adaptive_wavelet_filter(noisy_coeffs)
        
        print("Applying spectral enhancement...")
        enhanced_coeffs = self.spectral_enhancement(filtered_coeffs)
        
        print("Reconstructing audio...")
        restored_audio = self.wavelet_reconstruct(enhanced_coeffs, original_length)
        
        # Post-process
        restored_audio = np.clip(restored_audio, -1, 1)
        
        # Save
        restored_audio = restored_audio / np.max(np.abs(restored_audio)) * 0.95
        wavfile.write(output_path, self.sr, (restored_audio * 32767).astype(np.int16))
        
        print(f"Restored audio saved: {output_path}")
        
        return restored_audio, clean_audio, noisy_audio
    
    def evaluate_restoration(self, clean_audio, noisy_audio, restored_audio):
        """Evaluate results"""
        print("\n" + "=" * 50)
        print("EVALUATION")
        print("=" * 50)
        
        # Align lengths
        min_len = min(len(clean_audio), len(noisy_audio), len(restored_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]
        restored_audio = restored_audio[:min_len]
        
        # Calculate metrics
        def calc_snr_corr(ref, test):
            noise = test - ref
            signal_power = np.mean(ref ** 2)
            noise_power = np.mean(noise ** 2)
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
            corr = np.corrcoef(ref, test)[0, 1]
            return snr, corr if not np.isnan(corr) else 0
        
        orig_snr, orig_corr = calc_snr_corr(clean_audio, noisy_audio)
        rest_snr, rest_corr = calc_snr_corr(clean_audio, restored_audio)
        
        print(f"Original: SNR={orig_snr:.2f} dB, Correlation={orig_corr:.4f}")
        print(f"Restored: SNR={rest_snr:.2f} dB, Correlation={rest_corr:.4f}")
        print(f"Improvement: SNR={rest_snr-orig_snr:+.2f} dB, Correlation={rest_corr-orig_corr:+.4f}")
        
        # Wavelet domain analysis
        print(f"\nWavelet Analysis:")
        print(f"Used {self.levels} decomposition levels with {self.wavelet} wavelet")
        print(f"Learned patterns from {len(self.clean_patterns)} levels")
        
        # Plot results
        self.plot_results(clean_audio, noisy_audio, restored_audio)
        
        return {
            'snr_improvement': rest_snr - orig_snr,
            'correlation_improvement': rest_corr - orig_corr,
            'final_snr': rest_snr,
            'final_correlation': rest_corr
        }
    
    def plot_results(self, clean_audio, noisy_audio, restored_audio):
        """Plot comparison results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Time domain
        time = np.arange(min(4000, len(clean_audio))) / self.sr
        
        axes[0, 0].plot(time, clean_audio[:len(time)], 'b-', label='Clean')
        axes[0, 0].plot(time, noisy_audio[:len(time)], 'r-', alpha=0.7, label='Noisy')
        axes[0, 0].set_title('Original vs Noisy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(time, clean_audio[:len(time)], 'b-', label='Clean')
        axes[0, 1].plot(time, restored_audio[:len(time)], 'g-', alpha=0.7, label='Wavelet Restored')
        axes[0, 1].set_title('Clean vs Wavelet Restored')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Spectrograms
        D_clean = librosa.amplitude_to_db(np.abs(librosa.stft(clean_audio[:8000])))
        D_restored = librosa.amplitude_to_db(np.abs(librosa.stft(restored_audio[:8000])))
        
        im1 = axes[1, 0].imshow(D_clean, aspect='auto', origin='lower', cmap='viridis')
        axes[1, 0].set_title('Clean Spectrogram')
        
        im2 = axes[1, 1].imshow(D_restored, aspect='auto', origin='lower', cmap='viridis')
        axes[1, 1].set_title('Restored Spectrogram')
        
        plt.tight_layout()
        plt.savefig('fast_wavelet_results.png', dpi=150)
        plt.show()

def main():
    """Run fast wavelet restoration"""
    
    # Paths
    clean_path = "Music/Kimiko IshizakaGBVAria.mp3"
    noisy_path = "Music/1920s_78rpm_recording.wav"
    output_path = "Music/wavelet_restored.wav"
    
    # Run restoration
    restoration = FastWaveletRestoration(wavelet='db4', levels=4)
    
    restored_audio, clean_audio, noisy_audio = restoration.restore_audio(
        clean_path, noisy_path, output_path
    )
    
    # Evaluate
    results = restoration.evaluate_restoration(clean_audio, noisy_audio, restored_audio)
    
    print(f"\n🎯 WAVELET RESTORATION COMPLETE!")
    print(f"This approach builds on your multiresolution Fourier + adaptive Wiener work")
    print(f"but uses modern wavelets + pattern learning techniques!")
    
    return results

if __name__ == "__main__":
    main()