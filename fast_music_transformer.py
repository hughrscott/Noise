# Fast Music Transformer for Audio Restoration
# Simplified and optimized version

import numpy as np
import librosa
from scipy.io import wavfile
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import warnings
warnings.filterwarnings('ignore')

class FastMusicTransformer:
    """Fast Music Transformer for audio restoration"""
    
    def __init__(self, n_fft=1024, hop_length=256, n_mels=64):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sr = 16000
        self.learned_patterns = None
        
    def extract_features(self, audio, max_duration=30):
        """Extract mel-spectrogram features (limit duration for speed)"""
        # Limit audio duration for faster processing
        if len(audio) > max_duration * self.sr:
            audio = audio[:max_duration * self.sr]
            
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sr, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels
        )
        
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec
    
    def create_feature_vectors(self, spectrogram, window_size=8):
        """Create feature vectors from spectrogram"""
        vectors = []
        n_mels, n_frames = spectrogram.shape
        
        # Create sliding window features
        for i in range(0, n_frames - window_size + 1, window_size // 2):
            window = spectrogram[:, i:i+window_size]
            # Flatten and add statistics
            features = np.concatenate([
                window.flatten(),
                np.mean(window, axis=1),  # Frequency means
                np.std(window, axis=1),   # Frequency stds
                np.mean(window, axis=0),  # Time means
                np.std(window, axis=0)    # Time stds
            ])
            vectors.append(features)
        
        return np.array(vectors)
    
    def learn_patterns(self, clean_audio):
        """Learn musical patterns using clustering"""
        print("Learning musical patterns...")
        
        # Extract features
        mel_spec = self.extract_features(clean_audio)
        feature_vectors = self.create_feature_vectors(mel_spec)
        
        # Use MiniBatchKMeans for speed
        n_clusters = min(50, len(feature_vectors) // 4)
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, 
            random_state=42, 
            batch_size=100
        )
        
        cluster_labels = kmeans.fit_predict(feature_vectors)
        self.learned_patterns = kmeans.cluster_centers_
        
        print(f"Learned {n_clusters} musical patterns from {len(feature_vectors)} features")
        
        return {
            'patterns': self.learned_patterns,
            'labels': cluster_labels,
            'n_patterns': n_clusters
        }
    
    def pattern_based_enhancement(self, noisy_audio):
        """Apply pattern-based enhancement"""
        print("Applying pattern-based enhancement...")
        
        # Extract features from noisy audio
        noisy_mel_spec = self.extract_features(noisy_audio)
        noisy_features = self.create_feature_vectors(noisy_mel_spec)
        
        # Find closest patterns and compute enhancement weights
        enhancement_weights = np.ones(noisy_mel_spec.shape[1])
        
        for i, noisy_feature in enumerate(noisy_features):
            # Find closest learned pattern
            distances = np.linalg.norm(self.learned_patterns - noisy_feature, axis=1)
            closest_pattern_idx = np.argmin(distances)
            closest_distance = distances[closest_pattern_idx]
            
            # Convert distance to enhancement weight
            # Closer to learned patterns = more enhancement
            max_distance = np.max(distances)
            if max_distance > 0:
                similarity = 1 - (closest_distance / max_distance)
                enhancement_weights[i] = 1 + similarity * 0.5
        
        return enhancement_weights
    
    def spectral_enhancement(self, audio, enhancement_weights):
        """Apply spectral enhancement based on pattern analysis"""
        # Get STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Apply enhancement weights across time
        if len(enhancement_weights) != magnitude.shape[1]:
            # Interpolate weights to match STFT frames
            original_indices = np.linspace(0, magnitude.shape[1]-1, len(enhancement_weights))
            new_indices = np.arange(magnitude.shape[1])
            enhancement_weights = np.interp(new_indices, original_indices, enhancement_weights)
        
        # Apply weights
        enhanced_magnitude = magnitude * enhancement_weights[np.newaxis, :]
        
        # Frequency-domain enhancements
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        freq_enhancement = np.ones_like(freqs)
        
        # Boost mid-high frequencies (typical music content)
        for i, freq in enumerate(freqs):
            if 1000 <= freq <= 8000:  # Musical content range
                freq_enhancement[i] = 1.2
            elif freq > 8000:  # High frequencies
                freq_enhancement[i] = 1.1
        
        enhanced_magnitude *= freq_enhancement[:, np.newaxis]
        
        # Reconstruct audio
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.hop_length)
        
        return enhanced_audio
    
    def noise_reduction(self, audio):
        """Apply gentle noise reduction"""
        # Simple spectral subtraction
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise from first 0.5 seconds
        noise_frames = int(0.5 * self.sr / self.hop_length)
        noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # Gentle spectral subtraction
        alpha = 1.5  # Subtraction factor
        beta = 0.1   # Floor factor
        
        enhanced_magnitude = magnitude - alpha * noise_spectrum
        enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
        
        # Reconstruct
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.hop_length)
        
        return enhanced_audio
    
    def restore_audio(self, clean_audio_path, noisy_audio_path, output_path):
        """Complete restoration pipeline"""
        print("=" * 50)
        print("FAST MUSIC TRANSFORMER RESTORATION")
        print("=" * 50)
        
        # Load audio
        print(f"Loading audio files...")
        clean_audio, sr = librosa.load(clean_audio_path, sr=self.sr)
        noisy_audio, sr = librosa.load(noisy_audio_path, sr=self.sr)
        
        # Learn patterns from clean audio
        pattern_stats = self.learn_patterns(clean_audio)
        
        # Apply pattern-based enhancement
        enhancement_weights = self.pattern_based_enhancement(noisy_audio)
        
        # Apply spectral enhancement
        print("Applying spectral enhancement...")
        enhanced_audio = self.spectral_enhancement(noisy_audio, enhancement_weights)
        
        # Apply noise reduction
        print("Applying noise reduction...")
        restored_audio = self.noise_reduction(enhanced_audio)
        
        # Normalize and save
        print("Saving restored audio...")
        restored_audio = restored_audio / np.max(np.abs(restored_audio)) * 0.95
        restored_audio_int16 = (restored_audio * 32767).astype(np.int16)
        wavfile.write(output_path, self.sr, restored_audio_int16)
        
        print(f"Restored audio saved to: {output_path}")
        
        return restored_audio, clean_audio, noisy_audio, pattern_stats
    
    def analyze_improvement(self, clean_audio, noisy_audio, restored_audio):
        """Analyze the improvement achieved"""
        # Align audio lengths
        min_len = min(len(clean_audio), len(noisy_audio), len(restored_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]
        restored_audio = restored_audio[:min_len]
        
        # Calculate metrics
        def calculate_snr(signal, noisy):
            noise = noisy - signal
            signal_power = np.mean(signal ** 2)
            noise_power = np.mean(noise ** 2)
            if noise_power == 0:
                return float('inf')
            return 10 * np.log10(signal_power / noise_power)
        
        def calculate_correlation(sig1, sig2):
            return np.corrcoef(sig1, sig2)[0, 1]
        
        # Original metrics
        original_snr = calculate_snr(clean_audio, noisy_audio)
        original_corr = calculate_correlation(clean_audio, noisy_audio)
        
        # Restored metrics
        restored_snr = calculate_snr(clean_audio, restored_audio)
        restored_corr = calculate_correlation(clean_audio, restored_audio)
        
        # Improvements
        snr_improvement = restored_snr - original_snr
        corr_improvement = restored_corr - original_corr
        
        print("\nRESULTS:")
        print(f"Original SNR: {original_snr:.2f} dB")
        print(f"Restored SNR: {restored_snr:.2f} dB")
        print(f"SNR Improvement: {snr_improvement:.2f} dB")
        print(f"Original Correlation: {original_corr:.4f}")
        print(f"Restored Correlation: {restored_corr:.4f}")
        print(f"Correlation Improvement: {corr_improvement:.4f}")
        
        return {
            'original_snr': original_snr,
            'restored_snr': restored_snr,
            'snr_improvement': snr_improvement,
            'original_corr': original_corr,
            'restored_corr': restored_corr,
            'corr_improvement': corr_improvement
        }
    
    def plot_comparison(self, clean_audio, noisy_audio, restored_audio):
        """Plot comparison of audio signals"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
        
        # Time domain plots
        time_clean = np.arange(len(clean_audio)) / self.sr
        time_noisy = np.arange(len(noisy_audio)) / self.sr
        time_restored = np.arange(len(restored_audio)) / self.sr
        
        axes[0, 0].plot(time_clean, clean_audio)
        axes[0, 0].set_title('Clean Audio (Reference)')
        axes[0, 0].set_ylabel('Amplitude')
        
        axes[1, 0].plot(time_noisy, noisy_audio, 'r')
        axes[1, 0].set_title('Noisy 78 RPM Audio')
        axes[1, 0].set_ylabel('Amplitude')
        
        axes[2, 0].plot(time_restored, restored_audio, 'g')
        axes[2, 0].set_title('Restored Audio')
        axes[2, 0].set_ylabel('Amplitude')
        axes[2, 0].set_xlabel('Time (s)')
        
        # Spectrograms
        D_clean = librosa.amplitude_to_db(np.abs(librosa.stft(clean_audio)), ref=np.max)
        D_noisy = librosa.amplitude_to_db(np.abs(librosa.stft(noisy_audio)), ref=np.max)
        D_restored = librosa.amplitude_to_db(np.abs(librosa.stft(restored_audio)), ref=np.max)
        
        librosa.display.specshow(D_clean, y_axis='hz', x_axis='time', ax=axes[0, 1], sr=self.sr)
        axes[0, 1].set_title('Clean Spectrogram')
        
        librosa.display.specshow(D_noisy, y_axis='hz', x_axis='time', ax=axes[1, 1], sr=self.sr)
        axes[1, 1].set_title('Noisy Spectrogram')
        
        librosa.display.specshow(D_restored, y_axis='hz', x_axis='time', ax=axes[2, 1], sr=self.sr)
        axes[2, 1].set_title('Restored Spectrogram')
        
        plt.tight_layout()
        plt.savefig('transformer_comparison.png', dpi=150)
        plt.show()

def main():
    # File paths
    clean_audio_path = "Music/Kimiko IshizakaGBVAria.mp3"
    noisy_audio_path = "Music/1920s_78rpm_recording.wav"
    output_path = "Music/fast_transformer_restored.wav"
    
    # Initialize Fast Music Transformer
    transformer = FastMusicTransformer()
    
    # Restore audio
    restored_audio, clean_audio, noisy_audio, pattern_stats = transformer.restore_audio(
        clean_audio_path, noisy_audio_path, output_path
    )
    
    # Analyze improvement
    results = transformer.analyze_improvement(clean_audio, noisy_audio, restored_audio)
    
    # Create comparison plots
    transformer.plot_comparison(clean_audio, noisy_audio, restored_audio)
    
    print("\nFast Music Transformer restoration complete!")

if __name__ == "__main__":
    main()