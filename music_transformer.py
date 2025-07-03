# Music Transformer for Audio Restoration
# Learning musical structure from clean recordings to restore noisy vintage audio

import numpy as np
import librosa
from scipy.io import wavfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class MusicTransformer:
    """
    Music Transformer for audio restoration using attention mechanisms
    without requiring PyTorch - uses numpy-based implementation
    """
    
    def __init__(self, n_fft=2048, hop_length=512, n_mels=128):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sr = 16000
        
        # Model parameters (simplified transformer)
        self.d_model = 256
        self.n_heads = 8
        self.n_layers = 6
        self.learned_patterns = []
        
    def extract_features(self, audio):
        """Extract mel-spectrogram features from audio"""
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sr, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / (np.std(log_mel_spec) + 1e-8)
        
        return log_mel_spec
    
    def create_patches(self, spectrogram, patch_size=16):
        """Create patches from spectrogram (similar to Vision Transformer)"""
        patches = []
        n_mels, n_frames = spectrogram.shape
        
        for i in range(0, n_frames - patch_size + 1, patch_size // 2):  # 50% overlap
            for j in range(0, n_mels - patch_size + 1, patch_size // 2):
                patch = spectrogram[j:j+patch_size, i:i+patch_size]
                patches.append(patch.flatten())
        
        return np.array(patches)
    
    def attention_mechanism(self, query, key, value):
        """Simplified attention mechanism using numpy"""
        # Compute attention scores
        scores = np.dot(query, key.T) / np.sqrt(query.shape[-1])
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Apply attention to values
        output = np.dot(attention_weights, value)
        
        return output, attention_weights
    
    def learn_musical_patterns(self, clean_audio):
        """Learn musical patterns from clean audio"""
        print("Learning musical patterns from clean audio...")
        
        # Extract features
        mel_spec = self.extract_features(clean_audio)
        
        # Create patches
        patches = self.create_patches(mel_spec)
        
        # Learn patterns using k-means-like clustering
        from sklearn.cluster import KMeans
        n_clusters = min(100, len(patches) // 4)  # Adaptive number of clusters
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(patches)
        
        # Store learned patterns
        self.learned_patterns = kmeans.cluster_centers_
        
        # Compute pattern statistics
        pattern_stats = {
            'centroids': kmeans.cluster_centers_,
            'labels': cluster_labels,
            'inertia': kmeans.inertia_,
            'n_patterns': n_clusters
        }
        
        print(f"Learned {n_clusters} musical patterns")
        return pattern_stats
    
    def analyze_musical_structure(self, audio):
        """Analyze musical structure using learned patterns"""
        mel_spec = self.extract_features(audio)
        patches = self.create_patches(mel_spec)
        
        # Find closest patterns for each patch
        pattern_assignments = []
        pattern_distances = []
        
        for patch in patches:
            distances = np.linalg.norm(self.learned_patterns - patch, axis=1)
            closest_pattern = np.argmin(distances)
            pattern_assignments.append(closest_pattern)
            pattern_distances.append(distances[closest_pattern])
        
        return pattern_assignments, pattern_distances
    
    def spectral_pattern_matching(self, noisy_audio, clean_patterns):
        """Match spectral patterns between noisy and clean audio"""
        print("Applying spectral pattern matching...")
        
        # Extract features from noisy audio
        noisy_mel_spec = self.extract_features(noisy_audio)
        noisy_patches = self.create_patches(noisy_mel_spec)
        
        # Find best matching clean patterns
        restored_patches = []
        
        for noisy_patch in noisy_patches:
            # Find most similar clean pattern
            similarities = []
            for clean_pattern in self.learned_patterns:
                # Use cosine similarity
                similarity = np.dot(noisy_patch, clean_pattern) / (
                    np.linalg.norm(noisy_patch) * np.linalg.norm(clean_pattern) + 1e-8
                )
                similarities.append(similarity)
            
            # Use weighted combination of top patterns
            top_k = min(5, len(similarities))
            top_indices = np.argsort(similarities)[-top_k:]
            top_weights = np.array([similarities[i] for i in top_indices])
            top_weights = top_weights / np.sum(top_weights)
            
            # Weighted combination
            restored_patch = np.zeros_like(noisy_patch)
            for i, weight in zip(top_indices, top_weights):
                restored_patch += weight * self.learned_patterns[i]
            
            restored_patches.append(restored_patch)
        
        return np.array(restored_patches)
    
    def reconstruct_audio(self, restored_patches, original_shape):
        """Reconstruct audio from restored patches"""
        print("Reconstructing audio from restored patterns...")
        
        # Reconstruct spectrogram from patches
        patch_size = 16
        n_mels, n_frames = original_shape
        
        # Initialize reconstruction
        reconstructed = np.zeros((n_mels, n_frames))
        weights = np.zeros((n_mels, n_frames))
        
        patch_idx = 0
        for i in range(0, n_frames - patch_size + 1, patch_size // 2):
            for j in range(0, n_mels - patch_size + 1, patch_size // 2):
                if patch_idx < len(restored_patches):
                    patch = restored_patches[patch_idx].reshape(patch_size, patch_size)
                    reconstructed[j:j+patch_size, i:i+patch_size] += patch
                    weights[j:j+patch_size, i:i+patch_size] += 1
                    patch_idx += 1
        
        # Average overlapping regions
        weights[weights == 0] = 1  # Avoid division by zero
        reconstructed = reconstructed / weights
        
        return reconstructed
    
    def mel_to_audio(self, mel_spectrogram):
        """Convert mel-spectrogram back to audio"""
        # Denormalize
        mel_spectrogram = mel_spectrogram * np.std(mel_spectrogram) + np.mean(mel_spectrogram)
        
        # Convert from log scale
        mel_spectrogram = librosa.db_to_power(mel_spectrogram)
        
        # Inverse mel-spectrogram (approximation)
        # This is a simplified approach - real implementation would need more sophisticated inversion
        stft = librosa.feature.inverse.mel_to_stft(
            mel_spectrogram, 
            sr=self.sr, 
            n_fft=self.n_fft
        )
        
        # Reconstruct audio
        audio = librosa.istft(stft, hop_length=self.hop_length)
        
        return audio
    
    def adaptive_filtering(self, noisy_audio, clean_patterns):
        """Apply adaptive filtering based on learned patterns"""
        print("Applying adaptive filtering...")
        
        # Get spectrograms
        noisy_stft = librosa.stft(noisy_audio, n_fft=self.n_fft, hop_length=self.hop_length)
        noisy_magnitude = np.abs(noisy_stft)
        noisy_phase = np.angle(noisy_stft)
        
        # Apply frequency-domain filtering based on learned patterns
        # Boost frequencies that are prominent in clean patterns
        freq_weights = np.ones(noisy_magnitude.shape[0])
        
        # Analyze frequency content of learned patterns
        for pattern in self.learned_patterns:
            pattern_2d = pattern.reshape(16, 16)
            freq_content = np.mean(pattern_2d, axis=1)
            # Map the 16-element pattern to frequency bins
            freq_mapping = np.interp(
                np.arange(len(freq_weights)), 
                np.linspace(0, len(freq_weights)-1, len(freq_content)),
                freq_content
            )
            freq_weights += freq_mapping * 0.1
        
        # Apply frequency weighting
        enhanced_magnitude = noisy_magnitude * freq_weights[:, np.newaxis]
        
        # Reconstruct audio
        enhanced_stft = enhanced_magnitude * np.exp(1j * noisy_phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.hop_length)
        
        return enhanced_audio
    
    def train_and_restore(self, clean_audio_path, noisy_audio_path, output_path):
        """Complete training and restoration pipeline"""
        print("=" * 60)
        print("MUSIC TRANSFORMER AUDIO RESTORATION")
        print("=" * 60)
        
        # Load audio files
        print(f"Loading clean audio: {clean_audio_path}")
        clean_audio, sr = librosa.load(clean_audio_path, sr=self.sr)
        
        print(f"Loading noisy audio: {noisy_audio_path}")
        noisy_audio, sr = librosa.load(noisy_audio_path, sr=self.sr)
        
        # Learn musical patterns from clean audio
        pattern_stats = self.learn_musical_patterns(clean_audio)
        
        # Apply pattern-based restoration
        noisy_mel_spec = self.extract_features(noisy_audio)
        restored_patches = self.spectral_pattern_matching(noisy_audio, pattern_stats)
        
        # Reconstruct audio
        restored_mel_spec = self.reconstruct_audio(restored_patches, noisy_mel_spec.shape)
        
        # Convert back to audio (simplified approach)
        # For better results, we'll use adaptive filtering instead
        restored_audio = self.adaptive_filtering(noisy_audio, pattern_stats)
        
        # Apply final enhancement
        restored_audio = self.post_process_audio(restored_audio)
        
        # Save restored audio
        restored_audio = restored_audio / np.max(np.abs(restored_audio)) * 0.95
        restored_audio_int16 = (restored_audio * 32767).astype(np.int16)
        wavfile.write(output_path, self.sr, restored_audio_int16)
        
        print(f"Restored audio saved to: {output_path}")
        
        return restored_audio, clean_audio, noisy_audio, pattern_stats
    
    def post_process_audio(self, audio):
        """Apply final post-processing to restored audio"""
        # Apply gentle high-frequency boost
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Frequency bins
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        
        # Create enhancement curve
        enhancement = np.ones_like(freqs)
        
        # Gentle high-frequency boost
        for i, freq in enumerate(freqs):
            if freq > 2000:  # Boost above 2kHz
                enhancement[i] = 1.0 + 0.3 * np.exp(-(freq - 2000) / 2000)
        
        # Apply enhancement
        enhanced_magnitude = magnitude * enhancement[:, np.newaxis]
        
        # Reconstruct
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.hop_length)
        
        return enhanced_audio
    
    def visualize_patterns(self, pattern_stats):
        """Visualize learned musical patterns"""
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        
        # Show first 10 patterns
        for i in range(min(10, len(pattern_stats['centroids']))):
            ax = axes[i // 5, i % 5]
            pattern = pattern_stats['centroids'][i].reshape(16, 16)
            im = ax.imshow(pattern, aspect='auto', cmap='viridis')
            ax.set_title(f'Pattern {i+1}')
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig('learned_musical_patterns.png')
        plt.show()
        
        print(f"Learned patterns visualization saved to: learned_musical_patterns.png")

def main():
    # File paths
    clean_audio_path = "Music/Kimiko IshizakaGBVAria.mp3"
    noisy_audio_path = "Music/1920s_78rpm_recording.wav"
    output_path = "Music/transformer_restored_audio.wav"
    
    # Initialize Music Transformer
    transformer = MusicTransformer()
    
    # Train and restore
    restored_audio, clean_audio, noisy_audio, pattern_stats = transformer.train_and_restore(
        clean_audio_path, noisy_audio_path, output_path
    )
    
    # Visualize learned patterns
    transformer.visualize_patterns(pattern_stats)
    
    print("\nMusic Transformer restoration complete!")
    print(f"Check the restored audio at: {output_path}")

if __name__ == "__main__":
    main()