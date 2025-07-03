# LoRA-based Audio Restoration using Pre-trained Models
# Fine-tune pre-trained audio models with Low-Rank Adaptation

import numpy as np
import librosa
from scipy.io import wavfile
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class LoRAAdapter:
    """
    Simplified LoRA implementation for audio restoration
    Adapts pre-trained model features using low-rank matrices
    """
    
    def __init__(self, feature_dim=512, rank=16):
        self.feature_dim = feature_dim
        self.rank = rank
        
        # LoRA matrices: W = W_0 + B*A (where B and A are low-rank)
        self.lora_A = np.random.normal(0, 0.1, (rank, feature_dim))
        self.lora_B = np.random.normal(0, 0.1, (feature_dim, rank))
        
        # Learning rate and momentum for optimization
        self.learning_rate = 0.01
        self.momentum_A = np.zeros_like(self.lora_A)
        self.momentum_B = np.zeros_like(self.lora_B)
        self.beta = 0.9  # Momentum coefficient
    
    def forward(self, features):
        """Apply LoRA adaptation to features"""
        # Original features + low-rank adaptation
        delta_features = np.dot(self.lora_B, np.dot(self.lora_A, features.T)).T
        return features + delta_features
    
    def compute_loss(self, adapted_features, target_features):
        """Compute loss between adapted and target features"""
        return np.mean((adapted_features - target_features) ** 2)
    
    def update_weights(self, features, target_features):
        """Update LoRA weights using gradient descent"""
        adapted_features = self.forward(features)
        
        # Compute gradients (simplified)
        error = adapted_features - target_features
        
        # Gradient w.r.t B
        grad_B = np.dot(error.T, np.dot(features, self.lora_A.T)) / len(features)
        
        # Gradient w.r.t A  
        grad_A = np.dot(self.lora_B.T, np.dot(error.T, features)) / len(features)
        
        # Momentum update
        self.momentum_B = self.beta * self.momentum_B + (1 - self.beta) * grad_B
        self.momentum_A = self.beta * self.momentum_A + (1 - self.beta) * grad_A
        
        # Update weights
        self.lora_B -= self.learning_rate * self.momentum_B
        self.lora_A -= self.learning_rate * self.momentum_A
        
        return self.compute_loss(adapted_features, target_features)

class PretrainedAudioModel:
    """
    Simulated pre-trained audio model
    In practice, this would be wav2vec2, HuBERT, etc.
    """
    
    def __init__(self, n_fft=2048, hop_length=512):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = 16000
        
        # Simulate pre-trained weights (in reality, these would be loaded)
        self.conv_weights = np.random.normal(0, 0.1, (128, 64))  # Fixed dimensions
        self.feature_weights = np.random.normal(0, 0.1, (128, 512))
    
    def extract_features(self, audio):
        """Extract features using simulated pre-trained model"""
        # Convert to spectrogram
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Simulate convolutional feature extraction
        # In reality, this would be the actual pre-trained model forward pass
        
        # Mel-scale features
        mel_spec = librosa.feature.melspectrogram(
            S=magnitude**2, sr=self.sr, n_mels=64
        )
        mel_features = librosa.power_to_db(mel_spec)
        
        # Simulate learned features
        features = []
        for i in range(mel_features.shape[1]):
            frame = mel_features[:, i]
            # Simulate neural network layers
            hidden = np.tanh(np.dot(self.conv_weights, frame))
            feature_vector = np.tanh(np.dot(self.feature_weights.T, hidden))
            features.append(feature_vector)
        
        return np.array(features)
    
    def enhance_audio(self, audio, features):
        """Enhance audio using learned features"""
        # Convert features back to spectral domain
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Apply feature-guided enhancement
        enhanced_magnitude = magnitude.copy()
        
        # Use features to guide enhancement
        for i, feature_vector in enumerate(features):
            if i < magnitude.shape[1]:
                # Feature-guided spectral shaping
                # Map 512 features to frequency bins
                freq_weights = np.interp(
                    np.arange(magnitude.shape[0]),
                    np.linspace(0, magnitude.shape[0]-1, len(feature_vector)),
                    feature_vector
                )
                enhancement_weights = 1 + 0.2 * np.tanh(freq_weights)
                enhanced_magnitude[:, i] *= enhancement_weights
        
        # Reconstruct audio
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.hop_length)
        
        return enhanced_audio

class LoRAAudioRestoration:
    """Complete LoRA-based audio restoration system"""
    
    def __init__(self):
        self.pretrained_model = PretrainedAudioModel()
        self.lora_adapter = None
        self.baseline_features = None
    
    def baseline_restoration(self, noisy_audio_path, output_path):
        """Baseline restoration using pre-trained model only"""
        print("=" * 60)
        print("BASELINE: PRE-TRAINED MODEL ONLY")
        print("=" * 60)
        
        # Load noisy audio
        noisy_audio, sr = librosa.load(noisy_audio_path, sr=16000)
        
        # Extract features
        print("Extracting features with pre-trained model...")
        features = self.pretrained_model.extract_features(noisy_audio)
        self.baseline_features = features
        
        # Enhance audio
        print("Enhancing audio...")
        enhanced_audio = self.pretrained_model.enhance_audio(noisy_audio, features)
        
        # Save baseline result
        enhanced_audio = enhanced_audio / np.max(np.abs(enhanced_audio)) * 0.95
        enhanced_audio_int16 = (enhanced_audio * 32767).astype(np.int16)
        wavfile.write(output_path, 16000, enhanced_audio_int16)
        
        print(f"Baseline restoration saved to: {output_path}")
        return enhanced_audio, features
    
    def lora_fine_tuning(self, clean_audio_path, noisy_audio_path, epochs=20):
        """Fine-tune with LoRA using clean reference"""
        print("=" * 60)
        print("LORA FINE-TUNING")
        print("=" * 60)
        
        # Load audio files
        clean_audio, sr = librosa.load(clean_audio_path, sr=16000)
        noisy_audio, sr = librosa.load(noisy_audio_path, sr=16000)
        
        print(f"Clean audio: {len(clean_audio)/16000:.1f}s")
        print(f"Noisy audio: {len(noisy_audio)/16000:.1f}s")
        
        # Extract features from both
        print("Extracting features...")
        clean_features = self.pretrained_model.extract_features(clean_audio)
        noisy_features = self.pretrained_model.extract_features(noisy_audio)
        
        # Initialize LoRA adapter
        feature_dim = clean_features.shape[1]
        self.lora_adapter = LoRAAdapter(feature_dim=feature_dim, rank=32)
        
        # Training data: align features by taking overlapping segments
        print("Preparing training data...")
        min_len = min(len(clean_features), len(noisy_features))
        clean_features = clean_features[:min_len]
        noisy_features = noisy_features[:min_len]
        
        # Create training pairs
        segment_size = 50  # 50 frames per segment
        training_pairs = []
        
        for i in range(0, min_len - segment_size, segment_size // 2):
            clean_segment = clean_features[i:i+segment_size]
            noisy_segment = noisy_features[i:i+segment_size]
            training_pairs.append((noisy_segment, clean_segment))
        
        print(f"Created {len(training_pairs)} training pairs")
        
        # LoRA training
        print("Training LoRA adapter...")
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for noisy_segment, clean_target in training_pairs:
                # Apply LoRA adaptation to noisy features
                loss = self.lora_adapter.update_weights(noisy_segment, clean_target)
                epoch_loss += loss
            
            avg_loss = epoch_loss / len(training_pairs)
            losses.append(avg_loss)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        print("LoRA training complete!")
        return losses
    
    def lora_restoration(self, noisy_audio_path, output_path):
        """Restoration using LoRA-adapted model"""
        print("=" * 60)
        print("LORA-ADAPTED RESTORATION")
        print("=" * 60)
        
        if self.lora_adapter is None:
            raise ValueError("LoRA adapter not trained! Run lora_fine_tuning first.")
        
        # Load noisy audio
        noisy_audio, sr = librosa.load(noisy_audio_path, sr=16000)
        
        # Extract features
        print("Extracting features...")
        noisy_features = self.pretrained_model.extract_features(noisy_audio)
        
        # Apply LoRA adaptation
        print("Applying LoRA adaptation...")
        adapted_features = self.lora_adapter.forward(noisy_features)
        
        # Enhance audio with adapted features
        print("Enhancing audio with adapted features...")
        enhanced_audio = self.pretrained_model.enhance_audio(noisy_audio, adapted_features)
        
        # Save result
        enhanced_audio = enhanced_audio / np.max(np.abs(enhanced_audio)) * 0.95
        enhanced_audio_int16 = (enhanced_audio * 32767).astype(np.int16)
        wavfile.write(output_path, 16000, enhanced_audio_int16)
        
        print(f"LoRA restoration saved to: {output_path}")
        return enhanced_audio, adapted_features
    
    def compare_results(self, clean_path, noisy_path, baseline_audio, lora_audio):
        """Compare baseline vs LoRA results"""
        print("=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        
        # Load reference audio
        clean_audio, sr = librosa.load(clean_path, sr=16000)
        noisy_audio, sr = librosa.load(noisy_path, sr=16000)
        
        # Align lengths
        min_len = min(len(clean_audio), len(noisy_audio), len(baseline_audio), len(lora_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]
        baseline_audio = baseline_audio[:min_len]
        lora_audio = lora_audio[:min_len]
        
        # Calculate metrics
        def calculate_metrics(reference, restored):
            noise = restored - reference
            signal_power = np.mean(reference ** 2)
            noise_power = np.mean(noise ** 2)
            
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
            correlation = np.corrcoef(reference, restored)[0, 1]
            
            return snr, correlation if not np.isnan(correlation) else 0
        
        # Baseline metrics
        baseline_snr, baseline_corr = calculate_metrics(clean_audio, baseline_audio)
        
        # LoRA metrics
        lora_snr, lora_corr = calculate_metrics(clean_audio, lora_audio)
        
        # Original metrics (for reference)
        original_snr, original_corr = calculate_metrics(clean_audio, noisy_audio)
        
        print(f"ORIGINAL (Noisy):")
        print(f"  SNR: {original_snr:.2f} dB")
        print(f"  Correlation: {original_corr:.4f}")
        print()
        
        print(f"BASELINE (Pre-trained only):")
        print(f"  SNR: {baseline_snr:.2f} dB")
        print(f"  Correlation: {baseline_corr:.4f}")
        print(f"  SNR Improvement: {baseline_snr - original_snr:.2f} dB")
        print()
        
        print(f"LORA-ADAPTED:")
        print(f"  SNR: {lora_snr:.2f} dB")
        print(f"  Correlation: {lora_corr:.4f}")
        print(f"  SNR Improvement: {lora_snr - original_snr:.2f} dB")
        print()
        
        print(f"LORA vs BASELINE:")
        print(f"  SNR Improvement: {lora_snr - baseline_snr:.2f} dB")
        print(f"  Correlation Improvement: {lora_corr - baseline_corr:.4f}")
        
        # Plot comparison
        self.plot_comparison(clean_audio, noisy_audio, baseline_audio, lora_audio)
        
        return {
            'original': {'snr': original_snr, 'corr': original_corr},
            'baseline': {'snr': baseline_snr, 'corr': baseline_corr},
            'lora': {'snr': lora_snr, 'corr': lora_corr}
        }
    
    def plot_comparison(self, clean_audio, noisy_audio, baseline_audio, lora_audio):
        """Plot comparison of all approaches"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time domain plots
        time = np.arange(min(5000, len(clean_audio))) / 16000
        
        axes[0, 0].plot(time, clean_audio[:len(time)], 'b-', alpha=0.7, label='Clean')
        axes[0, 0].plot(time, noisy_audio[:len(time)], 'r-', alpha=0.7, label='Noisy')
        axes[0, 0].set_title('Original vs Noisy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(time, clean_audio[:len(time)], 'b-', alpha=0.7, label='Clean')
        axes[0, 1].plot(time, baseline_audio[:len(time)], 'g-', alpha=0.7, label='Baseline')
        axes[0, 1].set_title('Clean vs Baseline Restoration')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(time, clean_audio[:len(time)], 'b-', alpha=0.7, label='Clean')
        axes[1, 0].plot(time, lora_audio[:len(time)], 'm-', alpha=0.7, label='LoRA')
        axes[1, 0].set_title('Clean vs LoRA Restoration')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(time, baseline_audio[:len(time)], 'g-', alpha=0.7, label='Baseline')
        axes[1, 1].plot(time, lora_audio[:len(time)], 'm-', alpha=0.7, label='LoRA')
        axes[1, 1].set_title('Baseline vs LoRA Restoration')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        for ax in axes.flat:
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
        
        plt.tight_layout()
        plt.savefig('lora_comparison.png', dpi=150)
        plt.show()

def main():
    # File paths
    clean_audio_path = "Music/Kimiko IshizakaGBVAria.mp3"
    noisy_audio_path = "Music/1920s_78rpm_recording.wav"
    baseline_output = "Music/baseline_pretrained_restoration.wav"
    lora_output = "Music/lora_adapted_restoration.wav"
    
    # Initialize LoRA restoration system
    restoration_system = LoRAAudioRestoration()
    
    # Step 1: Baseline restoration (pre-trained model only)
    print("Step 1: Baseline restoration...")
    baseline_audio, baseline_features = restoration_system.baseline_restoration(
        noisy_audio_path, baseline_output
    )
    
    # Step 2: LoRA fine-tuning
    print("\nStep 2: LoRA fine-tuning...")
    losses = restoration_system.lora_fine_tuning(
        clean_audio_path, noisy_audio_path, epochs=25
    )
    
    # Step 3: LoRA-adapted restoration
    print("\nStep 3: LoRA-adapted restoration...")
    lora_audio, lora_features = restoration_system.lora_restoration(
        noisy_audio_path, lora_output
    )
    
    # Step 4: Compare results
    print("\nStep 4: Comparing results...")
    results = restoration_system.compare_results(
        clean_audio_path, noisy_audio_path, baseline_audio, lora_audio
    )
    
    print("\nLoRA Audio Restoration Experiment Complete!")
    print(f"Baseline result: {baseline_output}")
    print(f"LoRA result: {lora_output}")

if __name__ == "__main__":
    main()