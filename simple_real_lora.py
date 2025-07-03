# Simplified Real LoRA for Audio Restoration
# Robust implementation with proper error handling

import numpy as np
import librosa
from scipy.io import wavfile
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

class SimpleRealLoRA:
    """Simplified and robust LoRA implementation"""
    
    def __init__(self):
        self.sr = 16000
        self.scaler = RobustScaler()  # More robust to outliers
        self.pca = None
        self.lora_weights = None
        
    def extract_robust_features(self, audio, max_duration=30):
        """Extract robust audio features"""
        print("Extracting robust audio features...")
        
        # Limit duration for processing speed
        if len(audio) > max_duration * self.sr:
            audio = audio[:max_duration * self.sr]
        
        # Clean audio (remove NaN/inf values)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1, 1)  # Clip to valid range
        
        # Extract features in overlapping windows
        window_size = 4096
        hop_length = 2048
        features = []
        
        for i in range(0, len(audio) - window_size, hop_length):
            segment = audio[i:i + window_size]
            
            try:
                # MFCCs
                mfccs = librosa.feature.mfcc(y=segment, sr=self.sr, n_mfcc=13)
                mfcc_features = np.concatenate([np.mean(mfccs, axis=1), np.std(mfccs, axis=1)])
                
                # Mel spectrogram
                mel_spec = librosa.feature.melspectrogram(y=segment, sr=self.sr, n_mels=20)
                mel_features = np.concatenate([
                    np.mean(librosa.power_to_db(mel_spec), axis=1),
                    np.std(librosa.power_to_db(mel_spec), axis=1)
                ])
                
                # Zero crossing rate and RMS
                zcr = np.mean(librosa.feature.zero_crossing_rate(segment))
                rms = np.mean(librosa.feature.rms(y=segment))
                
                # Spectral rolloff and centroid (simplified)
                stft = librosa.stft(segment, n_fft=512, hop_length=256)
                magnitude = np.abs(stft)
                
                # Simple spectral features
                spectral_mean = np.mean(magnitude)
                spectral_std = np.std(magnitude)
                spectral_max = np.max(magnitude)
                
                # Combine features
                feature_vector = np.concatenate([
                    mfcc_features,    # 26 features
                    mel_features,     # 40 features  
                    [zcr, rms],       # 2 features
                    [spectral_mean, spectral_std, spectral_max]  # 3 features
                ])
                
                # Clean features
                feature_vector = np.nan_to_num(feature_vector, nan=0.0)
                features.append(feature_vector)
                
            except Exception as e:
                print(f"Warning: Feature extraction failed for segment {i}: {e}")
                # Add zero features if extraction fails
                features.append(np.zeros(71))  # 26+40+2+3 = 71 features
        
        return np.array(features)
    
    def fit_preprocessor(self, features):
        """Fit the preprocessing pipeline"""
        print(f"Fitting preprocessor on features shape: {features.shape}")
        
        # Remove any rows with all zeros or NaN
        valid_rows = ~np.all(features == 0, axis=1) & ~np.any(np.isnan(features), axis=1)
        features_clean = features[valid_rows]
        
        if len(features_clean) == 0:
            raise ValueError("No valid features found!")
        
        print(f"Using {len(features_clean)} valid feature vectors")
        
        # Fit scaler
        self.scaler.fit(features_clean)
        
        # Determine PCA components based on data
        n_components = min(20, features_clean.shape[1], features_clean.shape[0] - 1)
        self.pca = PCA(n_components=n_components)
        
        # Fit PCA
        scaled_features = self.scaler.transform(features_clean)
        self.pca.fit(scaled_features)
        
        processed_features = self.pca.transform(scaled_features)
        
        print(f"Features processed: {features.shape} -> {processed_features.shape}")
        print(f"Explained variance ratio: {np.sum(self.pca.explained_variance_ratio_):.3f}")
        
        return processed_features
    
    def process_features(self, features):
        """Process features through fitted pipeline"""
        if self.scaler is None or self.pca is None:
            raise ValueError("Preprocessor not fitted!")
        
        # Clean features
        features_clean = np.nan_to_num(features, nan=0.0)
        
        # Scale and transform
        scaled_features = self.scaler.transform(features_clean)
        processed_features = self.pca.transform(scaled_features)
        
        return processed_features
    
    def train_lora(self, clean_features, noisy_features, rank=16, epochs=50):
        """Train LoRA adapter"""
        print(f"\nTraining LoRA adapter...")
        print(f"Clean features: {clean_features.shape}")
        print(f"Noisy features: {noisy_features.shape}")
        
        # Align feature lengths
        min_len = min(len(clean_features), len(noisy_features))
        clean_features = clean_features[:min_len]
        noisy_features = noisy_features[:min_len]
        
        feature_dim = clean_features.shape[1]
        
        # Initialize LoRA matrices
        lora_A = np.random.normal(0, 0.01, (rank, feature_dim))
        lora_B = np.random.normal(0, 0.01, (feature_dim, rank))
        
        learning_rate = 0.001
        losses = []
        
        print(f"Training for {epochs} epochs...")
        
        # Create training batches
        batch_size = 32
        n_batches = len(clean_features) // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                batch_noisy = noisy_features[start_idx:end_idx]
                batch_clean = clean_features[start_idx:end_idx]
                
                # Forward pass: apply LoRA adaptation
                delta = np.dot(lora_B, np.dot(lora_A, batch_noisy.T)).T
                adapted_features = batch_noisy + delta
                
                # Compute loss
                loss = np.mean((adapted_features - batch_clean) ** 2)
                
                # Compute gradients (simplified)
                error = adapted_features - batch_clean
                
                # Gradient for B
                grad_B = np.dot(error.T, np.dot(batch_noisy, lora_A.T)) / batch_size
                
                # Gradient for A
                grad_A = np.dot(lora_B.T, np.dot(error.T, batch_noisy)) / batch_size
                
                # Update with small learning rate
                lora_B -= learning_rate * grad_B
                lora_A -= learning_rate * grad_A
                
                epoch_loss += loss
            
            avg_loss = epoch_loss / n_batches if n_batches > 0 else epoch_loss
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Store weights
        self.lora_weights = {'A': lora_A, 'B': lora_B}
        
        print("LoRA training complete!")
        return losses
    
    def apply_lora(self, features):
        """Apply trained LoRA adaptation"""
        if self.lora_weights is None:
            raise ValueError("LoRA not trained!")
        
        lora_A = self.lora_weights['A']
        lora_B = self.lora_weights['B']
        
        # Apply LoRA transformation
        delta = np.dot(lora_B, np.dot(lora_A, features.T)).T
        adapted_features = features + delta
        
        return adapted_features
    
    def features_to_audio_enhancement(self, original_audio, adapted_features):
        """Convert adapted features back to audio enhancement"""
        print("Converting features to audio enhancement...")
        
        # Clean audio
        original_audio = np.nan_to_num(original_audio, nan=0.0, posinf=0.0, neginf=0.0)
        original_audio = np.clip(original_audio, -1, 1)
        
        # Get spectrogram
        stft = librosa.stft(original_audio, n_fft=1024, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Use adapted features to create enhancement weights
        enhancement_weights = np.ones(magnitude.shape[1])
        
        for i in range(min(len(adapted_features), len(enhancement_weights))):
            # Use PCA components as enhancement factors
            feature_strength = np.mean(adapted_features[i])
            enhancement_factor = 1.0 + 0.2 * np.tanh(feature_strength)
            enhancement_weights[i] = enhancement_factor
        
        # Apply time-domain enhancement
        enhanced_magnitude = magnitude.copy()
        for i in range(min(len(enhancement_weights), magnitude.shape[1])):
            enhanced_magnitude[:, i] *= enhancement_weights[i]
        
        # Apply frequency-domain enhancement based on features
        freq_weights = np.ones(magnitude.shape[0])
        
        # Boost mid-high frequencies based on feature analysis
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=1024)
        for i, freq in enumerate(freqs):
            if 1000 <= freq <= 6000:  # Musical content range
                freq_weights[i] = 1.1
        
        enhanced_magnitude *= freq_weights[:, np.newaxis]
        
        # Reconstruct audio
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
        
        # Ensure output is finite and clean
        enhanced_audio = np.nan_to_num(enhanced_audio, nan=0.0, posinf=0.0, neginf=0.0)
        enhanced_audio = np.clip(enhanced_audio, -1, 1)
        
        return enhanced_audio
    
    def run_experiment(self, clean_path, noisy_path):
        """Run complete LoRA experiment"""
        print("=" * 60)
        print("SIMPLIFIED REAL LORA EXPERIMENT")
        print("=" * 60)
        
        # Load audio
        print("Loading audio files...")
        clean_audio, sr = librosa.load(clean_path, sr=self.sr)
        noisy_audio, sr = librosa.load(noisy_path, sr=self.sr)
        
        print(f"Clean audio: {len(clean_audio)/self.sr:.1f}s")
        print(f"Noisy audio: {len(noisy_audio)/self.sr:.1f}s")
        
        # Extract features
        clean_features = self.extract_robust_features(clean_audio)
        noisy_features = self.extract_robust_features(noisy_audio)
        
        # Fit preprocessor on combined data
        combined_features = np.vstack([clean_features, noisy_features])
        self.fit_preprocessor(combined_features)
        
        # Process features
        clean_processed = self.process_features(clean_features)
        noisy_processed = self.process_features(noisy_features)
        
        # Baseline restoration (no LoRA)
        print("\nBaseline restoration (no LoRA)...")
        baseline_audio = self.features_to_audio_enhancement(noisy_audio, noisy_processed)
        
        # Train LoRA
        losses = self.train_lora(clean_processed, noisy_processed)
        
        # LoRA restoration
        print("\nLoRA restoration...")
        adapted_features = self.apply_lora(noisy_processed)
        lora_audio = self.features_to_audio_enhancement(noisy_audio, adapted_features)
        
        # Save results
        baseline_path = "Music/simple_baseline_restoration.wav"
        lora_path = "Music/simple_lora_restoration.wav"
        
        # Normalize and save
        baseline_audio = baseline_audio / np.max(np.abs(baseline_audio)) * 0.95
        lora_audio = lora_audio / np.max(np.abs(lora_audio)) * 0.95
        
        wavfile.write(baseline_path, self.sr, (baseline_audio * 32767).astype(np.int16))
        wavfile.write(lora_path, self.sr, (lora_audio * 32767).astype(np.int16))
        
        print(f"\nBaseline saved: {baseline_path}")
        print(f"LoRA saved: {lora_path}")
        
        # Evaluate
        self.evaluate_results(clean_audio, noisy_audio, baseline_audio, lora_audio, losses)
        
        return baseline_audio, lora_audio, losses
    
    def evaluate_results(self, clean_audio, noisy_audio, baseline_audio, lora_audio, losses):
        """Evaluate restoration results"""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        
        # Align lengths
        min_len = min(len(clean_audio), len(noisy_audio), len(baseline_audio), len(lora_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]
        baseline_audio = baseline_audio[:min_len]
        lora_audio = lora_audio[:min_len]
        
        def calc_metrics(ref, test):
            noise = test - ref
            signal_power = np.mean(ref ** 2)
            noise_power = np.mean(noise ** 2)
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
            corr = np.corrcoef(ref, test)[0, 1] if not np.all(ref == 0) and not np.all(test == 0) else 0
            return snr, corr if not np.isnan(corr) else 0
        
        # Calculate metrics
        orig_snr, orig_corr = calc_metrics(clean_audio, noisy_audio)
        base_snr, base_corr = calc_metrics(clean_audio, baseline_audio)
        lora_snr, lora_corr = calc_metrics(clean_audio, lora_audio)
        
        print(f"Original (Noisy): SNR={orig_snr:.2f} dB, Corr={orig_corr:.4f}")
        print(f"Baseline:         SNR={base_snr:.2f} dB, Corr={base_corr:.4f}")
        print(f"LoRA:             SNR={lora_snr:.2f} dB, Corr={lora_corr:.4f}")
        print()
        print(f"Baseline improvement: SNR={base_snr-orig_snr:+.2f} dB, Corr={base_corr-orig_corr:+.4f}")
        print(f"LoRA improvement:     SNR={lora_snr-orig_snr:+.2f} dB, Corr={lora_corr-orig_corr:+.4f}")
        print(f"LoRA vs Baseline:     SNR={lora_snr-base_snr:+.2f} dB, Corr={lora_corr-base_corr:+.4f}")
        
        # Plot results
        self.plot_results(losses, clean_audio, noisy_audio, baseline_audio, lora_audio)
    
    def plot_results(self, losses, clean_audio, noisy_audio, baseline_audio, lora_audio):
        """Plot training and audio results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        axes[0, 0].plot(losses)
        axes[0, 0].set_title('LoRA Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('MSE Loss')
        axes[0, 0].grid(True)
        
        # Audio waveforms
        time = np.arange(min(8000, len(clean_audio))) / self.sr
        
        axes[0, 1].plot(time, clean_audio[:len(time)], 'b-', alpha=0.7, label='Clean')
        axes[0, 1].plot(time, noisy_audio[:len(time)], 'r-', alpha=0.7, label='Noisy')
        axes[0, 1].set_title('Original vs Noisy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(time, clean_audio[:len(time)], 'b-', alpha=0.7, label='Clean')
        axes[1, 0].plot(time, baseline_audio[:len(time)], 'g-', alpha=0.7, label='Baseline')
        axes[1, 0].set_title('Clean vs Baseline')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(time, clean_audio[:len(time)], 'b-', alpha=0.7, label='Clean')
        axes[1, 1].plot(time, lora_audio[:len(time)], 'm-', alpha=0.7, label='LoRA')
        axes[1, 1].set_title('Clean vs LoRA')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        for ax in axes.flat:
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
        
        plt.tight_layout()
        plt.savefig('simple_lora_results.png', dpi=150)
        plt.show()

def main():
    # File paths
    clean_path = "Music/Kimiko IshizakaGBVAria.mp3"
    noisy_path = "Music/1920s_78rpm_recording.wav"
    
    # Run experiment
    lora_system = SimpleRealLoRA()
    baseline_audio, lora_audio, losses = lora_system.run_experiment(clean_path, noisy_path)
    
    print("\nSimple Real LoRA experiment complete!")

if __name__ == "__main__":
    main()