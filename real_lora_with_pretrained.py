# Real LoRA with Pre-trained Audio Model
# Using OpenAI Whisper features (available via openai-whisper) 
# or falling back to pre-trained sklearn models

import numpy as np
import librosa
from scipy.io import wavfile
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class RealPretrainedModel:
    """
    Uses actual pre-trained models for audio feature extraction
    Falls back to sophisticated classical features if deep models unavailable
    """
    
    def __init__(self):
        self.sr = 16000
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.is_fitted = False
        
        # Try to load OpenAI Whisper if available
        self.whisper_model = self._try_load_whisper()
        
    def _try_load_whisper(self):
        """Try to load OpenAI Whisper for feature extraction"""
        try:
            import whisper
            print("Loading OpenAI Whisper model...")
            model = whisper.load_model("base")
            print("✅ Whisper model loaded successfully!")
            return model
        except ImportError:
            print("⚠️ OpenAI Whisper not available, using classical features")
            return None
        except Exception as e:
            print(f"⚠️ Could not load Whisper: {e}")
            return None
    
    def extract_whisper_features(self, audio):
        """Extract features using OpenAI Whisper encoder"""
        if self.whisper_model is None:
            return None
            
        try:
            # Whisper expects 30-second chunks
            chunk_size = 30 * self.sr
            features = []
            
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    # Pad the last chunk
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                
                # Use Whisper encoder to extract features
                chunk = whisper.pad_or_trim(chunk)
                mel = whisper.log_mel_spectrogram(chunk).to(self.whisper_model.device)
                
                with torch.no_grad():
                    encoded = self.whisper_model.encoder(mel.unsqueeze(0))
                    features.append(encoded.cpu().numpy().flatten())
            
            return np.array(features)
            
        except Exception as e:
            print(f"Whisper feature extraction failed: {e}")
            return None
    
    def extract_advanced_classical_features(self, audio):
        """Extract sophisticated classical audio features"""
        print("Extracting advanced classical features...")
        
        # Segment audio into overlapping windows
        window_size = 2048
        hop_length = 512
        
        features = []
        
        for i in range(0, len(audio) - window_size, hop_length):
            segment = audio[i:i + window_size]
            
            # Spectral features
            stft = librosa.stft(segment, n_fft=1024, hop_length=256)
            magnitude = np.abs(stft)
            
            # Mel-frequency cepstral coefficients (MFCCs)
            mfccs = librosa.feature.mfcc(y=segment, sr=self.sr, n_mfcc=13)
            
            # Mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(y=segment, sr=self.sr, n_mels=40)
            mel_features = librosa.power_to_db(mel_spec)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(segment)
            
            # RMS energy
            rms = librosa.feature.rms(y=segment)
            
            # Spectral features from STFT
            stft = librosa.stft(segment)
            magnitude = np.abs(stft)
            
            # Spectral statistics
            spectral_mean = np.mean(magnitude, axis=1)
            spectral_std = np.std(magnitude, axis=1)
            spectral_max = np.max(magnitude, axis=1)
            
            # Combine all features
            segment_features = np.concatenate([
                np.mean(mfccs, axis=1),           # 13 MFCC means
                np.std(mfccs, axis=1),            # 13 MFCC stds
                np.mean(mel_features, axis=1),    # 40 mel means
                np.std(mel_features, axis=1),     # 40 mel stds
                [np.mean(zcr), np.std(zcr)],      # 2 ZCR stats
                [np.mean(rms), np.std(rms)],      # 2 RMS stats
                spectral_mean[:20],               # 20 spectral means
                spectral_std[:20],                # 20 spectral stds
                spectral_max[:20]                 # 20 spectral maxes
            ])
            
            features.append(segment_features)
        
        return np.array(features)
    
    def extract_features(self, audio):
        """Extract features using best available model"""
        # Try Whisper first
        if self.whisper_model is not None:
            whisper_features = self.extract_whisper_features(audio)
            if whisper_features is not None:
                return whisper_features
        
        # Fall back to advanced classical features
        return self.extract_advanced_classical_features(audio)
    
    def fit_feature_processor(self, features):
        """Fit the feature processor (scaler + PCA)"""
        print("Fitting feature processor...")
        
        # Flatten features if needed
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        
        # Fit scaler and PCA
        scaled_features = self.scaler.fit_transform(features)
        self.pca.fit(scaled_features)
        self.is_fitted = True
        
        processed_features = self.pca.transform(scaled_features)
        print(f"Features processed: {features.shape} -> {processed_features.shape}")
        
        return processed_features
    
    def process_features(self, features):
        """Process features through fitted scaler and PCA"""
        if not self.is_fitted:
            raise ValueError("Feature processor not fitted! Call fit_feature_processor first.")
        
        # Flatten features if needed
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        
        scaled_features = self.scaler.transform(features)
        processed_features = self.pca.transform(scaled_features)
        
        return processed_features

class RealLoRAAdapter:
    """LoRA adapter for real pre-trained model features"""
    
    def __init__(self, feature_dim, rank=32):
        self.feature_dim = feature_dim
        self.rank = rank
        
        # LoRA matrices
        self.lora_A = np.random.normal(0, 0.02, (rank, feature_dim))
        self.lora_B = np.random.normal(0, 0.02, (feature_dim, rank))
        
        # Optimizer parameters
        self.learning_rate = 0.005
        self.momentum_A = np.zeros_like(self.lora_A)
        self.momentum_B = np.zeros_like(self.lora_B)
        self.beta = 0.9
        
        # Training history
        self.losses = []
    
    def forward(self, features):
        """Apply LoRA adaptation"""
        delta_features = np.dot(self.lora_B, np.dot(self.lora_A, features.T)).T
        return features + delta_features
    
    def compute_loss(self, adapted_features, target_features):
        """Compute MSE loss with regularization"""
        mse_loss = np.mean((adapted_features - target_features) ** 2)
        
        # Add L2 regularization
        l2_reg = 0.001 * (np.sum(self.lora_A ** 2) + np.sum(self.lora_B ** 2))
        
        return mse_loss + l2_reg
    
    def update_weights(self, features, target_features):
        """Update LoRA weights using momentum-based gradient descent"""
        adapted_features = self.forward(features)
        
        # Compute gradients
        error = adapted_features - target_features
        batch_size = len(features)
        
        # Gradient w.r.t B
        grad_B = np.dot(error.T, np.dot(features, self.lora_A.T)) / batch_size
        grad_B += 0.001 * self.lora_B  # L2 regularization
        
        # Gradient w.r.t A
        grad_A = np.dot(self.lora_B.T, np.dot(error.T, features)) / batch_size
        grad_A += 0.001 * self.lora_A  # L2 regularization
        
        # Momentum update
        self.momentum_B = self.beta * self.momentum_B + (1 - self.beta) * grad_B
        self.momentum_A = self.beta * self.momentum_A + (1 - self.beta) * grad_A
        
        # Update weights
        self.lora_B -= self.learning_rate * self.momentum_B
        self.lora_A -= self.learning_rate * self.momentum_A
        
        # Compute and store loss
        loss = self.compute_loss(adapted_features, target_features)
        self.losses.append(loss)
        
        return loss

class RealLoRAAudioRestoration:
    """Real LoRA-based audio restoration system"""
    
    def __init__(self):
        self.pretrained_model = RealPretrainedModel()
        self.lora_adapter = None
        self.enhancement_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
    
    def baseline_restoration(self, noisy_audio_path, output_path):
        """Baseline restoration using pre-trained model"""
        print("=" * 60)
        print("BASELINE: REAL PRE-TRAINED MODEL")
        print("=" * 60)
        
        # Load audio
        noisy_audio, sr = librosa.load(noisy_audio_path, sr=16000)
        print(f"Loaded audio: {len(noisy_audio)/16000:.1f}s")
        
        # Extract features
        print("Extracting features with pre-trained model...")
        features = self.pretrained_model.extract_features(noisy_audio)
        
        # Fit feature processor
        processed_features = self.pretrained_model.fit_feature_processor(features)
        
        # Apply basic enhancement based on features
        enhanced_audio = self.apply_feature_based_enhancement(noisy_audio, processed_features)
        
        # Save result
        enhanced_audio = enhanced_audio / np.max(np.abs(enhanced_audio)) * 0.95
        enhanced_audio_int16 = (enhanced_audio * 32767).astype(np.int16)
        wavfile.write(output_path, 16000, enhanced_audio_int16)
        
        print(f"Baseline restoration saved to: {output_path}")
        return enhanced_audio, processed_features
    
    def apply_feature_based_enhancement(self, audio, features):
        """Apply enhancement based on extracted features"""
        # Convert audio to spectrogram
        stft = librosa.stft(audio, n_fft=1024, hop_length=256)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Apply feature-guided enhancement
        enhanced_magnitude = magnitude.copy()
        
        # Use features to guide spectral enhancement
        for i in range(min(len(features), magnitude.shape[1])):
            # Use first few PCA components as enhancement weights
            enhancement_factor = 1 + 0.1 * np.tanh(features[i, :min(10, features.shape[1])])
            
            # Map to frequency bins
            freq_weights = np.interp(
                np.arange(magnitude.shape[0]),
                np.linspace(0, magnitude.shape[0]-1, len(enhancement_factor)),
                enhancement_factor
            )
            
            enhanced_magnitude[:, i] *= freq_weights
        
        # Reconstruct audio
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=256)
        
        return enhanced_audio
    
    def train_lora(self, clean_audio_path, noisy_audio_path, epochs=30):
        """Train LoRA adapter using clean reference"""
        print("=" * 60)
        print("REAL LORA TRAINING")
        print("=" * 60)
        
        # Load audio
        clean_audio, sr = librosa.load(clean_audio_path, sr=16000)
        noisy_audio, sr = librosa.load(noisy_audio_path, sr=16000)
        
        print(f"Clean audio: {len(clean_audio)/16000:.1f}s")
        print(f"Noisy audio: {len(noisy_audio)/16000:.1f}s")
        
        # Extract features
        print("Extracting features from both audio files...")
        clean_features = self.pretrained_model.extract_features(clean_audio)
        noisy_features = self.pretrained_model.extract_features(noisy_audio)
        
        print(f"Clean features shape: {clean_features.shape}")
        print(f"Noisy features shape: {noisy_features.shape}")
        
        # Process features
        if not self.pretrained_model.is_fitted:
            # Fit on combined features for better representation
            combined_features = np.vstack([clean_features, noisy_features])
            self.pretrained_model.fit_feature_processor(combined_features)
        
        clean_processed = self.pretrained_model.process_features(clean_features)
        noisy_processed = self.pretrained_model.process_features(noisy_features)
        
        # Initialize LoRA adapter
        feature_dim = clean_processed.shape[1]
        self.lora_adapter = RealLoRAAdapter(feature_dim=feature_dim, rank=64)
        
        # Create training pairs
        print("Creating training pairs...")
        min_len = min(len(clean_processed), len(noisy_processed))
        
        # Use overlapping segments for training
        segment_size = 32
        training_pairs = []
        
        for i in range(0, min_len - segment_size, segment_size // 2):
            noisy_segment = noisy_processed[i:i+segment_size]
            clean_segment = clean_processed[i:i+segment_size]
            training_pairs.append((noisy_segment, clean_segment))
        
        print(f"Created {len(training_pairs)} training pairs")
        
        # Training loop
        print(f"Training LoRA adapter for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_loss = 0
            for noisy_segment, clean_target in training_pairs:
                loss = self.lora_adapter.update_weights(noisy_segment, clean_target)
                epoch_loss += loss
            
            avg_loss = epoch_loss / len(training_pairs)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        self.is_trained = True
        print("LoRA training complete!")
        
        return self.lora_adapter.losses
    
    def lora_restoration(self, noisy_audio_path, output_path):
        """Restoration using LoRA-adapted features"""
        print("=" * 60)
        print("LORA-ADAPTED RESTORATION")
        print("=" * 60)
        
        if not self.is_trained:
            raise ValueError("LoRA adapter not trained!")
        
        # Load audio
        noisy_audio, sr = librosa.load(noisy_audio_path, sr=16000)
        
        # Extract and process features
        features = self.pretrained_model.extract_features(noisy_audio)
        processed_features = self.pretrained_model.process_features(features)
        
        # Apply LoRA adaptation
        adapted_features = self.lora_adapter.forward(processed_features)
        
        # Enhanced restoration using adapted features
        enhanced_audio = self.apply_feature_based_enhancement(noisy_audio, adapted_features)
        
        # Additional post-processing
        enhanced_audio = self.advanced_post_processing(enhanced_audio)
        
        # Save result
        enhanced_audio = enhanced_audio / np.max(np.abs(enhanced_audio)) * 0.95
        enhanced_audio_int16 = (enhanced_audio * 32767).astype(np.int16)
        wavfile.write(output_path, 16000, enhanced_audio_int16)
        
        print(f"LoRA restoration saved to: {output_path}")
        return enhanced_audio
    
    def advanced_post_processing(self, audio):
        """Advanced post-processing for better quality"""
        # Apply gentle noise reduction
        stft = librosa.stft(audio, n_fft=1024, hop_length=256)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise from quieter sections
        power = np.mean(magnitude ** 2, axis=0)
        noise_threshold = np.percentile(power, 25)  # Bottom 25% as noise
        noise_mask = power < noise_threshold
        
        if np.sum(noise_mask) > 0:
            noise_spectrum = np.mean(magnitude[:, noise_mask], axis=1, keepdims=True)
            
            # Gentle spectral subtraction
            alpha = 1.2
            beta = 0.2
            enhanced_magnitude = magnitude - alpha * noise_spectrum
            enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
        else:
            enhanced_magnitude = magnitude
        
        # Reconstruct
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=256)
        
        return enhanced_audio
    
    def evaluate_restoration(self, clean_path, noisy_path, baseline_audio, lora_audio):
        """Comprehensive evaluation of restoration quality"""
        print("=" * 60)
        print("RESTORATION EVALUATION")
        print("=" * 60)
        
        # Load reference
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
            # SNR
            noise = restored - reference
            signal_power = np.mean(reference ** 2)
            noise_power = np.mean(noise ** 2)
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
            
            # Correlation
            correlation = np.corrcoef(reference, restored)[0, 1]
            if np.isnan(correlation):
                correlation = 0
                
            # Spectral similarity
            ref_spec = np.abs(librosa.stft(reference))
            rest_spec = np.abs(librosa.stft(restored))
            spectral_corr = np.corrcoef(ref_spec.flatten(), rest_spec.flatten())[0, 1]
            if np.isnan(spectral_corr):
                spectral_corr = 0
            
            return snr, correlation, spectral_corr
        
        # Evaluate all versions
        original_snr, original_corr, original_spec = calculate_metrics(clean_audio, noisy_audio)
        baseline_snr, baseline_corr, baseline_spec = calculate_metrics(clean_audio, baseline_audio)
        lora_snr, lora_corr, lora_spec = calculate_metrics(clean_audio, lora_audio)
        
        # Print results
        print(f"ORIGINAL (Noisy):")
        print(f"  SNR: {original_snr:.2f} dB")
        print(f"  Correlation: {original_corr:.4f}")
        print(f"  Spectral Correlation: {original_spec:.4f}")
        print()
        
        print(f"BASELINE (Pre-trained only):")
        print(f"  SNR: {baseline_snr:.2f} dB ({baseline_snr - original_snr:+.2f} dB)")
        print(f"  Correlation: {baseline_corr:.4f} ({baseline_corr - original_corr:+.4f})")
        print(f"  Spectral Correlation: {baseline_spec:.4f} ({baseline_spec - original_spec:+.4f})")
        print()
        
        print(f"LORA-ADAPTED:")
        print(f"  SNR: {lora_snr:.2f} dB ({lora_snr - original_snr:+.2f} dB)")
        print(f"  Correlation: {lora_corr:.4f} ({lora_corr - original_corr:+.4f})")
        print(f"  Spectral Correlation: {lora_spec:.4f} ({lora_spec - original_spec:+.4f})")
        print()
        
        print(f"LORA vs BASELINE:")
        print(f"  SNR Improvement: {lora_snr - baseline_snr:+.2f} dB")
        print(f"  Correlation Improvement: {lora_corr - baseline_corr:+.4f}")
        print(f"  Spectral Improvement: {lora_spec - baseline_spec:+.4f}")
        
        # Plot results
        self.plot_training_and_results(lora_audio, baseline_audio, clean_audio, noisy_audio)
        
        return {
            'original': {'snr': original_snr, 'corr': original_corr, 'spec': original_spec},
            'baseline': {'snr': baseline_snr, 'corr': baseline_corr, 'spec': baseline_spec},
            'lora': {'snr': lora_snr, 'corr': lora_corr, 'spec': lora_spec}
        }
    
    def plot_training_and_results(self, lora_audio, baseline_audio, clean_audio, noisy_audio):
        """Plot training progress and audio comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        if self.lora_adapter and self.lora_adapter.losses:
            axes[0, 0].plot(self.lora_adapter.losses)
            axes[0, 0].set_title('LoRA Training Loss')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
        
        # Audio comparison
        time = np.arange(min(8000, len(clean_audio))) / 16000
        
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
        plt.savefig('real_lora_results.png', dpi=150)
        plt.show()

def main():
    # File paths
    clean_audio_path = "Music/Kimiko IshizakaGBVAria.mp3"
    noisy_audio_path = "Music/1920s_78rpm_recording.wav"
    baseline_output = "Music/real_baseline_restoration.wav"
    lora_output = "Music/real_lora_restoration.wav"
    
    # Initialize system
    restoration_system = RealLoRAAudioRestoration()
    
    # Step 1: Baseline restoration
    print("Step 1: Baseline restoration with real pre-trained features...")
    baseline_audio, baseline_features = restoration_system.baseline_restoration(
        noisy_audio_path, baseline_output
    )
    
    # Step 2: Train LoRA
    print("\nStep 2: Training LoRA on real pre-trained features...")
    training_losses = restoration_system.train_lora(
        clean_audio_path, noisy_audio_path, epochs=40
    )
    
    # Step 3: LoRA restoration
    print("\nStep 3: LoRA-adapted restoration...")
    lora_audio = restoration_system.lora_restoration(noisy_audio_path, lora_output)
    
    # Step 4: Evaluate
    print("\nStep 4: Evaluation...")
    results = restoration_system.evaluate_restoration(
        clean_audio_path, noisy_audio_path, baseline_audio, lora_audio
    )
    
    print("\n" + "="*60)
    print("REAL LORA EXPERIMENT COMPLETE!")
    print("="*60)
    print(f"Baseline: {baseline_output}")
    print(f"LoRA: {lora_output}")

if __name__ == "__main__":
    main()