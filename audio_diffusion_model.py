# Audio Diffusion Model for Noise Restoration
# Trains on clean audio to restore vintage 78 RPM recordings

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import warnings
warnings.filterwarnings('ignore')

class AudioDataset(Dataset):
    """Dataset for audio spectrograms"""
    def __init__(self, clean_audio_path, segment_length=2048, hop_length=512, n_fft=2048):
        self.clean_audio, self.sr = librosa.load(clean_audio_path, sr=16000)
        self.segment_length = segment_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        
        # Create overlapping segments
        self.segments = []
        for i in range(0, len(self.clean_audio) - segment_length, hop_length):
            segment = self.clean_audio[i:i + segment_length]
            if len(segment) == segment_length:
                self.segments.append(segment)
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        audio_segment = self.segments[idx]
        
        # Convert to spectrogram
        stft = librosa.stft(audio_segment, n_fft=self.n_fft, hop_length=self.hop_length//4)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Normalize magnitude
        magnitude = magnitude / (np.max(magnitude) + 1e-8)
        
        return torch.FloatTensor(magnitude), torch.FloatTensor(phase)

class UNet(nn.Module):
    """U-Net architecture for noise prediction"""
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.dec4 = self.upconv_block(1024, 512)
        self.dec3 = self.upconv_block(512, 256)
        self.dec2 = self.upconv_block(256, 128)
        self.dec1 = self.upconv_block(128, 64)
        
        # Final layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.MaxPool2d(2)(enc1))
        enc3 = self.enc3(nn.MaxPool2d(2)(enc2))
        enc4 = self.enc4(nn.MaxPool2d(2)(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(enc4))
        
        # Decoder with skip connections
        dec4 = self.dec4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        
        dec3 = self.dec3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        
        dec2 = self.dec2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        
        dec1 = self.dec1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        
        return self.final(dec1)

class AudioDiffusionModel:
    """Diffusion model for audio restoration"""
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Create noise schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Initialize U-Net model
        self.model = UNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"Using device: {self.device}")
    
    def forward_diffusion(self, x, t):
        """Add noise to clean audio (forward process)"""
        noise = torch.randn_like(x)
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod[t])
        
        # Add noise: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        return sqrt_alphas_cumprod * x + sqrt_one_minus_alphas_cumprod * noise, noise
    
    def reverse_diffusion(self, x, t):
        """Predict noise to remove (reverse process)"""
        return self.model(x.unsqueeze(1)).squeeze(1)
    
    def train(self, dataloader, epochs=100, learning_rate=1e-4):
        """Train the diffusion model"""
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        self.model.train()
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, (clean_spec, _) in enumerate(dataloader):
                clean_spec = clean_spec.to(self.device)
                
                # Random timestep for each sample
                t = torch.randint(0, self.timesteps, (clean_spec.shape[0],), device=self.device)
                
                # Forward diffusion (add noise)
                noisy_spec, noise = self.forward_diffusion(clean_spec, t)
                
                # Predict noise
                predicted_noise = self.reverse_diffusion(noisy_spec, t)
                
                # Calculate loss
                loss = criterion(predicted_noise, noise)
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}')
        
        return losses
    
    def denoise_audio(self, noisy_audio_path, output_path):
        """Restore noisy audio using trained model"""
        # Load noisy audio
        noisy_audio, sr = librosa.load(noisy_audio_path, sr=16000)
        
        # Convert to spectrogram
        stft = librosa.stft(noisy_audio, n_fft=2048, hop_length=128)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Normalize
        magnitude = magnitude / (np.max(magnitude) + 1e-8)
        
        # Convert to tensor
        magnitude_tensor = torch.FloatTensor(magnitude).unsqueeze(0).to(self.device)
        
        # Denoise using reverse diffusion
        self.model.eval()
        with torch.no_grad():
            # Start from noisy spectrogram and iteratively denoise
            denoised_spec = magnitude_tensor
            
            for t in reversed(range(0, self.timesteps, 50)):  # Sample every 50 steps
                t_tensor = torch.tensor([t], device=self.device)
                predicted_noise = self.reverse_diffusion(denoised_spec, t_tensor)
                
                # Remove predicted noise
                if t > 0:
                    alpha = self.alphas[t]
                    alpha_cumprod = self.alphas_cumprod[t]
                    alpha_cumprod_prev = self.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)
                    
                    # Denoising step
                    denoised_spec = (denoised_spec - predicted_noise * torch.sqrt(1 - alpha_cumprod) / torch.sqrt(alpha_cumprod)) / torch.sqrt(alpha)
        
        # Convert back to audio
        denoised_magnitude = denoised_spec.squeeze(0).cpu().numpy()
        
        # Reconstruct complex spectrogram
        denoised_stft = denoised_magnitude * np.exp(1j * phase)
        
        # Inverse STFT
        restored_audio = librosa.istft(denoised_stft, hop_length=128)
        
        # Save restored audio
        restored_audio = restored_audio / np.max(np.abs(restored_audio)) * 0.95
        restored_audio_int16 = (restored_audio * 32767).astype(np.int16)
        wavfile.write(output_path, 16000, restored_audio_int16)
        
        print(f"Restored audio saved to: {output_path}")
        return restored_audio

def main():
    # Paths
    clean_audio_path = "Music/Kimiko IshizakaGBVAria.mp3"
    noisy_audio_path = "Music/1920s_78rpm_recording.wav"
    output_path = "Music/restored_audio.wav"
    
    print("Setting up diffusion model for audio restoration...")
    
    # Create dataset
    dataset = AudioDataset(clean_audio_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"Dataset created with {len(dataset)} audio segments")
    
    # Initialize diffusion model
    diffusion_model = AudioDiffusionModel()
    
    # Train model
    print("Starting training...")
    losses = diffusion_model.train(dataloader, epochs=50)
    
    # Save model
    torch.save(diffusion_model.model.state_dict(), "audio_diffusion_model.pth")
    print("Model saved!")
    
    # Restore noisy audio
    print("Restoring noisy 78 RPM audio...")
    diffusion_model.denoise_audio(noisy_audio_path, output_path)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('training_loss.png')
    plt.show()
    
    print("Audio restoration complete!")

if __name__ == "__main__":
    main()