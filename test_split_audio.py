import numpy as np
from retriever.embedders.denoisers import DemucsDenoiser, VoiceFixerDenoiser

def test_split_audio():
    # Create a simple test audio array
    sample_rate = 44100
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Create a simple sine wave with some noise
    audio_array = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
    
    print(f"Original audio shape: {audio_array.shape}")
    print(f"Original audio range: [{audio_array.min():.4f}, {audio_array.max():.4f}]")
    
    # Test DemucsDenoiser
    print("\n=== Testing DemucsDenoiser ===")
    try:
        demucs = DemucsDenoiser()
        denoised, noise = demucs.split_audio(audio_array)
        
        print(f"Denoised shape: {denoised.shape}")
        print(f"Denoised range: [{denoised.min():.4f}, {denoised.max():.4f}]")
        print(f"Noise shape: {noise.shape}")
        print(f"Noise range: [{noise.min():.4f}, {noise.max():.4f}]")
        
        # Check if noise is all zeros or very small
        noise_magnitude = np.abs(noise).mean()
        print(f"Noise magnitude (mean abs): {noise_magnitude:.6f}")
        
        if noise_magnitude < 1e-6:
            print("⚠️ WARNING: Noise is essentially zero!")
        else:
            print("✅ Noise has reasonable magnitude")
            
    except Exception as e:
        print(f"❌ DemucsDenoiser failed: {e}")
    
    # Test VoiceFixerDenoiser
    print("\n=== Testing VoiceFixerDenoiser ===")
    try:
        voicefixer = VoiceFixerDenoiser()
        denoised, noise = voicefixer.split_audio(audio_array, sr=sample_rate)
        
        print(f"Denoised shape: {denoised.shape}")
        print(f"Denoised range: [{denoised.min():.4f}, {denoised.max():.4f}]")
        print(f"Noise shape: {noise.shape}")
        print(f"Noise range: [{noise.min():.4f}, {noise.max():.4f}]")
        
        # Check if noise is all zeros or very small
        noise_magnitude = np.abs(noise).mean()
        print(f"Noise magnitude (mean abs): {noise_magnitude:.6f}")
        
        if noise_magnitude < 1e-6:
            print("⚠️ WARNING: Noise is essentially zero!")
        else:
            print("✅ Noise has reasonable magnitude")
            
    except Exception as e:
        print(f"❌ VoiceFixerDenoiser failed: {e}")

if __name__ == "__main__":
    test_split_audio() 