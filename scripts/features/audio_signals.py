import librosa
import librosa.display
import matplotlib as plt
import numpy as np



def compute_spectrogram(noise, plot=False, sr=44100):
    D = librosa.stft(noise, n_fft=2048, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    if plot:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='log')
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectrogram")
        plt.show()

    return S_db

def compute_melspectrogram(noise, n_mels=128, plot=False, sr=44100):
    S = librosa.feature.melspectrogram(y=noise, sr=sr, n_fft=2048, hop_length=512, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)

    if plot:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
        plt.colorbar(format="%+2.0f dB")
        plt.title("Mel-Spectrogram")
        plt.show()

    return S_db

def compute_mfcc(noise, n_mfcc=13, plot=False, sr=44100):
    S_db = compute_melspectrogram(noise, plot=False, sr=sr)
    mfccs = librosa.feature.mfcc(S=S_db, n_mfcc=n_mfcc)

    if plot:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, sr=sr, x_axis='time')
        plt.colorbar()
        plt.title("MFCC")
        plt.show()

    return mfccs

def compute_chromagram(noise, plot=False, sr=44100):
    chroma = librosa.feature.chroma_stft(y=noise, sr=sr, n_fft=2048, hop_length=512)

    if plot:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(chroma, sr=sr, hop_length=512, x_axis='time', y_axis='chroma')
        plt.colorbar()
        plt.title("Chromagram")
        plt.show()

    return chroma

def get_time_segment(y, sr, start_time, end_time, feature_type, **kwargs):
    """Core function to extract time segment from any feature"""
    hop_length = kwargs.get('hop_length', 512)

    # Compute full feature
    if feature_type == 'spectrogram':
        D = librosa.stft(y, n_fft=kwargs.get('n_fft', 2048), hop_length=hop_length)
        feature = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    elif feature_type == 'melspectrogram':
        S = librosa.feature.melspectrogram(y=y, sr=sr,
                                         n_fft=kwargs.get('n_fft', 2048),
                                         hop_length=hop_length,
                                         n_mels=kwargs.get('n_mels', 128))
        feature = librosa.power_to_db(S, ref=np.max)
    elif feature_type == 'mfcc':
        S = librosa.feature.melspectrogram(y=y, sr=sr,
                                         n_fft=kwargs.get('n_fft', 2048),
                                         hop_length=hop_length)
        S_db = librosa.power_to_db(S, ref=np.max)
        feature = librosa.feature.mfcc(S=S_db, n_mfcc=kwargs.get('n_mfcc', 13))
    elif feature_type == 'chromagram':
        feature = librosa.feature.chroma_stft(y=y, sr=sr,
                                            n_fft=kwargs.get('n_fft', 2048),
                                            hop_length=hop_length)

    # Calculate frame range
    start_frame = int(start_time * sr / hop_length)
    end_frame = int(end_time * sr / hop_length)

    # Slice the feature
    feature_zoomed = feature[:, start_frame:end_frame]

    # Create time axis
    times = librosa.frames_to_time(np.arange(feature_zoomed.shape[1]),
                                 sr=sr,
                                 hop_length=hop_length)

    return feature_zoomed, times

def plot_zoomed_features(y, sr, start_time=1.0, end_time=2.0):
    """Plot all four zoomed features in subplots"""

    plt.figure(figsize=(15, 10))

    # Spectrogram
    plt.subplot(2, 2, 1)
    spec, times = get_time_segment(y, sr, start_time, end_time, 'spectrogram')
    librosa.display.specshow(spec, sr=sr, hop_length=512,
                           x_axis='time', y_axis='log',
                           x_coords=times)
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectrogram ({start_time}-{end_time}s)")

    # Mel Spectrogram
    plt.subplot(2, 2, 2)
    mel, times = get_time_segment(y, sr, start_time, end_time, 'melspectrogram')
    librosa.display.specshow(mel, sr=sr, hop_length=512,
                           x_axis='time', y_axis='mel',
                           x_coords=times)
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel Spectrogram ({start_time}-{end_time}s)")

    # MFCC
    plt.subplot(2, 2, 3)
    mfcc, times = get_time_segment(y, sr, start_time, end_time, 'mfcc')
    librosa.display.specshow(mfcc, sr=sr, hop_length=512,
                           x_axis='time', x_coords=times)
    plt.colorbar()
    plt.title(f"MFCC ({start_time}-{end_time}s)")

    # Chromagram
    plt.subplot(2, 2, 4)
    chroma, times = get_time_segment(y, sr, start_time, end_time, 'chromagram')
    librosa.display.specshow(chroma, sr=sr, hop_length=512,
                           x_axis='time', y_axis='chroma',
                           x_coords=times, cmap='coolwarm')
    plt.colorbar()
    plt.title(f"Chromagram ({start_time}-{end_time}s)")

    plt.tight_layout()
    plt.show()