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
