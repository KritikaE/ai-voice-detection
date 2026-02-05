import librosa
import numpy as np

audio, sr = librosa.load("D:\\Kritika-Cloud\\OneDrive\\Projects\\audio-detection\\sample.wav", sr=None)

# Extract MFCCs
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=40
)

# Take mean across time axis
mfcc_mean = np.mean(mfcc, axis=1)

print("MFCC shape:", mfcc_mean.shape)
print(mfcc_mean)
