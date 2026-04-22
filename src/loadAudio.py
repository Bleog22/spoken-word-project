import librosa
import librosa.display
import matplotlib.pyplot as plt

# Path to your audio file
file_path = "data/yes.wav"

# Load audio
audio, sr = librosa.load(file_path, sr=None)

print("Sample rate:", sr)
print("Audio shape:", audio.shape)

# Plot waveform
plt.figure()
plt.plot(audio)
plt.title("Waveform")
plt.show()

# Create spectrogram
spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
spectrogram_db = librosa.power_to_db(spectrogram, ref=max)

# Plot spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar()
plt.title("Mel Spectrogram")
plt.show()