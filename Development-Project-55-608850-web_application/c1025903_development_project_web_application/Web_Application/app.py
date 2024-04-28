import tensorflow
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from keras.models import load_model
import librosa
import numpy as np
from PIL import Image
import seaborn
import librosa.display

# Create an instance of Flask
app = Flask(__name__, template_folder='templates')

# Predictions and their labels
class_arr = np.array(['wren', 'blackbird', 'robin', 'great tit'])

# Load the model
model = load_model('CNN_model.h5')
model.make_predict_function()

def create_spectrogram_for_display(audio_path):
    SAMPLE_RATE = 48000

    birdcall, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    hop_length = 512
    S = librosa.feature.melspectrogram(y=birdcall, sr=SAMPLE_RATE, n_fft=2048,
                                       n_mels=128)
    mel_spec = librosa.power_to_db(S, ref=np.max)
    # Normalize
    mel_spec -= mel_spec.min()
    mel_spec /= mel_spec.max()
    plt.figure(figsize=(20, 5))
    librosa.display.specshow(mel_spec,
                             hop_length=512,
                             cmap='viridis',
                             x_axis="time",
                             y_axis="mel",
                             sr=sr)
    plt.colorbar(format='%+0.1f dB')
    plt.savefig("static/spectrogram.png")

def create_spectrogram(audio_path):
    RANDOM_SEED = 1337
    SAMPLE_RATE = 32000
    FMIN = 500
    FMAX = 12500

    birdcall, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    hop_length = 1007
    S = librosa.feature.melspectrogram(y=birdcall, sr=SAMPLE_RATE, n_fft=1024, hop_length=hop_length,
                                       n_mels=48, fmin=FMIN, fmax=FMAX)
    mel_spec = librosa.power_to_db(S, ref=np.max)
    # Normalize
    mel_spec -= mel_spec.min()
    mel_spec /= mel_spec.max()
    im = Image.fromarray(mel_spec * 255.0).convert("L")
    create_spectrogram_for_display(audio_path)
    im = np.array(im, dtype='float32')
    im -= im.min()
    im /= im.max()
    im = np.expand_dims(im, -1)
    return im  # Add this line to complete the function
def predict_label(image):
    new = np.reshape(image, (48, 159))
    p = model.predict(np.expand_dims(new, axis=0))
    return class_arr[np.argmax(p, axis=-1)]
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        audio_path = "static/" + img.filename
        img.save(audio_path)

        p = predict_label(create_spectrogram(audio_path))

    return render_template("index.html", prediction=p, img_path="static/spectrogram.png")


if __name__ == '__main__':
    app.run(debug=True)
