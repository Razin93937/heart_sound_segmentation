import soundfile as sf
import io
from flask import Flask, render_template, request, jsonify
import tensorflow.keras as keras
from tcn import TCN
import librosa
import numpy as np
import os

model = None
model_name = None


def load_model(new_model_name):
    global model, model_name
    if new_model_name != model_name:
        model_name = new_model_name
        model = keras.models.load_model(
            f"model/best_models/{new_model_name}", custom_objects={"TCN": TCN}
        )


sr = 2000


def plot_predict(y):
    size = 4000
    start = 0
    end = len(y)

    input = np.empty((0, size))

    while start + size < end:
        y_ = y[start : start + size]
        input = np.append(input, np.array([y_]), axis=0)
        start += size

    if start + size > end:
        input = np.append(input, np.array([y[end - size : end]]), axis=0)

    result = model.predict(input)

    pred = np.empty((0))

    for res in result[:-1, :]:
        pred = np.append(pred, np.repeat(np.argmax(res, axis=1), 16))

    if start + size > end:
        pred = np.append(
            pred,
            np.repeat(np.argmax(result[-1, :], axis=1), 16)[
                4000 - len(y) + len(pred) :
            ],
        )
    else:
        pred = np.append(pred, np.repeat(np.argmax(result[-1, :], axis=1), 16))

    s1_ts = np.empty((0, 2))
    s_ts = np.empty((0, 2))
    s2_ts = np.empty((0, 2))
    d_ts = np.empty((0, 2))

    ts = np.empty((0, 3))

    i = 1
    start = 0
    while i < len(pred):
        while (i < len(pred) - 1) and (pred[i] == pred[i - 1]):
            i += 1

        r = np.array([[start, i]]) / sr
        appender = 0
        if pred[i - 1] == 0:
            s1_ts = np.append(s1_ts, r, axis=0)
        elif pred[i - 1] == 1:
            if pred[i] == 0:
                d_ts = np.append(d_ts, r, axis=0)
                appender = 3
            else:
                s_ts = np.append(s_ts, r, axis=0)
                appender = 1
        else:
            s2_ts = np.append(s2_ts, r, axis=0)
            appender = 2

        ts = np.append(ts, np.array([[start / sr, i / sr, appender]]), axis=0)

        start = i
        i += 1

    return {
        "start": ts[:, 0].tolist(),
        "end": ts[:, 1].tolist(),
        "label": ts[:, 2].tolist(),
    }


app = Flask(__name__, static_folder="static")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.route("/")
def hello():
    return render_template("index.html")


@app.route("/models")
def get_models():
    models_dir = "model/best_models"
    if os.path.exists(models_dir):
        models = [f for f in os.listdir(models_dir) if f.endswith(".keras") or f.endswith(".h5")]
        return jsonify(models)
    return jsonify([])


@app.route("/upload", methods=["GET", "POST"])
def upload():
    global model
    if request.method == "POST":
        file = request.files["audio"]
        if file:
            model_name = request.form.get("model")
            load_model(model_name)
            if model is None:
                return jsonify(
                    {
                        "error": "Model not found on server. Please place a trained model at models/model.h5"
                    }
                ), 503
            tmp = io.BytesIO(file.read())
            data, sr_ = sf.read(tmp)
            data = librosa.resample(data, orig_sr=sr_, target_sr=sr)
            print(sr_)
            return jsonify(plot_predict(data))

    return ("", 204)


if __name__ == "__main__":
    app.run(debug=True, port=5555)
