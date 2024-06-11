from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import io
import base64

app = Flask(__name__)


def load_model_and_encoder():
    model = tf.keras.models.load_model('model/model.h5')
    with open('model/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, label_encoder


model, label_encoder = load_model_and_encoder()


def image_to_base64(image_data):
    return base64.b64encode(image_data).decode('utf-8')


def load_and_preprocess_image_from_memory(image_data):
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction='No file part', image_data=None)

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction='No selected file', image_data=None)

    try:
        image_data = file.read()
        base64_image = image_to_base64(image_data)
        image = load_and_preprocess_image_from_memory(image_data)
        prediction = model.predict(image)
        predicted_class_id = np.argmax(prediction, axis=-1)
        predicted_class = label_encoder.inverse_transform(predicted_class_id)
        max_probability = np.max(prediction)

        if max_probability < 0.5:
            return render_template('index.html', prediction='Unknown butterfly or no butterfly detected',
                                   image_data=base64_image)

        return render_template('index.html', prediction=predicted_class[0], image_data=base64_image)
    except Exception as e:
        return render_template('index.html', prediction=f'Error: {str(e)}', image_data=None)


if __name__ == '__main__':
    app.run(debug=True)
