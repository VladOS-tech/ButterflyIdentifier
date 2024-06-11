import tensorflow as tf
import numpy as np
import pickle


def load_model_and_encoder():
    model = tf.keras.models.load_model('model/model.h5')
    with open('model/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, label_encoder


def load_and_preprocess_image(filepath):
    image_string = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.resize(image, [150, 150])
    image = image / 255.0
    return np.expand_dims(image, axis=0)


def predict_image_class(filepath, model, label_encoder, threshold=0.5):
    image = load_and_preprocess_image(filepath)
    prediction = model.predict(image)
    max_prob = np.max(prediction)

    if max_prob < threshold:
        return "Unknown type or not butterfly", max_prob

    predicted_class_id = np.argmax(prediction, axis=-1)
    predicted_class = label_encoder.inverse_transform(predicted_class_id)
    return predicted_class[0], max_prob


if __name__ == "__main__":
    model, label_encoder = load_model_and_encoder()
    filepath = input("Enter path to butterfly image: ")
    predicted_class, confidence = predict_image_class(filepath, model, label_encoder)
    if predicted_class == "Unknown type or not butterfly":
        print(predicted_class)
    else:
        print(f"Predicted butterfly type: {predicted_class}, Confidence: {confidence:.2f}")
