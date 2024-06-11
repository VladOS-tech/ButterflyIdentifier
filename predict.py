import tensorflow as tf
import numpy as np
import pandas as pd
import pickle


def predict():
    model = tf.keras.models.load_model('model/model.h5')

    with open('model/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    test_dataset = load_data_for_predicting()
    predictions = model.predict(test_dataset)
    predicted_classes = np.argmax(predictions, axis=-1)

    predicted_labels = label_encoder.inverse_transform(predicted_classes)
    print(predicted_labels)


def load_data_for_predicting():
    test_df = pd.read_csv('data/Testing_set.csv')
    base_path_test = 'data/test/'
    test_paths = test_df['filename'].apply(lambda x: base_path_test + x).values

    def load_and_preprocess_image(filepath):
        image_string = tf.io.read_file(filepath)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize(image, [150, 150])
        image = image / 255.0
        return image

    test_dataset = tf.data.Dataset.from_tensor_slices(test_paths)
    test_dataset = test_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(32)
    return test_dataset


if __name__ == "__main__":
    predict()
