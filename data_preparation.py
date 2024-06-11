import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle


def load_data():
    train_df = pd.read_csv('data/Training_set.csv')
    test_df = pd.read_csv('data/Testing_set.csv')
    label_encoder = LabelEncoder()
    train_df['label'] = label_encoder.fit_transform(train_df['label'])

    with open('model/test_label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    return train_df, test_df, label_encoder


def load_and_preprocess_image(filepath, label=None):
    image_string = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.resize(image, [150, 150])
    image = image / 255.0
    if label is not None:
        return image, label
    else:
        return image


def create_datasets(train_df, test_df, batch_size=32):
    base_path = 'data/train/'
    train_paths = train_df['filename'].apply(lambda x: base_path + x).values
    train_labels = train_df['label'].values
    train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_dataset = train_dataset.map(lambda x, y: load_and_preprocess_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)

    base_path_test = 'data/test/'
    test_paths = test_df['filename'].apply(lambda x: base_path_test + x).values
    test_dataset = tf.data.Dataset.from_tensor_slices(test_paths)
    test_dataset = test_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    train_dataset = train_dataset.shuffle(buffer_size=len(train_paths)).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, test_dataset
