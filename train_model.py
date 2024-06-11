from matplotlib import pyplot as plt

import data_preparation as dp
import model as mod


def main():
    train_df, test_df, label_encoder = dp.load_data()
    train_dataset, test_dataset = dp.create_datasets(train_df, test_df)

    model = mod.create_model(len(label_encoder.classes_))

    history = model.fit(
        train_dataset,
        epochs=10,
        validation_data=train_dataset.take(int(0.2 * len(train_df)))
    )

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])

    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')

    plt.legend(['Training'], ['Validation'])
    plt.grid()
    plt.show()
    (loss, accuracy) = model.evaluate(test_dataset)
    print(loss)
    print(accuracy)
    model.save('test_model/model.h5')


if __name__ == "__main__":
    main()
