import cv2
import numpy as np
import os
import sys
import tensorflow as tf

EPOCHS = 10
IMG_WIDTH = 100
IMG_HEIGHT = 100

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    data_root = sys.argv[1]
    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")
    if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
        sys.exit(
            "Expected data_directory to contain train/ and test/ folders."
        )

    # Load training/testing data from class-named folders (e.g., Benign/Malignant)
    x_train, y_train, class_names = load_data(train_dir)
    x_test, y_test, _ = load_data(test_dir, class_names=class_names)

    # Normalize pixel values to [0, 1] to match prediction-time preprocessing
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Get a compiled neural network
    model = get_model(len(class_names))

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir, class_names=None):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category (e.g.,
    Benign, Malignant). Inside each category directory will be some number
    of image files.

    Return tuple `(images, labels, class_names)`. `images` is an array of
    resized images (IMG_WIDTH x IMG_HEIGHT x 3). `labels` is a one-hot array
    of integer class labels. `class_names` is the sorted list of class folder
    names used to map labels consistently.
    """
    if class_names is None:
        class_names = sorted(
            name
            for name in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, name))
        )

    class_to_index = {name: i for i, name in enumerate(class_names)}
    images = []
    labels = []

    for class_name in class_names:
        category_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(category_path):
            continue
        for image_name in os.listdir(category_path):
            file_path = os.path.join(category_path, image_name)
            img = cv2.imread(file_path)
            if img is None:
                continue
            res = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            images.append(res)
            labels.append(class_to_index[class_name])

    labels = tf.keras.utils.to_categorical(labels, num_classes=len(class_names))
    return np.array(images), np.array(labels), class_names


def get_model(num_categories):
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `num_categories` units, one for each category.
    """
    # Create a convolutional neural network
    model = tf.keras.models.Sequential([

        # Define input explicitly
        tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

        # Convolutional layer
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.BatchNormalization(),

        # Max-pooling layer, using 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Second Convolutional layer
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Third convolutional layer
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Global pooling instead of flatten
        tf.keras.layers.GlobalAveragePooling2D(),

        # Add a hidden layer with dropout
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Add an output layer
        tf.keras.layers.Dense(num_categories, activation="softmax")
    ])

    # Train neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    main()
