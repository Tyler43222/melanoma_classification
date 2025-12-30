import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

EPOCHS = 30
IMG_SIZE = 100

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

    # Create a stratified validation split so val metrics are stable
    y_labels = np.argmax(y_train, axis=1)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=y_labels,
    )

    # Get a compiled neural network
    model = get_model(len(class_names))

    # Fit model on training data
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        min_delta=0.01,
        restore_best_weights=True,
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=64,
        validation_data=(x_val, y_val),
        callbacks=[early_stop],
        verbose=2,
    )

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Plot accuracy/loss across epochs
    epochs = range(1, len(history.history.get("loss", [])) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history.get("accuracy", []), label="train")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history.get("loss", []), label="train")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.close()

    # Confusion matrix on test set
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    print("\nClassification report (test set):")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

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
            res = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
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
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        # Convolutional layer
        tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),

        # Max-pooling layer, using 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Second Convolutional layer
        tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Third convolutional layer
        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Global pooling instead of flatten
        tf.keras.layers.GlobalAveragePooling2D(),

        # Add a hidden layer with dropout
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.4),

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
