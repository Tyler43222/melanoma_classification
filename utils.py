import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def generate_graphs(history, class_names, y_test, x_test, model):
    # Plot accuracy across epochs
    epochs = range(1, len(history.history.get("loss", [])) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history.get("accuracy", []), label="train")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Plot loss across epochs
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history.get("loss", []), label="train")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.close()

    # Confusion matrix on testing set
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()