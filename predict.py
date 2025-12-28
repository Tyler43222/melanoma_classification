import cv2
import numpy as np
import sys
import tensorflow as tf

def analyze_image(uploaded_file):
    # Load the trained model
    try:
        model = tf.keras.models.load_model("model7.keras")
    except Exception as exc:
        message = str(exc)
        if "batch_input_shape" in message and "Conv2D" in message:
            raise RuntimeError(
                "Your saved model file (model7.keras) was created with an older Keras format and "
                "is not compatible with your current TensorFlow/Keras install. "
                "Re-train and re-save the model using the current environment (traffic.py), then try again."
            ) from exc
        raise

    # Read uploaded file into numpy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        sys.exit("Could not decode uploaded image")

    img = cv2.resize(img, (100, 100))
    img = img.astype("float32") / 255.0  # Normalize like training
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict
    pred = model.predict(img)
    probs = pred[0]  # [Benign, Malignant]
    return f"RESULTS:\nBenign: {int(probs[0] * 100)}%\nMalignant: {int(probs[1] * 100)}%"
   