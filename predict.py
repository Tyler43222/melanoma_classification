import cv2
import numpy as np
import tensorflow as tf
import functools

IMG_SIZE = 100

# Use 2nd to last conv layer for Grad-CAM.
CONV_LAYER = 2


def analyze_image(uploaded_file):
    model = load_model()

    # Read uploaded file into numpy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode uploaded image")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0 
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    probs = pred[0]  # [Benign, Malignant]
    return (int(probs[0] * 100), int(probs[1] * 100))


@functools.lru_cache(maxsize=1)
def load_model():
    return tf.keras.models.load_model("model25.keras")


def guided_gradcam_png(image_bytes, target_class=None):
    """Return a Guided Grad-CAM visualization PNG (bytes) for an image.

    - image_bytes: raw bytes of a PNG/JPG
    - target_class: optional int class index; defaults to predicted class
    """
    model = load_model()

    decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Could not decode uploaded image")

    # Model input (fixed size)
    img_resized = cv2.resize(decoded, (IMG_SIZE, IMG_SIZE))
    img_float = img_resized.astype("float32") / 255.0
    input_tensor = tf.convert_to_tensor(img_float[None, ...])

    # pick class
    preds = model(input_tensor, training=False)[0]
    class_index = int(tf.argmax(preds).numpy()) if target_class is None else int(target_class)

    # Choose an earlier Conv2D layer for a higher-resolution Grad-CAM.
    conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]

    idx_from_end = int(CONV_LAYER)
    if idx_from_end < 1:
        idx_from_end = 1
    if idx_from_end > len(conv_layers):
        idx_from_end = len(conv_layers)

    target_conv_name = conv_layers[-idx_from_end].name

    # Grad-CAM heatmap
    grad_input = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = grad_input
    conv_output = None
    for layer in model.layers:
        if hasattr(layer, '_batch_input_shape') or isinstance(layer, tf.keras.layers.InputLayer):
            continue  # skip input layer
        x = layer(x)
        if layer.name == target_conv_name:
            conv_output = x

    grad_model = tf.keras.Model(grad_input, [conv_output, x])
    with tf.GradientTape() as tape:
        conv_out, predictions = grad_model(input_tensor, training=False)
        score = predictions[:, class_index]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)
    cam = tf.nn.relu(cam)[0].numpy()

    # Normalize with percentile clipping for better contrast.
    cam = np.maximum(cam, 0.0)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    p = float(np.percentile(cam, 99.0))
    if p <= 1e-12:
        cam_norm = np.zeros_like(cam, dtype=np.float32)
    else:
        cam_norm = np.clip(cam / p, 0.0, 1.0).astype(np.float32)

    # Create a colored heatmap and overlay on the image.
    heatmap_u8 = (cam_norm * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_resized.astype(np.uint8), 0.55, heatmap_color, 0.45, 0)

    ok, buf = cv2.imencode(".png", overlay)
    if not ok:
        raise RuntimeError("Could not encode Grad-CAM overlay image")
    return buf.tobytes()
   