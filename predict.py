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

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0 
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    probs = pred[0]  # [Benign, Malignant]
    return (int(probs[0] * 100), int(probs[1] * 100))


@functools.lru_cache(maxsize=1)
def load_model():
    model = tf.keras.models.load_model("model25.keras")
    # force-build model by running a dummy inference
    dummy = tf.zeros((1, IMG_SIZE, IMG_SIZE, 3))
    model(dummy, training=False)
    return model


@functools.lru_cache(maxsize=1)
def load_guided_model():
    base_model = load_model()
    guided = _build_guided_model(base_model)
    _ = guided(tf.zeros((1, IMG_SIZE, IMG_SIZE, 3)), training=False)
    return guided


@tf.custom_gradient
def guided_relu(x):
    """ReLU with guided backpropagation: only propagate positive gradients."""
    def grad(dy):
        return dy * tf.cast(dy > 0, dtype=tf.float32) * tf.cast(x > 0, dtype=tf.float32)
    return tf.nn.relu(x), grad


def _build_guided_model(model):
    """Clone model but replace ReLU activations with guided ReLU."""
    # Clone architecture
    cloned = tf.keras.models.clone_model(model)

    # Build clone by running dummy inference
    dummy = tf.zeros((1, IMG_SIZE, IMG_SIZE, 3))
    cloned(dummy, training=False)

    # Copy weights
    cloned.set_weights(model.get_weights())

    # Replace activations
    for layer in cloned.layers:
        if hasattr(layer, "activation") and layer.activation == tf.keras.activations.relu:
            layer.activation = guided_relu

    return cloned


def guided_gradcam_png(image_bytes, target_class=None):
    """Return a Guided Grad-CAM visualization PNG (bytes) for an image.

    Guided Grad-CAM = Guided Backpropagation × Grad-CAM (element-wise)
    - image_bytes: raw bytes of a PNG/JPG
    - target_class: optional int class index; defaults to predicted class
    """
    model = load_model()
    _ = model(tf.zeros((1, IMG_SIZE, IMG_SIZE, 3)), training=False)

    decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Could not decode uploaded image")

    decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    # Use a higher-res image for visualization so overlays don't look pixelated
    display_img = decoded
    max_dim = max(display_img.shape[0], display_img.shape[1])
    if max_dim > 600:
        scale = 600 / max_dim
        new_w = int(display_img.shape[1] * scale)
        new_h = int(display_img.shape[0] * scale)
        display_img = cv2.resize(display_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    model_img = cv2.resize(decoded, (IMG_SIZE, IMG_SIZE))
    img_float = model_img.astype("float32") / 255.0
    input_tensor = tf.convert_to_tensor(img_float[None, ...])

    # Pick class
    preds = model(input_tensor, training=False)[0]
    class_index = int(tf.argmax(preds).numpy()) if target_class is None else int(target_class)

    # Select conv layer for Grad-CAM
    conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    idx_from_end = max(1, min(int(CONV_LAYER), len(conv_layers)))
    target_conv_name = conv_layers[-idx_from_end].name

    # Grad-CAM
    # Build a functional graph that guarantees the chosen conv activation feeds the final prediction
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs
    conv_out_tensor = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        x = layer(x)
        if layer.name == target_conv_name:
            conv_out_tensor = x
    if conv_out_tensor is None:
        raise ValueError(f"Could not find conv layer '{target_conv_name}' in model")
    grad_model = tf.keras.Model(inputs=inputs, outputs=[conv_out_tensor, x])

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        conv_out, predictions = grad_model(input_tensor, training=False)
        score = predictions[:, class_index]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)
    cam = tf.nn.relu(cam)[0].numpy()

    # Upsample Grad-CAM to model input size (for guided grad-cam multiplication)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)

    # --- Guided Backpropagation ---
    guided_model = load_guided_model()
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        guided_preds = guided_model(input_tensor, training=False)
        guided_score = guided_preds[:, class_index]
    guided_grads = tape.gradient(guided_score, input_tensor)[0].numpy()

    # --- Guided Grad-CAM visualization ---
    # Guided edges for fine detail
    guided_gradcam = guided_grads * cam[..., np.newaxis]
    guided_gradcam = np.maximum(guided_gradcam, 0)
    guided_gradcam = guided_gradcam / (guided_gradcam.max() + 1e-8)
    guided_gradcam = (guided_gradcam * 255).astype(np.uint8)

    # Upsample guided map to display resolution for overlay
    guided_display = cv2.resize(
        guided_gradcam,
        (display_img.shape[1], display_img.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )

    # Brighten both base image and guided visualization a bit
    display_bright = cv2.convertScaleAbs(display_img, alpha=1.15, beta=15)
    guided_bright = cv2.convertScaleAbs(guided_display, alpha=3.15, beta=75)

    # Overlay guided edges on the original image
    # `guided_gradcam` is already in RGB (same channel order as `img_resized`).
    guided_edges_overlay = cv2.addWeighted(display_bright, .5, guided_bright, .5, 0)
    ok2, buf_edges = cv2.imencode(".png", guided_edges_overlay)
    if not ok2:
        raise RuntimeError("Could not encode guided edges image")
    guided_edges_bytes = buf_edges.tobytes()

    return {"guided_edges_png": guided_edges_bytes}

