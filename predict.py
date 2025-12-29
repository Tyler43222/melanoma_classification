import cv2
import numpy as np
import tensorflow as tf

import functools

IMG_SIZE = 100

def analyze_image(uploaded_file):
    model = _load_model()

    # Read uploaded file into numpy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode uploaded image")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0  # Normalize like training
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict
    pred = model.predict(img)
    probs = pred[0]  # [Benign, Malignant]
    return f"RESULTS:\nBenign: {int(probs[0] * 100)}%\nMalignant: {int(probs[1] * 100)}%"


@functools.lru_cache(maxsize=1)
def _load_model():
    try:
        model = tf.keras.models.load_model("model7.keras")
        _ensure_model_called(model)
        return model
    except Exception as exc:
        message = str(exc)
        if "batch_input_shape" in message and "Conv2D" in message:
            raise RuntimeError(
                "Your saved model file (model7.keras) was created with an older Keras format and "
                "is not compatible with your current TensorFlow/Keras install. "
                "Re-train and re-save the model using the current environment (train.py), then try again."
            ) from exc
        raise


def _ensure_model_called(model):
    """Force-build/call a loaded Sequential model so model.inputs/model.output exist."""
    try:
        _ = model.inputs
        _ = model.outputs
        return
    except Exception:
        pass

    try:
        model.build((None, IMG_SIZE, IMG_SIZE, 3))
    except Exception:
        pass

    _ = model(tf.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32), training=False)


def guided_gradcam_png(image_bytes, target_class=None):
    """Return a Guided Grad-CAM visualization PNG (bytes) for an image.

    - image_bytes: raw bytes of a PNG/JPG
    - target_class: optional int class index; defaults to predicted class
    """
    model = _load_model()
    _ensure_model_called(model)

    decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Could not decode uploaded image")

    img_resized = cv2.resize(decoded, (IMG_SIZE, IMG_SIZE))
    img_float = img_resized.astype("float32") / 255.0
    input_tensor = tf.convert_to_tensor(img_float[None, ...])

    # pick class
    preds = model(input_tensor, training=False)[0]
    class_index = int(tf.argmax(preds).numpy()) if target_class is None else int(target_class)

    # last conv layer
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break
    if last_conv is None:
        raise RuntimeError("Could not find a Conv2D layer for Grad-CAM")

    # Grad-CAM heatmap
    grad_model = tf.keras.Model(model.inputs, [model.get_layer(last_conv).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, predictions = grad_model(input_tensor, training=False)
        score = predictions[:, class_index]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)
    cam = tf.nn.relu(cam)[0]
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    heatmap = cam.numpy()
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = heatmap[..., None]

    # Guided backprop (simple, by swapping relu gradients)
    @tf.custom_gradient
    def guided_relu(x):
        y = tf.nn.relu(x)

        def grad(dy):
            gate_f = tf.cast(x > 0, dy.dtype)
            gate_r = tf.cast(dy > 0, dy.dtype)
            return dy * gate_f * gate_r

        return y, grad

    def _clone_layer(layer):
        cfg = layer.get_config()
        if "activation" in cfg and cfg["activation"] == "relu":
            cfg["activation"] = guided_relu
        if isinstance(layer, tf.keras.layers.Activation) and cfg.get("activation") == "relu":
            cfg["activation"] = guided_relu
        return layer.__class__.from_config(cfg)

    guided_model = tf.keras.models.clone_model(model, clone_function=_clone_layer)
    guided_model.set_weights(model.get_weights())

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        guided_preds = guided_model(input_tensor, training=False)[:, class_index]
    guided_grads = tape.gradient(guided_preds, input_tensor)[0].numpy()  # (H,W,3)
    guided_grads = np.maximum(guided_grads, 0.0)

    guided_cam = guided_grads * heatmap
    guided_cam = guided_cam - guided_cam.min()
    guided_cam = guided_cam / (guided_cam.max() + 1e-8)
    guided_cam_u8 = (guided_cam * 255).astype(np.uint8)

    ok, buf = cv2.imencode(".png", guided_cam_u8)
    if not ok:
        raise RuntimeError("Could not encode Guided Grad-CAM image")
    return buf.tobytes()
   