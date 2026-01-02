import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 100

def analyze_image(uploaded_file):
    model = load_model()

    # Read uploaded file into numpy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode uploaded image")

    # IMPORTANT: Keep BGR here to match training preprocessing (train.py uses cv2.imread, which is BGR).
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0 
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    probs = pred[0]  # [Benign, Malignant]
    return (int(probs[0] * 100), int(probs[1] * 100))


def load_model():
    model = tf.keras.models.load_model("model25.keras")
    # force-build model by running a dummy inference
    dummy = tf.zeros((1, IMG_SIZE, IMG_SIZE, 3))
    model(dummy, training=False)
    return model


def guided_gradcam_png(image_bytes):
    """Returns a Grad-CAM gradient attribution PNG for uploaded image.
       - image_bytes: raw bytes of a PNG/JPG
    """
    model = load_model()

    decoded_bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded_bgr is None:
        raise ValueError("Could not decode uploaded image")

    # Keep a BGR copy for model inference (matches training), and an RGB copy for display.
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)

    # Use a higher-res image for visualization so overlays don't look pixelated
    display_img = decoded_rgb
    max_dim = max(display_img.shape[0], display_img.shape[1])
    if max_dim > 600:
        scale = 600 / max_dim
        new_w = int(display_img.shape[1] * scale)
        new_h = int(display_img.shape[0] * scale)
        display_img = cv2.resize(display_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    model_img = cv2.resize(decoded_bgr, (IMG_SIZE, IMG_SIZE))
    img_float = model_img.astype("float32") / 255.0
    input_tensor = tf.convert_to_tensor(img_float[None, ...])

    # Pick class
    preds = model(input_tensor, training=False)[0]
    class_index = int(tf.argmax(preds).numpy())

    # Build a functional graph that captures the conv layer output
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    x = inputs
    conv_output = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == "conv2d_2": #last conv layer
            conv_output = x
    outputs = x

    grad_model = tf.keras.Model(inputs=inputs, outputs=[conv_output, outputs])

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        conv_out, predictions = grad_model(input_tensor, training=False)
        score = predictions[:, class_index]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)
    cam = tf.nn.relu(cam)[0].numpy()

    # Upsample Grad-CAM to model input size
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)

    # Remove border artifacts (padding / resize bias)
    border = int(0.08 * cam.shape[0])
    cam[:border, :] = 0
    cam[-border:, :] = 0
    cam[:, :border] = 0
    cam[:, -border:] = 0

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        preds = model(input_tensor, training=False)
        score = preds[:, class_index]
    grads = tape.gradient(score, input_tensor)[0].numpy()

    input_arr = img_float
    # Filter activations by Grad-CAM strength
    cam_gate = cam > 0.3
    attribution = np.mean(np.abs(grads * input_arr), axis=-1)
    guided_gradcam = attribution * cam_gate

    # Normalize
    guided_gradcam = guided_gradcam / (guided_gradcam.max() + 1e-8)
    guided_gradcam = (guided_gradcam * 255).astype(np.uint8)
    
    guided_display = cv2.resize(
        guided_gradcam,
        (display_img.shape[1], display_img.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )

    # Expand activations for easier viewing
    guided_display = cv2.dilate(guided_display, np.ones((3, 3), np.uint8), iterations=1)

    # Make a colored attribution map
    colored_bgr = cv2.applyColorMap(guided_display, cv2.COLORMAP_TURBO)
    colored_bgr = cv2.convertScaleAbs(colored_bgr, alpha=1.4, beta=40)

    # Build alpha from attribution strength
    alpha = guided_display.astype(np.float32) / 255.0
    alpha = np.power(alpha, 0.8)
    alpha = np.clip(alpha * 1.0, 0, 0.85)
    alpha[alpha < 0.10] = 0.0             

    # Dim the background so the overlay stands out
    base_bgr = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
    base_bgr = (base_bgr.astype(np.float32) * 0.9).clip(0, 255).astype(np.uint8)

    # Blend only where alpha > 0
    out = base_bgr.astype(np.float32).copy()
    colored_f = colored_bgr.astype(np.float32)
    for c in range(3):
        out[..., c] = out[..., c] * (1 - alpha) + colored_f[..., c] * alpha

    guided_edges_overlay_bgr = out.astype(np.uint8)

    ok2, buf_edges = cv2.imencode(".png", guided_edges_overlay_bgr)
    if not ok2:
        raise RuntimeError("Could not encode guided edges image")
    
    guided_edges_bytes = buf_edges.tobytes()
    return guided_edges_bytes

