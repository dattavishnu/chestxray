from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
import cv2

app = FastAPI(title="Grad-CAM API")

# Load your pneumonia detection model (adjust path as needed)
MODEL_PATH = "model_Pneumonia_detection.keras"
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Dynamically find the last Conv2D layer
conv_layer_name = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        conv_layer_name = layer.name
        break
if conv_layer_name is None:
    raise RuntimeError("No Conv2D layer found in model for Grad-CAM")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Specify response_class and content type so Swagger knows it's an image
@app.post(
    "/gradcam",
    response_class=StreamingResponse,
    responses={200: {"content": {"image/png": {}}}}
)
async def gradcam_endpoint(file: UploadFile = File(...)):
    try:
        # Validate uploaded file
        if file.content_type.split('/')[0] != 'image':
            raise HTTPException(status_code=400, detail="Invalid file type. Image required.")
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        heatmap = make_gradcam_heatmap(img_array, model, conv_layer_name)
        heatmap = cv2.resize(heatmap, (img_resized.shape[1], img_resized.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(img_resized, 0.6, heatmap_color, 0.4, 0)
        _, buffer = cv2.imencode('.png', cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
