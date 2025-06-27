from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import numpy as np
import sys

def predict_image(model_path, image_path):
    # Load model
    model = load_model(model_path)
    
    # Load image, convert to grayscale, resize to 256x256
    img = Image.open(image_path).convert("L").resize((256, 256))
    
    # Convert to array and normalize
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Predict
    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])
    diagnosis = "Pneumonia" if confidence > 0.5 else "Normal"
    
    print(f"Prediction: {diagnosis}")
    print(f"Confidence: {confidence:.4f}")

    
predict_image('model_Pneumonia_detection.keras', 'normal.jpg')
