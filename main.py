from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
import io
import google.generativeai as genai
from keras.models import load_model
from keras.preprocessing import image
import os

os.makedirs("static", exist_ok=True)

# Load your trained model
model = load_model("model_Pneumonia_detection.keras")

# Configure Gemini API
genai.configure(api_key="AIzaSyBO1wXdIaUR0MAbgczp-UgS_eKCHktO1J4")
gemini_model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

# Initialize FastAPI app
app = FastAPI()

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use '*' for testing, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (e.g. HTML)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    with open("static/xray_insight_final_with_report_page.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("L")
        img_resized = pil_img.resize((256, 256))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])
        diagnosis = "Pneumonia" if confidence > 0.5 else "Normal"

        report_text = ""
        if confidence >= 0.2:
            prompt = (
                f"The AI model predicts that the chest X-ray has a {confidence:.2%} chance of Pneumonia. "
                f"Generate a short, human-readable diagnostic report based on this result and suggest tests if needed."
            )
            report_response = gemini_model.generate_content([{"text": prompt}])
            report_text = report_response.text.strip()

        return {
            "prediction": diagnosis,
            "confidence": confidence,
            "report": report_text
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
