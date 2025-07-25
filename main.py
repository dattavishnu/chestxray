from fastapi import FastAPI, Form, Request, HTTPException, UploadFile, File, Depends, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from passlib.context import CryptContext
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import sqlite3
import os
import io
import google.generativeai as genai

# ---------------------- Setup ----------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.makedirs("static/uploads", exist_ok=True)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
model = load_model("model_Pneumonia_detection.keras")
genai.configure(api_key="AIzaSyBO1wXdIaUR0MAbgczp-UgS_eKCHktO1J4")
gemini_model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

# ---------------------- Auth ----------------------
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def require_login(session_user: str = Cookie(default=None)):
    if not session_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return session_user

@app.on_event("startup")
def init_db():
    # users table
    conn = sqlite3.connect("users.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT
        )""")
    conn.commit()
    conn.close()

    # predictions table
    conn = sqlite3.connect("images.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            filepath TEXT,
            prediction TEXT,
            confidence REAL,
            report TEXT
        )""")
    conn.commit()
    conn.close()

@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("SELECT password FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()

    if row and verify_password(password, row[0]):
        response = RedirectResponse(url="/dashboard", status_code=302)
        response.set_cookie(key="session_user", value=username, httponly=True)
        return response

    return HTMLResponse("Invalid credentials", status_code=401)

@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...), email: str = Form(...)):
    hashed = hash_password(password)
    conn = sqlite3.connect("users.db")
    try:
        conn.execute(
            "INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
            (username, hashed, email)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return {"message": "Username already exists"}
    conn.close()
    return {"message": "Registration successful!"}

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, user: str = Depends(require_login)):
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/")
    response.delete_cookie("session_user")
    return response

# ---------------------- Prediction ----------------------
@app.post("/predict/")
async def predict(file: UploadFile = File(...), user: str = Depends(require_login)):
    try:
        contents = await file.read()

        # Save image
        image_path = f"static/uploads/{file.filename}"
        with open(image_path, "wb") as f:
            f.write(contents)

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
                f"Generate a short, human-readable diagnostic report."
            )
            report_response = gemini_model.generate_content([{"text": prompt}])
            report_text = report_response.text.strip()

        # Save prediction to DB
        conn = sqlite3.connect("images.db")
        conn.execute(
            "INSERT INTO predictions (filename, filepath, prediction, confidence, report) VALUES (?, ?, ?, ?, ?)",
            (file.filename, image_path, diagnosis, confidence, report_text)
        )
        conn.commit()
        conn.close()

        return {
            "filename": file.filename,
            "prediction": diagnosis,
            "confidence": confidence,
            "report": report_text
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
