#chestxray

This is a project hosts website to pneumonia to detect by taking chestxray as input.  

## Requirements

To get started, you will need the following Python packages:

- **TensorFlow**: For machine learning models.
- **Matplotlib**: For plotting and visualizations.
- **Pillow**: For image processing.
- **FastAPI**: For serving the application.

### How to Install

1. Create a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    ```

2. Activate the virtual environment:
    - On Linux/macOS:
        ```bash
        source venv/bin/activate
        ```
    - On Windows:
        ```bash
        .\venv\Scripts\activate
        ```

3. Install dependencies:
    ```bash
    pip install tensorflow matplotlib pillow fastapi
    ```

4. Run the FastAPI server:
    ```bash
    uvicorn main:app --reload
    ```

## File Structure



├── .keras/                # Keras folder (for storing model files, etc.)
├── static/                # Folder containing static files
│   └── index.html         # Example HTML file
├── main.py                # FastAPI app
└── README.md              # Project requirements


