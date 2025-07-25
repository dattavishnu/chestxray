# ChestXray        <!-- Largest heading -->

This project hosts a website that detects pneumonia by analyzing chest X-ray images as input.

## Requirements

To get started, you will need the following Python packages(use python 3.7-3.11):

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

```
my_project/
├── .keras                 # Example model file
├── static/                # Folder containing static files
│   └── index.html         # Example HTML file
├── main.py                # FastAPI app
└── README.md              # Project documentation
```


### Output
![alt text](<Screenshot 2025-07-24 190144-1.png>)

![alt text](<Screenshot 2025-07-24 190217.png>)

![alt text](<Screenshot 2025-07-24 190200.png>)

