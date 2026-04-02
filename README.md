---

# ASD Detection API 🧠
WEBSITE LINK - https://zeenu002-asd-detection-api.hf.space/

An intelligent, fast, and scalable Application Programming Interface (API) built to detect Autism Spectrum Disorder (ASD) in both adults and children using various pre-trained Machine Learning models.

## 🚀 What We Built
We have developed a robust machine learning API capable of receiving clinical/behavioral features (18 features) and returning a prediction on whether the subject is ASD Positive or ASD Negative. 

It supports multiple state-of-the-art models customized for both **Child** and **Adult** datasets:
- Random Forest
- XGBoost
- AdaBoost
- Decision Tree (CART)
- Gradient Boosting

The API provides detailed insights including confidence scores and probabilities along with every prediction.

## 🛠️ Tech Stack & Tools Used
- **Language**: Python 3.10, HTML, CSS, JavaScript (Vanilla Frontend)
- **Web Server/Framework**: FastAPI (High-performance API framework), Uvicorn & Pydantic.
- **Machine Learning**: Scikit-Learn, XGBoost, Numpy
- **Model Serialization**: Joblib
- **Containerization**: Docker (Dockerfile)
- **Deployment & Hosting**: Hugging Face Spaces & `huggingface_hub` Python library.

## 🖥️ NeuroScreen Frontend UI (Client)

To provide an accessible and visual experience, we built **NeuroScreen** — a beautiful, glassmorphism-inspired single-page web application (`frontend/index.html`) that allows users to take the standard AQ-10 clinical screening test interactively.

**Update:** The frontend UI is now **native to the backend**, served directly by FastAPI via `StaticFiles` on the root (`/`) endpoint!

### How it Connects & Works with the Model (Step-by-Step)
1. **User Input Selection**: The frontend presents the 10 behavioral questions (A1–A10) and basic demographic fields (Age & Gender) mimicking a clinical form.
2. **Data Parsing & Validation**: Upon clicking "Analyze & Predict", embedded JavaScript checks that all 10 questions are answered, avoiding silent prediction failures.
3. **Feature Construction**: Since our ML model strictly requires a vector of exactly **18 features**, the script creates an array containing the 10 question scores, Age, and Gender. It then automatically pads the remaining 6 unsupported demographic fields (e.g. Ethnicity, Jaundice history, etc.) with baseline zeros (`0`), maintaining the model's structural integrity.
4. **API Communication (`fetch`)**: The UI triggers an asynchronous HTTP `POST` request via JavaScript's `fetch()` to the FastAPI backend (`http://127.0.0.1:7860/predict`). It dynamically decides to ping either `xgboost_child` or `xgboost_adult` models based purely on the patient's entered age.
5. **Result Rendering**: The Uvicorn-served API returns the evaluated JSON response, containing the prediction label and confidence metrics. The JavaScript catches this and dynamically modifies the DOM to show an elegant result Modal ("ASD Positive" or "ASD Negative") with color-coordinated UI elements.

*(Note: We integrated `CORSMiddleware` within our FastAPI `app.py` script so that local or third-party web clients can hit the API seamlessly without triggering Cross-Origin Resource Sharing (CORS) blocks).*

## 🌐 Where is it Deployed?
The project is successfully deployed on **Hugging Face Spaces**. 
- **Repository/Space Name**: `zeenu002/asd-detection-api`
- **Environment**: Containerized Docker Space running on port `7860`.

## ⚙️ How We Deployed It (Step-by-Step)

We used a seamless, programmatic approach to deploy our API directly to Hugging Face Spaces using a Python deployment script.

**Step 1: Containerize the Application**
We created a `Dockerfile` to configure the production environment. We used the lightweight `python:3.10-slim` image, installed all necessary dependencies from `requirements.txt`, and launched the FastAPI app on port `7860` using Uvicorn.

**Step 2: Setup Hugging Face Token**
A Write-Access token was generated from the Hugging Face account settings to authenticate and allow pushing code programmatically. 

**Step 3: Define the Deployment Script (`hf_upload.py`)**
We utilized the `huggingface_hub` library to automate the deployment. The script uses the `HfApi().upload_folder` method to directly push the current directory to the Hugging Face Space, while cleverly ignoring unnecessary local files like virtual environments and cache directories:

```python
from huggingface_hub import HfApi
api = HfApi()

api.upload_folder(
    folder_path='.',
    repo_id='zeenu002/asd-detection-api',
    repo_type='space',
    token='[REDACTED]',
    ignore_patterns=['saved_models/*', '__pycache__/*', '.git/*', '.venv/*']
)
```

**Step 4: Execute the Push**
By simply running `python hf_upload.py`, the code was zipped and sent to Hugging Face. Once received, the Hugging Face CI/CD pipeline automatically detected the `Dockerfile`, built the Docker image, and started the container. 

## 📌 API Usage & Endpoints

- `GET /`: Serves the interactive **NeuroScreen Frontend Web UI**.
- `GET /api/info`: Health check, returns API status, available models, and expected features.
- `GET /features`: Returns the names and count of all 18 features expected.
- `POST /predict`: Submit a JSON payload to get an ASD prediction.

**Example Request Payload (`/predict`)**:
```json
{
  "features": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0], 
  "model_name": "xgboost_adult"
}
```

**Example Response**:
```json
{
  "model_used": "xgboost_adult",
  "prediction": 1,
  "prediction_label": "ASD Positive ✅",
  "confidence": 89.5,
  "probabilities": {
    "ASD_negative": 0.105,
    "ASD_positive": 0.895
  }
}
```
