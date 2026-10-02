# Task 8: Integrating Streamlit Frontend with Flask API

## MNIST Handwritten Digit Recognition Using CNN

**Programme:** MSc Artificial Intelligence – Part II  
**Institution:** L&T Edutech  
**Student Name:** Saikumar Padala  
**Roll Number:** 15  
**PRN / Student ID:** 5711387  
**Task:** Task 8 – Streamlit Frontend and Flask API Integration

---

## 1. Project Overview

This project integrates a Streamlit frontend with a Flask REST API backend to perform real-time handwritten digit recognition using a Convolutional Neural Network (CNN).

Users can upload an image of a handwritten digit through the Streamlit web interface. The frontend sends the image to the Flask API using an HTTP POST request. The backend preprocesses the image, loads it into the trained CNN model, generates a prediction, and returns the predicted digit and class probabilities in JSON format.

The Streamlit frontend displays the prediction results and visualizes the probability distribution for digits from 0 to 9.

## 2. Objectives

- Integrate a Streamlit frontend with a Flask backend.
- Develop REST API endpoints for health checking and digit prediction.
- Transfer uploaded images between the frontend and backend.
- Generate predictions using a trained CNN model.
- Display predicted digits and confidence values dynamically.
- Visualize prediction probabilities.
- Validate API responses and application functionality.
- Document the implementation, testing, results, and observations.

## 3. System Architecture

The application follows a frontend–backend architecture.

1. **User Interface:** Streamlit provides the interface for uploading handwritten digit images.
2. **HTTP Communication:** The frontend sends the uploaded image to the Flask API.
3. **Image Preprocessing:** The backend converts the image into the format expected by the trained model.
4. **CNN Inference:** TensorFlow/Keras performs digit classification.
5. **JSON Response:** The API returns the predicted digit and prediction probabilities.
6. **Result Visualization:** Streamlit displays the prediction and a probability chart.

```text
User
  |
  v
Streamlit Frontend
  |
  | POST /predict
  | Image upload
  v
Flask REST API
  |
  v
Image Preprocessing
  |
  v
Trained CNN Model
  |
  v
Prediction and Probabilities
  |
  | JSON Response
  v
Streamlit Results and Chart
```

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development |
| Streamlit | Frontend and interactive user interface |
| Flask | Backend REST API |
| TensorFlow/Keras | Loading and executing the CNN model |
| NumPy | Numerical operations and model input preparation |
| Pillow (PIL) | Image loading and preprocessing |
| Requests | HTTP communication between frontend and backend |
| MNIST | Handwritten digit recognition dataset |
| Matplotlib / Streamlit charts | Prediction probability visualization |

## 5. Application Features

### 5.1 Streamlit Frontend

- Upload PNG, JPG, and JPEG images.
- Preview the uploaded image.
- Send the image to the backend for prediction.
- Display the predicted digit.
- Display the model's reported confidence.
- Visualize probabilities for the ten digit classes.
- Display errors when the API is unavailable or a request fails.

### 5.2 Flask Backend

- Provide an API health-check endpoint.
- Receive uploaded images through HTTP requests.
- Preprocess images before inference.
- Generate CNN predictions.
- Return predictions and probabilities in JSON format.
- Handle invalid input and prediction errors.

## 6. API Endpoints

### Health Check

**Method:** `GET`  
**Endpoint:** `/health`

Example URL:

```text
http://127.0.0.1:5000/health
```

Observed response from the local application:

```json
{
  "model_loaded": true,
  "status": "healthy",
  "success": true
}
```

This response indicates that the health endpoint returned a healthy status and that the API's model-loaded flag was true when the screenshot was captured.

### Digit Prediction

**Method:** `POST`  
**Endpoint:** `/predict`

The frontend sends an image to the prediction endpoint. The backend processes the image and returns the predicted digit and associated prediction probabilities.

The exact request field names and JSON response structure depend on the implementation in `flask_api.py`.

## 7. Model and Dataset

The project uses a Convolutional Neural Network (CNN) trained for MNIST handwritten digit classification.

The MNIST dataset contains grayscale images representing handwritten digits from 0 to 9. The images are commonly represented at a resolution of 28 × 28 pixels.

The model receives a preprocessed image and generates scores or probabilities for the ten digit classes. The class with the highest model output is selected as the predicted digit.

**Important:** Uploaded images must be preprocessed consistently with the training pipeline. Differences in image polarity, resizing, centering, normalization, and background can affect prediction accuracy and confidence.

## 8. Project Structure

```text
Task-8-Streamlit-Flask-API-Integration/
│
├── app.py
├── flask_api.py
├── deep_learning_model.h5
├── requirements.txt
├── README.md
│
├── screenshots/
│   ├── streamlit_home.png
│   ├── prediction_result.png
│   ├── probability_chart.png
│   ├── browser_health_check.png
│   └── flask_terminal_logs.png
│
└── report/
    └── Task_8_Streamlit_Flask_API_Integration_Report_Times_New_Roman_With_Health_Check.pdf
```

This is the recommended repository layout. Ensure that the filenames match your actual files before uploading them.

## 9. Installation and Setup

### Prerequisites

- Python installed on your computer.
- The trained CNN model file.
- The Flask backend source code.
- The Streamlit frontend source code.
- Required Python packages.

### Step 1: Clone the Repository

After creating the public GitHub repository, run:

```bash
git clone https://github.com/Saipadala2003/Task-8-Streamlit-Flask-API-Integration.git
cd Task-8-Streamlit-Flask-API-Integration
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

The requirements file should contain the packages used by your application, such as Flask, Streamlit, TensorFlow, NumPy, Pillow, and Requests, with compatible versions.

## 10. Running the Application

Run the frontend and backend in separate terminals.

### Terminal 1: Start the Flask API

```bash
python flask_api.py
```

The API should start at:

```text
http://127.0.0.1:5000
```

Check its health endpoint in a browser:

```text
http://127.0.0.1:5000/health
```

### Terminal 2: Start the Streamlit Frontend

```bash
streamlit run app.py
```

Open the local Streamlit URL displayed in the terminal. By default, this is commonly:

```text
http://localhost:8501
```

Keep both processes running while testing the integrated application.

## 11. Testing and Validation

The following checks should be performed to validate the application.

| Test Case | Expected Result |
|---|---|
| Flask API startup | Backend starts without a fatal error |
| Health endpoint | Returns a healthy JSON response |
| Valid image upload | Streamlit displays the uploaded image |
| Prediction request | Flask receives the image and processes it |
| Prediction result | API returns a predicted digit |
| Probability visualization | Frontend displays the returned class probabilities |
| Invalid image | Application displays a meaningful error |
| Backend unavailable | Frontend reports an API connection error |
| Repeated requests | Application continues responding without crashing |

Actual test results should be recorded after running each test. A successful health check does not, by itself, prove that all prediction requests are accurate.

## 12. Observed Results

During local testing, the Flask health endpoint returned the following values:

- `status`: `healthy`
- `success`: `true`
- `model_loaded`: `true`

The Streamlit frontend also displayed a handwritten digit image, a prediction result, and a chart of prediction probabilities.

For the uploaded image named `7.png`, the observed prediction was **digit 7 with a reported confidence of 75.85%** in the captured interface.

This value is the observed application output, not a guarantee that every handwritten digit will receive the same confidence. Confidence values depend on the trained model and the preprocessing of each uploaded image.

## 13. Observations

- Streamlit and Flask can communicate through HTTP requests.
- The Flask backend separates model inference from the frontend interface.
- The health endpoint provides a simple way to check backend availability.
- The probability chart helps users inspect the model's outputs across all ten digit classes.
- Image preprocessing has a significant effect on predictions.
- A high confidence value does not necessarily guarantee a correct prediction.
- Testing must cover both API functionality and prediction quality.

## 14. Limitations

- Prediction quality depends on the trained model and the similarity of uploaded images to the training data.
- Unusual handwriting, low-resolution images, incorrect image polarity, and poor centering may reduce accuracy.
- The local API address is intended for local development and is not automatically accessible from other computers.
- The Flask development server should not be used as the production deployment server.
- Model confidence should not be interpreted as a calibrated probability unless calibration has been evaluated.

## 15. Future Enhancements

- Improve image preprocessing and digit centering.
- Evaluate accuracy using a separate test dataset.
- Add request timeouts and clearer API error messages.
- Add structured logging and performance measurements.
- Introduce automated API tests.
- Deploy the frontend and backend to suitable hosting platforms.
- Add model versioning and input validation.
- Evaluate probability calibration and model robustness.

## 16. Conclusion

This project demonstrates the integration of a Streamlit frontend with a Flask REST API for real-time MNIST handwritten digit classification. The frontend accepts image uploads and communicates with the backend, while the backend performs inference using a trained CNN model and returns prediction results.

The implementation demonstrates frontend–backend communication, API endpoint testing, image processing, model inference, JSON responses, and interactive visualization. The project provides practical experience in integrating machine-learning models into Python web applications.

## 17. Screenshots

Add the actual screenshots captured during your implementation:

1. Streamlit home page.
2. Uploaded handwritten digit and prediction result.
3. Prediction probability chart.
4. Browser health-check response.
5. Flask terminal showing successful startup and API requests.

The screenshots should correspond to the application being submitted.

## 18. Academic Submission

**Task:** Task 8 – Integrating Streamlit Frontend with Flask API  
**Programme:** MSc Artificial Intelligence – Part II  
**Institution:** L&T Edutech  
**Student:** Saikumar Padala

The submission consists of the Python source files, trained model file where permitted, dependency list, screenshots, testing evidence, and PDF report.

---

**Repository:** `Task-8-Streamlit-Flask-API-Integration`

**GitHub Username:** `Saipadala2003`

**Note:** Replace any suggested filenames with the actual filenames in your project. Confirm that the repository is public and that the code, model, screenshots, and PDF have been uploaded before submitting the repository link through the L&T Edutech LMS.
