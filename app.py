import streamlit as st
import requests
from PIL import Image

# Flask API URL
API_URL = "http://127.0.0.1:5000/predict"

st.set_page_config(
    page_title="MNIST CNN - Streamlit + Flask",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 MNIST Digit Recognition")
st.subheader("Streamlit Frontend + Flask API")

st.write(
    "Upload a handwritten digit image. "
    "The Streamlit frontend sends the image to the Flask backend "
    "for real-time CNN prediction."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload a handwritten digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=250
    )

    if st.button("Predict Digit", type="primary"):

        try:

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            with st.spinner("Sending image to Flask API..."):

                response = requests.post(
                    API_URL,
                    files=files,
                    timeout=30
                )

            if response.status_code == 200:

                result = response.json()

                if result.get("success"):

                    digit = result["predicted_digit"]
                    confidence = result["confidence"]

                    st.success(
                        f"Predicted Digit: {digit}"
                    )

                    st.metric(
                        "Prediction Confidence",
                        f"{confidence:.2f}%"
                    )

                    st.subheader("Prediction Probabilities")

                    probabilities = result["probabilities"]

                    st.bar_chart(probabilities)

                else:

                    st.error(
                        result.get(
                            "error",
                            "Prediction failed"
                        )
                    )

            else:

                st.error(
                    f"Flask API returned status code "
                    f"{response.status_code}"
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "Unable to connect to Flask API. "
                "Please make sure flask_api.py is running."
            )

        except requests.exceptions.Timeout:

            st.error(
                "The Flask API request timed out."
            )

        except Exception as e:

            st.error(
                f"Unexpected error: {e}"
            )

st.divider()

st.info(
    "Backend API: http://127.0.0.1:5000"
)