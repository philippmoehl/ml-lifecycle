import logging
import matplotlib.pyplot as plt
from PIL import Image
import requests
import streamlit as st

from src.model import ImageClassifier
from src.solution import generate_solution
from src.utils import setup_app

INFERENCE_API = "http://localhost:8080"
MANAGEMENT_API = "http://localhost:8081"

logger = logging.getLogger(__name__)


def ping_serve():
    try:
        res = requests.post(INFERENCE_API + "/ping")
        status = res.json()["status"]
        logger.info(f"Ping status: {status}")
    except Exception as e:
        logger.error(e)
        raise ConnectionError("torchserve is not running")
    

def list_models():
    try:
        res = requests.get(MANAGEMENT_API + "/models")
        models = res.json()["models"]
        logger.info(f"Available models: {models}")
        models = {d["modelName"]: d["modelUrl"] for d in models}
        return models
    except Exception as e:
        logger.error(e)
        raise ConnectionError("torchserve is not running")
    

def predict(url, file):
    try:
        res = requests.post(INFERENCE_API + f"/predictions/{url}", data=file)
        prediction = res.json()
        logger.info(prediction)
        return res.json()
    except Exception as e:
        logger.error(e)
        raise ConnectionError("torchserve is not running")
    

def show_image(file):
    st.image(file.getvalue(), caption=file.name)


def main():
    print(open("image.jpg", "rb"))
    st.header("PLANT DISEASE")

    models = list_models()
    model_name = st.selectbox(
        'Choose Model', (name for name, _ in models.items()))
    
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])

    if file_uploaded:
        show_image(file_uploaded)
        if st.button("Predict"):
            out = predict(model_name, file=file_uploaded.getvalue())
            st.write("Prediction:")
            st.write(out)
        if st.button("Solution"):
            # TODO: link highest conf class to name
            # solution = generate_solution(out)
            solution = "Use more water"
            st.write(solution)



if __name__ == "__main__":
    setup_app()
    ping_serve()
    main()
