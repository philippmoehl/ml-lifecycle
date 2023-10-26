import logging
import matplotlib.pyplot as plt
import os
from PIL import Image
import requests
import streamlit as st

from supabase import create_client, Client

from src.model import ImageClassifier
from src.solution import generate_solution
from src.utils import setup_app, get_datetime_stamp
from src.app_utils import page_config

INFERENCE_API = "http://localhost:8080"
MANAGEMENT_API = "http://localhost:8081"

logger = logging.getLogger(__name__)


def setup_supabase():
    supabase_client = init_connection()
    sign_in(supabase_client)
    return supabase_client


@st.cache_resource
def init_connection():
    # TODO: use env variables (setup_app loads them anyways)
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    return create_client(url, key)


def sign_in(client):
    mail = os.getenv("SUPABASE_MAIL")
    password = os.getenv("SUPABASE_PSWD")
    return client.auth.sign_in_with_password(
        {"email": mail, "password": password})
    

def sign_out():
    supabase_client.auth.sign_out()


def store_image(path, image, bucket="plants"):
    return supabase_client.storage.from_(bucket).upload(
        path, image, { "content-type": "image/jpeg"})


def store_prediction(path, cls, conf, table="mytable"):
    supabase_client.table(table).insert(
        {"image": path, "label": cls, "confidence": conf}).execute()


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
            timestamp = get_datetime_stamp()
            file = f"{timestamp}.jpg"
            store_image(file, file_uploaded.getvalue())
            cls, conf = max(out.items(), key=lambda item: item[1])
            store_prediction(file, cls, conf)
        if st.button("Solution"):
            # TODO: link highest conf class to name
            # solution = generate_solution(out)
            solution = "Use more water"
            st.write(solution)



if __name__ == "__main__":
    setup_app()
    page_config("App", "🌱")
    ping_serve()
    supabase_client = setup_supabase()
    main()
