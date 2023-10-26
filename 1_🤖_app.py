import datetime
import logging
import os
from PIL import Image
import requests
import streamlit as st

from src.solution import generate_solution
from src.utils import setup_app, get_datetime_stamp
from src.app_utils import page_config, setup_supabase,  show_image

INFERENCE_API = "http://localhost:8080"
MANAGEMENT_API = "http://localhost:8081"

logger = logging.getLogger(__name__)


def store_image(path, image, bucket="plants"):
    return supabase_client.storage.from_(bucket).upload(
        path, image, { "content-type": "image/jpeg"})


def store_prediction(path, cls, conf, id, table="plants"):
    supabase_client.table(table).insert(
        {"id": id, "name": path, "label": cls, "confidence": conf}).execute()


def ping_serve():
    try:
        res = requests.post(INFERENCE_API + "/ping")
        status = res.json()["status"]
        logger.info(f"Ping status: {status}")
    except Exception as e:
        logger.warning(e)
        st.write("torchserve is not running yet, start serving")
        st.link_button("serve", "http://localhost:8501/serve")
        st.stop()
    

def list_models():
    try:
        res = requests.get(MANAGEMENT_API + "/models")
        models = res.json()["models"]
        logger.info(f"Available models: {models}")
        models = {d["modelName"]: d["modelUrl"] for d in models}
        return models
    except Exception as e:
        logger.warning(e)
        raise ConnectionError("torchserve may not running")
    

def predict(url, file):
    try:
        res = requests.post(INFERENCE_API + f"/predictions/{url}", data=file)
        prediction = res.json()
        logger.info(prediction)
        return res.json()
    except Exception as e:
        logger.warning(e)
        raise ConnectionError("torchserve may not running")


def main():
    st.header("PLANT DISEASE")
    solution = st.sidebar.toggle("solution")
    models = list_models()
    if models:
        model_name = st.selectbox(
            'Choose Model', (name for name, _ in models.items()))
    else:
        st.write("torchserve has no registered model yet, register one")
        st.link_button("serve", "http://localhost:8501/serve")
        logger.warning("No torchserve model is registered")
        st.stop()
    
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])

    if file_uploaded:
        show_image(file_uploaded.getvalue(), file_uploaded.name)
        if st.button("Predict"):
            out = predict(model_name, file=file_uploaded.getvalue())
            st.write("Prediction:")
            st.write(out)
            timestamp = get_datetime_stamp()
            date_format = datetime.datetime.strptime(
                timestamp, '%Y-%m-%d_%H-%M-%S')
            unix_timestamp = int(datetime.datetime.timestamp(date_format))
            file = f"{timestamp}.jpg"
            store_image(file, file_uploaded.getvalue())
            cls, conf = max(out.items(), key=lambda item: item[1])
            store_prediction(file, cls, conf, id=unix_timestamp)
            if solution:
                # TODO: link highest conf class to name
                # solution = generate_solution(out)
                solution_text = f"Using more water helps with {cls}"
                st.write(solution_text)


if __name__ == "__main__":
    page_config("App", "🤖")
    setup_app()
    ping_serve()
    supabase_client = setup_supabase()
    main()
