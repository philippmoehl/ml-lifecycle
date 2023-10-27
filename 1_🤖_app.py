import argparse
import datetime
import logging
import os
import streamlit as st

from src.utils import get_datetime_stamp
from src.app_utils import (
    page_config, show_image, setup_app
    )

logger = logging.getLogger(__name__)


@st.cache_resource
def handle_parse():
    parser = argparse.ArgumentParser(description="Torchserve Dashboard")

    parser.add_argument(
        "--config-file", 
        default="app_config.yaml", 
        help="App configuration file."
    )
    try:
        args = parser.parse_args()
        return args
    except SystemExit as e:
        os._exit(e.code)


def main():
    st.title("PLANT DISEASE PREDICTIONS")

    # ----sidebar----
    solution = st.sidebar.toggle("solution")
    # ---------------

    # check if torch-serve is running
    if not st.session_state.inference.ping_serve():
        st.write("torchserve is not running yet, start serving")
        st.link_button("serve", "http://localhost:8501/serve")
        st.stop()

    # list available models
    models = st.session_state.api.get_loaded_models()["models"]
    if models:
        logger.info(f"Available models: {models}")
        models_dict = {d["modelName"]: d["modelUrl"] for d in models}
        model_name = st.selectbox(
            'Choose Model', (name for name, _ in models_dict.items()))
    else:
        st.write("torchserve has no registered model yet, register one")
        st.link_button(
            "register", "http://localhost:8501/serve#register-a-model-docs")
        logger.warning("No torchserve model is registered")
        st.stop()
    
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    if file_uploaded:
        show_image(file_uploaded.getvalue(), file_uploaded.name)
        if st.button("Predict"):
            out = st.session_state.inference.predict(model_name, file=file_uploaded.getvalue())
            if not out:
                st.write("torchserve may not be running")
                st.stop()
            cls, conf = max(out.items(), key=lambda item: item[1])
            st.markdown("### Prediction")
            st.write(out)
            
            # create unique index and file name for buckets
            timestamp = get_datetime_stamp()
            date_format = datetime.datetime.strptime(
                timestamp, '%Y-%m-%d_%H-%M-%S')
            unix_timestamp = int(datetime.datetime.timestamp(date_format))
            file_path = f"{timestamp}.jpg"

            st.session_state.supabase.store_image(file_path, file_uploaded.getvalue())
            st.session_state.supabase.store_prediction(file_path, cls, conf, id=unix_timestamp)

            if solution:
                solution = st.session_state.solution.generate_solution(cls)
                st.markdown("### Solution")
                st.write(solution)


if __name__ == "__main__":
    page_config("App", "🤖")
    args = handle_parse()
    setup_app(args.config_file)
    main()
