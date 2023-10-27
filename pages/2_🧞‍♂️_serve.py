import logging
import streamlit as st

from src.app_utils import auth, handle_status, page_config

logger = logging.getLogger(__name__)


@st.cache_resource()
def initialize():
    api = st.session_state.api
    ts = st.session_state.ts
    torchserve_status = api.get_loaded_models()
    
    if not torchserve_status:
       st.write("Starting Torchserve")
       last_res()[0] = ts.start_torchserve()
       rerun()
    return api, ts


@st.cache_resource()
def last_res():
    return ["Nothing"]


def rerun():
    st.experimental_rerun()


def dashboard():
    st.title("Serving Management")
    default_key = "None"
    api, ts = initialize()

    # ----sidebar----
    st.sidebar.markdown("## Controls")
    start = st.sidebar.button("Start")
    if start:
        last_res()[0] = ts.start_torchserve()
        rerun()

    stop = st.sidebar.button("Stop")
    if stop:
        last_res()[0] = ts.stop_torchserve()
        rerun()

    torchserve_status = api.get_loaded_models()
    if torchserve_status:
        loaded_models_names = [
            m["modelName"] for m in torchserve_status["models"]]
    st.sidebar.subheader("Loaded models")
    st.sidebar.write(torchserve_status)

    stored_models = ts.get_model_store()
    st.sidebar.subheader("Available models")
    st.sidebar.write(stored_models)
    # ---------------

    st.markdown(f"**Last Message**: {last_res()[0]}")

    if torchserve_status:

        with st.expander(label="Register a model", expanded=False):

            st.header(
                "Register a model"
            )
            placeholder = st.empty()
            mar_path = placeholder.selectbox(
                "Choose mar file *", [default_key] + stored_models, index=0
            )
            p = st.checkbox("manually enter location")
            if p:
                mar_path = placeholder.text_input("Input mar file path*")
            
            model_name = st.text_input(
                label="Model name (overrides predefined)")
            col1, col2 = st.columns(2)
            batch_size = col1.number_input(
                label="batch_size", value=1, min_value=1, step=1)
            max_batch_delay = col2.number_input(
                label="max_batch_delay", value=0, min_value=0, step=100
            )
            initial_workers = col1.number_input(
                label="initial_workers", value=1, min_value=0, step=1
            )
            response_timeout = col2.number_input(
                label="response_timeout", value=0, min_value=0, step=100
            )
            handler = col1.text_input(label="handler")
            runtime = col2.text_input(label="runtime")
            proceed = st.button("Register")
            if proceed:
                if mar_path != default_key:
                    st.write(f"Registering Model...{mar_path}")
                    res = api.register_model(
                        mar_path,
                        model_name,
                        handler=handler,
                        runtime=runtime,
                        batch_size=batch_size,
                        max_batch_delay=max_batch_delay,
                        initial_workers=initial_workers,
                        response_timeout=response_timeout,
                    )
                    last_res()[0] = res
                    rerun()
                else:
                    st.warning(":octagonal_sign: Fill the required fileds!")

        with st.expander(label="Remove a model", expanded=False):

            st.header("Remove a model")
            model_name = st.selectbox(
                "Choose model to remove", 
                [default_key] + loaded_models_names, index=0)
            if model_name != default_key:
                default_version = api.get_model(model_name)[0]["modelVersion"]
                st.write(f"default version {default_version}")
                versions = api.get_model(model_name, list_all=True)
                versions = [m["modelVersion"] for m in versions]
                version = st.selectbox(
                    "Choose version to remove", 
                    [default_key] + versions, index=0)
                proceed = st.button("Remove")
                if proceed:
                    if model_name != default_key and version != default_key:
                        res = api.delete_model(model_name, version)
                        last_res()[0] = res
                        rerun()
                    else:
                        st.warning(":octagonal_sign: Pick a model & version!")

        with st.expander(label="Get model details", expanded=False):

            st.header("Get model details")
            model_name = st.selectbox(
                "Choose model", [default_key] + loaded_models_names, index=0
            )
            if model_name != default_key:
                default_version = api.get_model(model_name)[0]["modelVersion"]
                st.write(f"default version {default_version}")
                versions = api.get_model(model_name, list_all=False)
                versions = [m["modelVersion"] for m in versions]
                version = st.selectbox(
                    "Choose version", [default_key, "All"] + versions, index=0
                )
                if model_name != default_key:
                    if version == "All":
                        res = api.get_model(model_name, list_all=True)
                        st.write(res)
                    elif version != default_key:
                        res = api.get_model(model_name, version)
                        st.write(res)

        with st.expander(label="Scale workers", expanded=False):
            st.header(
                "Scale workers"
            )
            model_name = st.selectbox(
                "Pick model", [default_key] + loaded_models_names, index=0
            )
            if model_name != default_key:
                default_version = api.get_model(model_name)[0]["modelVersion"]
                st.write(f"default version {default_version}")
                versions = api.get_model(model_name, list_all=False)
                versions = [m["modelVersion"] for m in versions]
                version = st.selectbox("Choose version", 
                                       ["All"] + versions, index=0)

                col1, col2, col3 = st.columns(3)
                min_worker = col1.number_input(
                    label="min_worker(optional)", value=-1, min_value=-1, step=1
                )
                max_worker = col2.number_input(
                    label="max_worker(optional)", value=-1, min_value=-1, step=1
                )
                proceed = st.button("Apply")
                if proceed and model_name != default_key:
                    if version == "All":
                        version = None
                    if min_worker == -1:
                        min_worker = None
                    if max_worker == -1:
                        max_worker = None


                    res = api.change_model_workers(model_name,
                                                version=version,
                                                min_worker=min_worker,
                                                max_worker=max_worker,
                                                )
                    last_res()[0] = res
                    rerun()
        

def main():
    if handle_status(status):
        dashboard()


if __name__ == "__main__":
    page_config("Serving", "🧞‍♂️")
    _, status, _ = auth()
    main()