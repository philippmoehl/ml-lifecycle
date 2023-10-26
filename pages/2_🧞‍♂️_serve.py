import logging
import os
import streamlit as st
import subprocess
from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import httpx
from httpx import Response

from src.app_utils import auth, handle_status, page_config

logger = logging.getLogger(__name__)


ENVIRON_WHITELIST = [
    "LOG_LOCATION", "METRICS_LOCATION"
]


class LocalTS:
    def __init__(self,
                 model_store: str,
                 config_path: Optional[str] = None,
                 log_location: Optional[str] = None,
                 metrics_location: Optional[str] = None,
                 log_config: Optional[str] = None) -> None:
        new_env = {}
        env = os.environ
        for x in ENVIRON_WHITELIST:
            if x in env:
                new_env[x] = env[x]
        if config_path:
            new_env["TS_CONFIG_FILE"] = config_path
        if log_location:
            new_env["LOG_LOCATION"] = log_location
            if not os.path.isdir(log_location):
                os.makedirs(log_location, exist_ok=True)
        if metrics_location:
            new_env["METRICS_LOCATION"] = metrics_location
            if not os.path.isdir(metrics_location):
                os.makedirs(metrics_location, exist_ok=True)

        self.model_store = model_store
        self.config_path = config_path
        self.log_location = log_location
        self.metrics_location = metrics_location
        self.log_config = log_config
        self.env = new_env

    def start_torchserve(self) -> str:

        if not os.path.exists(self.model_store):
            return "Can't find model store path"
        elif not os.path.exists(self.config_path):
            return "Can't find configuration path"
        dashboard_log_path = os.path.join(
            self.log_location, "ts-app.log"
        ) if self.log_location is not None else None
        torchserve_cmd = f"torchserve --start --ncs --model-store {self.model_store}"
        if self.log_config:
            torchserve_cmd += f" --log-config {self.log_config}"
        p = subprocess.Popen(
            torchserve_cmd.split(" "),
            env=self.env,
            stdout=subprocess.DEVNULL,
            stderr=open(dashboard_log_path, "a+")
            if dashboard_log_path else subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True
        )
        p.communicate()
        if p.returncode == 0:
            return f"Torchserve is starting (PID: {p.pid})..please refresh page"
        else:
            return f"Torchserve is already started. Check {dashboard_log_path} for errors"

    def stop_torchserve(self) -> Union[str, Exception]:
        try:
            p = subprocess.run(["torchserve", "--stop"],
                               check=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True)
            return p.stdout
        except (subprocess.CalledProcessError, OSError) as e:
            return e

    def get_model_store(self) -> List[str]:
        return os.listdir(self.model_store)


class ManagementAPI:
    def __init__(self, address: str, error_callback: Callable = None) -> None:
        self.address = address
        if not error_callback:
            error_callback=self.default_error_callback
        self.client = httpx.Client(timeout=1000,
                                   event_hooks={"response": [error_callback]})
    @staticmethod
    def default_error_callback(response: Response) -> None:
        if response.status_code != 200:
            logger.warning(f"status code: {response.status_code},{response}")

    def get_loaded_models(self) -> Optional[Dict[str, Any]]:
        try:
            res = self.client.get(self.address + "/models")
            return res.json()
        except httpx.HTTPError:
            return None

    def get_model(self,
                  model_name: str,
                  version: Optional[str] = None,
                  list_all: bool = False) -> List[Dict[str, Any]]:
        req_url = self.address + "/models/" + model_name
        if version:
            req_url += "/" + version
        elif list_all:
            req_url += "/all"


        res = self.client.get(req_url)
        return res.json()

    def register_model(
        self,
        mar_path: str,
        model_name: Optional[str] = None,
        handler: Optional[str] = None,
        runtime: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_batch_delay: Optional[int] = None,
        initial_workers: Optional[int] = None,
        response_timeout: Optional[int] = None,
        is_encrypted: Optional[bool] = None,
    ) -> Dict[str, str]:

        req_url = self.address + "/models?url=" + mar_path + "&synchronous=false"
        if model_name:
            req_url += "&model_name=" + model_name
        if handler:
            req_url += "&handler=" + handler
        if runtime:
            req_url += "&runtime=" + runtime
        if batch_size:
            req_url += "&batch_size=" + str(batch_size)
        if max_batch_delay:
            req_url += "&max_batch_delay=" + str(max_batch_delay)
        if initial_workers:
            req_url += "&initial_workers=" + str(initial_workers)
        if response_timeout:
            req_url += "&response_timeout=" + str(response_timeout)
        if is_encrypted:
            req_url += "&s3_sse_kms=true"

        res = self.client.post(req_url)
        return res.json()

    def delete_model(self,
                     model_name: str,
                     version: Optional[str] = None) -> Dict[str, str]:
        req_url = self.address + "/models/" + model_name
        if version:
            req_url += "/" + version
        res = self.client.delete(req_url)
        return res.json()

    def change_model_default(self,
                             model_name: str,
                             version: Optional[str] = None):
        req_url = self.address + "/models/" + model_name
        if version:
            req_url += "/" + version
        req_url += "/set-default"
        res = self.client.put(req_url)
        return res.json()

    def change_model_workers(
            self,
            model_name: str,
            version: Optional[str] = None,
            min_worker: Optional[int] = None,
            max_worker: Optional[int] = None,
            number_gpu: Optional[int] = None) -> Dict[str, str]:
        req_url = self.address + "/models/" + model_name
        if version:
            req_url += "/" + version
        req_url += "?synchronous=false"
        if min_worker:
            req_url += "&min_worker=" + str(min_worker)
        if max_worker:
            req_url += "&max_worker=" + str(max_worker)
        if number_gpu:
            req_url += "&number_gpu=" + str(number_gpu)
        res = self.client.put(req_url)
        return res.json()
    

def error_callback(response:Response):
    if response.status_code != 200:
        st.write("There was an error!")
        st.write(response)


@st.cache_resource()
def initialize():
    api_address = "http://127.0.0.1:8081"
    config_path = str(Path("config.properties").resolve())
    config = open(config_path, "r").readlines()
    model_store = str(Path("model_archive").resolve())
    log_location = str(Path("./logs").resolve())
    metrics_location = str(Path("./logs/metrics/").resolve())
    log_config = None
    
    api = ManagementAPI(api_address, error_callback)
    ts = LocalTS(
        model_store, config_path, log_location, metrics_location, log_config)
    torchserve_status = api.get_loaded_models()
    
    if not torchserve_status:
       st.write("Starting Torchserve")
       last_res()[0] = ts.start_torchserve()
       rerun() 
    return (config, api, ts)


@st.cache_resource()
def last_res():
    return ["Nothing"]


def rerun():
    st.experimental_rerun()


def dashboard():
    st.title("Serving Management")
    default_key = "None"
    # TODO: use a config.yaml in the main app loader
    config, api, ts = initialize()

    ##########Sidebar##########
    st.sidebar.markdown("## Controls")
    start = st.sidebar.button("Start")
    if start:
        last_res()[0] = ts.start_torchserve()
        rerun()

    stop = st.sidebar.button("Stop")
    # TODO: should not show the model regiuster etc after stop
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
    ####################

    st.markdown(f"**Last Message**: {last_res()[0]}")

    with st.expander(label="Show torchserve config", expanded=False):
        st.write(config)
        st.markdown("[configuration docs](https://pytorch.org/serve/configuration.html)")

    if torchserve_status:

        with st.expander(label="Register a model", expanded=False):

            st.header(
                "Register a model [(docs)](https://pytorch.org/serve/management_api.html#register-a-model)"
            )
            placeholder = st.empty()
            mar_path = placeholder.selectbox(
                "Choose mar file *", [default_key] + stored_models, index=0
            )
            p = st.checkbox("manually enter location")
            if p:
                mar_path = placeholder.text_input("Input mar file path*")
            
            model_name = st.text_input(label="Model name (overrides predefined)")
            col1, col2 = st.columns(2)
            batch_size = col1.number_input(label="batch_size", value=0, min_value=0, step=1)
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
            is_encrypted = st.checkbox("SSE-KMS Encrypted", help="Refer to https://github.com/pytorch/serve/blob/v0.5.0/docs/management_api.md#encrypted-model-serving")
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
                        is_encrypted=is_encrypted,
                    )
                    last_res()[0] = res
                    rerun()
                else:
                    st.warning(":octagonal_sign: Fill the required fileds!")

        with st.expander(label="Remove a model", expanded=False):

            st.header("Remove a model")
            model_name = st.selectbox(
                "Choose model to remove", [default_key] + loaded_models_names, index=0
            )
            if model_name != default_key:
                default_version = api.get_model(model_name)[0]["modelVersion"]
                st.write(f"default version {default_version}")
                versions = api.get_model(model_name, list_all=True)
                versions = [m["modelVersion"] for m in versions]
                version = st.selectbox(
                    "Choose version to remove", [default_key] + versions, index=0
                )
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
                "Scale workers [(docs)](https://pytorch.org/serve/management_api.html#scale-workers)"
            )
            model_name = st.selectbox(
                "Pick model", [default_key] + loaded_models_names, index=0
            )
            if model_name != default_key:
                default_version = api.get_model(model_name)[0]["modelVersion"]
                st.write(f"default version {default_version}")
                versions = api.get_model(model_name, list_all=False)
                versions = [m["modelVersion"] for m in versions]
                version = st.selectbox("Choose version", ["All"] + versions, index=0)

                col1, col2, col3 = st.columns(3)
                min_worker = col1.number_input(
                    label="min_worker(optional)", value=-1, min_value=-1, step=1
                )
                max_worker = col2.number_input(
                    label="max_worker(optional)", value=-1, min_value=-1, step=1
                )
                #             number_gpu = col3.number_input(label="number_gpu(optional)", value=-1, min_value=-1, step=1)
                proceed = st.button("Apply")
                if proceed and model_name != default_key:
                    # number_input can't be set to None
                    if version == "All":
                        version = None
                    if min_worker == -1:
                        min_worker = None
                    if max_worker == -1:
                        max_worker = None
                    #                 if number_gpu == -1:
                    #                     number_gpu=None

                    res = api.change_model_workers(model_name,
                                                version=version,
                                                min_worker=min_worker,
                                                max_worker=max_worker,
                                                #                     number_gpu=number_gpu,
                                                )
                    last_res()[0] = res
                    rerun()
        

def main():
    if handle_status(status):
        dashboard()


if __name__ == "__main__":
    page_config("Serving", "🧞‍♂️")
    authenticator, (name, status, user) = auth()
    main()