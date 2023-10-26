import logging
import os
import supabase # TODO: in utils
import streamlit as st
import json

from supabase import create_client

from src.app_utils import auth, handle_status, page_config, setup_supabase


logger = logging.getLogger(__name__)


with open("tmp.json", "r") as f:
    _labels_dict = json.load(f)
    _labels = [v for k,v in _labels_dict.items()]
    _labels_to_ints = {
        v: int(k) for k,v in _labels_dict.items()
    }


def fetch_table(conf, table="plants"):
    label_subjects = []
    (_, rows), _ = supabase_client.table(table).select("*").execute()
    rows = rows
    for row in rows:
        if float(row["confidence"]) <= conf:
            label_subjects.append(row)
    return label_subjects


def update_table(id, label, conf=1.0, table="plants"):
    data, count = supabase_client.table(table).update(
        {"label": label, 'confidence': conf}
        ).eq('id', id).execute()
    
    return data, count


def fetch_bucket(name, bucket="plants"):
    try:
        res = supabase_client.storage.from_(bucket).download(name)
        st.image(res, caption=name, use_column_width="always")
    except Exception as e:
        logger.warning(e)


def main():
    if handle_status(status):
        st.title("Labeling")
        st.sidebar.markdown("## Info")

        conf = st.sidebar.slider('Confidence threshold', 0.0, 0.99, 0.9)
        label_subjects = fetch_table(conf)
        st.sidebar.markdown("###  No. of selected images")
        st.sidebar.write(f"{len(label_subjects)}")
        if len(label_subjects) > 0:
            _name = label_subjects[0]["name"]
            _conf = label_subjects[0]["confidence"]
            st.sidebar.markdown("###  Current image")
            st.sidebar.write(_name)
            st.sidebar.markdown("###  Current Confidence")
            st.sidebar.write(_conf)
            fetch_bucket(_name)
            label = st.selectbox(
                'Label',
                _labels,
                index=_labels_to_ints[label_subjects[0]["label"]],
                )
            if st.button('Next'):
                # TODO: does not update correctly
                update_table(id=label_subjects[0]["id"], label=label)
        else:
            st.write("No saved images below the selected confidence threshold")


if __name__ == "__main__":
    page_config("Labeling", "👉")
    authenticator, (name, status, user) = auth()
    supabase_client = setup_supabase()
    main()
