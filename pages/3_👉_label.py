import os
import supabase # TODO: in utils
import streamlit as st
import json

from supabase import create_client, Client

from src.app_utils import auth, handle_status, page_config


with open("tmp.json", "r") as f:
    _labels_dict = json.load(f)
    _labels = [v for k,v in _labels_dict.items()]
    _labels_to_ints = {
        v: int(k) for k,v in _labels_dict.items()
    }
    print(_labels_to_ints)

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
        print(e)



def main():
    if handle_status(status):
        conf = st.slider('Confidence threshold', 0.0, 0.99, 0.9)
        label_subjects = fetch_table(conf)
        if len(label_subjects) > 0:
            fetch_bucket(label_subjects[0]["name"])
            label = st.selectbox(
                'Label',
                _labels,
                index=_labels_to_ints[label_subjects[0]["label"]],
                )
            if st.button('Next'):
                update_table(id=label_subjects[0]["id"], label=label)
        else:
            st.write("No saved images below the selected confidence threshold")


if __name__ == "__main__":
    page_config("Labeling", "👉")
    authenticator, (name, status, user) = auth()
    supabase_client = setup_supabase()
    main()
