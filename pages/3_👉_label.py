import logging
import streamlit as st

from src.app_utils import auth, handle_status, page_config, show_image

logger = logging.getLogger(__name__)


def main():
    if handle_status(status):
        st.title("Labelling")

        # ----sidebar----
        st.sidebar.markdown("## Info")
        conf = st.sidebar.slider('Confidence threshold', 0.0, 0.99, 0.9)
        label_subjects = st.session_state.supabase.fetch_table(conf)
        st.sidebar.markdown("### No. of selected images")
        st.sidebar.write(f"{len(label_subjects)}")
        # ---------------
        if len(label_subjects) > 0:
            _name = label_subjects[0]["name"]
            _conf = label_subjects[0]["confidence"]
            st.sidebar.markdown("### Current image")
            st.sidebar.write(_name)
            st.sidebar.markdown("### Current Confidence")
            st.sidebar.write(_conf)
            res = st.session_state.supabase.fetch_bucket(_name)
            show_image(res, _name)
            label = st.selectbox(
                'Label',
                [k for k, _ in st.session_state.labels_to_ints.items()],
                index=st.session_state.labels_to_ints[
                    label_subjects[0]["label"]],
                )
            if st.button('Next'):
                st.session_state.supabase.update_table(
                    id=label_subjects[0]["id"], label=label)
        else:
            st.write("No saved images below the selected confidence threshold")


if __name__ == "__main__":
    page_config("Labelling", "👉")
    _, status, _ = auth()
    main()
