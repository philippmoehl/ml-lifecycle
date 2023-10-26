import os
import streamlit as st

from src.app_utils import auth, handle_status, page_config



def main():
    if handle_status(status):
        st.write(f'Welcome *{name}*')



if __name__ == "__main__":
    page_config("Serving", "🧞‍♂️")
    authenticator, (name, status, user) = auth()
    main()