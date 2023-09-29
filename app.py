import streamlit as st

from src.model import ImageClassifier


st.header("Predict if plant is healthy")


def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])

    if file_uploaded is not None:
        classifier = ImageClassifier(file_name=file_uploaded)
        fig = classifier.visalize()
        predictions = classifier.predict()

        st.write(predictions)
        st.pyplot(fig)


if __name__ == "__main__":
    main()
