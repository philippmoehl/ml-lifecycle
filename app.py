import streamlit as st

from src.model import ImageClassifier
from src.solution import generate_solution
from src.utils import setup


st.header("Predict if plant is healthy")


def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])

    if file_uploaded is not None:
        classifier = ImageClassifier(file_name=file_uploaded)
        fig = classifier.visalize()
        prediction, conf_score = classifier.predict()

        st.write(f"{prediction} with a {conf_score} percent confidence."
)
        if st.button("Solution"):
            solution = generate_solution(prediction)
            st.write(solution)
        st.pyplot(fig)


if __name__ == "__main__":
    setup()
    main()
