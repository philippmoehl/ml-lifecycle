import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras import preprocessing


st.header("Predict if plant is healthy")


def predict(image):
    classifier_model = "https://tfhub.dev/agripredict/disease-classification/1"
    IMAGE_SHAPE = (300, 300,3)
    model = tf.keras.Sequential([
    hub.KerasLayer(classifier_model,input_shape=IMAGE_SHAPE)])
    test_image = image.resize((300,300))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = [
          'Tomato Healthy',
          'Tomato Septoria Leaf Spot',
          'Tomato Bacterial Spot',
          'Tomato Blight',
          'Cabbage Healthy',
          'Tomato Spider Mite',
          'Tomato Leaf Mold',
          'Tomato_Yellow Leaf Curl Virus',
          'Soy_Frogeye_Leaf_Spot',
          'Soy_Downy_Mildew',
          'Maize_Ravi_Corn_Rust',
          'Maize_Healthy',
          'Maize_Grey_Leaf_Spot',
          'Maize_Lethal_Necrosis',
          'Soy_Healthy',
          'Cabbage Black Rot']
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    results = {
          'Tomato Healthy':0,
          'Tomato Septoria Leaf Spot':0,
          'Tomato Bacterial Spot':0,
          'Tomato Blight':0,
          'Cabbage Healthy':0,
          'Tomato Spider Mite':0,
          'Tomato Leaf Mold':0,
          'Tomato_Yellow Leaf Curl Virus':0,
          'Soy_Frogeye_Leaf_Spot':0,
          'Soy_Downy_Mildew':0,
          'Maize_Ravi_Corn_Rust':0,
          'Maize_Healthy':0,
          'Maize_Grey_Leaf_Spot':0,
          'Maize_Lethal_Necrosis':0,
          'Soy_Healthy':0,
          'Cabbage Black Rot':0
}

    result = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } percent confidence."
    return result


def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])

    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        fig = plt.figure()
        plt.imshow(image)
        plt.axis("off")
        predictions = predict(image)
        st.write(predictions)
        st.pyplot(fig)


if __name__ == "__main__":
    main()
