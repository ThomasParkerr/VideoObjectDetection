import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import tempfile
import os

# Load the pre-trained InceptionV3 model
@st.cache_resource
def load_model():
    return tf.keras.applications.InceptionV3(weights='imagenet')

model = load_model()

# Set the maximum file size (in bytes)
MAX_FILE_SIZE = 40 * 1024 * 1024  # 25 MB

def predict_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return decode_predictions(preds, top=3)[0]

def process_video(video_path, search_object=None):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    results = []
    frame_count = 0
    object_found = False
    object_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process 1 frame per second
        if frame_count % fps == 0:
            predictions = predict_image(frame)
            frame_result = {
                'frame': frame_count,
                'time': frame_count / fps,
                'predictions': [{'label': label, 'score': float(score)} for _, label, score in predictions]
            }
            results.append(frame_result)

            # Check if the searched object is in this frame
            if search_object and not object_found:
                for _, label, score in predictions:
                    if search_object.lower() in label.lower():
                        object_found = True
                        object_frame = frame
                        break

        frame_count += 1

    cap.release()
    return results, object_found, object_frame

def main():
    st.title("Video Object Detection")

    uploaded_file = st.file_uploader("Choose a video file (max 40 MB)", type=["mp4", "avi", "mov"])
    search_object = st.text_input("Search for object (optional)")

    if uploaded_file is not None:
        file_size = uploaded_file.size

        if file_size > MAX_FILE_SIZE:
            st.error(f"File size exceeds the limit of {MAX_FILE_SIZE / 1024 / 1024} MB")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            if uploaded_file.name.lower().endswith(('.mp4', '.avi', '.mov')):
                with st.spinner('Processing video...'):
                    results, object_found, object_frame = process_video(tmp_file_path, search_object)

                if search_object:
                    if object_found:
                        st.success(f'Object "{search_object}" found!')
                        st.image(object_frame, channels="BGR", caption=f'Frame containing "{search_object}"')
                    else:
                        st.warning(f'Object "{search_object}" not found in the video.')
                else:
                    for result in results:
                        st.write(f"Frame {result['frame']} (Time: {result['time']:.2f}s):")
                        for pred in result['predictions']:
                            st.write(f"  {pred['label']}: {pred['score']*100:.2f}%")
                        st.write("")
            else:
                img = cv2.imread(tmp_file_path)
                predictions = predict_image(img)
                st.image(img, channels="BGR", caption="Uploaded Image")
                st.write("Predictions:")
                for _, label, score in predictions:
                    st.write(f"  {label}: {score*100:.2f}%")

            # Clean up the temporary file
            os.unlink(tmp_file_path)

if __name__ == '__main__':
    main()
