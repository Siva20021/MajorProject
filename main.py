import streamlit as st
import librosa 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

import librosa.display

def extract_features(audio, sample_rate):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mean_mfcc = np.mean(mfccs, axis=1)
    return mean_mfcc.reshape(1, -1)



def plot_waveform(audio, sr):
    plt.figure(figsize=(10, 4))
    # Calculate the time axis in seconds
    time = np.linspace(0, len(audio) / sr, num=len(audio))
    # Plot the waveform with time on the x-axis and amplitude on the y-axis
    plt.plot(time, audio)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform of Audio')
    plt.grid(True)
    st.pyplot(plt.gcf())


# Streamlit app title
st.title('Audio Fake Detection')
# Upload the file 
uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

model=load_model('./my_model.h5')
print(model)

if uploaded_file is not None:
    
    # Load the model 
    audio, sample_rate = librosa.load(uploaded_file, sr=None)
    st.text(f"Sample rate: {sample_rate} Hz")

    # Display audio player
    st.audio(uploaded_file, format='audio/wav', start_time=0)

    # Plot waveform
    plot_waveform(audio, sample_rate)

    # Extract features
    with st.spinner('Extracting features...'):
        mfcc_reshaped = extract_features(audio, sample_rate)
        st.write('Features extracted')


    with st.spinner('Making prediction...'):
        prediction = model.predict(mfcc_reshaped)
        st.write('Prediction made', prediction[0])
        if prediction[0]<0.5:
            st.success('Prediction: Real')
        else:
            st.error('Prediction: Fake')

