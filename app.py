from emotion_recognition import EmotionRecognizer
from utils import get_audio_config, extract_feature
import os
from sys import byteorder
from array import array
from struct import pack
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
#from deep_emotion_recognition import DeepEmotionRecognizer
#import tensorflow as tf
from convert_wavs import convert_audio

import soundfile
import librosa
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
import streamlit as st
import base64
import gc

import sounddevice as sd
from scipy.io.wavfile import write
from scipy.io import wavfile
import stqdm
#print("stqdm",stqdm.__version__)
from time import sleep
import streamlit as st
from stqdm import stqdm

import pydub
from pydub import AudioSegment
from pydub.utils import make_chunks
#print("pydub",pydub.__version__)


#Enable garbage collection
gc.enable()

#Hide warnings
st.set_option("deprecation.showfileUploaderEncoding", False)

#App title
st.title(" EMOTION RECOGNITION APP")

#Read files
my_path= '.'
audio1 = "/"
image1 = "emojis/angry_emoji.gif"
image2= "emojis/happy_emoji.gif"
image3= "emojis/neutral_emoji.gif"
image4= "emojis/background2.jpeg"
image5= "emojis/sad_emoji.gif"
image6= "emojis/ps.gif"



#Read sidebar image and title
st.sidebar.markdown("# PREDICT YOUR MOOD ! ! !")
st.sidebar.image(image4, width=300)

#Set 2 different pages: Upload and Record
pages= ['UPLOAD AUDIO', 'RECORD AUDIO']
page= st.radio('Navigation', pages)


def loadGif(image):
     file_ = open(image, "rb")
     contents = file_.read()
     data_url = base64.b64encode(contents).decode("utf-8")
     file_.close()
     st.markdown(
     f'<img src="data:image/gif;base64,{data_url}" alt="cat gif" width="300">',
     unsafe_allow_html=True,
     )

#Deploy function to run the model and print the predictions
def deploy(FILE, upload=False):
    ## Limit the uploaded audio file to only 10seconds
    if upload:
        myaudio = AudioSegment.from_wav(FILE) 
        chunk_length_ms = 10000 # pydub calculates in millisec
        chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

        #Export all of the individual chunks as wav files
        for i, chunk in enumerate(chunks):
            if i > 0:
                break  #Keep only the 1st slice (10s) of the upload audio
            FILE = f"chunk{i}.wav"
            chunk.export(FILE, format="wav")

    #Display the audio player to reproduce the uplaoded file
    st.sidebar.markdown('***')
    st.sidebar.markdown("# AUDIO PLAYER")   
    st.sidebar.audio(FILE)
    
    #Getting predictions by clicking on the predict button
    # if predict_button:
    #     #Initilizaing the model
    #     new_model=DeepEmotionRecognizer(emotions=['angry', 'sad', 'neutral', 'ps', 'happy'],
    #                                     verbose=2)
    #     #Loading the training weights
    #     new_model.train()

    #     #Predicting the final result
    #     result = new_model.predict(FILE)
        
    #     #Display the prediction + emoji
    #     if result == 'ps':
    #         st.write(f"The predicted emotion is: **Pleasant/Surprised**")
    #     else:
    #         st.write(f"The predicted emotion is: **{result}**")
    #     if result == "happy":
    #         loadGif(image2)
    #     elif result == "angry":
    #         loadGif(image1)

    #     elif result == "neutral":
    #         loadGif(image3)
    #     elif result == "sad":
    #         loadGif(image5)
    #     elif result == "ps":
    #         loadGif(image6)

    #     #Delete the model to free the memory
    #     del new_model
    #     gc.collect()



##Plot spectrogram function
def plot_specgram(FILE, WIDTH=500):
    samplingFrequency, signalData = wavfile.read(FILE)

    #duration= librosa.get_duration(filename=FILE) #Get file duration
    plt.title('Spectrogram')    
    Pxx, freqs, bins, im = plt.specgram(signalData,Fs=samplingFrequency,NFFT=512)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.xlim(left=0,right=5)

    plt.savefig('spectro2.png')
    st.image("spectro2.png", width=WIDTH)


#Select between 2 pages: UPLAOD or RECORD
if page == "UPLOAD AUDIO":
    #Set the box for the user to upload an image
    st.warning(" **Upload your Audio**")
    uploaded_audio = st.file_uploader("Upload your audio in WAV format", type=["wav", "mp3"])
    if uploaded_audio:
        plot_specgram(uploaded_audio)
    #Get predictions only after clicking on the predict button
    predict_button= st.button('🎲 PREDICT ')
    if uploaded_audio is not None:
        #Start the model inference
        deploy(uploaded_audio, upload=True)
        #Free the memory
        del uploaded_audio
        gc.collect()

#If the selected page is: RECORD       
else:
    #Set the record button to start recording
    st.warning('**Record your voice**')
    record_mode=st.button("🎤 RECORD ")
    
    if record_mode:
        fs = 44100  # Sample rate
        seconds = 5  # Duration of the recording: MAX 10s
        st.write('**START TALKING!**')
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        
        #Print the pregression bar
        for _ in stqdm(range(5)):
            sleep(1)
        #Save the recorded audio file to the disk
        st.write('*Saving the recorded file...*')
        sd.wait()  # Wait until recording is finished
        write('output.wav', fs, myrecording)
        
        #Open the recorded file and play it in the audio player
        recorded_file='output.wav'
        st.sidebar.markdown('# AUDIO PLAYER')
        # st.sidebar.audio(recorded_file)
        st.success('**Audio recorded successfully!**')
    
    #Finally, predict the user recorded audio
    if os.path.isfile('output.wav'):
        plot_specgram('output.wav')
        predict_button= st.button('🎲 PREDICT')
        deploy('output.wav')
        gc.collect()    



