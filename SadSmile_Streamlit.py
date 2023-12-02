import cv2
import os
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import pickle
from pathlib import Path
from PIL import Image
import random
import plotly.express as px
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.svm import SVC


def reduce_array(dataset):
    reduceddata = np.zeros((len(dataset),len(dataset[0])))
    for i in tqdm(range(len(dataset))):
        for j in range(len(dataset[i])):
            #converting each array to a single float of the mean value of each value in the array
            reduceddata[i,j] = dataset[i,j].mean()
    return reduceddata

def select_bar_column_pallete(values, select, sel_color = 'orange', other_color = 'lightgrey'):
    pal = []
    for item in values:
        if item == select:
            pal.append(sel_color)
        else:
            pal.append(other_color)
    return pal

st.header("Facial Emotion Classification!")

OUTPUT_DIR = Path('output')
IMG_SIZE = (48,48)
labels = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Suprise'}

with st.sidebar:
    selected = option_menu("Main Menu", ["Introduction", "Emotion Classifier", ], 
        icons=['house', 'file-bar-graph'], menu_icon="cast", default_index=0)
    selected

    
    st.subheader("Lets explore Student Success!")

    st.markdown("Student success is a multifaceted affair combining various psychological, demographical, and social factors. The purpose of the current project is to facilitate the exploration of some of these variables in hopes that one might gain a more nuanced understanding regarding the ways in which these components influence and inform the academic acheivements of high school students.")
    st.subheader("What Defines Success?")
    st.markdown("We will define \'Success\', as a students ability to understand the material distrubuted to them throughout their time taking the course. For the purposes of this project, we will measure success using a students final grade in the respective class, with a score equal or above 10 representing \'success\' and a score below 10 representing \'failure\'. It is important to note that this method of defining student success is problematically reductionalist for multiple reasons, but due to constraints in both resoruces and understanding, this was the best metric available.")
    st.markdown("Continue reading in the \'About this Dataset\' tab to learn more about the dataset used in this project, or click around in the Interactive Data Explorer to start exploring!")
    
    describe = st.selectbox("Would you like to view summary statistics of the numerical variables?", ["No", "Yes, summarize the Math dataset", "Yes, summarize the Portuguese dataset"])


if selected == "Introduction":
    tab1, tab2= st.tabs(["Introduction", "About this Dataset"])
    with tab1: 
        st.markdown("Student success is a multifaceted affair combining various psychological, demographical, and social factors. The purpose of the current project is to facilitate the exploration of some of these variables in hopes that one might gain a more nuanced understanding regarding the ways in which these components influence and inform the academic acheivements of high school students.")
        st.subheader("What Defines Success?")
        
    with tab2: 
        st.subheader("About this Dataset:")
        
        st.markdown("These datasets were collected by Paulo Cortez and Alice Silva in 2006 during a study aimed at assessing student achievement in secondary school. Specifically, data was collected from a total of 788 high school students located in two public schools from the Alentejo region of Portugal, Gabriel Pereira and Mousinho da Silveira. A total of two datasets were collected, a collection of Math Scores (with 395 records) and Portuguese Scores (with 649 records) These two classes in particular were chosen due to the critical role they play in subsequential classes. For example, researchers identify physics and history as core classes that build off of the fundamental understanding of Math and Portuguese (in Portugal specifically). Thus, a student's success in their academic future, as predicted by researchers, is contingent on their understanding of these two core classes.")
        st.markdown("Continue below to view the datasets in their original states, and a description of each variable and what it represents.")
        viewdata = st.selectbox("Which dataset would you like to display?", ["None", "Math Scores Dataset", "Portuguese Scores Dataset"])


if selected == "Emotion Classifier":
    
    model_choice = st.selectbox("##### Choose a model:", ["SGD", "MLP"])
    
    if model_choice == "SGD":
        with open(OUTPUT_DIR / 'SGDmodel_LDAdata.pickle', "rb") as f:
                model = pickle.load(f)
        with open(OUTPUT_DIR / 'trainedlda.pickle', "rb") as f:
                fittedlda = pickle.load(f)
    
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        col1, col2 = st.columns([1,1])
        uploaded_image = Image.open(uploaded_file)
        uploaded_image = np.array(uploaded_image)
        uploaded_image = uploaded_image.astype('float32') / 255
        img = cv2.resize(uploaded_image, IMG_SIZE, interpolation = cv2.INTER_AREA)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_lda = fittedlda.transform(gray_image.reshape(1,48*48))
        predictions = model.predict_proba(img_lda)
        with col1:
            fig, ax = plt.subplots()
            fig = plt.imshow(gray_image, cmap = 'gray')
            ax.axis('off')
            st.subheader(labels[np.argmax(predictions.reshape(6))])
            st.pyplot(fig.get_figure())
        with col2:
            color = ['lightgrey', 'orange', 'lightgrey', 'lightgrey', 'lightgrey', 'lightgrey']
            #select_bar_column_pallete(labels.keys(), np.argmax(predictions.reshape(6))
            fig = px.bar(y=list(labels.values()), x=predictions.reshape(6))
            st.plotly_chart(fig, use_container_width=True)