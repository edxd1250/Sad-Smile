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

st.set_option('deprecation.showPyplotGlobalUse', False)

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

def load_data(dir_path, IMG_SIZE):
   
    X = []
    y = []
    i = 0
    labels = dict()
    #Loading images folder by folder
    for path in tqdm(sorted(os.listdir(dir_path))):
        if not path.startswith('.'):
            #Setting image label to the name of folder image is in
            labels[i] = path
            for file in os.listdir(dir_path +'/'+ path):
                if not file.startswith('.'):
                    #loading in each individual image
                    img = Image.open(dir_path + '/'+ path + '/' + file)
                    imagedat = np.array(img)
                    imagedat = imagedat.astype('float32') / 255
                    resized = cv2.resize(imagedat, IMG_SIZE, interpolation = cv2.INTER_AREA)
                    X.append(resized)
                    y.append(i)
            i += 1
    #Converting lists to np arrays
    X = np.array(X)
    y = np.array(y)
    print(f'{len(X)} images loaded from {dir_path} directory.')
    return X, y, labels

def check_labels(dataset, y_labels, labels, sample_size = (3,5)):
    samples = np.random.randint(0, len(dataset), size=sample_size)
    fig, ax = plt.subplots(len(samples),len(samples[0]))
    for i in range(len(samples)):
        for j in range(len(samples[0])):
            ax[i,j].imshow(dataset[samples[i,j]], cmap='gray')
            ax[i,j].set_title(f'{labels[y_labels[samples[i,j]]]} Face')
            ax[i,j].axis('off')

def augment_images(dataset, tilt=20, flip=True, vshift=10, hshift=10):
    augmented_images = []
    for image in dataset:
        # Randomly flip the image horizontally or vertically
        if flip == True:
            if random.choice([0,1]) == 1:
                image = cv2.flip(image, 1)

        # Randomly rotate the image up to 20 degrees
        angle = random.uniform(-tilt, tilt)
        rows, cols = image.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

        # Randomly shift horizontally or vertically
        tx = random.randint(-vshift, vshift)
        ty = random.randint(-hshift, hshift)
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, translation_matrix, (cols, rows))

        augmented_images.append(image)

    return np.array(augmented_images)

def getclasssamples(y, percent=.3):
    samples = np.empty_like(np.random.choice(np.argwhere(y==1).reshape(len(np.argwhere(y==1))), replace=False))
    for i in np.unique(y):
        labels = np.argwhere(y==i)
        num_samples = int(percent * len(labels))
        selected_indices = np.random.choice(labels.reshape(len(labels)), size=num_samples, replace=False)
        samples = np.append(samples,selected_indices)
    #for some reason this index keeps getting generated??
    if samples[0] > len(y):
        samples = np.delete(samples, 0)

    return samples
OUTPUT_DIR = Path('output')
TRAIN_DIR = ('imagedata/Training/Training')
TEST_DIR = ('imagedata/Testing/Testing')
IMG_SIZE = (48,48)
labels = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Suprise'}

#X_train, y_train, train_labels = load_data(TRAIN_DIR, IMG_SIZE)
#X_test, y_test, test_labels = load_data(TEST_DIR,IMG_SIZE)

with st.sidebar:
    selected = option_menu("Main Menu", ["Introduction", "Emotion Classifier", ], 
        icons=['house', 'file-bar-graph'], menu_icon="cast", default_index=0)
 

if selected == "Introduction":
    tab1, tab2, tab3= st.tabs(["Introduction", "About this Dataset", "Dataset Augmention"])
    with tab1: 
        st.header("Facial Emotion Classification!")
        st.image("behavior_facial_expressions_emotions-100718798-orig.webp")
        st.markdown("Understanding human emotions is often a highly complicated task that requires situational awareness, an understanding of how emotions are typically displayed, and an intuition towards how that emotion might look on the person in question. Our amazing brains are able to take this information, processes and analyze it within seconds to give a (sometimes) acurate estimation on what emotion an individual might be feeling. But what about computers? How might a computer understand the complexities of facial emotions and be able to detect the differences? And how might this technology be used in the future? The current project aims to address these problems. Click through to learn more!.")
 
        
    with tab2: 
        st.subheader("About this Dataset:")
        link = "https://www.kaggle.com/datasets/apollo2506/facial-recognition-dataset"
        st.markdown("This dataset used to train and test the models in this project was collected from the Kaggle competition: \"Challenges in Representation Learning: Facial Expression Recognition Challenge\". and originally used in the comp contains folders pertaining to different expressions of the human face, namely. Surprise, Anger, Happiness, Sad, Neutral, and Fear. The dataset was presplit into training and testing subcategories at about an 80\% ratio. The dataset contained 35257 images, with the training set containing  28,079 samples in total and the testing set consisting of 7,178 samples in total. The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The dataset was originally curated by Pierre-Luc Carrier and Aaron Courville, as part of a research project. A link to the dataset can be found below:")
        st.markdown("[Kaggle Facial Recognition Dataset](%s)" %link)
        
        st.markdown("Use the controls below to view a sample of the images used during training:")
        col1, col2, col3 = st.columns([2,2,1])
        with col1:
            numberrows = st.number_input("Number of Rows:", value=3, placeholder="3")
        with col2:
            numbercols = st.number_input("Number of Columns:", value=5, placeholder="5")
        with col3:
            generate = st.button("Generate", type="primary")
        if generate:
            X_train, y_train, train_labels = load_data(TRAIN_DIR, IMG_SIZE)
            fig = check_labels(X_train, y_train, train_labels, sample_size = (numberrows,numbercols))
            st.pyplot(fig)

    with tab3:
        st.subheader("Data augmentations:")
        augview = st.radio("Would you like to:", ["View training augments", "Generate training augments"], horizontal=True,)
        if augview == "View training augments":
            st.write("As part of the preprocessing process, data was augmented in the following ways:")
            string = '''- 30% augmented data added\\
                Height randomly shifted by up to 20%\\
                    Width randomly shifted by up to 20%\\
                    Images randomly flipped horizontally\\
                    Images randomly rotated left/right by up to 20 degrees\\
                        RGB values normalized (0-255) -> (0-1)'''
            st.write(string)
            st.write("Click the button below to preview some of the augmented data:")
            col1, col2, col3 = st.columns([2,2,1])
            with col1:
                numberrows = st.number_input("Number of Rows:", value=3, placeholder="3",key='row2')
            with col2:
                numbercols = st.number_input("Number of Columns:", value=5, placeholder="5",key='col2')
            with col3:
                generate = st.button("Generate", type="primary",key='but2')
            if generate:
                X_train, y_train, train_labels = load_data(TRAIN_DIR, IMG_SIZE)
                augmentindicies = getclasssamples(y_train)
                X_augmented = augment_images(X_train[augmentindicies])
                fig = check_labels(X_augmented, y_train, train_labels)
                st.pyplot(fig)
        if augview == "Generate training augments":
            tilt = st.number_input("##### Max tilt:", value=20)
            vshift = st.number_input("##### Max Vertical Shift (in pixels):", value=10)
            hshift = st.number_input("##### Max Horizontal Shift (in pixels):", value=10)
            flip = st.toggle("Randomly flip images?")
            col1, col2, col3 = st.columns([2,2,1])
            with col1:
                numberrows = st.number_input("Number of Rows:", value=3, placeholder="3",key='row2')
            with col2:
                numbercols = st.number_input("Number of Columns:", value=5, placeholder="5",key='col2')
            with col3:
                generate = st.button("Generate", type="primary",key='but2')
            if generate:
                X_train, y_train, train_labels = load_data(TRAIN_DIR, IMG_SIZE)
                augmentindicies = getclasssamples(y_train)
                X_augmented = augment_images(X_train[augmentindicies],flip=flip,vshift=vshift,hshift=hshift,tilt=tilt)
                st.session_state['X_augmented'] = np.concatenate((X_train, X_augmented), axis=0)
                st.session_state['y_fulltrain'] = np.concatenate((y_train, y_train[augmentindicies]), axis=0)
                fig = check_labels(X_augmented, y_train, train_labels)
                st.pyplot(fig)


            



        #samples = check_labels(X_train, y_train, train_labels, sample_size = (3,5))
       


if selected == "Emotion Classifier":
    tab1, tab2= st.tabs(["Model Explorer", "Model Builder"])
    with tab1:
        with open(OUTPUT_DIR / 'trainedscaler.pickle', "rb") as f:
            fittedscaler = pickle.load(f)
        with open(OUTPUT_DIR / 'trainedPCA.pickle', "rb") as f:
            fittedpca = pickle.load(f)
        with open(OUTPUT_DIR / 'trainedLDA.pickle', "rb") as f:
            fittedlda = pickle.load(f)
        
        comparemode = st.toggle('Compare Models?')
        if comparemode == True:
            model_choice = st.selectbox("##### Model 1:", ["Stochastic Gradient Descent","Gaussian Naive Bayes","Support Vector Machine","Multilayer Perceptron", "User Model 1", "User Model 2"])
            model_choice2 = st.selectbox("##### Model 2:", ["Stochastic Gradient Descent","Gaussian Naive Bayes","Support Vector Machine","Multilayer Perceptron", "User Model 1", "User Model 2"])
            
            
            if model_choice == "Stochastic Gradient Descent":
                with open(OUTPUT_DIR / 'SGDmodel_LDAdata.pickle', "rb") as f:
                        model = pickle.load(f)
            if model_choice == "Gaussian Naive Bayes":
                with open(OUTPUT_DIR / 'NBmodel_LDAdata.pickle', "rb") as f:
                        model = pickle.load(f)
            if model_choice == "Support Vector Machine":
                with open(OUTPUT_DIR / 'SVMmodel_LDAdata.pickle', "rb") as f:
                        model = pickle.load(f)
            if model_choice == "Multilayer Perceptron":
                with open(OUTPUT_DIR / 'SGDmodel_LDAdata.pickle', "rb") as f:
                        model = pickle.load(f)
            if model_choice == "User Model 1":
                if 'usermodel1' in st.session_state:
                    model = st.session_state['usermodel1']
                else:
                    st.write("User Model 1 has not been defined!")
            if model_choice == "User Model 2":
                if 'usermodel2' in st.session_state:
                    model = st.session_state['usermodel2']
                else:
                    st.write("User Model 2 has not been defined!")
                
            if model_choice2 == "Stochastic Gradient Descent":
                with open(OUTPUT_DIR / 'SGDmodel_LDAdata.pickle', "rb") as f:
                        model2 = pickle.load(f)
            if model_choice2 == "Gaussian Naive Bayes":
                with open(OUTPUT_DIR / 'NBmodel_LDAdata.pickle', "rb") as f:
                        model2 = pickle.load(f)
            if model_choice2 == "Support Vector Machine":
                with open(OUTPUT_DIR / 'SVMmodel_LDAdata.pickle', "rb") as f:
                        model2 = pickle.load(f)
            if model_choice2 == "Multilayer Perceptron":
                with open(OUTPUT_DIR / 'SGDmodel_LDAdata.pickle', "rb") as f:
                        model2 = pickle.load(f)
            if model_choice2 == "User Model 1":
                if 'usermodel1' in st.session_state:
                    model2 = st.session_state['usermodel1']
                else:
                    st.write("User Model 1 has not been defined!")
            if model_choice2 == "User Model 2":
                if 'usermodel2' in st.session_state:
                    model2 = st.session_state['usermodel2']
                else:
                    st.write("User Model 2 has not been defined!")
            
            uploaded_file = st.file_uploader("Upload Image")
            col1, col2 = st.columns([1,1])
            
            with col1:
                if uploaded_file is not None and 'model' in globals():
                    colA, colB = st.columns([1,1]) 
                    uploaded_image = Image.open(uploaded_file)
                    uploaded_image = np.array(uploaded_image)
                    uploaded_image = uploaded_image.astype('float32') / 255
                    img = cv2.resize(uploaded_image, IMG_SIZE, interpolation = cv2.INTER_AREA)
                    #handling non grayscale images s
                    if len(img.shape) == 3:
                        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_image = img
                    img_scale = fittedscaler.transform(gray_image.reshape(1,48*48))
                    img_pca = fittedpca.transform(img_scale)
                    img_lda = fittedlda.transform(img_pca)
                    predictions = model.predict_proba(img_lda)
                    with colA:
                        fig, ax = plt.subplots()
                        fig = plt.imshow(gray_image, cmap = 'gray')
                        ax.axis('off')
                        st.subheader(labels[np.argmax(predictions.reshape(6))])
                        st.pyplot(fig.get_figure())
                    with colB:
                        color = ['lightgrey', 'orange', 'lightgrey', 'lightgrey', 'lightgrey', 'lightgrey']
                        #select_bar_column_pallete(labels.keys(), np.argmax(predictions.reshape(6))
                        fig = px.bar(y=list(labels.values()), x=predictions.reshape(6))
                        st.plotly_chart(fig, use_container_width=True)
                with col2:
                    if uploaded_file is not None and 'model2' in globals():
                        colC, colD = st.columns([1,1]) 
                        uploaded_image = Image.open(uploaded_file)
                        uploaded_image = np.array(uploaded_image)
                        uploaded_image = uploaded_image.astype('float32') / 255
                        img = cv2.resize(uploaded_image, IMG_SIZE, interpolation = cv2.INTER_AREA)
                        #handling non grayscale images s
                        if len(img.shape) == 3:
                            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        else:
                            gray_image = img
                        img_scale = fittedscaler.transform(gray_image.reshape(1,48*48))
                        img_pca = fittedpca.transform(img_scale)
                        img_lda = fittedlda.transform(img_pca)
                        predictions = model2.predict_proba(img_lda)
                        with colC:
                            fig, ax = plt.subplots()
                            fig = plt.imshow(gray_image, cmap = 'gray')
                            ax.axis('off')
                            st.subheader(labels[np.argmax(predictions.reshape(6))])
                            st.pyplot(fig.get_figure())
                        with colD:
                            color = ['lightgrey', 'orange', 'lightgrey', 'lightgrey', 'lightgrey', 'lightgrey']
                            #select_bar_column_pallete(labels.keys(), np.argmax(predictions.reshape(6))
                            fig = px.bar(y=list(labels.values()), x=predictions.reshape(6))
                            st.plotly_chart(fig, use_container_width=True)
                
        else:
            model_choice = st.selectbox("##### Choose a Model:", ["Stochastic Gradient Descent","Gaussian Naive Bayes","Support Vector Machine","Multilayer Perceptron", "User Model 1", "User Model 2"])
            if model_choice == "Stochastic Gradient Descent":
                with open(OUTPUT_DIR / 'SGDmodel_LDAdata.pickle', "rb") as f:
                        model = pickle.load(f)
            if model_choice == "Gaussian Naive Bayes":
                with open(OUTPUT_DIR / 'NBmodel_LDAdata.pickle', "rb") as f:
                        model = pickle.load(f)
            if model_choice == "Support Vector Machine":
                with open(OUTPUT_DIR / 'SVMmodel_LDAdata.pickle', "rb") as f:
                        model = pickle.load(f)
            if model_choice == "Multilayer Perceptron":
                with open(OUTPUT_DIR / 'SGDmodel_LDAdata.pickle', "rb") as f:
                        model = pickle.load(f)
            if model_choice == "User Model 1":
                if 'usermodel1' in st.session_state:
                    model = st.session_state['usermodel1']
                else:
                    st.write("User Model 1 has not been defined!")
            if model_choice == "User Model 2":
                if 'usermodel2' in st.session_state:
                    model = st.session_state['usermodel2']
                else:
                    st.write("User Model 2 has not been defined!")
            uploaded_file = st.file_uploader("Upload Image")
            if uploaded_file is not None and 'model' in globals():
                col1, col2 = st.columns([1,1])
                uploaded_image = Image.open(uploaded_file)
                uploaded_image = np.array(uploaded_image)
                uploaded_image = uploaded_image.astype('float32') / 255
                img = cv2.resize(uploaded_image, IMG_SIZE, interpolation = cv2.INTER_AREA)
                #handling non grayscale images s
                if len(img.shape) == 3:
                    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray_image = img
                img_scale = fittedscaler.transform(gray_image.reshape(1,48*48))
                img_pca = fittedpca.transform(img_scale)
                img_lda = fittedlda.transform(img_pca)
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
    with tab2:
        saveslot = st.radio("Model Save Slot:", ["Save Slot 1", "Save Slot 2"], horizontal=True,)
        model_choice = st.selectbox("##### Choose a model to build:", ["Stochastic Gradient Descent","Gaussian Naive Bayes","Support Vector Machine"])
        if model_choice == "Stochastic Gradient Descent":
            col1, col2 = st.columns([5,1])
            with col1:
                loss = st.selectbox("##### Loss:", ['log_loss', 'modified_huber'])
            with col2:
                helpbutton = st.checkbox('Help!')
            if helpbutton == True:
                advice = '''The loss function to be used. \\
                    **'hinge'** gives a linear SVM. \\
                    **‘log_loss’** gives logistic regression, a probabilistic classifier.\\
                    **‘modified_huber’** is another smooth loss that brings tolerance to outliers as well as probability estimates.\\
                    **‘squared_hinge’** is like hinge but is quadratically penalized.\\
                    **‘perceptron’** is the linear loss used by the perceptron algorithm.'''
                st.markdown(advice)
            col3, col4 = st.columns([5,1])
            with col3:
                penalty = st.selectbox("##### Penalty:", ['l2','l1','elasticnet',None])
            with col4:
                helpbutton = st.checkbox('Help!', key='helppenalty')
            if helpbutton == True:
                advice = '''The penalty (aka regularization term) to be used. Defaults to ‘l2’ which is the standard regularizer for linear SVM models. ‘l1’ and ‘elasticnet’ might bring sparsity to the model (feature selection) not achievable with ‘l2’. No penalty is added when set to None.'''
                st.markdown(advice)
            col5, col6 = st.columns([5,1])
            with col5:
                alpha = st.number_input("##### Alpha:", value=0.0001, placeholder="default")
            with col6:
                helpbutton = st.checkbox('Help!', key='helpalpha')
            if helpbutton == True:
                advice = '''Constant that multiplies the regularization term. The higher the value, the stronger the regularization. Also used to compute the learning rate when learning_rate is set to ‘optimal’. Values must be in the range [0.0, inf).'''
                st.markdown(advice)
            col7, col8 = st.columns([5,1])
            with col7:
                max_iter = st.number_input("##### Maximum iterations:", value=1000, placeholder="default")
            with col8:
                helpbutton = st.checkbox('Help!', key='helpmaxitr')
            if helpbutton == True:
                advice = '''The maximum number of passes over the training data (aka epochs). Default value is 1000.'''
                st.markdown(advice)
            usermodel = SGDClassifier(loss=loss, penalty=penalty,alpha=alpha,max_iter=max_iter)
        if model_choice == "Gaussian Naive Bayes":
            col1, col2 = st.columns([5,1])
            with col1:
                var_smoothing = st.number_input("##### Var_smoothing:", value=1e-9, placeholder="default")
            with col2:
                helpbutton = st.checkbox('Help!')
            if helpbutton == True:
                advice = '''Portion of the largest variance of all features that is added to variances for calculation stability. Default value 1e-9'''
                st.markdown(advice)
            usermodel = GaussianNB(var_smoothing=var_smoothing)
        if model_choice == "Support Vector Machine":
            col1, col2 = st.columns([5,1])
            with col1:
                C = st.number_input("##### C:", value=1, placeholder="default")
            with col2:
                helpbutton = st.checkbox('Help!',key='svcc')
            if helpbutton == True:
                advice = '''Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty. Default value 1.0.'''
                st.markdown(advice)
            col3, col4 = st.columns([5,1])
            with col3:
                kernel = st.selectbox("##### Kernel:", ['linear','poly','rbf','sigmoid','precomputed'])
            with col4:
                helpbutton = st.checkbox('Help!', key='helpsvckernel')
            if helpbutton == True:
                advice = '''Specifies the kernel type to be used in the algorithm. If none is given, ‘rbf’ will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples).'''
                st.markdown(advice)
            col5, col6 = st.columns([5,1])
            with col5:
                gamma = st.selectbox("##### Gamma:", ['scale','auto'])
            with col6:
                helpbutton = st.checkbox('Help!', key='helpsvcgamma')
            if helpbutton == True:
                advice = '''Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.\\
                if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,
                \\if ‘auto’, uses 1 / n_features'''
                st.markdown(advice)
            usermodel = SVC(C=C,kernel=kernel,gamma=gamma)

        generate = st.button("Generate Model", type="primary")
        if generate:
            st.write("Loading training/testing data...")
            if 'X_augmented' in st.session_state and 'y_fulltrain' in st.session_state:
                st.write("Augmented Dataset Detected! Loading Augmented set instead...")
                X_train = st.session_state['X_augmented']
                y_train = st.session_state['y_fulltrain']
                X_test, y_test, test_labels = load_data(TEST_DIR,IMG_SIZE)
            else:
                X_train, y_train, train_labels = load_data(TRAIN_DIR, IMG_SIZE)
                X_test, y_test, test_labels = load_data(TEST_DIR,IMG_SIZE)
            model_train_labels = [test_labels[i] for i in y_train]
            model_test_labels = [test_labels[i] for i in y_test]
            scaledtrain = fittedscaler.transform(X_train.reshape(len(X_train),48*48))
            pcatrain = fittedpca.transform(scaledtrain)
            ldatrain = fittedlda.transform(pcatrain)
            st.write("Generating Model...")
            usermodel.fit(ldatrain, model_train_labels)
            st.write("Model Generated! Testing Model...")
            scaledtest = fittedscaler.transform(X_test.reshape(len(X_test),48*48))
            pcatest = fittedpca.transform(scaledtest)
            ldatest = fittedlda.transform(pcatest)
            predictions = usermodel.predict_proba(ldatest)
            st.write(f"Model Accuracy: {usermodel.score(ldatest, model_test_labels)}")
            st.write(f"Model saved in {saveslot}, and can be accessed in the model explorer!")
            if saveslot == "Save Slot 1":
                st.session_state['usermodel1'] = usermodel
            if saveslot == "Save Slot 2":
                usermodel2 = usermodel
                st.session_state['usermodel2'] = usermodel
            
            