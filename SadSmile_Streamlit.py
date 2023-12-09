
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
import cv2
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

def apply_pca(dataset, ratio=.95, sample=True):
    #reshape image data into 2d array for pca imput
    shape = (len(dataset[0])*len(dataset[0,0]))
    pcaprep = np.reshape(dataset, (len(dataset),shape))
    #apply pca (default 95% of variance)
    pca = PCA(ratio)
    pca.fit(pcaprep)
    transform = pca.transform(pcaprep)
    if sample == True:
        fig, ax = plt.subplots(1,2)
        inverse = pca.inverse_transform(transform)
        ax[0].imshow(dataset[10], cmap = 'gray')
        ax[0].set_title("Original")
        ax[1].imshow(inverse[10].reshape(48,48), cmap='gray')
        ax[1].set_title("Transformed")
        
    return pca, transform, fig, ax

def visualize_pca(dataset, ratio=.95):
    shape = (len(dataset[0])*len(dataset[0,0]))
    pcaprep = np.reshape(dataset, (len(dataset),shape))
    #running pca without components to capture all component data
    pca = PCA()
    pca.fit(pcaprep)
    #plotting components vs cumulative explained ratio
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    #making ratio line
    x = np.linspace(0, (len(dataset[0])*len(dataset[0,0])), 100)
    y = ratio * np.ones_like(x)
    plt.plot(x, y, label=f'y = {ratio}')


OUTPUT_DIR = Path('output')
TRAIN_DIR = ('imagedata/Training/Training')
TEST_DIR = ('imagedata/Testing/Testing')
IMG_SIZE = (48,48)
labels = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Suprise'}

#X_train, y_train, train_labels = load_data(TRAIN_DIR, IMG_SIZE)
#X_test, y_test, test_labels = load_data(TEST_DIR,IMG_SIZE)

with st.sidebar:
    selected = option_menu("Main Menu", ["Introduction", "Model Explorer", "Conclusion","About Me"], 
        icons=['house', 'layers', 'text-left', 'union'], menu_icon="cast", default_index=0)
 

if selected == "Introduction":
    tab1, tab2, tab3, tab4, tab5= st.tabs(["Introduction", "About this Dataset", "Dataset Augmention", "Dimensionality Reduction", "Model Selection and Tuning"])
    with tab1: 
        st.header("Facial Emotion Classification!")
        st.image("behavior_facial_expressions_emotions-100718798-orig.webp")
        st.markdown("Understanding human emotions is often a highly complicated task that requires situational awareness, an understanding of how emotions are typically displayed, and an intuition towards how that emotion might look on the person in question. Our amazing brains are able to take this information, processes and analyze it within seconds to give a (sometimes) acurate estimation on what emotion an individual might be feeling. But what about computers? How might a computer understand the complexities of facial emotions and be able to detect the differences? And how might this technology be used in the future? The current project aims to address these problems. Click through to learn more!.")
 
        
    with tab2: 
        st.header("About this Dataset")
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
        st.header("Data Augmentation")
        augview = st.radio("Would you like to:", ["View training augments", "Generate training augments"], horizontal=True,)
        if augview == "View training augments":
            st.write('''In order to maximize model performance, we chose to augment data in several ways (Nelson, 2022). 
            For this project, we created a function that randomly selected 30% of the samples from each class 
            (in order to preserve class frequencies), and performed the following augmentations: randomly shift x-axis values by up to 10 pixels, 
            randomly shift y-axis values by up to 10 pixels, randomly flip images across y-axis, and randomly rotate images by up to 20 degrees. 
            Additionally, pixel values were normalized from integers ranging from (0-255) to floats ranging from (0-1)
            Augmented images were combined with the original training set, 
            creating a new total training set of 35,754 images. For both sets, data was rescaled so that the mean pixel value (about 129) was 0.''')
          
            st.write("Click the button below to preview some of the augmented data, or click above to generate your own augmented dataset! The augmented dataset will be saved and can be used later on for training your own model!")
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
                fig = check_labels(X_augmented, y_train[augmentindicies], train_labels)
                st.pyplot(fig)
        if augview == "Generate training augments":
            
            tilt = st.number_input("##### Max tilt:", value=20)
            vshift = st.number_input("##### Max Vertical Shift (in pixels):", value=10)
            hshift = st.number_input("##### Max Horizontal Shift (in pixels):", value=10)
            
            cola, colb = st.columns([1,2])
            with cola:
                flip = st.toggle("Randomly flip images?")
            with colb:
                colc, cold = st.columns([1,1])
                with colc:
                    add = st.toggle("Replace augmented images?")
                with cold:
                    numaugment = st.number_input("Augmentation Percentage:", value = .30, min_value=0.0, max_value=1.0, step=.05)
                    helpbutton = st.checkbox('Help!', key='helpalpha')
            if helpbutton == True:
                advice = '''Augmentation Percentage refers to the subset of images selected from the training set for augmentation. By default, this subset will
                be added on top of the origininal training set to increase the size of the training set, however, if replace augmented images is selected, this subset
                will instead replace its respective original images. The algorithm will randomly augment images up to the max values specified above.'''
                st.markdown(advice)
            
            col1, col2, col3 = st.columns([2,2,1])
            with col1:
                numberrows = st.number_input("Number of Rows:", value=3, placeholder="3",key='row2')
            with col2:
                numbercols = st.number_input("Number of Columns:", value=5, placeholder="5",key='col2')
            with col3:
                generate = st.button("Generate", type="primary",key='but2')
            if generate:
                st.write("Augmenting images...")
                X_train, y_train, train_labels = load_data(TRAIN_DIR, IMG_SIZE)
                augmentindicies = getclasssamples(y_train, numaugment)
                if add:
                    X_augmented = augment_images(X_train[augmentindicies],flip=flip,vshift=vshift,hshift=hshift,tilt=tilt)
                    X_train[augmentindicies] = X_augmented
                    st.session_state['X_augmented'] = X_train
                    st.session_state['y_fulltrain'] = y_train
                    
                else:
                    X_augmented = augment_images(X_train[augmentindicies],flip=flip,vshift=vshift,hshift=hshift,tilt=tilt)
                    st.session_state['X_augmented'] = np.concatenate((X_train, X_augmented), axis=0)
                    st.session_state['y_fulltrain'] = np.concatenate((y_train, y_train[augmentindicies]), axis=0)

                st.write(f"New training set of length {len(st.session_state['X_augmented'])} saved, and can be accessed in the model builder!")
                fig = check_labels(X_augmented, y_train[augmentindicies], train_labels)
                st.pyplot(fig)

    with tab4:
        st.header('Dimentionality Reduction')

        st.write('''In this project, in order to reduce the dimentionality of the dataset,
        a two stage approach was employed, in which we first applied a Principal Component Analysis (PCA) 
        where we attempted to maintain 99% of the explained variance of the data,
         and then applied a Linear Discriminant Analysis (LDA), where we intended to reduce data dimensionality even 
         further by projecting it to the most ‘discriminative’ directions.
        ''')
        st.write('''Below is a graph of explained variance ratio vs number of components. Using the slider, select a desired explained variance ratio, and click generate to see a visualization of its effect on the data!''')
        if 'PCAgraph' in st.session_state:
            pca = st.session_state["PCAgraph"]
            dataset = st.session_state["dataset"]
        else:
            X_test, y_test, test_labels = load_data(TEST_DIR,IMG_SIZE)
            dataset = X_test
            pca = PCA()
            pca.fit(dataset.reshape(len(dataset),48*48))
            st.session_state["PCAgraph"] = pca
            st.session_state["dataset"] = dataset
        col1, col2 = st.columns([6,1])
        with col1:
            ratio = st.slider('Select Variance Ratio', min_value=0.500, max_value=1.000, value=0.95, step=0.001, format="%.3f")
        with col2:
            generatepca = st.button("Generate PCA") 
        #running pca without components to capture all component data
        fig = plt.figure()
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        x = np.linspace(0, (len(dataset[0])*len(dataset[0,0])), 100)
        y = ratio * np.ones_like(x)
        plt.plot(x, y, label=f'y = {ratio}')
        st.pyplot(fig)

        
        if generatepca:
            pca2, transform2, fig2, ax2 = apply_pca(dataset, ratio=ratio, sample=True)
            st.pyplot(fig2)

   

    with tab5:
        st.header("Model Selection and Tuning")
        st.write("Below is a summary of the models used, as well as the hyperparameters tested using a gridsearch.")
        st.subheader("Model 1: Gaussian Naïve Bayes Classification")
        st.write("A classification method assuming continuous features follow a Gaussian distribution. Using Bayes' theorem, it predicts class probabilities based on these assumptions, proving effective for classification tasks with continuous data despite its simple assumption of feature independence. Hyperparameters tested include: 'var_smoothing': [5e-9,1e-9,5e-8,1e-8,5e-7,1e-7,5e-6, 1e-6,5e-5,1e-5]")
        st.subheader("Model 2: Stochastic Gradient Descent Classification")
        st.write("An iterative optimization algorithm used in machine learning for training models. It updates parameters by considering individual data samples or small batches, making it efficient for large datasets. The algorithm aims to minimize the loss function by adjusting model parameters in the direction that reduces the error, ultimately leading to a well-performing classifier. Tested parameters include: 'alpha': [0.0001, 0.001, 0.01], 'loss': ['hinge', 'log_loss', 'modified_huber'], 'penalty': ['l1', 'l2', 'elasticnet'], 'max_iter': [1000, 2000, 3000],  'tol': [1e-3, 1e-4, 1e-5]")
        st.subheader("Model 3: C-Support Vector Classification")
        st.write("A type of model that explores different settings and kernel functions to effectively classify data. By varying parameters like regularization strength, kernel types, and gamma settings, it aims to determine the most suitable configuration for accurate classification in diverse scenarios. Tested hyperparameters include:  'C': [0.1, .5, 1, 10],  'kernel': ['linear', 'rbf'], and  'gamma': ['scale', 'auto']")        
        st.subheader("Model 4: Multilayer Perceptron Classification")
        st.write("A type of neural network that uses multiple layers of interconnected nodes to learn and classify data. By passing information through these layers, it can identify patterns and relationships within complex datasets, making it effective for various classification tasks. Tested hyperparameters include: 'hidden_layer_sizes': [(50,), (100,), (50,50), (100,50)],  'activation': ['relu', 'tanh'], 'solver': ['adam', 'sgd'], and  'alpha': [0.0001, 0.001, 0.01]")


        #samples = check_labels(X_train, y_train, train_labels, sample_size = (3,5))
       


if selected == "Model Explorer":
    tab1, tab2= st.tabs(["Model Explorer", "Model Builder"])
    with tab1:
        st.header("Model Explorer")
        st.write('''Welcome to the model explorer. In this tab you can test and compare different models and the different ways they classify emotions. Below you have the option to upload an image, as well as select a model to use for classification. The image should be of a person's face. This image will be rescaled to be 48x48 pixels (it is suggested to use an already square image) and converted to grayscale. 
        After uploading the image, you will see the class probability reported by the selected model. In the next tab, you have the option to create your own models, using the augmented dataset from before! Once generated, you will have the option to test them here. User models will be stored in the "User Model" slots. Have fun!''')
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
                    st.write("User Model 1 has not been defined! Go to the next tab and create a model!")
            if model_choice2 == "User Model 2":
                if 'usermodel2' in st.session_state:
                    model2 = st.session_state['usermodel2']
                else:
                    st.write("User Model 2 has not been defined! Go to the next tab and create a model!")
            
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
                    st.write("User Model 1 has not been defined! Go to the next tab and create a model!")
            if model_choice == "User Model 2":
                if 'usermodel2' in st.session_state:
                    model = st.session_state['usermodel2']
                else:
                    st.write("User Model 2 has not been defined! Go to the next tab and create a model!")
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
            usermodel = SVC(C=C,kernel=kernel,gamma=gamma, probability=True)

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
if selected == "Conclusion":
    tab1,tab2 = st.tabs(["Results/Discussion", "Future Work"])
    with tab1:
        st.header("Results/Discussion")
        st.write("Model performace was measured via accuracy score collected by running models against the testing set described earlier. Unfortunatly, model performance for all models was poor, with accuracies around 36-38% for all four models. We discuss some possibilities for this below:")
        st.write("**Quality of Training Data**")
        st.write("An unforeseen challenge surfaced from potential mislabeling within the dataset used for both training and testing. The accuracy of any machine learning model critically hinges on the quality and accuracy of its training data. In our case, the presence of mislabeled samples might have significantly impacted the models' ability to accurately discern and classify facial expressions.")
        st.write("**Complexity of Facial Emotion Recognition**")
        st.write('''Facial emotion recognition is an intricate task that demands nuanced feature extraction and pattern recognition. The scikit-learn models, while versatile, 
        might struggle to capture the intricate and hierarchical patterns present in facial expressions due to their inherent 
        simplicity and linearity. Additionally, these models relied heavily on handcrafted features, which might not adequately capture the rich information embedded in facial images. 
        Extracting relevant features from raw pixels in facial images is a challenging task, and the limitations of 
        manually engineered features might have hindered the models' ability to generalize well on this complex dataset.''')
    with tab2:
        st.header("Future Work")
        st.write('''Despite the current limitations, the potential applications of accurate facial expression recognition technology are vast. 
        In the realm of mental health, these models could assist therapists and healthcare professionals in monitoring patients' emotional states, 
        aiding in diagnoses or treatment plans. Moreover, this technology holds promise for human-computer interaction, facilitating more intuitive and responsive interfaces. Applications in gaming, virtual reality, and user experience design could benefit from systems capable of understanding and responding to users' emotional cues.

''')

if selected == "About Me": 
    st.header("About Me")
    col1, col2 = st.columns([1,1])
    with col1:
        st.image("goodheadshot.jpeg")
    with col2:  
        st.write("Hello! My name is Edmond Anderson and I am a second year in the Masters of Data Science program at Michigan State University. I got my undergraduate degrees in Mathematics and Psychology, and I am interested in the ways humans experience computers, and the ways cpmputers experience humans. Specifically, I am curious about the ways in which Artificial Intellegence can understand and manipulate human emotions, as well as the ethicallity behind such applications of machine learning. I am also interested in studying the ways in which recommender systems and generative models can alter and affect human behaviour and beliefs. I currently work in the Shiu Lab here at MSU, where I study applications of topic modeling and knowledge graphs on very large scale datasets.")  
            