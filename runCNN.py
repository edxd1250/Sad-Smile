import cv2
import os
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import pickle
from pathlib import Path
import random
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.svm import SVC

from tensorflow.keras.utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19
from tensorflow.keras.backend import clear_session
import tensorflow as tf





def main():
    
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
                        img = cv2.imread(dir_path + '/'+ path + '/' + file)
                        img
                        img = img.astype('float32') / 255
                        resized = cv2.resize(img, IMG_SIZE, interpolation = cv2.INTER_AREA)
                        X.append(resized)
                        y.append(i)
                i += 1
    #Converting lists to np arrays
        X = np.array(X)
        y = np.array(y)
        print(f'{len(X)} images loaded from {dir_path} directory.')
        return X, y, labels

    OUTPUT_DIR = Path('output')
    TRAIN_DIR = ('imagedata/Training/Training')
    TEST_DIR = ('imagedata/Testing/Testing')
    IMG_SIZE= (48, 48)

    
    X_train, y_train, train_labels = load_data(TRAIN_DIR, IMG_SIZE)
    X_test, y_test, test_labels = load_data(TEST_DIR,IMG_SIZE)

    
    clear_session()
    

    base_model = VGG19(
            weights=None,
            include_top=False, 
            input_shape=IMG_SIZE + (3,)
        )

    base_model.summary()
    NUM_CLASSES = 6

    NUM_CLASSES = 6

    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(1000, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(NUM_CLASSES, activation="softmax"))
   
    def deep_model(model, X_train, Y_train, epochs, batch_size):
   
        model.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(learning_rate=1e-4),
        metrics=['accuracy'])
        
    #     es = EarlyStopping(monitor='val_loss',
    #                            restore_best_weights=True,
    #                            mode='min'
    #                           min_delta=1.5)
        history = model.fit(X_train
                        , Y_train
                        , epochs=epochs
                        , batch_size=batch_size
                        , verbose=1)
        return history


    epochs = 40
    batch_size = 128
    Y_train = to_categorical(y_train, num_classes=6)
    Y_test = to_categorical(y_test, num_classes=6)


    history = deep_model(model, X_train, Y_train, epochs, batch_size)

    with open(OUTPUT_DIR / 'VGG19m4history.pickle', "wb") as f:
       pickle.dump(history, f)
    with open(OUTPUT_DIR / 'VGG19m4model.pickle', "wb") as f:
       pickle.dump(model, f)

if __name__ == '__main__':
    main()