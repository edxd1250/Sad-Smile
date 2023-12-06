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
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.svm import SVC

from tensorflow.keras.utils import to_categorical

import tensorflow as tf

#!export KERAS_BACKEND="tensorflow"
import keras
import keras_tuner as kt

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras import layers
from keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19
from tensorflow.keras.backend import clear_session



!module load CUDA
!export PATH=/opt/software/CUDA/12.3.0/bin:$PATH
!export LD_LIBRARY_PATH=/opt/software/CUDA/12.3.0/lib64:$LD_LIBRARY_PATH
!export CUDA_HOME=/opt/software/CUDA/12.3.0/
!export XLA_FLAGS=--xla_gpu_cuda_data_dir=/opt/software/CUDA/12.3.0/


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

   
    epochs = 40
    batch_size = 32
    IMG_SIZE = (48,48)
    NUM_CLASSES = 6
    oglabels = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Suprise'}
    OUTPUT_DIR = Path('output')
    TRAIN_DIR = ('imagedata/Training/Training')
    TEST_DIR = ('imagedata/Testing/Testing')
    X_test, y_test, test_labels = load_data(TEST_DIR,IMG_SIZE)

    datagen = ImageDataGenerator(
   # featurewise_center=True,
   # featurewise_std_normalization=True,
    rescale = 1/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)

    train_generator = datagen.flow_from_directory(TRAIN_DIR,
                                                    target_size=IMG_SIZE,
                                                    color_mode="rgb",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    subset='training',
                                                    shuffle=True)
                                                    
    val_generator = datagen.flow_from_directory(TRAIN_DIR,
                                                    target_size=IMG_SIZE,
                                                    color_mode="rgb",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    subset='validation',
                                                    shuffle=True)
    
   # datagen.fit(train_generator)

    

    def build_model(hp):
        base_model = VGG19(
            weights='imagenet',
            include_top=False,
            input_shape=IMG_SIZE + (3,)
        )
        model = Sequential()
        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(hp.Int('units', min_value=256, max_value=1024, step=32), activation="relu"))
        model.add(Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(Dense(NUM_CLASSES, activation="softmax"))

        optimizer = Adam(learning_rate=hp.Float('learning_rate', min_value=1e-6, max_value=1e-4, sampling='log'))
        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        return model

    # Set up the tuner
    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=10,
        factor=3,
        directory='my_dir',
        project_name='intro_to_kt'
    )

    # Perform the hyperparameter search
    tuner.search(train_generator, validation_data=val_generator, epochs=10)

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build the best model using the best hyperparameters
    best_model = tuner.hypermodel.build(best_hps)

    # Train the model using the best hyperparameters
    best_model.fit(train_generator, validation_data=val_generator, epochs=40)
   

   
    #with open(OUTPUT_DIR / 'trainingGenVGG19.pickle', "wb") as f:
      # pickle.dump(train_generator, f)
    best_model.save(OUTPUT_DIR / 'VGG19m6model.h5')
   
    with open(OUTPUT_DIR / 'VGG19m6model.pickle', "wb") as f:
       pickle.dump(best_model, f)
    

    predictions = best_model.predict(X_test)
    y_pred = [np.argmax(probas) for probas in predictions]


    accuracy = accuracy_score(y_test, y_pred)
    print('Test Accuracy = %.2f' % accuracy)
    clear_session()

if __name__ == '__main__':
    main()