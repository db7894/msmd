from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.metrics import binary_accuracy
import matplotlib.pyplot as plt
import os
import csv
import ast
import cv2
import random
from random import shuffle
from PIL import Image
import numpy as np
import tensorflow as tf

def get_file_vector(filename):
    # search for filename row in vector CSV file
    vals = []
    with open('vectors.csv', 'r') as myfile:
        reader = csv.reader(myfile,delimiter=":")
        for row in reader:
            name = row[0]
            rmPng = filename[:-4]
            if name != rmPng:
                continue
            vals = row[1]
            vals = ast.literal_eval(vals)
        
    img = Image.open(filename)
    width, height = img.size
    print("vals type is: " + str(type(vals)))
    vect = np.array([1 if x in vals else 0 for x in range(0, width)])
    return vect
        

def generate_vectors():
    vectors = {}
    with open('data.csv', 'r') as myfile:
        for row in myfile:
            name = row.split()[0]
            print("name is :",name)
            vals = row.split()[1]
        
            img = Image.open(name)
            width, height = img.size
            vect = np.array([1 if x in vals else 0 for x in range(0, width)])
            vectors[img] = vect # use img vector as key
    return vectors

def generate_train_test(train_names, test_names):
    train = {}
    test = {}
    with open('data.csv', 'r') as myfile:
        reader = csv.reader(myfile,delimiter=":")
        for row in reader:
            name = row[0]
            vals = ast.literal_eval(row[1]) 
            vals1 = [x+1 for x in vals]
            vals2 = [x+2 for x in vals]
            vals3 = [x+3 for x in vals]
            vals4 = [x+4 for x in vals]
            vals1m = [x-1 for x in vals]
            vals2m = [x-2 for x in vals]
            vals3m = [x-3 for x in vals]
            newVals = []
            newVals.extend(vals)
            newVals.extend(vals1)
            newVals.extend(vals2)
            newVals.extend(vals3)
            newVals.extend(vals4)
            newVals.extend(vals1m)
            newVals.extend(vals2m)
            newVals.extend(vals3m)
            img = Image.open(name)
            im = np.asarray(img)
            #im = img.getdata()
            width, height = img.size
            vect = np.array([1 if x in newVals else 0 for x in range(0, 800)])
            if name in train_names:
                # print("in train names")
                train[name] = vect # use img vector as key
            elif name in test_names:
                test[name] = vect
    return train, test

def main():
    random.seed(1)
    filelist= []
    with open('data.csv', 'r') as myfile:
        reader = csv.reader(myfile,delimiter=":")
        for row in reader:
            name = row[0]
            filelist.append(name)
    print(len(filelist))    
    shuffle(filelist)
    size = len(filelist)
    train_size = int(0.8 * size)
    train_list= filelist[:train_size]
    test_list = filelist[train_size:]
    train_dict, test_dict = generate_train_test(train_list, test_list)
    y_train = np.asarray(list(train_dict.values()))

    x_train = []
    for name in train_list:
        img= cv2.imread(name,0)
        height, width = img.shape
        WHITE=[255,255,255]
        image = cv2.copyMakeBorder(img, top=0, bottom=0, left=0, right = (800-width), borderType=cv2.BORDER_CONSTANT, value=WHITE)
        x_train.append(image.tolist())
    x_train = np.asarray(x_train)
    x_test = []
    for name in test_list:
        img= cv2.imread(name,0)
        height, width = img.shape
        WHITE=[255,255,255]
        image = cv2.copyMakeBorder(img, top=0, bottom=0, left=0, right = (800-width), borderType=cv2.BORDER_CONSTANT, value=WHITE)
        x_test.append(image.tolist())
    x_test = np.asarray(x_test)
    y_test = np.asarray(list(test_dict.values()))
    #x_test = [ast.literal_eval(key) for key in test_dict.keys()]



    timesteps = 8
    data_dim = 800
    model = Sequential()
    model.add(LSTM(800, input_shape=(timesteps,data_dim)))

    #model.add(Dropout(0.5))
    model.add(Dense(800, activation='sigmoid'))
    model.summary()
    
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    print(x_train.shape)
    history = model.fit(x_train, y_train, batch_size=256, epochs=5, validation_data=(x_test,y_test))
   
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('acc.png')
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')

    print("predicting")
    #print(model.predict(np.expand_dims(x_test[0],axis=0)))
    # pred = model.predict(np.expand_dims(x_test[0],axis=0))
    # pred = pred[0]
    # print(pred)
    # print("length: " + str(len(pred)))
    # print(y_test[0])
    # print("y length: " + str(len(y_test[0])))
    for i in range(5):
        pred = model.predict(np.expand_dims(x_train[i],axis=0))
        pred = pred[0]
        ans = y_test[i]
        ones = [x for x in range(len(ans)) if ans[x] == 1]
        pred_ones = [pred[x] for x in ones]
        print("filename")
        print(train_list[i])
        print("prediction: ")
        print(pred)
        # print("ground truth")
        # print(ans)
        # print("values predicted for ones:")
        # print(pred_ones)
        # print("binary accuracy:")
        # print(binary_accuracy(tf.cast(ans, tf.float32),pred))
main()
