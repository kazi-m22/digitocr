import cv2 as cv
import numpy
import imutils
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization


class Detector:
    def __init__(self):
      global model

      model = Sequential()
      model.add(Conv2D(64, (5, 5), input_shape=(64, 64, 1), activation='relu', padding='same'))
      model.add(Conv2D(64, (5, 5), activation='relu'))
      model.add(MaxPooling2D((2, 2), strides=(2, 2)))
      model.add(Conv2D(128, (3, 3), activation='relu'))
      model.add(BatchNormalization())
      model.add(Conv2D(128, (3, 3), activation='relu'))
      model.add(MaxPooling2D((2, 2), strides=(2, 2)))
      model.add(Conv2D(512, (3, 3), activation='relu'))
      model.add(BatchNormalization())
      model.add(Conv2D(512, (3, 3), activation='relu'))
      model.add(BatchNormalization())
      model.add(Conv2D(512, (3, 3), activation='relu'))
      model.add(MaxPooling2D((2, 2), strides=(2, 2)))
      model.add(Flatten())
      model.add(Dense(256, activation='relu'))
      model.add(Dropout(0.4))
      model.add(Dense(256, activation='relu'))
      model.add(Dropout(0.25))
      model.add(Dense(10, activation='softmax'))
      model.load_weights('model/bangla_1.h5')

    def detectObject(self, imName):
        font = cv.FONT_HERSHEY_SIMPLEX
        image = numpy.array(imName)
        gray = cv.cvtColor(numpy.array(imName), cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (7, 7), 0)
        ret,thresh1 = cv.threshold(gray ,200,255,cv.THRESH_BINARY_INV)
        # thresh1 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
        dilate = cv.dilate(thresh1, None, iterations=2)
        cnts = cv.findContours(dilate.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1] if imutils.is_cv2() else cnts[0]
        i = 0
        t = 0
        c=0
        x_all = []
        w_all = []
        lines = []
        parts = []

        for cnt in cnts:
          if(cv.contourArea(cnt) < 100):
              continue
          x,y,w,h = cv.boundingRect(cnt)
          x_all.append(x)
          w_all.append(w)
          i = i + 1 

        comb = numpy.zeros((2,len(x_all)))
        comb[0,:]=x_all
        comb[1,:]=w_all
        comb = comb.T
        comb = comb[comb[:,0].argsort()]

        x_all = comb[:,0]
        w_all = comb[:,1]

        for i, item in enumerate (x_all):
          if i < len(x_all)-1:
              lines.append((item+w_all[i]+x_all[i+1])/2)

        for i in range(len(lines)):
          parts.append(cv.resize(thresh1[:,t:int(lines[i])]/255.0,(64,64)))
          t = int(lines[i])
          if i == len(lines)-1:
              parts.append(cv.resize(thresh1[:,t:]/255.0,(64,64)))
        # temp = parts[0]
        # for i in range(1,len(parts)):
        #   temp = numpy.concatenate((temp, parts[i]), axis=1)


        temp = parts[0]*255.0
        for i in range(1,len(parts)):
          temp = numpy.concatenate((temp, parts[i]*255.0), axis=1)
        temp = numpy.array(temp)


        pred = ""
        for i in parts:
          # print(i.shape)
          pred = pred + str(numpy.argmax(model.predict(i.reshape(1,64,64,1))))
          # pred.append(numpy.argmax(model.predict(i.reshape(1,64,64,1))))
        pad = numpy.zeros((100,gray.shape[1]))
        cv.putText(pad, pred, (0,pad.shape[0]-20), font, 2, (255, 255, 255), 2, cv.LINE_AA)
        gray = numpy.concatenate((gray, pad), axis=0)
        print('*********', pred)
        img = cv.imencode('.jpg', gray)[1].tobytes()
        return img
