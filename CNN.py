from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Convolution2D, MaxPool2D
import numpy as np
import pandas as pd
import cv2
input_file = 'label.csv'
a = pd.read_csv(input_file)
y = []
x = []
key_words = {'Effusion':1, 'Infiltration':0}
counts = 0
for index, row in a.iterrows():
    imglabel = row['Finding Labels']
    y.append(key_words[imglabel])
    imgname = row['Image Index']
    array = cv2.imread("data/{}".format(imgname))
    array = cv2.resize(array,(256,256))
    x.append(array.tolist())
    counts += 1
    print("Successfully load image:{}, there are {} images totally.".format(imgname, counts))

y_train = np.array(y)
x_train = np.array(x)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, test_size=0.3, random_state=42)
from keras.utils import to_categorical
y_binary = to_categorical(y_train)


model = Sequential()
model.add(Convolution2D(filters=64, kernel_size=(5, 5), strides=(1, 1), input_shape=[RESIZE_NUM, RESIZE_NUM, 1]))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D())
model.add(Convolution2D(filters=128, kernel_size=(5, 5), strides=(1, 1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D())
model.add(Convolution2D(filters=256, kernel_size=(5, 5), strides=(1, 1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D())
model.add(Convolution2D(filters=512, kernel_size=(5, 5), strides=(1, 1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D())
model.add(Convolution2D(filters=1024, kernel_size=(5, 5), strides=(1, 1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="softmax"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"])

model.fit(x_train, y_binary, batch_size=128, epochs=10)



y_pred = model.predict(x_test)
#y_test = y_train

from sklearn.metrics import roc_auc_score
def recall_precision(y_test,y_score,threshold = 0.5):
    true_pos = 0
    recall_num = 0
    precision_num = 0
    for i in range(len(y_score)):
        score = y_score[i]
        true = y_test[i]
        if score > threshold:
            precision_num += 1
            if true == 1:
                true_pos += 1
        if true == 1:
            recall_num += 1
    precision = true_pos/float(precision_num)
    recall = true_pos/float(recall_num)
    return recall, precision

print('auc is:',roc_auc_score(y_test,y_pred[:,1]))
for rand in range(1,99):
    try:
        recall, precision=recall_precision(y_test,y_pred[:,1],rand/100)
        f_score = 2*precision*recall/(precision+recall)
        print("P:{:.5f}, R:{:.5f}, F:{:.5f}, threshold:{}".format(precision,recall,f_score,rand/100))
    except:
        continue
    