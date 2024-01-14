# !pip install tensorflow==2.8.0
# !pip install keras==2.8.0
# !pip install pillow
# !pip install opencv-python
# !pip install scikit-learn
## Step 1: Import Dataset & data preparation
import numpy as np
import pandas as pd
import keras
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import os
import random
df = pd.read_csv("train.csv")
base_path = "./images/"
df
df = df.loc[df["id"].str.startswith(('00', '7d', 'b1'), na=False), :]
num_classes = len(df["landmark_id"].unique())
num_data = len(df)
num_classes
num_data
data = pd.DataFrame(df["landmark_id"].value_counts())
data
data.reset_index(inplace=True)
data.columns=["landmark_id", "count"]
data
plt.hist(data['count'], 100, range= (0,32), label = 'test')
data['count'].between(0,5).sum()
data['count'].between(5,10).sum()
plt.hist(df["landmark_id"], bins=df["landmark_id"].unique())
from sklearn.preprocessing import LabelEncoder
lencoder = LabelEncoder()
lencoder.fit(df["landmark_id"])
def encode_label(label):
    return lencoder.transform(label)
def decode_label(label):
    return lencoder.inverse_transform(label)
def get_image_from_numbers(num, df):
    fname, label = df.iloc[num, :]
    fname = fname + '.jpg'
    f1 = fname[0]
    f2 = fname[1]
    f3 = fname[2]
    full_path = os.path.join(base_path,f1,f2,f3, fname)
    im = cv2.imread(full_path)
    return im, label
print("4 sample Images from random classes")
fig = plt.figure(figsize=(15,15))
for i in range(1,5):
    ri = random.choices(os.listdir(base_path), k=3)
    folder = base_path + random.choice(["0/0/", "b/1/", "7/d/"]) + ri[2]
    random_img = random.choice(os.listdir(folder))
    img = np.array(Image.open(folder + "/" + random_img))
    fig.add_subplot(1,4,i)
    plt.imshow(img)
    plt.axis("off")
plt.show()
## Step 2: Build  the Model
from keras.applications.vgg19 import VGG19
from keras.layers import *
from keras import Sequential
tf.compat.v1.disable_eager_execution()
# Hyper parameter
learning_rate = 0.0001
decay_speed = 1e-6
momentum = 0.9
loss_function = "sparse_categorical_crossentropy"
source_model = VGG19(weights=None)
drop_layer = Dropout(0.5)
model = Sequential()
for layer in source_model.layers[:-1]:
    if layer == source_model.layers[-25]:
        model.add(BatchNormalization())
    model.add(layer)
model.add(Dense(num_classes, activation="softmax"))
model.summary()
optim1 = keras.optimizer_v1.RMSprop(lr= learning_rate)
model.compile(optimizer=optim1, loss=loss_function, metrics= ["accuracy"])
def image_reshape(im, target_size):
    return cv2.resize(im, target_size)
def get_batch(dataframe, start, batch_size):
    image_array = []
    label_array = []
    
    end_img = start+batch_size
    if(end_img) > len(dataframe):
        end_img = len(dataframe)
    
    for idx in range(start, end_img):
        n = idx
        im, label = get_image_from_numbers(n, dataframe)
        im = image_reshape(im, (224, 224)) / 255.0
        image_array.append(im)
        label_array.append(label)
    
    label_array = encode_label(label_array)
    
    return np.array(image_array), np.array(label_array)
# split
train, val = np.split(df.sample(frac=1),[int(0.8*len(df))])
print(len(train))
print(len(val))
batch_size = 16
epoch_shuffle = True
weight_classes = True
epochs = 1

for e in range(epochs):
    print("Epoch :" + str (e+1) + "/"+ str(epochs))
    if epoch_shuffle:
        train = train.sample(frac = 1)
    for it in range(int(np.ceil(len(train)/batch_size))):
        X_train, y_train = get_batch(train, it*batch_size, batch_size)
        
        model.train_on_batch(X_train, y_train)
        
model.save("Model")
# Test
batch_size = 16

errors = 0
good_preds = []
bad_preds = []


for it in range(int(np.ceil(len(val)/batch_size))):
    X_val, y_val = get_batch(val, it*batch_size, batch_size)
    
    result = model.predict(X_val)
    cla = np.argmax(result, axis=1)
    for idx, res in enumerate(result):
        if cla[idx] != y_val[idx]:
            errors = errors + 1
            bad_preds.append([batch_size*it + idx, cla[idx], res[cla[idx]]])
        else:
            good_preds.append([batch_size*it + idx, cla[idx], res[cla[idx]]])


good_preds = np.array(good_preds)
good_preds = np.array(sorted(good_preds, key = lambda x: x[2], reverse=True))
len(good_preds)
fig=plt.figure(figsize=(16,16))
for i in range(7, 12):
    n = int(good_preds[i,0])
    img, lbl = get_image_from_numbers(n, val)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig.add_subplot(1, 5, i-6)
    plt.imshow(img)
    lbl2 = np.array(int(good_preds[i,1])).reshape(1,1)
    sample_cnt = list(df.landmark_id).count(lbl)
    plt.title("Label: " + str(lbl) + "\nClassified as: " + str(decode_label(lbl2)) + "\nSamples in class " + str(lbl) + ": " + str(sample_cnt))
    plt.axis('off')
plt.show()
