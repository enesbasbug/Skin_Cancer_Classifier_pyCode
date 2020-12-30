
from PIL import ImageTk, Image

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D
from keras.models import load_model
from keras.optimizers import Adam #


#%%
"""skin_df = pd.read_csv("skin_db/HAM10000_metadata.csv")

skin_df.head()
#sns.countplot(x= "dx", data=skin_df)

#%% Preprocess

data_folder_name = "skin_db/HAM10000_images/"
ext = ".jpg"

# data_folder_name + image_id[i] + ext

skin_df["path"] = [data_folder_name + i + ext for i in skin_df["image_id"]]
skin_df["image"] = skin_df["path"].map( lambda x: np.asarray(Image.open(x).resize((100,75))))
#plt.imshow(skin_df["image"][0])

skin_df["dx_idx"] = pd.Categorical(skin_df["dx"]).codes
skin_df.to_pickle("skin_df_2.pkl")"""

#%% PKL

skin_df_2 = pd.read_pickle("skin_df_2.pkl")

#%% standardization

x_train = np.asarray(skin_df_2["image"].to_list())

x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)
x_train = (x_train - x_train_mean) / x_train_std

#one hot encoding
y_train = to_categorical(skin_df_2["dx_idx"], num_classes = 7) #kanser hucresi sinifimiz 7 tane


#%%
"""input_shape = (75,100,3)
num_classes = 7

model = Sequential()
model.add(Conv2D(32,kernel_size = (3,3), activation ="relu", padding ="Same", input_shape = input_shape))
model.add(Conv2D(32,kernel_size = (3,3), activation ="relu", padding ="Same"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,kernel_size = (3,3), activation ="relu", padding ="Same"))
model.add(Conv2D(64,kernel_size = (3,3), activation ="relu", padding ="Same"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation="softmax"))
model.summary()

optimizer = Adam(lr=0.00001) # Model1 lr=0.0001 yaptik bunu Model2 icin
model.compile(optimizer = optimizer, loss="categorical_crossentropy", metrics = ["accuracy"])

epochs = 20
batch_size = 30

history = model.fit(x = x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose = 1,shuffle = True)

#model.save("my_model1.h5")
model.save("my_model4.h5")
"""