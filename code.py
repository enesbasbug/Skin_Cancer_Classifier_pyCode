#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: senesbasbug
"""

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

from PIL import ImageTk, Image

import pandas as pd
import numpy as np
from keras.models import load_model
from keras.utils.np_utils import to_categorical
import TrainingModel



#%%
"""
skin_df = pd.read_csv("skin_db/HAM10000_metadata.csv")

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
skin_df.to_pickle("skin_df.pkl")"""

#%% PKL

skin_df = pd.read_pickle("skin_df.pkl")

#%% standardization

x_train = np.asarray(skin_df["image"].to_list())

x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)
x_train = (x_train - x_train_mean) / x_train_std

#one hot encoding
y_train = to_categorical(skin_df["dx_idx"], num_classes = 7) #kanser hucresi sinifimiz 7 tane

#%%

"""
train val ve test diye ayirip yap en bastan
"""
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

epochs = 5
batch_size = 25

history = model.fit(x = x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose = 1,shuffle = True)

#model.save("my_model1.h5")
model.save("my_model2.h5")
"""

#%%
    
model1 = load_model("my_model1.h5")
model2 = load_model("my_model2.h5")
model3 = load_model("my_model3.h5")

#%%
index = 5
y_pred = model3.predict(x_train[index].reshape(1,75,100,3)) # MODEL MODEL MODEL
np.argmax(y_pred, axis =1)

#%%

def ImageResize(img):
    basewidth = 500
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    return img


def OpenImage():
    global img_name 
    global img_jpg
    global count
    
    count += 1
    if count != 1:
        messagebox.showinfo(title="Warning", message= "Only one image can be opened")
    else:
        img_name = filedialog.askopenfilename(initialdir = "/Users/senesbasbug/Desktop/Deri_Kanseri_DataI/", title="Select an image")
        
        img_jpg = img_name.split("/")[-1].split(".")[0]
        
        tk.Label(frame1, text=img_jpg, bd =3).pack(pady=30)
        
        img = Image.open(img_name)
        img = ImageResize(img)
        img = ImageTk.PhotoImage(img)
        panel = tk.Label(frame1, image = img)
        panel.image = img
        panel.pack(padx=15,pady=10)
        
        # IMG feature          
        data = pd.read_csv("skin_db/HAM10000_metadata.csv")
        cancer = data[data.image_id == img_jpg]
        for i in range(cancer.size):
            x = 0.4
            y = i/20
            print("geldim")
            tk.Label(frame3, font=("Times", 15), text = str(cancer.iloc[0,i])).place(relx =x, rely= y)

    
window = tk.Tk()
window.geometry("1100x680")
window.wm_title("Skin Cancer Classification")


#global variables
img_name = ""
count = 0
img_jpg = ""


#frame
frame_left = tk.Frame(window, width=540, height=780, bd="2")
frame_left.grid(row =0, column=0)

frame_right = tk.Frame(window, width=560, height=640, bd="2")
frame_right.grid(row =0, column=1)

frame1= tk.LabelFrame(frame_left, text="Image", width=540,height=500)
frame1.grid(row=0, column=0)

frame2= tk.LabelFrame(frame_left, text="Model and Save", width=540, height=240)
frame2.grid(row=2, column=0)

frame3= tk.LabelFrame(frame_right, text="Feature", width=270,height=590)
frame3.grid(row=0, column=0)

frame4= tk.LabelFrame(frame_right, text="Results", width=270,height=590)
frame4.grid(row=0, column=1, padx= 10)



      
B = tk.Button(frame_left, text ="File", command = OpenImage)
B.grid(row=1, column=0, pady=10 )


#%%
def classification():
    
    if img_name != "" and models.get != "":
        
        if models.get == "Model1":
            classification_model= model1 # yukadaridan geliyor model1 kaydetmistik
        elif models.get == "Model2":
            classification_model= model2
        else:   # Model 3 
            classification_model= model3
        
        z = skin_df[skin_df.image_id == img_jpg ]
        z = z.image.values[0].reshape(1, 75,  100, 3)
        
        # standarte ettigim halde yukarida tekrar etmem gerekli dedi....
        z = (z - x_train_mean) / x_train_std
        h = classification_model.predict(z)[0]
        h_index = np.argmax(h)
        predicted_cancer = list(skin_df.dx.unique())[h_index]
        
        for i in range(len(h)):
            x = 0.5
            y = i/20
            
            if i != h_index:
                tk.Label(frame4, text = str(h[i])).place(relx = x, rely = y)
            else:
                tk.Label(frame4, bg="green", text = str(h[i])).place(relx = x, rely = y)

        if chvar.get() == 1:
            
            val = entry.get()
            entry.config(state = "disable")
            path_name = val + ".txt" 
            
            save_txt = img_name + "--" + str(predicted_cancer)
            text_file = open(path_name,"w")
            text_file.write(save_txt)
            text_file.close()
        else:
            print(" Save is not selected.")
    else:
        messagebox.showinfo(title = "Warning", message = "Choose a image and a model first.")
        tk.Label(frame3, text ="Choose a image and a model").place(relx = 0.1, rely = 0.6)
        
    
columns = ["lesion_id","image_id","dx","dx_type","age","sex","localization" ]
for i in range(len(columns)):
    x = 0.1
    y = i/20
    tk.Label(frame3, font=("Times", 15), text = columns[i] + " :" ).place(relx =x, rely= y)
    
classify_button = tk.Button(frame3, bg = "red", bd = 4, font = ("Times", 14), activebackground = "orange", text="Classify", command = classification)
classify_button.place(relx = 0.1, rely = 0.5) # rel oran olarak calisiyor



#%%  frame 4

labels = skin_df.dx.unique()
for i in range(len(labels)):
    x = 0.1
    y = i/20
    tk.Label(frame4, font=("Times", 15), text = labels[i] + " :" ).place(relx =x, rely= y)


#%% frame 2
# Combo box

model_selection_label = tk.Label(frame2, text = "Choose classification model : ")
model_selection_label.grid(row = 0, column = 0, padx= 5)

models = tk.StringVar()
model_selection = ttk.Combobox(frame2, textvariable=models, values = ("Model1","Model2","Model3"), state="readonly")
model_selection.grid(row = 0, column = 1, padx= 5)

# Check box
chvar = tk.IntVar()
chvar.set(0)
xbox = tk.Checkbutton(frame2, text = "Save Classification Results", variable = chvar)
xbox.grid(row = 1, column = 0, pady= 5)

# Entry
entry = tk.Entry(frame2, width = 23)
entry.insert(index = 0, string = "file title")
entry.grid(row = 1, column = 1)




window.mainloop()


