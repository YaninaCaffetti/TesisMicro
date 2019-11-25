#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Librerías 
import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import applications
#Librerías para la creación Clases de Imagenes
import skimage
from skimage import data
from skimage.filters import threshold_li
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage import io
from skimage import color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import os


# In[3]:


DATADIR_Peces = "./data/entrenamiento/Peces"


# In[4]:


CATEGORIES = ["AN","Peces","Moluscos"]


# In[6]:


fig, ax = plt.subplots(figsize=(10.6666, 10.6666))
image = "./data/entrenamiento/Peces/Blebed.tif"
image = io.imread(image)
ax.imshow(image)


# In[12]:


def IC_Peces (im, box):
    box=int(box)
    image_label_overlay = im
    imagen_peces = color.rgb2gray(image_label_overlay[:,:,2])
    img=imagen_peces
    thresh = threshold_li(imagen_peces)
    #Obtenemos los objetos detectados en la imagen eliminando el fondo 
    bw = closing(imagen_peces > thresh, square(3))

    # Etiquetamos los objetos detectados en la imagen
    label_image_blue = label(bw)
    #Contamos el numero de nucleos detectados
    region_image_blue = label_image_blue.copy()

    cell_num=1
    for region in regionprops(label_image_blue, imagen_peces):
        if region.area > 200:#regiones de color azul con un minimo de pixeles: un nucleo
            region_image_blue[:,:] = 0
            region_image_blue[label_image_blue == region.label] = region.label
            region_image_blue[region_image_blue != 0] = imagen_peces[region_image_blue != 0]
            centre=region.centroid
                    
            bbox = [(int(centre[0]-box/2)),(int(centre[0]+box/2)),(int(centre[1]-box/2)),(int(centre[1]+box/2))]
        
            for i,j in enumerate(bbox):
                if j<0:
                    bbox[i+1]=box
                    bbox[i]=0
                if j>1024:
                    bbox[i-1]=1024-box
                    bbox[i]=1024
        
            sel_cel = region_image_blue[bbox[0]:bbox[1],bbox[2]:bbox[3]]
            col_sel_cel=np.matrix(sel_cel, "uint8")
            plt.close("all")
            cell_num+=1
    return col_sel_cel

image = IC_Peces(image, 292)


# In[10]:


from skimage import transform as tf
box=292
tform =tf.AffineTransform(scale=(1,1.5), rotation=-0.5,translation=(0,0), shear=0)
img_warp = tf.warp(image, tform)

img_warp = img_warp*255
plt.subplots( figsize=(20,20))
plt.subplot(1,2,1)
plt.imshow(image)
plt.subplot(1,2,2)
plt.imshow(img_warp)
plt.show( )


# In[11]:


fig, ax = plt.subplots(figsize=(10.6666, 10.6666))
image = "./data/entrenamiento/Moluscos/BlebbedG.tif"
image = io.imread(image)
ax.imshow(image)


# In[18]:


def IC_Moluscos (im, box):
    box=int(box)
    image_label_overlay = im
    imagen_moluscos = color.rgb2gray(image_label_overlay[:,:,2])
    img=imagen_moluscos
    thresh = threshold_li(imagen_moluscos)
    #Obtenemos los objetos detectados en la imagen eliminando el fondo 
    bw = closing(imagen_moluscos > thresh, square(3))

    # Etiquetamos los objetos detectados en la imagen
    label_image_blue = label(bw)
    #Contamos el numero de nucleos detectados
    region_image_blue = label_image_blue.copy()

    cell_num=1
    for region in regionprops(label_image_blue, imagen_moluscos):
        if region.area > 200:#regiones de color azul con un minimo de pixeles: un nucleo
            region_image_blue[:,:] = 0
            region_image_blue[label_image_blue == region.label] = region.label
            region_image_blue[region_image_blue != 0] = imagen_peces[region_image_blue != 0]
            centre=region.centroid
                    
            bbox = [(int(centre[0]-box/2)),(int(centre[0]+box/2)),(int(centre[1]-box/2)),(int(centre[1]+box/2))]
        
            for i,j in enumerate(bbox):
                if j<0:
                    bbox[i+1]=box
                    bbox[i]=0
                if j>1024:
                    bbox[i-1]=1024-box
                    bbox[i]=1024
        
            sel_cel = region_image_blue[bbox[0]:bbox[1],bbox[2]:bbox[3]]
            col_sel_cel=np.matrix(sel_cel, "uint8")
            plt.close("all")
            cell_num+=1
    return col_sel_cel

image = IC_Moluscos (image, 192)


# In[ ]:


from skimage import transform as tf
box=292
tform =tf.AffineTransform(scale=(1,1.5), rotation=-0.5,translation=(0,0), shear=0)
img_warp = tf.warp(image, tform)

img_warp = img_warp*255
plt.subplots( figsize=(20,20))
plt.subplot(1,2,1)
plt.imshow(image)
plt.subplot(1,2,2)
plt.imshow(img_warp)
plt.show( )

