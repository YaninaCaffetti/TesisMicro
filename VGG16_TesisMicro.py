#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
#Librerías de tratamiento de imágenes
import skimage
from skimage import data
from skimage.filters import threshold_li
from skimage.filters import gaussian
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.transform import resize
from skimage.color import label2rgb
from skimage import io
from skimage import color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import time


# In[2]:


#Creación VGG16
vgg= applications.vgg16.VGG16()


# In[4]:


#Creación VGG16 propia denominada cnn_micro
cnn_micro=Sequential()
for capa in vgg.layers:
    cnn_micro.add (capa)


# In[5]:


cnn_micro.summary()


# In[6]:


cnn_micro.pop()


# In[7]:


cnn_micro.summary()


# In[8]:


#Las capas ya entrenadas de la VGG16 no vuelven a entrenarse para reutilizar el entrenamiento en la segmentación y detección de las imágenes en bruto
for layer in cnn_micro.layers:
    layer.trainable=False 


# In[9]:


cnn_micro.add(Dense(3,activation='softmax')) #3 clases, moluscos, peces, y anormalidades nucleares.


# In[10]:


cnn_micro.summary()


# In[11]:


K.clear_session()
data_entrenamiento = './data/entrenamiento'
data_validacion = './data/validacion'


# In[12]:


epocas=20
longitud, altura = 224, 224 #224 pixeles
batch_size = 32
pasos = 1000 # una época tendrá 1000 pasos
pasos_validacion = 200 # al finalizar la época se corren 200 pasos con datos de validación
filtrosConv1 = 32 # profundidad 
filtrosConv2 = 64
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2) #tamaño del filtro maxpooling
clases = 3 #clases moluscos, peces y an.
lr = 0.0004 # ajustes de solución óptima


# In[13]:


CATEGORIES = ["AN","Peces","Moluscos"]


# In[15]:


#for category in CATEGORIES:
 #   etiquetas = os.path.join(data_entrenamiento,CATEGORIES)
  #  for img in os.listdir(etiquetas):
   #     img_array = cv2.imread(os.path.join(etiquetas,img)cv2.IMREAD_GRAYSCALE)
    #    plt.imshow (img_array,cmap = 'gray')
     #   plt.show ()
   # break
#break


# In[14]:


fig, ax = plt.subplots(figsize=(10.6666, 10.6666))
image_peces = './data/entrenamiento/Moluscos/BlebbedG.tif'
image = io.imread(image_peces)
ax.imshow(image) 


# In[18]:


def IC(im, box):
    box=int(box)
    image_label_overlay = im
    image_MN = color.rgb2gray(image_label_overlay[:,:,2])
    img=image_MN
    
    # Eliminamos el fondo de la imagen mediante el metodo de Li
    thresh = threshold_li(image_MN)
    #Obtenemos los objetos detectados en la imagen eliminando el fondo -utilizando el valor limite- y uniendo huecos de un pixel para obtener continuidad.
    bw = closing(image_MN > thresh, square(3))

    # Etiquetamos los objetos detectados
    label_image_blue = label(bw)
    #Contamos el numero de nucleos detectados
    region_image_blue = label_image_blue.copy()

    cell_num=1
    for region in regionprops(label_image_blue, image_MN):
        if region.area > 200:#Solo detectamos aquellos regiones de color azul con un minimo de pixeles para poder ser considerado un nucleo
            
            region_image_blue[:,:] = 0
            region_image_blue[label_image_blue == region.label] = region.label
            region_image_blue[region_image_blue != 0] = image_MN[region_image_blue != 0]
            centre=region.centroid
            #centre = region.coords[np.random.choice(range(region.coords.shape[0])),:]#random centre
            
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
            #img=Image.fromarray(col_sel_cel)
            #img.save("".join(["./",im[:len(im)-4],"_cell",str(cell_num),".tif"])) #save plots
            plt.close("all")
            cell_num+=1
    return col_sel_cel

image = IC(image, 192)


# In[19]:


from skimage import transform as tf
box=192
tform =tf.AffineTransform(scale=(1,1.5), rotation=-0.5,translation=(0,0), shear=0)
img_warp = tf.warp(image, tform)

img_warp = img_warp*255
plt.subplots( figsize=(20,20))
plt.subplot(1,2,1)
plt.imshow(image)
plt.subplot(1,2,2)
plt.imshow(img_warp)
plt.show( )


# In[94]:


fig, ax = plt.subplots(figsize=(5, 5))
sigma = 7
def watershed (image, sigma):
    t0 = time.time()
    T_min = threshold_li(image)
    bin_image = image > T_min
    dnaf = mh.gaussian_filter(image.astype(float), sigma)
    maxima = mh.regmax(mh.stretch(dnaf))
    maxima,_ = mh.label(maxima)
    dist = mh.distance(bin_image)
    dist = 255 - mh.stretch(dist)
    watershed = mh.cwatershed(dist, maxima)
    watershed *= bin_image
    t1 = time.time()
    return (t1-t0, watershed)

plt.imshow(watershed(image, sigma)[7])


# In[20]:


##Preparamos nuestras imagenes

entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2, #posibilidad de detectar MN en base a zoom
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255) 

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical') # Del directorio entrenamiento procesa las imagenes, y la clasificación será de etiqueta categorica. 

validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')


# In[21]:


cnn_micro.compile(loss='categorical_crossentropy',
            optimizer=optimizers.Adam(lr=lr),
            metrics=['accuracy'])


# In[ ]:


cnn_micro.fit_generator(
    entrenamiento_generador,
    steps_per_epoch=pasos,
    epochs=epocas,
    data_validacion=validacion_generador,
    validation_steps= pasos_validacion)


# In[ ]:




