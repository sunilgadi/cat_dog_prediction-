import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import cv2
img = cv2.imread('../input/cat-and-dog/training_set/training_set/cats/cat.100.jpg')
print(img.shape)
plt.imshow(img)

from keras.models import Sequential # To intitialize our neural network
from keras.layers import Convolution2D  # To add convolution layers of CNN
from keras.layers import MaxPooling2D  # To add pooling layers
from keras.layers import Flatten # To flatten pooled layers to a large array for input of ANN
from keras.layers import Dense # To add fully constructed layers in ANN

classifier = Sequential()

classifier.add(Convolution2D(32 , (3 , 3) , input_shape = (64 , 64 , 3) ,activation = 'relu'))

classifier.add(MaxPooling2D(pool_size= (2,2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 128 , activation= 'relu')) # here there is no need to add input_dim since there is a prev layer before this hidden layer.

classifier.add(Dense(output_dim = 1 , activation= 'sigmoid'))

classifier.compile(optimizer='adam', loss = 'binary_crossentropy' , metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator # import for ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,       #### Parameters for Image augmentation part
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        '../input/cat-and-dog/training_set/training_set',
        target_size=(64, 64),     ## Dimensions expected by our CNN , which we made as (64 x 64)
        batch_size=32,
        class_mode='binary')  # Since it is a binary output

test_set = test_datagen.flow_from_directory(
        '../input/cat-and-dog/test_set/test_set',
        target_size=(64, 64),  ## Dimensions expected by our CNN , which we made as (64 x 64)
        batch_size=32,
        class_mode='binary')    # Since it is a binary output 

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000, ## Total number of images in our training set
        epochs=5,
        validation_data=test_set,
        validation_steps=2000)   ## Total number of images in our test set
      
 training_set.class_indices
 from keras.preprocessing import image
test_image = image.load_img('../input/test-dog/1.jpeg' , target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image , axis =0 )
result = classifier.predict(test_image)
if result[0][0]==0 :
    print('cat')
else:
    print('dog')
