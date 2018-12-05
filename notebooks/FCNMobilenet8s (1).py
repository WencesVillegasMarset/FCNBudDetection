
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import cv2
from skimage.io import imread,imshow
from utils_fcn import DataGeneratorMobileNet
import matplotlib.pyplot as plt
import keras.callbacks
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dataset = pd.read_csv(os.path.join('/home','wvillegas','dataset-mask', 'full_masks.csv'))


# In[3]:


from sklearn.model_selection import train_test_split


# In[4]:


X_train, X_test, Y_train, Y_test = train_test_split(dataset['orig'],dataset['mask'],
                                                    test_size = 0.2, random_state=1)


# In[5]:


partition = {'train':list(X_train),
             'test': list(X_test)}


# In[6]:


img_list = list(X_train) + list(X_test)
mask_list = list(Y_train) + list(Y_test)


# In[7]:


labels = dict(zip(img_list, mask_list))


# In[8]:


img_path = os.path.join('/home','wvillegas','dataset-mask','dataset_resize', 'images_resize')
masks_path = os.path.join('/home','wvillegas','dataset-mask','dataset_resize', 'masks_resize')


# In[9]:


batch_size = 4
# dim_img = (1728, 2304)
# dim_mask = (658,874)


# In[10]:


train_generator = DataGeneratorMobileNet(batch_size=batch_size,img_path=img_path,
                                labels=labels,list_IDs=partition['train'],n_channels=3,
                                n_channels_label=1,shuffle=True,mask_path=masks_path)
valid_generator = DataGeneratorMobileNet(batch_size=1,img_path=img_path,
                                labels=labels,list_IDs=partition['test'],n_channels=3,
                                n_channels_label=1,shuffle=False,mask_path=masks_path)


# In[11]:


from keras.applications import MobileNet
from keras.layers import Conv2DTranspose,Conv2D,Add
from keras import Model
from keras.models import load_model


# In[12]:


net = MobileNet(include_top=False, weights=None)


# In[13]:


net.load_weights('/home/wvillegas/DLProjects/PatchClassificationModelsResults/modelosV2/mobilenet_weights_detection.h5', by_name=True)


# In[14]:


for layer in net.layers:
    layer.trainable = True


# In[15]:


net.summary()


# In[16]:


# test arquitectura paper de FCN 
# deconv1 = Conv2D(filters=256,kernel_size=3,strides=1,activation='relu')(mobilenet.output)
predict = Conv2D(filters=1,kernel_size=1,strides=1)(net.output)

deconv2 = Conv2DTranspose(filters=1,kernel_size=4,strides=2, padding='same', use_bias=False)(predict)

pred_conv_pw_11_relu = Conv2D(filters=1,kernel_size=1,strides=1)(net.get_layer('conv_pw_11_relu').output)

fuse1 = Add()([deconv2, pred_conv_pw_11_relu])

pred_conv_pw_5_relu = Conv2D(filters=1,kernel_size=1,strides=1)(net.get_layer('conv_pw_5_relu').output)

deconv2fuse1 = Conv2DTranspose(filters=1,kernel_size=4,strides=2, padding='same', use_bias=False)(fuse1)

fuse2 = Add()([deconv2fuse1, pred_conv_pw_5_relu])

deconv8 = Conv2DTranspose(filters=1,kernel_size=16,strides=8, padding='same', use_bias=False, activation='sigmoid')(fuse2)


# In[17]:


fcn = Model(inputs=net.input,outputs=deconv8)


# In[18]:


from keras.optimizers import SGD, RMSprop


# In[19]:


sgd = SGD(lr=0.001,momentum=0.9, decay=0.0005)
fcn.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[20]:


# reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                               patience=3, min_lr=0.00001)


# In[21]:


history = fcn.fit_generator(generator=train_generator,validation_data=valid_generator,
                            use_multiprocessing=True,workers=6, epochs=10)


# In[22]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[23]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[24]:


list_img = partition['test'][11:21]


# In[25]:


inp_img = []
for img in list_img:
    temp = cv2.imread(img_path + '/' + img)
    temp = cv2.resize(temp, (0,0), fx=0.5, fy=0.5)
    temp = cv2.normalize(temp, temp, 0, 255, cv2.NORM_MINMAX)
    inp_img.append(temp)
    
ground_truth = []
for img in list_img:
    mask = labels[img]
    temp = cv2.imread(masks_path + '/' + mask)
    temp = cv2.resize(temp, (0,0), fx=0.5, fy=0.5)
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    temp = temp.astype(bool).astype(int)
    ground_truth.append(temp)


# In[26]:


inp_img = np.asarray(inp_img)
ground_truth = np.asarray(ground_truth)


# In[27]:


pred = fcn.predict(inp_img)


# In[28]:


fig=plt.figure(figsize=(20, 5))
columns = 5
rows = 2
for i in range(1, columns*rows +1):
    img = inp_img[i-1,:,:,:]
    fig.add_subplot(rows, columns, i)
    imshow(img)
plt.show()


# In[29]:


fig=plt.figure(figsize=(20, 5))
columns = 5
rows = 2
for i in range(1, columns*rows +1):
    img = pred[i-1,:,:,0]
    fig.add_subplot(rows, columns, i)
    imshow(img)
plt.show()


# In[30]:


fig=plt.figure(figsize=(20, 5))
columns = 5
rows = 2
for i in range(1, columns*rows +1):
    img = ground_truth[i-1,:,:]
    fig.add_subplot(rows, columns, i)
    plt.gray()
    imshow(img)
plt.show()


# Model Validation

# In[31]:


# test_iou = {
#     'sample':[],
#     'iou':[],
# }


# In[32]:


# prediction = fcn.predict_generator(generator=valid_generator,use_multiprocessing=True,workers=6, verbose=True)


# In[33]:


# prediction.shape


# In[34]:


# for i in np.arange(0,prediction.shape[0]):
#     pred = (prediction[i,:,:,0] > 0.2)
#     pred = pred.astype(bool) 
#     sample_name = partition['test'][i]
#     test_iou['sample'].append(sample_name)
#     mask_name = labels[sample_name]
#     mask = cv2.imread(masks_path + '/' + mask_name)
#     mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)
#     mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#     mask = mask.astype(bool)
#     #compute iou
#     interesection = np.sum(np.logical_and(pred, mask))
#     union = np.sum(np.logical_or(pred, mask))
#     iou = intersection / union
#     test_iou['iou'].append(iou)


# In[35]:


# test_frame = pd.DataFrame(test_iou)


# In[36]:


# test_frame.describe()

