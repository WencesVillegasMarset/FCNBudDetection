
# coding: utf-8

# In[1]:


from keras.models import load_model
import pandas as pd
import cv2
from skimage.io import imread, imshow, imsave
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
from utils_fcn import DataGeneratorMobileNet


# Para medir el diametro de la mascara, sacar la distancia en el centro de masa al pixel mas lejano de la mascara.

# In[3]:


fcn = load_model('/home/wvillegas/DLProjects/DetectionModels/models/FCNMNSGD_100ep.h5')


# In[6]:


img_path = os.path.join('/home','wvillegas','dataset-mask','dataset_resize', 'images_resize')
masks_path = os.path.join('/home','wvillegas','dataset-mask','dataset_resize', 'masks_resize')


# In[7]:


dataset = pd.read_csv(os.path.join('/home','wvillegas','dataset-mask', 'full_masks.csv'))


# In[8]:


test_set = pd.read_csv('/home/wvillegas/DLProjects/DetectionModels/models/testSGD_100ep.csv', header=None)


# In[9]:


img_list = list(dataset['orig'].values)
mask_list = list(dataset['mask'].values)


# In[10]:


test_images = list(test_set[1].values)


# In[11]:


labels = dict(zip(img_list, mask_list))


# In[12]:


list_img = test_set[1][11:21].values
labels = test_set[2][11:21].values


# In[13]:


inp_img = []
for img in list_img:
    temp = cv2.imread(img_path + '/' + img)
    temp = cv2.resize(temp, (0,0), fx=0.5, fy=0.5)
    temp = cv2.normalize(temp, temp, 0, 255, cv2.NORM_MINMAX)
    inp_img.append(temp)
    
ground_truth = []
for img in np.arange(0,len(list_img)):
    mask = labels[img]
    temp = cv2.imread(masks_path + '/' + mask)
    temp = cv2.resize(temp, (0,0), fx=0.5, fy=0.5)
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    temp = temp.astype(bool).astype(int)
    ground_truth.append(temp)


# In[14]:


inp_img = np.asarray(inp_img)
ground_truth = np.asarray(ground_truth)


# In[15]:


pred = fcn.predict(inp_img)


# In[16]:


fig=plt.figure(figsize=(20, 5))
columns = 5
rows = 2
for i in range(1, columns*rows +1):
    img = inp_img[i-1,:,:,:]
    fig.add_subplot(rows, columns, i)
    imshow(img)
plt.show()


# In[17]:


fig=plt.figure(figsize=(20, 5))
columns = 5
rows = 2
for i in range(1, columns*rows +1):
    img = pred[i-1,:,:,0]
    fig.add_subplot(rows, columns, i)
#     plt.gray()
    imshow(img)
plt.show()


# In[18]:


fig=plt.figure(figsize=(20, 5))
columns = 5
rows = 2
for i in range(1, columns*rows +1):
    img = pred[i-1,:,:,0]
    img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
    fig.add_subplot(rows, columns, i)
#     plt.gray()
    imshow(img)
plt.show()


# In[16]:


# fig=plt.figure(figsize=(20, 5))
# columns = 5
# rows = 2
# for i in range(1, columns*rows +1):
#     img = pred[i-1,:,:,0]
#     img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
#     img = img > 0.5
#     fig.add_subplot(rows, columns, i)
# #     plt.gray()
#     imshow(img)
# plt.show()


# In[17]:


# fig=plt.figure(figsize=(20, 5))
# columns = 5
# rows = 2
# for i in range(1, columns*rows +1):
#     img = pred[i-1,:,:,0]
#     img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
#     img = img > 0.7
#     fig.add_subplot(rows, columns, i)
# #     plt.gray()
#     imshow(img)
# plt.show()


# In[18]:


# fig=plt.figure(figsize=(20, 5))
# columns = 5
# rows = 2
# for i in range(1, columns*rows +1):
#     img = ground_truth[i-1,:,:]
#     fig.add_subplot(rows, columns, i)
#     plt.gray()
#     imshow(img)
# plt.show()


# In[19]:


# normalized=cv2.normalize(pred[0,:,:,0], None, 0, 1, cv2.NORM_MINMAX)


# In[20]:


# thresholded = normalized > 0.6


# In[21]:


valid_generator = DataGeneratorMobileNet(batch_size=1,img_path=img_path,
                                labels=labels,list_IDs=test_images,n_channels=3,
                                n_channels_label=1,shuffle=False,mask_path=masks_path)
prediction = fcn.predict_generator(generator=valid_generator,use_multiprocessing=True,workers=6, verbose=True)


# In[22]:


threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# In[23]:


test_iou = {
    'threshold':[],
    'sample':[],
    'iou':[],
    'intersection':[],
    'union':[],
    'gt_area':[],
    'segmentation_area':[],
    'gt_x_center':[],
    'gt_y_center':[],
    'segmentation_x_center':[],
    'segmentation_y_center':[],
    'x_distance':[],
    'y_distance':[],
    'euclidean_distance':[],
    'x_size':[],
    'y_size':[]
}


# In[24]:


def mass_center(mask):
    #calculate mass center from top-left corner
    x_by_mass = 0
    y_by_mass = 0
    total_mass = np.sum(mask)
    for x in np.arange(0,mask.shape[0]):
        x_by_mass += np.sum(x * mask[:,x])
        y_by_mass += np.sum(x * mask[x,:])

    return((x_by_mass/total_mass, y_by_mass/total_mass))


# In[25]:


for threshold in threshold_list:    
    for i in np.arange(0,prediction.shape[0]):
        test_iou['threshold'].append(threshold)
        #get prediction and normalize
        pred = cv2.normalize(prediction[i,:,:,0], None, 0, 1, cv2.NORM_MINMAX)
        pred = (pred > threshold).astype(bool)
        #save sample name
        test_iou['sample'].append(test_images[i])
        #get mask and preprocess
        mask_name = labels[test_images[i]]
        mask = cv2.imread(masks_path + '/' + mask_name)
        mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask.astype(bool)

        #compute iou and areas
        intersection = np.sum(np.logical_and(pred, mask))
        union = np.sum(np.logical_or(pred, mask))
        prediction_area = np.sum(pred)
        mask_area = np.sum(mask)
        iou = intersection / union
        #add them to dict
        test_iou['iou'].append(iou)
        test_iou['intersection'].append(intersection)
        test_iou['union'].append(union)
        test_iou['gt_area'].append(mask_area)
        test_iou['segmentation_area'].append(prediction_area)
        gt_center = mass_center(mask)
        segmentation_center = mass_center(pred)
        distance = np.subtract(gt_center,segmentation_center)
        test_iou['gt_x_center'].append(gt_center[0])
        test_iou['gt_y_center'].append(gt_center[1])
        test_iou['segmentation_x_center'].append(segmentation_center[0])
        test_iou['segmentation_y_center'].append(segmentation_center[1])
        test_iou['x_distance'].append(distance[0])
        test_iou['y_distance'].append(distance[1])
        test_iou['euclidean_distance'].append(np.linalg.norm(distance))
        test_iou['x_size'].append(pred.shape[0])
        test_iou['y_size'].append(pred.shape[1])


# In[26]:


data = pd.DataFrame(test_iou)


# In[27]:


data.describe()


# In[28]:


data.to_csv(os.path.join('.','metrics','FCNMNAdam_50ep','mobile_fcn8_metrics_adam_50ep.csv'))


# In[115]:


th_dict = dict(zip(threshold_list, ['01', '02', '03', '04','05', '06', '07', '08', '09']))


# In[118]:


mean_th_iou = []
for threshold in threshold_list:
    for i in np.arange(10):
        pred = cv2.normalize(prediction[i,:,:,0], None, 0, 1, cv2.NORM_MINMAX)
        pred = (pred > threshold).astype(int)*0.5
        
        mask_name = labels[test_images[i]]
        mask = cv2.imread(masks_path + '/' + mask_name)
        mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask.astype(bool)
        result = mask + pred
        imsave(arr=cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(int),
               fname='./metrics/FCNMNAdam_50ep/sample_images/'+ th_dict[threshold] + mask_name)


# In[110]:


pred = cv2.normalize(prediction[0,:,:,0], None, 0, 1, cv2.NORM_MINMAX)
pred = (pred > 0.5).astype(int)*0.5

mask_name = labels[test_images[0]]
mask = cv2.imread(masks_path + '/' + mask_name)
mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
mask = mask.astype(bool)
result = mask + pred
imshow(cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(int))


# In[112]:


plt.gray()
plt.imshow(cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(int))


# In[114]:


imsave(arr=cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(int), fname='./metrics/FCNMNAdam_50ep/sample_images/'+ mask_name + str(th))


# In[2]:


metricas = pd.read_csv(os.path.join('.','metrics','FCNMNAdam_50ep','mobile_fcn8_metrics_adam_50ep.csv'))


# In[4]:


metricas.loc[]


# Recall = Intersection/GT  
# Precision = Intersection/Prediction
