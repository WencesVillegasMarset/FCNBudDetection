import pandas as pd
import os
import models
from utils import DataGeneratorMobileNet
import numpy as np
import cv2

def mass_center(mask):
    #calculate mass center from top-left corner
    x_by_mass = 0
    y_by_mass = 0
    total_mass = np.sum(mask)
    for x in np.arange(0,mask.shape[0]):
        x_by_mass += np.sum(x * mask[:,x])
        y_by_mass += np.sum(x * mask[x,:])

    return((x_by_mass/total_mass, y_by_mass/total_mass))

def train_model(**kwargs):
    '''
        Model Training Script
        kwargs:
            'model' = Codename for model (8, 16 or 32) (int)
            'batch size' = batch size to be used in training (int)
            'train_encoder': wether or not train the pretrained encoder layers of the net (bool) 
            'img_path': path to the images (X) folder (string)
            'masks_path': path to the masks (Y) folder (string)
            'partition': dictionary with two lists one with training samples and one with validation samples (dict)
            'labels': dictionary key:value where img_name:mask_name (dict)
            'optimizer': name of the optimizer to use (sgd, adam, rmsprop or nadam) (string)
            'lr': learning rate (float)
            'momentum': momentum (float)
            'decay': lr decay (float)
            'epochs': number of epochs (int)
            'models_folder': target folder and filename for h5 model file (string)
            'history_folder': target folder and filename for csv history file (string)
            'final_layer': name of the final layer activation, linear or sigmoid (str)
    '''

    if kwargs['model'] == 8:
        model = models.mobilenet_8s(train_encoder=kwargs['train_encoder'],
            final_layer_activation=kwargs['final_layer'])
    elif kwargs['model'] == 16:
        model = models.mobilenet_16s(train_encoder=kwargs['train_encoder'],
            final_layer_activation=kwargs['final_layer'])
    elif kwargs['model'] == 32:
        model = models.mobilenet_32s(train_encoder=kwargs['train_encoder'],
            final_layer_activation=kwargs['final_layer'])

    train_generator = DataGeneratorMobileNet(batch_size=kwargs['batch_size'],img_path=kwargs['img_path'],
                                labels=kwargs['labels'],list_IDs=kwargs['partition']['train'],n_channels=3,
                                n_channels_label=1,shuffle=True,mask_path=kwargs['masks_path'])
    if kwargs['optimizer'] == 'sgd':
        from keras.optimizers import SGD
        optimizer = SGD(lr=kwargs['lr'], momentum=kwargs['momentum'], decay=kwargs['decay'])
    elif kwargs['optimizer'] == 'rmsprop':
        from keras.optimizers import RMSprop
        optimizer = RMSprop()
    elif kwargs['optimizer'] == 'adam':
        from keras.optimizers import Adam
        optimizer = Adam()    

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    train_history = model.fit_generator(generator=train_generator, use_multiprocessing=True,workers=6, epochs=kwargs['epochs'])
    model_name = str(kwargs['fold'])+'FCMN'+str(kwargs['model'])+kwargs['optimizer']+'_lr'+str(kwargs['lr'])+'_decay'+str(kwargs['decay'])+'_ep'+str(kwargs['epochs'])

    model.save(os.path.join(kwargs['models_folder'], model_name + '.h5'))

    history_csv = pd.DataFrame(train_history.history)
    history_csv.to_csv(os.path.join(kwargs['history_folder'], model_name +'.csv'))
    #compute validation metrics

    valid_generator = DataGeneratorMobileNet(batch_size=1,img_path=kwargs['img_path'], labels=kwargs['labels'],
        list_IDs=kwargs['partition']['valid'],n_channels=3, n_channels_label=1,shuffle=False,mask_path=kwargs['masks_path'])

    prediction = model.predict_generator(generator=valid_generator,use_multiprocessing=True,workers=6, verbose=True)


    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    valid_metrics = {
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
    for threshold in threshold_list:    
        for i in np.arange(0,prediction.shape[0]):
            valid_metrics['threshold'].append(threshold)
            #get prediction and normalize
            pred = (prediction[i,:,:,0] > threshold).astype(bool)
            #save sample name
            sample_name = kwargs['partition']['valid'][i]
            valid_metrics['sample'].append()
            #get mask and preprocess
            mask_name = kwargs['labels'][sample_name]
            mask = cv2.imread(kwargs['masks_path'] + '/' + mask_name)
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
            valid_metrics['iou'].append(iou)
            valid_metrics['intersection'].append(intersection)
            valid_metrics['union'].append(union)
            valid_metrics['gt_area'].append(mask_area)
            valid_metrics['segmentation_area'].append(prediction_area)
            gt_center = mass_center(mask)
            segmentation_center = mass_center(pred)
            distance = np.subtract(gt_center,segmentation_center)
            valid_metrics['gt_x_center'].append(gt_center[0])
            valid_metrics['gt_y_center'].append(gt_center[1])
            valid_metrics['segmentation_x_center'].append(segmentation_center[0])
            valid_metrics['segmentation_y_center'].append(segmentation_center[1])
            valid_metrics['x_distance'].append(distance[0])
            valid_metrics['y_distance'].append(distance[1])
            valid_metrics['euclidean_distance'].append(np.linalg.norm(distance))
            valid_metrics['x_size'].append(pred.shape[0])
            valid_metrics['y_size'].append(pred.shape[1])
    data = pd.DataFrame(valid_metrics)
    data.to_csv(os.path.join('/home','wvillegas','DLProjects','DetectionModels','validation_metrics', 'valid' + model_name +'.csv'))
    return model_name


    
        
        