import pandas as pd
import os, cv2
import numpy as np
import models
from utils import DataGeneratorMobileNetKeras, DataGeneratorMobileNet
import re

def mass_center(mask):
    #calculate mass center from top-left corner
    x_by_mass = 0
    y_by_mass = 0
    total_mass = np.sum(mask)
    for x in np.arange(0,mask.shape[0]):
        x_by_mass += np.sum(x * mask[:,x])
        y_by_mass += np.sum(x * mask[x,:])

    return((x_by_mass/total_mass, y_by_mass/total_mass))

#compute validation metrics
def validate(**kwargs):

    if kwargs['preprocessing'] == True:
        valid_generator = DataGeneratorMobileNetKeras(batch_size=1,img_path=kwargs['img_path'], labels=kwargs['labels'],
            list_IDs=kwargs['partition']['valid'],n_channels=3, n_channels_label=1,shuffle=False,mask_path=kwargs['masks_path'])
    else:
        valid_generator = DataGeneratorMobileNet(batch_size=1,img_path=kwargs['img_path'], labels=kwargs['labels'],
            list_IDs=kwargs['partition']['valid'],n_channels=3, n_channels_label=1,shuffle=False,mask_path=kwargs['masks_path'])
        
    model = models.load_model(os.path.join('.','output', 'models', kwargs['model_name']+'.h5'))
    prediction = model.predict_generator(generator=valid_generator, use_multiprocessing=True, workers=6, verbose=True)

    #TODO Guardar mascaras resultantes
    mask_output_path = os.path.join(kwargs['validation_folder'], kwargs['model_name'])
    if not os.path.exists(mask_output_path):
        os.makedirs(mask_output_path)
    


    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    test_images = kwargs['partition']['valid']
    labels = kwargs['labels']
    valid_metrics = {
            'model_name': [],
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
        array_pred = np.copy(prediction)
        for i in np.arange(0,prediction.shape[0]):
            valid_metrics['model_name'].append(kwargs['model_name'])
            valid_metrics['threshold'].append(threshold)
            #get prediction and normalize
            pred = array_pred[i,:,:,0]
            pred = (pred > threshold).astype(bool)
            #save sample name
            valid_metrics['sample'].append(test_images[i])
            cv2.imwrite(os.path.join(mask_output_path, str(threshold) + '_' + test_images[i] + '.png'), np.uint8(pred)*255)
            #get mask and preprocess
            mask_name = labels[test_images[i]]
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
            #compute mass centers without binarizing
        array_pred = np.copy(prediction) #reset prediction array
        for i in np.arange(0,prediction.shape[0]):
            pred = array_pred[i,:,:,0]
            pred[pred < threshold] = 0
            #get mask
            mask_name = labels[test_images[i]]
            mask = cv2.imread(kwargs['masks_path'] + '/' + mask_name)
            mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = mask.astype(bool)
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
    csv_path = os.path.join(kwargs['validation_folder'], kwargs['model_name'] + '.csv')
    data.to_csv(csv_path)
    print(kwargs['model_name'] + ' report finished!')
    return csv_path

if __name__ == "__main__":
    list_models = pd.read_csv('models_to_validate.csv', header=None)
    list_models = list_models.iloc[:,0].values
    print(list_models)
    args = {}
    for model in list_models:
        if(re.search(r'mobilenet', model) != None):
            args['preprocessing'] = True
        else:
            args['preprocessing'] = False

        test_set = pd.read_csv(os.path.join('/home','wvillegas','dataset-mask', 'single_instance_test.csv'))
        test_set_array = test_set['imageOrigin'].values
        args['partition'] = {
            'train':[],
            'valid':test_set_array
        }

        args['labels'] = dict(zip(list(test_set['imageOrigin'].values), list(test_set['mask'].values)))
        args['model_name'] = model
        args['img_path'] = os.path.join('/home','wvillegas','dataset-mask','dataset_resize', 'images_resize')
        args['masks_path'] = os.path.join('/home','wvillegas','dataset-mask','dataset_resize', 'masks_resize')
        args['validation_folder'] = os.path.join('.','output','validation')
        validate(**args)