import pandas as pd
import os, cv2
import numpy as np
import models
from utils import DataGeneratorMobileNetKeras, DataGeneratorMobileNet
import re
def validate(**kwargs):

    if kwargs['preprocessing'] == True:
        valid_generator = DataGeneratorMobileNetKeras(batch_size=1,img_path=kwargs['img_path'], labels=kwargs['labels'],
            list_IDs=kwargs['partition']['valid'],n_channels=3, n_channels_label=1,shuffle=False,mask_path=kwargs['masks_path'])
    else:
        valid_generator = DataGeneratorMobileNet(batch_size=1,img_path=kwargs['img_path'], labels=kwargs['labels'],
            list_IDs=kwargs['partition']['valid'],n_channels=3, n_channels_label=1,shuffle=False,mask_path=kwargs['masks_path'])
        
    model = models.load_model(os.path.join('.','output', 'models', kwargs['model_name']+'.h5'))
    prediction = model.predict_generator(generator=valid_generator, use_multiprocessing=True, workers=6, verbose=True)

    #mask_output_path = os.path.join(kwargs['validation_folder'], kwargs['model_name'], 'prediction_masks')
    #if not os.path.exists(mask_output_path):
    #    os.makedirs(mask_output_path)
    


    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    test_images = kwargs['partition']['valid']
    labels = kwargs['labels']
    valid_metrics = {
            'model_name': [],
            'threshold':[],
            'sample_name': [],
            'iou':[],
            'precision':[],
            'recall':[],
            'intersection':[],
            'union':[],
            'gt_area':[],
            'segmentation_area':[],
            'num_components':[]
            }
    for threshold in threshold_list:  
        array_pred = np.copy(prediction)
        for i in np.arange(0,prediction.shape[0]):
            valid_metrics['model_name'].append(kwargs['model_name'])
            valid_metrics['threshold'].append(threshold)
            valid_metrics['sample_name'].append(test_images[i])
            #get prediction and normalize
            pred = array_pred[i,:,:,0]
            pred = (pred > threshold).astype(bool)
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
            valid_metrics['intersection'].append(intersection)
            valid_metrics['union'].append(union)
            valid_metrics['gt_area'].append(mask_area)
            valid_metrics['segmentation_area'].append(prediction_area)
            valid_metrics['iou'].append(intersection / union)
            valid_metrics['precision'].append(intersection/prediction_area)
            valid_metrics['recall'].append(intersection / mask_area)
            if (prediction_area == 0):
                valid_metrics['num_components'].append(0)
            _, num_components = cv2.connectedComponents(pred.astype(np.uint8))
            valid_metrics['num_components'].append(num_components-1)


    data = pd.DataFrame(valid_metrics)
    csv_path = os.path.join(kwargs['validation_folder'],'final_cv', kwargs['model_name'] + '.csv')
    data.to_csv(csv_path)
    print(kwargs['model_name'] + ' report finished!')
    return csv_path


if __name__ == "__main__":
    list_models = pd.read_csv('models_to_validate.csv', header=None)
    list_models = list_models.iloc[:,0].values
    print(list_models)
    args = {}
    for model in list_models:
        args['preprocessing'] = True

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