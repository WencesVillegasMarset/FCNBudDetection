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

def connected_components_with_threshold(image, threshold):
    '''
        Function that takes a mask and filters its component given a provided threshold
        this returns the number of resulting components and a new filtered mask (tuple) 
    '''
    num_components, mask = cv2.connectedComponents(image)
    filtered_mask = np.zeros_like(image, dtype=np.uint8)
    component_list = []
    mass_center_array = []

    for component in np.arange(1, num_components):
        isolated_component = (mask == component)
        if np.sum(isolated_component) >= threshold:
            mass_center_array.append(mass_center(isolated_component.astype(int)))
            filtered_mask += isolated_component
            component_list.append(component)
    if len(component_list) == 0:   
        mass_center_array = np.nan
    return len(component_list), filtered_mask, (np.asarray(mass_center_array))



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

    mask_output_path = os.path.join(kwargs['validation_folder'], kwargs['model_name'], 'prediction_masks')
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
            'true_positive_x_center':[],
            'true_positive_y_center':[],
            'euclidean_distance':[],
            'x_size':[],
            'y_size':[],
            'buds_predicted':[],
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
            #get mask and preprocess
            mask = cv2.imread(kwargs['masks_path'] + '/' + labels[test_images[i]])
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

            gt_center = mass_center(mask) #grab ground truth center
            valid_metrics['gt_x_center'].append(gt_center[0])
            valid_metrics['gt_y_center'].append(gt_center[1])

            
            valid_metrics['x_size'].append(pred.shape[0])
            valid_metrics['y_size'].append(pred.shape[1])


            num_labels, labeled_img, centers = connected_components_with_threshold(mask, 0)

            if not np.any(np.isnan(centers)):
                valid_metrics['buds_predicted'].append(centers.shape[0])
                temp_correspondence = {}
                for c in np.arange(centers.shape[0]):
                    pred_center = centers[c]   
                    distance_list.append(np.linalg.norm(np.subtract(gt_center,pred_center)))
                    temp_correspondence[distance_list[c]] = pred_center
                valid_metrics['euclidean_distance'].append(min(distance_list))
                valid_metrics['true_positive_x_center'].append(temp_correspondence[min(distance_list)][0])
                valid_metrics['true_positive_y_center'].append(temp_correspondence[min(distance_list)][1])
            else: #no buds detected register it in the metrics dict
                valid_metrics['euclidean_distance'].append(np.nan)
                valid_metrics['true_positive_x_center'].append(np.nan)
                valid_metrics['true_positive_y_center'].append(np.nan)
                valid_metrics['buds_predicted'].append(0)
            
    data = pd.DataFrame(valid_metrics)
    csv_path = os.path.join(kwargs['validation_folder'], kwargs['model_name'] + '.csv')
    data.to_csv(csv_path)
    print(kwargs['model_name'] + ' report finished!')
    return csv_path

if __name__ == "__main__":
    list_models = pd.read_csv('models_to_validate.csv', header=None)
    list_models = list_models.iloc[:,0].values
    print(list_models)
    test_set = pd.read_csv(os.path.join('/home','wvillegas','dataset-mask', 'single_instance_test.csv'))
    test_set_array = test_set['imageOrigin'].values
    args = {}
    for model in list_models:
        args['preprocessing'] = True
        
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