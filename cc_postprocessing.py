import faulthandler
faulthandler.enable()
from sklearn.cluster import DBSCAN
import cv2
import numpy as np
#import matplotlib as mpl
#mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import time
import argparse
from keras.models import load_model
from utils import DataGeneratorMobileNetKeras, DataGeneratorMobileNet

'''
    TODO : Adaptar a las mascaras generadas por mobilenet
    TODO : Mantener correspondencia haciendo inferencia una por una a costa de tiempo  
'''
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

def connected_components(img):
    num_components, labeled_img =  cv2.connectedComponents(img, connectivity=8)
    label_array = np.arange(0,num_components)[1::]
    return label_array, labeled_img

def cluster_mass_center(mask, labels):
    if labels == 0:
        return np.nan
    mass_center_array = []
    for label in labels:
        cluster_array = (mask == label)
        mass_center_array.append(mass_center(cluster_array.astype(int)))
    return (np.asarray(mass_center_array))

def labeled_img_to_rgb(mask, labels):
    if labels.shape[0] == 0:
        return utils_cluster.grayscale_to_rgb((np.zeros_like(mask)).astype(float))
    cluster_hue = np.linspace(0,255,labels.shape[0]+1)
    cluster_array_list = []
    for label in labels:
        cluster_array = (mask == label)
        cluster_array_list.append(cluster_array)
    grayscale_img = (np.zeros_like(mask)).astype(float)
    for c in np.arange(len(cluster_array_list)):
        grayscale_img += (cluster_array_list[c] * cluster_hue[c+1]) 
    return utils_cluster.grayscale_to_rgb(grayscale_img)

def run(args):
    start = time.clock()
    model_name = os.path.split(args.h5)[1]
    model_name = model_name[:-3]
    #create model output folder if it doesnt exist already
    output_path = os.path.join('.', 'output', 'validation', model_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ground_truth_csv = pd.read_csv('./single_instance_dataset_wradius.csv')
    test_set_csv = pd.read_csv(os.path.join('/home','wvillegas','dataset-mask', 'single_instance_test.csv'))
    test_set_images = test_set['imageOrigin'].values
    partition = {
            'train':[],
            'valid':test_set_images
        }
    labels = dict(zip(list(test_set_csv['imageOrigin'].values), list(test_set_csv['mask'].values)))
    img_path = os.path.join('/home','wvillegas','dataset-mask','dataset_resize', 'images_resize')
    masks_path = os.path.join('/home','wvillegas','dataset-mask','dataset_resize', 'masks_resize')

    valid_generator = DataGeneratorMobileNetKeras(batch_size=1,img_path=img_path, labels=labels,
            list_IDs=partition['valid'],n_channels=3, n_channels_label=1,shuffle=False,mask_path=masks_path)




    model = models.load_model(args.h5)

    prediction = model.predict_generator(generator=valid_generator, use_multiprocessing=True, workers=6, verbose=True)


    
    threshold_range = (np.arange(args.min, args.max, args.step)).tolist()

    metrics = {
            'model_name':[],
            'image_name':[],
            'threshold':[],
            'buds_predicted':[],
            'true_positive_x':[],
            'true_positive_y':[],
            'true_positive_distance':[],
            'true_positive_norm_distance': []
            }

    for threshold in threshold_range:
           
        print("Threshold " + str(threshold))
        for i in len(test_set_images):
            metrics['model_name'].append(model_name)
            metrics['image_name'].append(test_set_images[i])
            metrics['threshold'].append(threshold)
            
            
            print('Processing :' + test_set_images[i])
            num_labels, labeled_img, centers = connected_components_with_threshold((prediction[i,:,:,0], threshold)
          
            sample_data = {}
            sample_data['sample_name'] = img
            sample_data['threshold'] = threshold
            sample_data['clustered_sample_path'] = os.path.join(model_validation_folder,'clustered_masks', 'cluster_'+img)
            if not np.any(np.isnan(centers)):
                    sample_data['centers'] = centers.tolist()
            else:
                sample_data['centers'] = centers

            row = ground_truth_csv.loc[ground_truth_csv['imageOrigin'] == test_set_images[i], :]
            gt_center = np.ndarray([1,2])
            gt_center[0,0] = (row['x_center_resize']/2)
            gt_center[0,1] = (row['y_center_resize']/2)
            distance_list = []

            if not np.any(np.isnan(centers)):
                metrics['buds_predicted'].append(centers.shape[0])
                temp_correspondence = {}
                for c in np.arange(centers.shape[0]):
                    pred_center = centers[c]   
                    distance_list.append(np.linalg.norm(np.subtract(gt_center,pred_center)))
                    temp_correspondence[distance_list[c]] = pred_center
                metrics['true_positive_distance'].append(min(distance_list))
                metrics['true_positive_x'].append(temp_correspondence[min(distance_list)][0])
                metrics['true_positive_y'].append(temp_correspondence[min(distance_list)][1])
                metrics['true_positive_norm_distance'].append(min(distance_list)/(ground_truth_csv['diam_resize']/2))
            else: #no buds detected register it in the metrics dict
                metrics['true_positive_distance'].append(np.nan)
                metrics['true_positive_x'].append(np.nan)
                metrics['true_positive_y'].append(np.nan)
                metrics['buds_predicted'].append(0)
                metrics['true_positive_norm_distance'].append(np.nan)


            sample_data['gt_center'] = gt_center.tolist()
            sample_data['distances'] = distance_list

            #with open(os.path.join(model_validation_folder,'clustered_masks', 'cluster_'+utils_cluster.remove_extension_from_filename(img) + '.json'), 'w') as fp:
            #    json.dump(sample_data, fp, indent=4)




    print(str(time.clock() - start) + ' seconds') 
    data = pd.DataFrame(metrics)
    data.to_csv(os.path.join(output_path, model_name+'_postprocessing.csv'))


    print("Generating plots!")

    threshold_list = data['threshold'].unique()
    nan_list = []
    for threshold in threshold_list:
        th_extracted = data.loc[data['threshold'] == threshold,:]
        nan_list.append(th_extracted.loc[th_extracted['buds_predicted'] == 0,:].shape[0])

    prec_list = []
    for th in threshold_list:
        th_extracted = data.loc[data['threshold'] == th,:]
        true_positives = th_extracted.loc[th_extracted['buds_predicted']>=1,:].shape[0]
        false_positives = th_extracted['buds_predicted'].sum() - true_positives
        prec_list.append(true_positives / (true_positives + false_positives))
    plt.bar(x=threshold_list, height=prec_list)
    plt.title(nan_list)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(os.path.join(output_path, model_name+'_precision.png'))

    plt.clf()


    rec_list = []
    for threshold in threshold_list:
        th_extracted = data.loc[data['threshold'] == threshold,:]
        #aquellos patches en los que se predijo una yema
        true_positives = th_extracted.loc[th_extracted['buds_predicted']==1,:].shape[0] 
        # aquellos en los que no se predijeron ninguna
        false_negatives = th_extracted.loc[th_extracted['buds_predicted']==0,:].shape[0] 
        rec_list.append(true_positives / (true_positives + false_negatives))
    plt.bar(x=threshold_list, height=rec_list, align='center',tick_label=threshold_list)
    plt.title(nan_list)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(os.path.join(output_path, model_name+'_recall.png'))
    
        

        
def main():
    parser = argparse.ArgumentParser(
        description="Run clustering for sliding windows binary masks")
    parser.add_argument("-h5", help="path of .h5 model file",
                        dest="h5", type=str, required=True)
    parser.add_argument("-min", help="min of range of thresholds",
                        dest="min", type=int, required=True)
    parser.add_argument("-max", help="max of range of thresholds",
                        dest="max", type=int, required=True)
    parser.add_argument("-step", help="step for threshold range",
                        dest="step", type=int, required=True)
    

    parser.set_defaults(func=run)
    args = parser.parse_args()

    if (not os.path.exists(args.h5)):
        parser.error('Invalid path to csv')
    args.func(args)


if __name__ == "__main__":
    main()
