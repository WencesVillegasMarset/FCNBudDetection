import faulthandler
faulthandler.enable()
from sklearn.cluster import DBSCAN
import cv2
import numpy as np
#import matplotlib as mpl
#mpl.use('TkAgg')  # or whatever other backend that you want
#import matplotlib.pyplot as plt
import pandas as pd
import os
from utils import utils_cluster
import json


def dbscan(img, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(img)
    labeled_img = (np.reshape(db.labels_, [1024, 1024]))
#     print(np.unique(labeled_img))
    num_labels = np.unique(labeled_img)
    num_labels = num_labels[np.where( num_labels > 0 )]
    return labeled_img, num_labels

def cluster_mass_center(mask, labels):
    if labels.shape[0] == 0:
        return np.nan
    mass_center_array = []
    for label in labels:
        cluster_array = (mask == label)
        mass_center_array.append(utils_cluster.mass_center(cluster_array.astype(float)))
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

if __name__ == "__main__":
    '''
        TODO 1: Configurar que se mande un csv con la ruta absoluta a las imagenes, y el nombre del modelo a validar
        TODO 2: crear un directorio donde se guarden las versiones rgb de las mascaras pasadas por dbscan
        TODO 3: crear un json para cada imagen (1:1) y se guarde toda las info sobre el resultado de la clusterizacion
    '''
    ground_truth_csv = pd.read_csv('../single_instance_dataset_wradius.csv')
    route_csv = pd.read_csv(os.path.join('..', 'cluster_route.csv'), header=None)
    base_images_path = route_csv.iloc[0,0]
    print(base_images_path)

    model_validation_folder = os.path.split(base_images_path)[0]
    model_name = os.path.split(model_validation_folder)[1]
    print("Processing images in: " + base_images_path)
    img_list = sorted(os.listdir(base_images_path))
    metrics = {
        'model_name':[],
        'mask_name':[],
        'eps':[],
        'min_samples':[],
        'buds_predicted':[],
        'true_positive_x':[],
        'true_positive_y':[],
        'true_positive_distance':[]
    }
    if not os.path.exists(os.path.join(model_validation_folder,'clustered_masks')):
            os.makedirs(os.path.join(model_validation_folder,'clustered_masks'))
    for img in img_list:
        metrics['model_name'].append(model_name)
        metrics['mask_name'].append(img)
        metrics['eps'].append(10)
        metrics['min_samples'].append(50)
        
        print('Processing :' + img)
        # cluster image and get a labeled image where each pixel has a label value and a number of labels (ignoring 0 and -1)
        labeled_img, num_labels = dbscan(utils_cluster.preprocess_image(utils_cluster.read_image_grayscale(os.path.join(base_images_path, img))),10,50)
        
        
        utils_cluster.save_image(labeled_img_to_rgb(labeled_img, num_labels), os.path.join(model_validation_folder,'clustered_masks'), 'cluster_'+img)
        
        #get array of bidimensional center arrays [xcoord, ycoord]
        centers = cluster_mass_center(labeled_img, num_labels)
    #     print(centers)
        #check if there have been buds detected and process them
        sample_data = {}
        sample_data['sample_name'] = img
        sample_data['clustered_sample_path'] = os.path.join(model_validation_folder,'clustered_masks', 'cluster_'+img)
        if not np.any(np.isnan(centers)):
                sample_data['centers'] = centers.tolist()
        else:
            sample_data['centers'] = centers


        if not np.any(np.isnan(centers)):

            metrics['buds_predicted'].append(centers.shape[0])
            distance_list = []
            temp_correspondence = {}
            for c in np.arange(centers.shape[0]):
                pred_center = centers[c]
                row = utils_cluster.get_sample_ground_truth(img, ground_truth_csv)
                gt_center = np.ndarray([1,2])
                gt_center[0,0] = (row['x_center_resize'].values[0])/2
                gt_center[0,1] = (row['y_center_resize'].values[0])/2
                distance_list.append(np.linalg.norm(np.subtract(gt_center,pred_center)))
                temp_correspondence[distance_list[c]] = pred_center
            metrics['true_positive_distance'].append(min(distance_list))
            metrics['true_positive_x'].append(temp_correspondence[min(distance_list)][0])
            metrics['true_positive_y'].append(temp_correspondence[min(distance_list)][1])
        else: #no buds detected register it in the metrics dict
            metrics['true_positive_distance'].append(np.nan)
            metrics['true_positive_x'].append(np.nan)
            metrics['true_positive_y'].append(np.nan)
            metrics['buds_predicted'].append(0)


        sample_data['gt_center'] = gt_center.tolist()
        sample_data['distances'] = distance_list
        with open(os.path.join(model_validation_folder,'clustered_masks', 'cluster_'+utils_cluster.remove_extension_from_filename(img) + '.json'), 'w') as fp:
            json.dump(sample_data, fp, indent=4)



    data = pd.DataFrame(metrics)
    data.to_csv(os.path.join(model_validation_folder,'metrics_cluster_'+model_name+'.csv'))
