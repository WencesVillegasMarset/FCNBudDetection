import matplotlib.pyplot as plt
import json
from utils import utils_cluster
import os
import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import hdbscan
import faulthandler
faulthandler.enable()
#import matplotlib as mpl
# mpl.use('TkAgg')  # or whatever other backend that you want


def mass_center(mask):
    # calculate mass center from top-left corner
    x_by_mass = 0
    y_by_mass = 0
    total_mass = np.sum(mask)
    for x in np.arange(0, mask.shape[0]):
        x_by_mass += np.sum(x * mask[:, x])
        y_by_mass += np.sum(x * mask[x, :])

    return((x_by_mass/total_mass, y_by_mass/total_mass))


def dbscan(img, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(img)
    num_labels = np.unique(db.labels_)
    num_labels = num_labels[np.where(num_labels > 0)]
    return db.labels_+1, num_labels


def hdbscan_clustering(img):
    hd = hdbscan.HDBSCAN(min_samples=1)
    hd.fit(img)
    num_labels = np.unique(hd.labels_)
    num_labels = num_labels[np.where(num_labels > 0)]
    return hd.labels_+1, num_labels


def connected_components_with_threshold(image, threshold, ground_truth):
    '''
        Function that takes a mask and filters its component given a provided threshold
        this returns the number of resulting components and a new filtered mask (tuple) 
    '''

    num_components, mask = cv2.connectedComponents(image)
    filtered_mask = np.zeros_like(image, dtype=np.uint8)
    component_list = []
    mass_center_array = []
    iou_array = []

    for component in np.arange(1, num_components):
        isolated_component = (mask == component)
        if np.sum(isolated_component) >= threshold:
            iou_array.append(np.sum(np.logical_and(isolated_component, ground_truth)
                                    ) / np.sum(np.logical_or(isolated_component, ground_truth)))
            mass_center_array.append([component, mass_center(
                isolated_component.astype(int))])
            filtered_mask += isolated_component.astype(np.uint8)*component
            component_list.append(component)
    if len(component_list) == 0:
        mass_center_array = None
        return len(component_list), filtered_mask, mass_center_array, np.asarray(iou_array)
    else:
        print((num_components))
        return len(component_list), filtered_mask, np.asarray(mass_center_array, dtype=object), np.asarray(iou_array)


def cluster_mass_center(mask, labels):
    if labels.shape[0] == 0:
        return np.nan
    mass_center_array = []
    for label in labels:
        cluster_array = (mask == label)
        mass_center_array.append(
            utils_cluster.mass_center(cluster_array.astype(float)))
    return (np.asarray(mass_center_array))


def labeled_img_to_rgb(mask, labels):
    if labels.shape[0] == 0:
        return utils_cluster.grayscale_to_rgb((np.zeros_like(mask)).astype(float))
    cluster_hue = np.linspace(0, 255, labels.shape[0]+1)
    cluster_array_list = []
    for label in labels:
        cluster_array = (mask == label)
        cluster_array_list.append(cluster_array)
    grayscale_img = (np.zeros_like(mask)).astype(float)
    for c in np.arange(len(cluster_array_list)):
        grayscale_img += (cluster_array_list[c] * cluster_hue[c+1])
    return utils_cluster.grayscale_to_rgb(grayscale_img)


def relabel_centers(clustered_centers, labeled_img):
    new_clustered_image = np.zeros_like(labeled_img, dtype=np.int64)
    for clustered_center in clustered_centers:
        original_label = clustered_center[0]
        isolated_component = (labeled_img == original_label)
        new_clustered_image += np.where(isolated_component,
                                        clustered_center[2], 0)
    return new_clustered_image
    # retorna nueva imagen con labels clusterizados


if __name__ == "__main__":

    '''
        TODO GRAL: armar clustering sobre componentes, grilla de parametros y reporte con csv compatible con las otras scripts
            TODO 1 : Recomputar los centros sobre los nuevos labels.
            TODO 2 : Integrar sobre logica de calculo de distancia anterior
            TODO 3 : Hacer un per sample report no un TP + los que sobran no los queremos ver

    '''

    ground_truth_csv = pd.read_csv('../single_instance_dataset_wradius.csv')

    hdb = True

    route_csv = pd.read_csv(os.path.join(
        '..', 'cluster_route.csv'), header=None)
    base_images_path = route_csv.iloc[0, 0]
    print(base_images_path)

    model_validation_folder = os.path.split(base_images_path)[0]
    model_name = os.path.split(model_validation_folder)[1]
    print("Processing images in: " + base_images_path)
    img_list = sorted(os.listdir(base_images_path))
    metrics = {
        'model_name': [],
        'mask_name': [],
        'eps': [],
        'min_samples': [],
        'buds_predicted': [],
        'clusters': [],
        'component_x': [],
        'component_y': [],
        'component_iou': [],
        'cluster_label': [],
        'cluster_x': [],
        'cluster_y': [],
        'pixel_distance': [],
        'norm_distance': [],
        'TP': [],
        'FP': [],
        'FN': []
    }
    # if not os.path.exists(os.path.join(model_validation_folder, 'clustered_masks')):
    #   os.makedirs(os.path.join(model_validation_folder, 'clustered_masks'))

    min_samples = 1
    eps_range = [1, 150, 300, 500]

    if hdb:
        eps_range = [1]

    for eps in eps_range:
        for img in img_list:
            print('Processing :' + img)
            # cluster image and get a labeled image where each pixel has a label value and a number of labels (ignoring 0 and -1)
            #image_data = utils_cluster.preprocess_image(utils_cluster.read_image_grayscale(os.path.join(base_images_path, img)))
            image_data = (utils_cluster.read_image_grayscale(
                os.path.join(base_images_path, img)))
            # LOAD GROUND TRUTH MASK
            ground_truth = (utils_cluster.read_image_grayscale(
                os.path.join('/home/wences/Documents/temp/dataset_resize/masks_resize/', 'mask_'+img[3:-3]+'png')))
            ground_truth = cv2.resize(ground_truth, (0, 0), fx=0.5, fy=0.5)

            #image_data = utils_cluster.filter_out_background_px(image_data)
            num_labels, labeled_img, centers, iou_array = connected_components_with_threshold(
                (image_data > 100).astype(np.uint8), 0, ground_truth)
            if centers is not None:
                if centers.shape[0] > 1:
                    center_array = []
                    for c in centers:
                        center_array.append(c[1])
                    center_array = np.asarray(center_array)
                    if hdb:
                        labeled_centers, num_clusters = hdbscan_clustering(
                            center_array)
                    else:
                        labeled_centers, num_clusters = dbscan(
                            center_array, eps, min_samples)

                    num_clusters = np.unique(labeled_centers)
                    # coloca en la ultima columna el label de cada centro
                    clustered_centers = np.append(centers, np.reshape(
                        labeled_centers, (centers.shape[0], 1)), axis=1)
                    # coloca en la ultima columna el iou
                    clustered_centers = np.append(clustered_centers, np.reshape(
                        iou_array, (centers.shape[0], 1)), axis=1)
                    clustered_image = relabel_centers(
                        clustered_centers, labeled_img)
                    new_centers = []
                    # recomputa los centros
                    for cluster_label in num_clusters:
                        new_centers.append(mass_center(
                            (clustered_image == cluster_label).astype(np.uint8)))
                    new_centers = np.asarray(new_centers)
                    new_centers = np.append(new_centers, np.reshape(
                        num_clusters, (len(num_clusters), 1)), axis=1)
                    # en new_centers tendremos cada centro con su lable de cluster, el label de cluster se usa como foreign key para matchear entre arreglos

            #utils_cluster.save_image(labeled_img_to_rgb(labeled_img, num_labels), os.path.join(model_validation_folder,'clustered_masks'), 'cluster_'+img)

            # get array of bidimensional center arrays [xcoord, ycoord]
        #     print(centers)
            # check if there have been buds detected and process them

            if centers is not None:
                row = utils_cluster.get_sample_ground_truth(
                    img, ground_truth_csv)
                gt_center = np.ndarray([1, 2])
                gt_center[0, 0] = (row['x_center_resize'].values[0])/2
                gt_center[0, 1] = (row['y_center_resize'].values[0])/2
                diam_resize = (row['diam_resize'].values[0])/2
                if len(centers) > 1:  # si hay mas de un componente pueden haber clusters
                    for center in clustered_centers:
                        metrics['model_name'].append(model_name)
                        metrics['mask_name'].append(img)
                        metrics['eps'].append(eps)
                        metrics['min_samples'].append(min_samples)
                        metrics['buds_predicted'].append(
                            clustered_centers.shape[0])
                        metrics['clusters'].append(new_centers.shape[0])
                        metrics['component_x'].append(center[1][0])
                        metrics['component_y'].append(center[1][1])
                        cluster_label = center[2]
                        metrics['cluster_label'].append(cluster_label)
                        xy_cluster = new_centers[np.where(
                            new_centers[:, 2] == float(cluster_label))]
                        metrics['cluster_x'].append(xy_cluster[0, 0])
                        metrics['cluster_y'].append(xy_cluster[0, 1])
                        pixel_distance = np.linalg.norm(
                            np.subtract(gt_center, xy_cluster[0, 0:2]))
                        metrics['pixel_distance'].append(pixel_distance)
                        metrics['norm_distance'].append(
                            pixel_distance/diam_resize)
                        metrics['component_iou'].append(center[3])
                        if(center[3] > 0):
                            metrics['TP'].append(1)
                            metrics['FP'].append(0)
                        else:
                            metrics['TP'].append(0)
                            metrics['FP'].append(1)
                        metrics['FN'].append(0)

                else:  # tengo solo un componente en la mascara resultante
                    for c in range(centers.shape[0]):
                        metrics['model_name'].append(model_name)
                        metrics['mask_name'].append(img)
                        metrics['eps'].append(eps)
                        metrics['min_samples'].append(min_samples)
                        metrics['buds_predicted'].append(0)
                        metrics['clusters'].append(0)
                        metrics['component_x'].append((centers[c])[1][0])
                        metrics['component_y'].append((centers[c])[1][1])
                        metrics['cluster_label'].append(1)
                        metrics['cluster_x'].append((centers[c])[1][0])
                        metrics['cluster_y'].append((centers[c])[1][1])
                        pixel_distance = np.linalg.norm(
                            np.subtract(gt_center, (centers[c])[1]))
                        metrics['pixel_distance'].append(pixel_distance)
                        metrics['norm_distance'].append(
                            pixel_distance/diam_resize)
                        metrics['component_iou'].append(iou_array[c])
                        if(iou_array[c] > 0):
                            metrics['TP'].append(1)
                            metrics['FP'].append(0)
                        else:
                            metrics['TP'].append(0)
                            metrics['FP'].append(1)
                        metrics['FN'].append(0)
            else:  # no buds detected register it in the metrics dict
                metrics['model_name'].append(model_name)
                metrics['mask_name'].append(img)
                metrics['eps'].append(eps)
                metrics['min_samples'].append(min_samples)
                metrics['buds_predicted'].append(0)
                metrics['clusters'].append(0)
                metrics['component_x'].append(np.nan)
                metrics['component_y'].append(np.nan)
                metrics['cluster_label'].append(0)
                metrics['cluster_x'].append(np.nan)
                metrics['cluster_y'].append(np.nan)
                metrics['pixel_distance'].append(np.nan)
                metrics['norm_distance'].append(np.nan)
                metrics['component_iou'].append(0)
                metrics['TP'].append(0)
                metrics['FP'].append(0)
                metrics['FN'].append(1)
    data = pd.DataFrame(metrics)
    data.to_csv(os.path.join(model_validation_folder,
                             'clustering_validation'+model_name+'.csv'))
