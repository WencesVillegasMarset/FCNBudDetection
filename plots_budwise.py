import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re 

def generate_plots(csv_path):

    validation_data = pd.read_csv(csv_path)
    ground_truth = pd.read_csv('single_instance_dataset_wradius.csv')
    threshold_list = validation_data['threshold'].unique()
    ########################

    mean_norm_dist = []
    acum = []
    error_norm_dist = []
    for i in np.arange(0,len(threshold_list)):
        acum = []
        th_slice = validation_data.loc[validation_data['threshold'] == threshold_list[i]]
        for j in np.arange(th_slice.shape[0]):
            if (np.isnan(th_slice['euclidean_distance'].values[j])):
                continue
            else:
                row = ground_truth.loc[ground_truth['imageOrigin'] == th_slice['sample'].values[j]]
                acum.append(th_slice['euclidean_distance'].values[j] / ((row['diam_resize'].values[0]/2)))
        mean_norm_dist.append(np.mean(acum))
        error_norm_dist.append(np.std(acum))
    # for i in np.arange(0,len(threshold_list)):
    #     yerr.append(th_list[i]['euclidean_distance'].std())
    plt.errorbar(threshold_list, mean_norm_dist,yerr=error_norm_dist, linestyle='None', marker='.', ecolor='orange',markersize=10)
    plt.xticks(threshold_list)
    plt.xlabel('Threshold')
    plt.ylabel('Euclidean Distance Normalized by Bud Diameter')
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig('errorbar_norm_dist_by_threshold.png')
    ###########
    for i in np.arange(0,len(threshold_list)):
        acum = []
        th_slice = validation_data.loc[validation_data['threshold'] == threshold_list[i]]
        for j in np.arange(th_slice.shape[0]):
                if (np.isnan(th_slice['euclidean_distance'].values[j])):
                    acum.append(9)
                else:
                    row = ground_truth.loc[ground_truth['imageOrigin'] == th_slice['sample'].values[j]]
                    acum.append(th_slice['euclidean_distance'].values[j] / ((row['diam_resize'].values[0]/2)))
    plt.scatter(x=np.arange(0,len(acum)),y=acum)
    plt.yticks(np.arange(0,12,0.3))
    plt.ylabel('normalized euclidean distance between mass centers')
    plt.title('Normalized Distance at ' + str((i+1)/10) + ' threshold')
    plt.xlabel('Samples')
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig('scatter_norm_dist_' + str((i+1)/10) + '.png')