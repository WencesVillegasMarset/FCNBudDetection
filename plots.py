import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re 

def generate_plots(**kwargs):

    csv_list = pd.read_csv('list.csv')
    csv_list = csv_list.iloc[:,0].values
    frames = []
    for csv in csv_list:
        frames.append(pd.read_csv(csv))

    full_validation = pd.concat(frames)
    threshold_list = full_validation['threshold'].unique()

    fcn8 = full_validation.loc[full_validation.model_name.str.contains('FCMN8'),:]
    fcn16 = full_validation.loc[full_validation.model_name.str.contains('FCMN16'),:]
    fcn32 = full_validation.loc[full_validation.model_name.str.contains('FCMN32'),:]

'''
    Lista de Plots a Generar:
        PixelWise Metrics:
            Barplots de Precision, Recall e IoU
                x: 3 arquitecturas a cada threshold
                y: valor de la metrica [0,1]
        BudWise Metrics:
            Distancia Euclideana entre centros de masa normalizada
                x: thresholds
                y: valor de distancia normalizado promedio y error promedio
            Una vez definido cuando se detecta una yema
            ROC Curve Precision X Recall:
                x: Recall
                y: Precision
                z: Threshold
'''
    iou_mean_8s = []
    iou_std_8s = []
    iou_mean_16s = []
    iou_std_16s = []
    iou_mean_32s = []
    iou_std_32s = []

    for th in threshold_list:
        th_slice = fcn8.loc[fcn8.threshold == th, :]
        iou_mean_8s.append(th_slice.iou.mean())
        iou_std_8s.append(th_slice.iou.std())
    for th in threshold_list:
        th_slice = fcn16.loc[fcn16.threshold == th, :]
        iou_mean_16s.append(th_slice.iou.mean())
        iou_std_16s.append(th_slice.iou.std())
    for th in threshold_list:
        th_slice = fcn32.loc[fcn32.threshold == th, :]
        iou_mean_32s.append(th_slice.iou.mean())
        iou_std_32s.append(th_slice.iou.std())
    ind = np.arange(len(threshold_list))*3  # the x locations for the groups
    width = 0.85  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(ind - width, iou_mean_8s, width, yerr=iou_std_8s,
                    color='#cecece', label='FCN8s')
    ax.bar(ind, iou_mean_16s, width, yerr=iou_std_16s,
                    color='#827f7f', label='FCN16s')
    ax.bar(ind + width, iou_mean_32s, width, yerr=iou_std_32s,
                    color='#000000', label='FCN32s')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('FCN8s - FCN16s - FCN32s')
    ax.set_ylabel('Mean IoU')
    ax.set_title('Average IoU for the three implemented architectures')
    ax.set_xticks(ind)
    ax.set_xticklabels(threshold_list)
    ax.legend()
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig('barplot_iou.png')
######################################
    prec_mean_8s = []
    prec_std_8s = []
    prec_mean_16s = []
    prec_std_16s = []
    prec_mean_32s = []
    prec_std_32s = []

    for th in threshold_list:
        prec_list = []
        th_slice = fcn8.loc[fcn8.threshold == th, :]

        for i in np.arange(th_slice.shape[0]):
            if(th_slice.segmentation_area.values[i] == 0):
                prec_list.append(0)
            else:
                prec_list.append(th_slice.intersection.values[i] / th_slice.segmentation_area.values[i])

        prec_mean_8s.append(np.mean(prec_list))
        prec_std_8s.append(np.std(prec_list))

    for th in threshold_list:
        prec_list = []
        th_slice = fcn16.loc[fcn16.threshold == th, :]

        for i in np.arange(th_slice.shape[0]):
            if(th_slice.segmentation_area.values[i] == 0):
                prec_list.append(0)
            else:
                prec_list.append(th_slice.intersection.values[i] / th_slice.segmentation_area.values[i])

        prec_mean_16s.append(np.mean(prec_list))
        prec_std_16s.append(np.std(prec_list))

    for th in threshold_list:
        prec_list = []
        th_slice = fcn32.loc[fcn32.threshold == th, :]

        for i in np.arange(th_slice.shape[0]):
            if(th_slice.segmentation_area.values[i] == 0):
                prec_list.append(0)
            else:
                prec_list.append(th_slice.intersection.values[i] / th_slice.segmentation_area.values[i])

        prec_mean_32s.append(np.mean(prec_list))
        prec_std_32s.append(np.std(prec_list))

    ind = np.arange(len(threshold_list))*3  # the x locations for the groups
    width = 0.85  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(ind - width, prec_mean_8s, width, yerr=prec_std_8s,
                    color='#cecece', label='FCN8s')
    ax.bar(ind, prec_mean_16s, width, yerr=prec_std_16s,
                    color='#827f7f', label='FCN16s')
    ax.bar(ind + width, prec_mean_32s, width, yerr=prec_std_32s,
                    color='#000000', label='FCN32s')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('FCN8s - FCN16s - FCN32s')
    ax.set_title('Average Precision for the three implemented architectures')
    ax.set_xticks(ind)
    ax.set_xticklabels(threshold_list)
    ax.legend()
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig('barplot_precision.png')
##################
    rec_mean_8s = []
    rec_std_8s = []
    rec_mean_16s = []
    rec_std_16s = []
    rec_mean_32s = []
    rec_std_32s = []

    for th in threshold_list:
        th_slice = fcn8.loc[fcn8.threshold == th, :]
        rec_mean_8s.append(np.mean(th_slice.intersection.values / th_slice.gt_area.values))
        rec_std_8s.append(np.std(th_slice.intersection.values / th_slice.gt_area.values))
    for th in threshold_list:
        th_slice = fcn16.loc[fcn16.threshold == th, :]
        rec_mean_16s.append(np.mean(th_slice.intersection.values / th_slice.gt_area.values))
        rec_std_16s.append(np.std(th_slice.intersection.values / th_slice.gt_area.values))
    for th in threshold_list:
        th_slice = fcn32.loc[fcn32.threshold == th, :]
        rec_mean_32s.append(np.mean(th_slice.intersection.values / th_slice.gt_area.values))
        rec_std_32s.append(np.std(th_slice.intersection.values / th_slice.gt_area.values))

    ind = np.arange(len(threshold_list))*3  # the x locations for the groups
    width = 0.85  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(ind - width, rec_mean_8s, width, yerr=rec_std_8s,
                    color='#cecece', label='FCN8s')
    ax.bar(ind, rec_mean_16s, width, yerr=rec_std_16s,
                    color='#827f7f', label='FCN16s')
    ax.bar(ind + width, rec_mean_32s, width, yerr=rec_std_32s,
                    color='#000000', label='FCN32s')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('FCN8s - FCN16s - FCN32s')
    ax.set_ylabel('Mean Recall')
    ax.set_title('Average Recall for the three implemented architectures')
    ax.set_xticks(ind)
    ax.set_xticklabels(threshold_list)
    ax.legend()
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig('barplot_recall.png')