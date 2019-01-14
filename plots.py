import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re 

def generate_plots(**kwargs):

    csv_list = pd.read_csv(kwargs['csv_path'])
    csv_list = csv_list.iloc[:,0].values
    frames = []
    for csv in csv_list:
        frames.append(pd.read_csv('./csv/'+csv))

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

