import os
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import train

if __name__ == "__main__":
    model_list = [8,16,32]
    lr_list = [0.0001]#[0.001,0.0001,0.00001]
    batch_size_list = [4]
    optimizer_list = ['rmsprop','adam']
    lr_decay_list = [0] #['0', '0.0005'] 
    preprocessing_list = [True, False]
    epoch_list = [150,110,75]
    #load dataset csv
    train_set_full = pd.read_csv(os.path.join('/home','wvillegas','dataset-mask', 'single_instance_train.csv'))
    train_set_array = train_set_full['imageOrigin'].values
#     split it into 4 folds
    kf = KFold(n_splits = 4, shuffle=False)
    train_indexes = []
    test_indexes = []
    for train_index, test_index in kf.split(train_set_array):
        train_indexes.append(train_index)
        test_indexes.append(test_index)
    #save indexes arrays
    np.save('./train_indexes.npy', np.asarray(train_indexes))
    np.save('./test_indexes.npy', np.asarray(test_indexes))


    labels = dict(zip(list(train_set_full['imageOrigin'].values), list(train_set_full['mask'].values)))

    models = []
    for model in model_list:
        for fold in np.arange(0,len(train_indexes)):
            partition = {'train':list(train_set_array[train_indexes[fold]]),
                'valid': list(train_set_array[test_indexes[fold]])}
#                             partition = {'train':list(train_set_full['imageOrigin'].values),
#                                            'valid':[] }
            for bs in batch_size_list:
                for decay in lr_decay_list:
                    for lr in lr_list:
                        for epoch in epoch_list:
                            for optimizer in optimizer_list:
                                for prep in preprocessing_list:
                                        args = {
                                            'fold': fold,
                                            'model':model,
                                            'batch_size':bs,
                                            'train_encoder':True,
                                            'img_path':os.path.join('/home','wvillegas','dataset-mask','dataset_resize', 'images_resize'),
                                            'masks_path':os.path.join('/home','wvillegas','dataset-mask','dataset_resize', 'masks_resize'),
                                            'partition':partition,
                                            'labels':labels,
                                            'optimizer':optimizer,
                                            'lr':lr,
                                            'momentum':0.9,
                                            'decay':decay,
                                            'epochs':epoch,
                                            'models_folder':os.path.join('/home','wvillegas','DLProjects','DetectionModels', 'scriptmodels'),
                                            'history_folder':os.path.join('/home','wvillegas','DLProjects','DetectionModels', 'trainhist'),
                                            'final_layer':'sigmoid',
                                            'preprocessing':prep
                                        }
                                        train.train_model(**args)