import os
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import train, validation 

if __name__ == "__main__":
    model_list = [8,16,32]
    lr_list = [0.0001]#[0.001,0.0001,0.00001]
    batch_size_list = [4]
    optimizer_list = ['rmsprop']
    lr_decay_list = [0] #['0', '0.0005'] 
    preprocessing_list = [True]
    epoch_list = [150]
    #load dataset csv
    train_set_full = pd.read_csv(os.path.join('/home','wvillegas','dataset-mask', 'single_instance_train.csv'))
    train_set_array = train_set_full['imageOrigin'].values
    test_set = pd.read_csv(os.path.join('/home','wvillegas','dataset-mask', 'single_instance_test.csv'))
#     split it into 4 folds
#     kf = KFold(n_splits = 4, shuffle=False)
#     train_indexes = []
#     test_indexes = []
#     for train_index, test_index in kf.split(train_set_array):
#         train_indexes.append(train_index)
#         test_indexes.append(test_index)
#     #save indexes arrays
#     np.save('./train_indexes.npy', np.asarray(train_indexes))
#     np.save('./test_indexes.npy', np.asarray(test_indexes))
    #create output directories
    out_models = os.path.join('.','output', 'models')
    out_history = os.path.join('.','output', 'history')
    out_validation = os.path.join('.','output', 'validation')
    if not os.path.exists(out_models):
        os.makedirs(out_models)
    if not os.path.exists(out_history):
        os.makedirs(out_history) 
    if not os.path.exists(out_validation):
        os.makedirs(out_validation)
    
    dataset_full = pd.read_csv(os.path.join('/home','wvillegas','dataset-mask', 'single_instance_corpus.csv'))
    labels = dict(zip(list(dataset_full['imageOrigin'].values), list(dataset_full['mask'].values)))

    models = []
    for model in model_list:
#         for fold in np.arange(0,len(train_indexes)):
#             partition = {'train':list(train_set_array[train_indexes[fold]]),
#                 'valid': list(train_set_array[test_indexes[fold]])}
            partition = {'train':list(train_set_full['imageOrigin'].values),
                           'valid':list(test_set['imageOrigin'].values) }
            for bs in batch_size_list:
                for decay in lr_decay_list:
                    for lr in lr_list:
                        for epoch in epoch_list:
                            for optimizer in optimizer_list:
                                for prep in preprocessing_list:
                                        args = {
                                            'fold': 0,
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
                                            'models_folder':out_models,
                                            'history_folder':out_history,
                                            'final_layer':'sigmoid',
                                            'preprocessing':prep
                                        }
                                        args['model_name'] = train.train_model(**args)
                                        args['validation_folder'] = out_validation
                                        
                                        args['csv_path'] = validation.validate(**args)