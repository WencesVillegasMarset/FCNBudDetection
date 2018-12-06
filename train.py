import pandas as pd
import os
import models
from utils import DataGeneratorMobileNet

def train_model(**kwargs):
    '''
        Model Training Script
        kwargs:
            'model' = Codename for model (8, 16 or 32) (int)
            'batch size' = batch size to be used in training (int)
            'train_encoder': wether or not train the pretrained encoder layers of the net (bool) 
            'img_path': path to the images (X) folder (string)
            'masks_path': path to the masks (Y) folder (string)
            'partition': dictionary with two lists one with training samples and one with validation samples (dict)
            'labels': dictionary key:value where img_name:mask_name (dict)
            'optimizer': name of the optimizer to use (sgd, adam, rmsprop or nadam) (string)
            'lr': learning rate (float)
            'momentum': momentum (float)
            'decay': lr decay (float)
            'epochs': number of epochs (int)
            'models_folder': target folder and filename for h5 model file (string)
            'history_folder': target folder and filename for csv history file (string)
            'final_layer': name of the final layer activation, linear or sigmoid (str)
    '''

    if kwargs['model'] == 8:
        model = models.mobilenet_8s(train_encoder=kwargs['train_encoder'],
            final_layer_activation=kwargs['final_layer'])
    elif kwargs['model'] == 16:
        model = models.mobilenet_16s(train_encoder=kwargs['train_encoder'],
            final_layer_activation=kwargs['final_layer'])
    elif kwargs['model'] == 32:
        model = models.mobilenet_32s(train_encoder=kwargs['train_encoder'],
            final_layer_activation=kwargs['final_layer'])

    train_generator = DataGeneratorMobileNet(batch_size=kwargs['batch_size'],img_path=kwargs['img_path'],
                                labels=kwargs['labels'],list_IDs=kwargs['partition']['train'],n_channels=3,
                                n_channels_label=1,shuffle=True,mask_path=kwargs['masks_path'])
    valid_generator = DataGeneratorMobileNet(batch_size=kwargs['batch_size'],img_path=kwargs['img_path'],
                                labels=kwargs['labels'],list_IDs=kwargs['partition']['valid'],n_channels=3,
                                n_channels_label=1,shuffle=True,mask_path=kwargs['masks_path'])
    
    if kwargs['optimizer'] == 'sgd':
        from keras.optimizers import SGD
        optimizer = SGD(lr=kwargs['lr'], momentum=kwargs['momentum'], decay=kwargs['decay'])
    elif kwargs['optimizer'] == 'rmsprop':
        from keras.optimizers import RMSprop
        optimizer = RMSprop()
    elif kwargs['optimizer'] == 'adam':
        from keras.optimizers import Adam
        optimizer = Adam()    

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    train_history = model.fit_generator(generator=train_generator,validation_data=valid_generator,
                            use_multiprocessing=True,workers=6, epochs=kwargs['epochs'])
    model_name = str(kwargs['fold'])+'FCMN'+str(kwargs['model'])+kwargs['optimizer']+'_lr'+str(kwargs['lr'])+'_decay'+str(kwargs['decay'])+'_ep'+str(kwargs['epochs'])

    model.save(os.path.join(kwargs['models_folder'], model_name + '.h5'))

    history_csv = pd.DataFrame(train_history.history)
    history_csv.to_csv(os.path.join(kwargs['history_folder'], model_name +'.csv'))
    return model_name


    
        
        