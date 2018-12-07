import numpy as np
import pandas as pd
import tensorflow as tf
import random as rn

from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LambdaCallback
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from segmentation_models import Unet, Linknet
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm_notebook
import os


# from albumentations import (
#     RandomSizedCrop,
#     VerticalFlip,
#     HorizontalFlip,
#     Transpose,
#     RandomRotate90,
#     OneOf,
#     ElasticTransform,
#     GridDistortion,
#     OpticalDistortion,
#     RandomContrast,
#     RandomBrightness,
#     RandomGamma,
#     Compose
# )

from telepyth import TelepythClient
tp = TelepythClient('6866613901398991037')


def size_up(x):
    return np.pad(x[:-1,:-1], 14, 'reflect')


def size_down(x):
    return x[14:115, 14:115]


def data_reader(fold):
    data = {}
    for name in ['X_train', 'X_valid', 'y_train', 'y_valid']:
        data[name] = np.load('data/{}_fold_{}.npy'.format(name, fold))
    data['X_test'] = np.load('data/X_test.npy')
    return data


def find_tresh(y_real, y_preds):
    thresholds = np.linspace(0.3, 0.7, 41)
    ious = np.array([get_iou_vector(y_real, y_preds > thresh) for thresh in thresholds])
    return thresholds[np.argmax(ious)], max(ious)
    
    
def TTA(estimator, X):
        return (estimator.predict(X) + np.flip(estimator.predict(np.flip(X, axis=0)), axis=0) + np.flip(estimator.predict(np.flip(X, axis=1)), axis=1))/3
    

def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch]>0, B[batch]>0
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10 )/ (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))
        
    return np.mean(metric)


def my_iou_metric(label, pred):
    return tf.py_func(get_iou_vector, [label, pred > 0.5], tf.float64)


class Segmentation():
    def __init__(self, model_name, backbone_name, fold, loss='binary_crossentropy', optimizer='rmsprop'):
        self.model_name = model_name
        self.backbone_name = backbone_name
        self.fold = fold
        self.loss = loss
        self.optimizer = optimizer
        self.model = eval(model_name)(backbone_name=self.backbone_name, encoder_weights='imagenet')
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[my_iou_metric])
    
    def fit(self, data, load_name, batch_size=64, epochs=1000000, augmentation=True, verbose=1):
        self.data = data
        self.load_name = load_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.augmentation = augmentation
        self.verbose = verbose
        
        X_train = data['X_train']
        y_train = data['y_train']
        
        X_valid = data['X_valid']
        y_valid = data['y_valid']
        
        y_valid_real = np.array([size_down(x) for x in y_valid])
        
        self.train_size = len(X_train)
        self.steps_per_epoch = int(self.train_size / self.batch_size)
        
        early_stopping = EarlyStopping(monitor='val_my_iou_metric', mode='max', patience=400, verbose=self.verbose)
        model_checkpoint = ModelCheckpoint(self.load_name, monitor='val_my_iou_metric', mode='max', save_best_only=True, verbose=self.verbose)
        reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric', mode='max', factor=0.5, patience=150, verbose=1)
        
        def generate_report(self, epoch, logs):
            if epoch % 500 == 0:
                tp.send_text('{} {} fold = {}\nepoch = {} lr = {}\nscore = {}'.format(self.model_name, 
                                                                                      self.backbone_name,
                                                                                      self.fold,
                                                                                      epoch, logs['lr'],
                                                                                      logs['val_my_iou_metric']))
            return True
        
        my_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: generate_report(self, epoch, logs))
        
        callbacks=[early_stopping, model_checkpoint, reduce_lr, my_callback]
                
        generator = self.augmentation_generator(X_train, y_train, self.batch_size)
        
        history = self.model.fit_generator(generator,
                                  steps_per_epoch=self.steps_per_epoch,
                                  validation_data=[X_valid, y_valid],
                                  epochs=self.epochs,
                                  callbacks=callbacks,
                                  verbose=self.verbose)
        if self.loss == lovasz_loss:
            self.model = load_model(self.load_name, custom_objects={'my_iou_metric': my_iou_metric, 'lovasz_loss':lovasz_loss})
        else:
            self.model = load_model(self.load_name, custom_objects={'my_iou_metric': my_iou_metric})
        
        preds_val = self.predict_proba(X_valid)
        best_thres, max_iou = find_tresh(y_valid_real, preds_val)
        self.best_thres = best_thres
        self.max_iou = max_iou
    
    
    def predict_proba(self, X):
        preds = TTA(self.model, X)
        preds = np.array([size_down(x) for x in preds])
        return preds
    
    
    def predict(self, X):
        preds = self.predict_proba(X)
        return np.int8(preds > self.best_thres)   
        

    def augmentation_generator(self, X, y, batch_size):
        data_gen_args = dict(rotation_range=10,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2,
                         fill_mode='reflect',
                         horizontal_flip=True,
                         vertical_flip=True
                         )
        
        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)
        image_datagen.fit(X, augment=True, seed=42)
        mask_datagen.fit(y, augment=True, seed=42)

        im_gen = image_datagen.flow(X, batch_size=batch_size, seed=42)
        mask_gen = mask_datagen.flow(y, batch_size=batch_size, seed=42)
        
        return zip(im_gen, mask_gen)
    

    
def rle_encode(im):
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def create_submit(X):
    tmp_list = []
    for x in X:
        tmp_list.append(rle_encode(x))

    name_list = [(Path('test/images/')/f).name[0:-4] for f in os.listdir('input/test/images/')]
    sub = pd.DataFrame(list(zip(name_list, tmp_list)), columns = ['id', 'rle_mask'])
    sub.to_csv('submission.csv', index=False)