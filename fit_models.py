import os
import sys

CUDA_N = int(sys.argv[1])

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(CUDA_N)

import numpy as np
import tensorflow as tf
import random as rn

np.random.seed(42)
rn.seed(42)
tf.set_random_seed(42)

from utils import *

model_name_array = ['Unet', 'Linknet']
backbone_name_array = ['resnet18', 'resnet34', 'densenet121', 'densenet169', 
                       'inceptionv3', 'inceptionresnetv2']
backbone_name = backbone_name_array[CUDA_N]

for fold in [0,1,2,3]:
    data = data_reader(fold)
    for model_name in model_name_array:
        load_name = 'models/{}_{}_fold_{}.model'.format(model_name, backbone_name, fold)
        model = Segmentation(model_name, backbone_name, fold)
        model.fit(data, load_name, verbose=1)

        preds_val = model.predict_proba(data['X_valid'])
        preds_test = model.predict_proba(data['X_test'])

        np.save('prediction/val_{}_{}_fold_{}.npy'.format(model_name, backbone_name, fold), preds_val)
        np.save('prediction/test_{}_{}_fold_{}.npy'.format(model_name, backbone_name, fold), preds_test)
        np.save('results/{}_{}_fold_{}.npy'.format(model_name, backbone_name, fold), np.array([round(model.max_iou, 5), round(model.best_thres, 2)]))

        tp.send_text('\n.\n.\n {} {} fold = {}\nscore = {} tresh={}\n.\n.'.format(model_name, backbone_name, fold, round(model.max_iou, 5), round(model.best_thres, 2)))
