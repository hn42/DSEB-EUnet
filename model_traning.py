
import numpy as np
from tensorflow.python.keras.callbacks import ReduceLROnPlateau

import outher_Value_and_func as dg
import os
np.random.randint(0 ,25)
os.environ['SM_FRAMEWORK'] = 'tf.keras'
from keras.optimizers import *
import cv2
from datetime import datetime
from os import listdir
from os.path import isfile, join
image_dim = (192, 192)
from tensorflow import keras

def DAGM_org(names, Hatasiz, esit):
    path_defect_name="addPAth"
    if names==None:
        names=[]
        for i in range(1,11):
            names.append('Class'+str(i))
    global image_dim
    list_image = []
    list_mask = []
    traint_index = []
    test_index = []
    index_sayac=0
    sayac_hatali=0
    sayac_hatasiz=0
    for alt_name in ['Train','Test']:
        for name in names:
            path_defect=path_defect_name+name+'/'+alt_name
            onlyfiles = [f for f in listdir(path_defect) if isfile(join(path_defect, f))]

            # print(onlyfiles)
            for img_p in onlyfiles:
                # if '.PNG' in img_p:
                #     continue

                path_img = path_defect + "/" + img_p
                img = cv2.imread(path_img)

                path_img = path_defect + "/Label/" + img_p.split('.')[0] + '_label.PNG'
                mask= cv2.imread(path_img, 0)
                if img is None:
                    continue

                if Hatasiz and mask is None:
                    continue

                if mask is None and esit and sayac_hatali<=sayac_hatasiz:
                    continue
                if mask is None:
                    mask = np.zeros(img.shape[0:2], np.uint8)
                    # print("hatasız goruntu -->",img_p)
                    sayac_hatasiz+=1
                else:
                    sayac_hatali+=1

                img = cv2.resize(img, image_dim, interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, image_dim, interpolation=cv2.INTER_AREA)
                # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


                print(mask.max())
                mask=mask>155


                list_image.append(img)
                list_mask.append(mask)
                if alt_name=='Train':
                    traint_index.append(index_sayac)
                else:
                    test_index.append(index_sayac)
                index_sayac+=1
                # dataGenerator_GetValue(img,mask)
        # print(sayac_____________)
    images_img = converNumpy(list_image)
    images_mask = converNumpy(list_mask)


    return images_img,images_mask,traint_index,test_index
def converNumpy(list):
    try:
        array = np.zeros((len(list), list[0].shape[0], list[0].shape[1], list[0].shape[2]), dtype=np.float32)
        for i in range(len(list)):
            array[i, :, :, :] = list[i]
    except:
        array = np.zeros((len(list), list[0].shape[0], list[0].shape[1], 1), dtype=np.float32)
        for i in range(len(list)):
            array[i, :, :, 0] = list[i]
    return array


from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score

from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
def result_treshold_05_jaccard_score_f1score(sonuc, y_test, pred_test):
    try:
        y_pred = pred_test[:, :, :].ravel()
        y_true = ( y_test[:, :, :].ravel() >= 0.5) * 1
        y_pred_binary = (y_pred >= 0.5) * 1
        # jaccard_score(y_true, y_pred_binary) * 10000 // 1

        AP=average_precision_score(y_true, y_pred)* 10000 // 1
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        # for idx in range(70):
        #     img=y_test[idx, :, :, 1]
        #     if img.sum()>10:
        #         print(img.sum())
        #         dg.visualize(y_test[idx, :, :, 1], pred_test[idx, :, :, 1])
        # dg.visualize(y_test[4, :, :, 1], pred_test[1, :, :, 1])

        f1_s = f1_score(y_true, y_pred_binary) * 10000 // 1
        iou = jaccard_score(y_true, y_pred_binary) * 10000 // 1
    except:
        iou=0
        f1_s=0
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    # tpr_sensiti=tp/(tp+fp)*10000//1
    # tnr=tn/(tn+fn)*10000//1

    sonuc.append('my_iou_score')
    sonuc.append(str(iou))
    sonuc.append('my_f1-score')
    sonuc.append(str(f1_s))
    sonuc.append('my_roc_auc')
    sonuc.append(str(roc_auc))
    sonuc.append('AP_'+str(AP))

    return sonuc

def GetModel(model_idx, input_shape, classes, activation):
    import DSEB_EUnet as dseb_eunet
    if model_idx == 13:
        model = dseb_eunet.DSEB_EUNET_MODEL(input_shape=(input_shape),
                                    classes=classes, activation=activation,
                                     type='DSEB_mout_3x3')
        modelName = 'DSEB_EUNET_3x3_mout'
    if model_idx == 15:
        model = dseb_eunet.DSEB_EUNET_MODEL(input_shape=(input_shape),
                                    classes=classes, activation=activation, type='DSEB_5x5_mout')
        modelName = 'DSEB_EUNET_5x5_mout'
    if model_idx == 17:
        model = dseb_eunet.DSEB_EUNET_MODEL(input_shape=(input_shape),
                                    classes=classes, activation=activation, type='DSEB_7x7_mout')
        modelName = 'DSEB_EUNET_7x7_mout'
    return model, modelName


batch_size = 8
Dataset = 'DAGM_class_your_123456'
names = []
for i in [1, 2, 3, 4, 5, 6]:
    names.append('Class' + str(i))
# names = []
# for i in [1]:
#     names.append('Class' + str(i))
images_img, images_mask, traint_index, test_index = DAGM_org(names, False, False)

train, test= traint_index, test_index

X_train, X_test, y_train, y_test = images_img[train], images_img[test], images_mask[train], images_mask[test]


print('+++++++++++++++++++++++++++++++++++++++++++++++++')
print('X_train   Shape=', X_train.shape)
print('y_train   Shape=', y_train.shape)

print('X_test   Shape=', X_test.shape)
print('y_test   Shape=', y_test.shape)
print('+++++++++++++++++++++++++++++++++++++++++++++++++')
# ]

X_train = X_train.astype(np.float32) / 255
y_train = y_train.astype(np.float32)
X_test = X_test.astype(np.float32) / 255
y_test = y_test.astype(np.float32)
import segmentation_models as sm

# model_idx_listeeee

pred_test_list = []
pred_test_id = []
# continue
model_idx=15
keras.backend.clear_session()

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

if y_train.shape[-1]==1:
    classes = 1
    activation='sigmoid'
else:
    classes = 2
    activation='softmax'

img_row, img_colum, img_channels = X_train.shape[1], X_train.shape[2], X_train.shape[3]
input_shape = (img_row, img_colum, img_channels)

model = None
# tf.tpu.experimental.initialize_tpu_system(tpu)
keras.backend.clear_session()
tf.keras.backend.clear_session()


model, modelName = GetModel(model_idx, input_shape, classes, activation)
model.summary()

lr = 0.001
reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
                                         factor=0.1,
                                         patience=5,
                                         verbose=1,
                                         epsilon=0.001,
                                         cooldown=0,
                                         min_lr=0.000001)

print("---------> lr:", lr)
model.compile(optimizer=Adam(lr=lr), loss=sm.losses.binary_crossentropy,
              metrics=[sm.metrics.iou_score, sm.metrics.f1_score])


print('-----------------------------------------------------------------------------------------------------------------------')
callback, logdir=dg.getColback('Dnm_' + modelName +'_' + str(model_idx) + '_' + Dataset, "./", model)

print('tensorboard --logdir=\''+logdir+'\'')

now = datetime.now()
current_time_bas = now.strftime("%d %m %H:%M:%S.%f")

autoencoder_train = model.fit(X_train, y_train,batch_size=batch_size,callbacks=callback
                                ,epochs=100,verbose=2)

now = datetime.now()
current_time_son = now.strftime("%H:%M:%S.%f")

now = datetime.now()
current_time_bas_evaluate= now.strftime("%H:%M:%S.%f")
TestSonuc = model.evaluate(X_test, y_test,
                           batch_size=batch_size, verbose=2)

now = datetime.now()
current_time_son_evaluate = now.strftime("%H:%M:%S.%f")
# sonuc=[]
sonuc=[modelName+'_'+'/'+str(model.optimizer.lr.numpy())]

sonuc.append(Dataset)
pred_test= model.predict(X_test, batch_size=batch_size)
sonuc = result_treshold_05_jaccard_score_f1score(sonuc, y_test, pred_test)

print(sonuc)
print(sonuc)





