
import  numpy as np

bu_modeli_bitir=False
epochs=150
model_kaydet=True
sonuc_goster=True
batchsizeList=[16]
# PYTHONUNBUFFERED=1;LD_LIBRARY_PATH="/usr/lib/cuda/include:/usr/lib/cuda/lib64:"

def DegiskenEkranaGoster():
    print("****************************")
    print("****************************")
    print("****************************")
    print(bu_modeli_bitir)
    print(sonuc_goster)
    print(batchsizeList)
    print("****************************")
    print("****************************")
    print("****************************")


def listToString(s):
    # s=list(s)
    # initialize an empty string
    str1 = ""
    # traverse in the string
    for ele in s:
        str1 += ' '+str(ele)
        # return string
    return str1

DegiskenEkranaGoster()
import os
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/cuda/include:/usr/lib/cuda/lib64:"
import pandas as pd
def sonucKaydet(katityeri,veri):
    try:
        df=pd.read_csv(katityeri, header=None)
        listSonuclar = df.values.tolist()
    except:
        listSonuclar=[]

    listSonuclar.append(veri)

    sonuclar = np.array(listSonuclar)
    np.savetxt(katityeri, sonuclar, delimiter=',', fmt='%s')


from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from keras.utils import plot_model
def getColback(name,path,model):
    # plot_model(model, to_file=path+name+  '_model_.png', show_shapes=True,
    #            show_layer_names=True)
    model_checkpoint = ModelCheckpoint(filepath=path+name+'-{epoch:02d}_loss-{loss:.4f}.h5',
                                       monitor='loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='auto',
                                       period=1)

    csv_logger = CSVLogger(filename=path+name+'_training_log.csv',
                           separator=',',
                           append=True)

    early_stopping = EarlyStopping(monitor='loss',
                                   min_delta=0.0,
                                   patience=10,
                                   verbose=1)

    #
    reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
                                             factor=0.2,
                                             patience=3,
                                             verbose=1,
                                             epsilon=0.001,
                                             cooldown=0,
                                             min_lr=0.000001)


    from datetime import datetime
    from tensorflow import keras
    logdir = path+name+'logs/'
    print(logdir)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir+"fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    callbacks = [
        # model_checkpoint,
        #          csv_logger,
                 early_stopping,
                 reduce_learning_rate
                # tensorboard_callback
    ]
    return callbacks ,logdir

# print(part_epoc())

import math
import albumentations as A
import random

from matplotlib import pyplot as plt
def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)

        #
        # fig_2 = plt.figure(figsize=(21, 21))
        # ax = fig_2.add_subplot(1, 3, 1)
        # ax.imshow(mask)
        # ax.axis('off')
        # ax.title.set_text('mask' + str(mask_idx))
        # ax = fig_2.add_subplot(1, 3, 2)
        # ax.imshow(dilation)
        # ax.title.set_text('dilation')
        # ax.axis('off')
        # ax = fig_2.add_subplot(1, 3, 3)
        # ax.imshow(y_train[mask_idx])
        # ax.axis('off')
        # plt.show()

def dataGenerator_GetValue_org(image, mask):
    original_height, original_width = image.shape[:2]

    oran=0.7
    min_max_height = (int(min(original_height * oran, original_width * oran)), min(original_height, original_width))

    alpha = original_height
    sigma = original_height * 0.15
    alpha_affine = original_height * 0.03


    ElasticTransform=-1
    GridDistortion=3
    OpticalDistortion=3
    RandomSizedCrop=3
    aug_medium_size=3
    aug_heavy_size=-1
    RandomRotate90=2

    images=[]
    masks=[]
    # https: // albumentations.ai / docs / examples / example_kaggle_salt /
    # min_height=2**(int(math.log(original_height,2))+1)
    # min_width=2**(int(math.log(original_height,2))+1)
    # min_height=64
    # min_width=64
    # aug = A.PadIfNeeded(min_height=min_height, min_width=min_width, p=1)
    # augmented = aug(image=image, mask=mask)
    # image_padded = augmented['image']
    # mask_padded = augmented['mask']



    aug = A.CenterCrop(p=1, height=original_height, width=original_width)
    augmented = aug(image=image, mask=mask)
    image_center_cropped = augmented['image']
    mask_center_cropped = augmented['mask']
    images.append(image_center_cropped)
    masks.append(mask_center_cropped)


    # x_min = (original_width)
    # y_min = (min_height - original_height) // 2
    # x_max = (x_min + original_width)
    # y_max = (y_min + original_height)
    # aug = A.Crop(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, p=1)
    # augmented = aug(image=image_padded, mask=mask_padded)
    # image_cropped = augmented['image']
    # mask_cropped = augmented['mask']
    # visualize(image_cropped, mask_cropped, original_image=image, original_mask=mask)



    aug = A.HorizontalFlip(p=1)
    augmented = aug(image=image, mask=mask)
    image_h_flipped = augmented['image']
    mask_h_flipped = augmented['mask']
    images.append(image_h_flipped)
    masks.append(mask_h_flipped)

    aug = A.VerticalFlip(p=1)
    augmented = aug(image=image, mask=mask)
    image_v_flipped = augmented['image']
    mask_v_flipped = augmented['mask']
    images.append(image_v_flipped)
    masks.append(mask_v_flipped)

    aug = A.RandomRotate90(p=1)
    for i in range(RandomRotate90):
        augmented = aug(image=image, mask=mask)
        image_rot90 = augmented['image']
        mask_rot90 = augmented['mask']
        images.append(image_rot90)
        masks.append(mask_rot90)

    aug = A.Transpose(p=1)
    augmented = aug(image=image, mask=mask)
    image_transposed = augmented['image']
    mask_transposed = augmented['mask']
    images.append(image_transposed)
    masks.append(mask_transposed)


    aug = A.ElasticTransform(p=1, alpha=alpha, sigma=sigma, alpha_affine=alpha_affine)
    for i in range(ElasticTransform):
        augmented = aug(image=image, mask=mask)
        image_elastic = augmented['image']
        mask_elastic = augmented['mask']
        images.append(image_elastic)
        masks.append(mask_elastic)


    # visualize(images[-1], images[-2], original_image=images[-3], original_mask=images[-4])
    # visualize(images[-1], masks[-1], original_image=images[-2], original_mask=masks[-2])

    aug = A.GridDistortion(p=1)
    for i in range(GridDistortion):
        augmented = aug(image=image, mask=mask)
        image_grid = augmented['image']
        mask_grid = augmented['mask']
        images.append(image_grid)
        masks.append(mask_grid)

    aug = A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1)
    for i in range(OpticalDistortion):
        augmented = aug(image=image, mask=mask)
        image_optical = augmented['image']
        mask_optical = augmented['mask']
        images.append(image_optical)
        masks.append(mask_optical)

    aug = A.RandomSizedCrop(min_max_height=min_max_height, height=original_height, width=original_width, p=1)
    for i in range(RandomSizedCrop):
        augmented = aug(image=image, mask=mask)
        image_scaled = augmented['image']
        mask_scaled = augmented['mask']
        images.append(image_scaled)
        masks.append(mask_scaled)

    aug = A.Compose([
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5)]
    )
    augmented = aug(image=image, mask=mask)
    image_light = augmented['image']
    mask_light = augmented['mask']
    images.append(image_light)
    masks.append(mask_light)

    aug_medium = A.Compose([
        A.OneOf([
            A.RandomSizedCrop(min_max_height=min_max_height, height=original_height, width=original_width, p=0.5),
            A.PadIfNeeded(min_height=original_height, min_width=original_width, p=0.5)
        ], p=1),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=alpha, sigma=sigma, alpha_affine=alpha_affine),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1),
        ], p=0.8)])

    for i in range(aug_medium_size):
        augmented = aug_medium(image=image, mask=mask)
        image_medium = augmented['image']
        mask_medium = augmented['mask']
        images.append(image_medium)
        masks.append(mask_medium)

    aug_heavy= A.Compose([
        A.OneOf([
            A.RandomSizedCrop(min_max_height=min_max_height, height=original_height, width=original_width, p=0.5),
            A.PadIfNeeded(min_height=original_height, min_width=original_width, p=0.5)
        ], p=1),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=alpha, sigma=sigma, alpha_affine=alpha_affine),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1)
        ], p=0.8),
        A.CLAHE(p=0.8),
        A.RandomBrightnessContrast(p=0.8),
        A.RandomGamma(p=0.8)])
    for i in range(aug_heavy_size):
        augmented = aug_heavy(image=image, mask=mask)
        image_heavy = augmented['image']
        mask_heavy = augmented['mask']
        images.append(image_heavy)
        masks.append(mask_heavy)


    # visualize(images[-1], masks[-1], original_image=image, original_mask=mask)

    return images, masks



def to_catacorial(mask):
    array = np.zeros((mask.shape[0], mask.shape[1], 2), dtype=np.uint8)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j]==0:
                array[i,j,0]=1
            else:
                array[i,j,1]=1

    return  array




def catogarial_toimage(mask):
    if mask.shape[2]==1:
        return mask
    array = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j,0]<mask[i,j,1]:
                array[i,j]=255
                # print(mask[i,j,0],"------------------",mask[i,j,1])
            else:
                pass
    return array

ElasticTransform=-2
GridDistortion=2
OpticalDistortion=2
RandomSizedCrop=3
aug_medium_size=2
aug_heavy_size=-1
RandomRotate90=2
def dataGenerator_GetValue(image, mask):
    original_height, original_width = image.shape[:2]

    oran=0.7
    min_max_height = (int(min(original_height * oran, original_width * oran)), min(original_height, original_width))

    alpha = original_height
    sigma = original_height * 0.15
    alpha_affine = original_height * 0.03




    images=[]
    masks=[]
    # https: // albumentations.ai / docs / examples / example_kaggle_salt /
    # min_height=2**(int(math.log(original_height,2))+1)
    # min_width=2**(int(math.log(original_height,2))+1)
    # min_height=64
    # min_width=64
    # aug = A.PadIfNeeded(min_height=min_height, min_width=min_width, p=1)
    # augmented = aug(image=image, mask=mask)
    # image_padded = augmented['image']
    # mask_padded = augmented['mask']



    aug = A.CenterCrop(p=1, height=original_height, width=original_width)
    augmented = aug(image=image, mask=mask)
    image_center_cropped = augmented['image']
    mask_center_cropped = augmented['mask']
    images.append(image_center_cropped)
    masks.append(mask_center_cropped)


    # x_min = (original_width)
    # y_min = (min_height - original_height) // 2
    # x_max = (x_min + original_width)
    # y_max = (y_min + original_height)
    # aug = A.Crop(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, p=1)
    # augmented = aug(image=image_padded, mask=mask_padded)
    # image_cropped = augmented['image']
    # mask_cropped = augmented['mask']
    # visualize(image_cropped, mask_cropped, original_image=image, original_mask=mask)



    aug = A.HorizontalFlip(p=1)
    augmented = aug(image=image, mask=mask)
    image_h_flipped = augmented['image']
    mask_h_flipped = augmented['mask']
    images.append(image_h_flipped)
    masks.append(mask_h_flipped)

    aug = A.VerticalFlip(p=1)
    augmented = aug(image=image, mask=mask)
    image_v_flipped = augmented['image']
    mask_v_flipped = augmented['mask']
    images.append(image_v_flipped)
    masks.append(mask_v_flipped)

    aug = A.RandomRotate90(p=1)
    for i in range(RandomRotate90):
        augmented = aug(image=image, mask=mask)
        image_rot90 = augmented['image']
        mask_rot90 = augmented['mask']
        images.append(image_rot90)
        masks.append(mask_rot90)

    aug = A.Transpose(p=1)
    augmented = aug(image=image, mask=mask)
    image_transposed = augmented['image']
    mask_transposed = augmented['mask']
    images.append(image_transposed)
    masks.append(mask_transposed)


    aug = A.ElasticTransform(p=1, alpha=alpha, sigma=sigma, alpha_affine=alpha_affine)
    for i in range(ElasticTransform):
        augmented = aug(image=image, mask=mask)
        image_elastic = augmented['image']
        mask_elastic = augmented['mask']
        images.append(image_elastic)
        masks.append(mask_elastic)


    # visualize(images[-1], images[-2], original_image=images[-3], original_mask=images[-4])
    # visualize(images[-1], masks[-1], original_image=images[-2], original_mask=masks[-2])

    aug = A.GridDistortion(p=1)
    for i in range(GridDistortion):
        augmented = aug(image=image, mask=mask)
        image_grid = augmented['image']
        mask_grid = augmented['mask']
        images.append(image_grid)
        masks.append(mask_grid)

    aug = A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1)
    for i in range(OpticalDistortion):
        augmented = aug(image=image, mask=mask)
        image_optical = augmented['image']
        mask_optical = augmented['mask']
        images.append(image_optical)
        masks.append(mask_optical)

    aug = A.RandomSizedCrop(min_max_height=min_max_height, height=original_height, width=original_width, p=1)
    for i in range(RandomSizedCrop):
        augmented = aug(image=image, mask=mask)
        image_scaled = augmented['image']
        mask_scaled = augmented['mask']
        images.append(image_scaled)
        masks.append(mask_scaled)

    aug = A.Compose([
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5)]
    )
    augmented = aug(image=image, mask=mask)
    image_light = augmented['image']
    mask_light = augmented['mask']
    images.append(image_light)
    masks.append(mask_light)

    aug_medium = A.Compose([
        A.OneOf([
            A.RandomSizedCrop(min_max_height=min_max_height, height=original_height, width=original_width, p=0.5),
            A.PadIfNeeded(min_height=original_height, min_width=original_width, p=0.5)
        ], p=1),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=alpha, sigma=sigma, alpha_affine=alpha_affine),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1),
        ], p=0.8)])

    for i in range(aug_medium_size):
        augmented = aug_medium(image=image, mask=mask)
        image_medium = augmented['image']
        mask_medium = augmented['mask']
        images.append(image_medium)
        masks.append(mask_medium)

    aug_heavy= A.Compose([
        A.OneOf([
            A.RandomSizedCrop(min_max_height=min_max_height, height=original_height, width=original_width, p=0.5),
            A.PadIfNeeded(min_height=original_height, min_width=original_width, p=0.5)
        ], p=1),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=alpha, sigma=sigma, alpha_affine=alpha_affine),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1)
        ], p=0.8),
        A.CLAHE(p=0.8),
        A.RandomBrightnessContrast(p=0.8),
        A.RandomGamma(p=0.8)])
    for i in range(aug_heavy_size):
        augmented = aug_heavy(image=image, mask=mask)
        image_heavy = augmented['image']
        mask_heavy = augmented['mask']
        images.append(image_heavy)
        masks.append(mask_heavy)


    # visualize(images[-1], masks[-1], original_image=image, original_mask=mask)

    return images, masks

def converNumpy(list):
    try:
        array = np.zeros((len(list), list[0].shape[0], list[0].shape[1], list[0].shape[2]), dtype=np.uint8)
        for i in range(len(list)):
            array[i, :, :, :] = list[i]
    except:
        array = np.zeros((len(list), list[0].shape[0], list[0].shape[1], 1), dtype=np.uint8)
        for i in range(len(list)):
            array[i, :, :, 0] = list[i]
    return array
def dataGenerator(X_train, y_train):

    images_bos, maskes_bos = dataGenerator_GetValue(X_train[0], y_train[0])
    DG_times=len(images_bos)
    list_images=[]
    list_maskes=[]
    sayac=0
    for i in range(X_train.shape[0]):
        if y_train[i].sum()<5:
            sayac+=1
            # continue
        images, maskes=dataGenerator_GetValue(X_train[i],y_train[i])
        list_images=list_images+images
        list_maskes=list_maskes+maskes
    X_train=converNumpy(list_images)
    y_train=converNumpy(list_maskes)
    print(sayac, ' tane atlatıldı ')
    return X_train,y_train, DG_times




def DravPred(X_train,X_test,y_test,y_train,model_idx, model, kayitYeri=None, ismask_line=None,batch_size=16):

    (img_height, img_width,img_channels) = (128, 128,1)
    def GoruntuIcinFgureAyarla(figure_axis):
        for axD in figure_axis:
            for ax in axD:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

    def DravİmageFgure(X, y, fgure, fgrure_axis, pred=None, name=None, kayitYeri=None, pause=None):


        if ismask_line:
            y = y.reshape(y.shape[0], img_height, img_width,img_channels)
        for j in range(0, X.shape[0]):

            fgrure_axis[j, 0].imshow(X[j, :, :, 0], cmap='gray')
            if y is not None:
                y_temp = catogarial_toimage(y[j, :, :, :])
                fgrure_axis[j, 1].imshow(y_temp, cmap='gray')
            if pred is not None:
                pred_temp = catogarial_toimage(pred[j, :, :, :])
                fgrure_axis[j, 2].imshow(pred_temp, cmap='gray')
        fgure.suptitle(name, fontsize=16)
        fgure.canvas.draw()
        if kayitYeri is not None:
            fgure.savefig(kayitYeri + name)
        if pause is not None:
            plt.pause(pause)

    f_S_Limite = 15
    fgr_pl, fgr_pl_axis = plt.subplots(f_S_Limite, 3, figsize=(15, 10))
    GoruntuIcinFgureAyarla(fgr_pl_axis)
    fgr_pl_2, fgr_pl_axis_2 = plt.subplots(f_S_Limite, 3, figsize=(15, 10))
    GoruntuIcinFgureAyarla(fgr_pl_axis_2)
    fgr_pl_3, fgr_pl_axis_3 = plt.subplots(f_S_Limite, 2)
    GoruntuIcinFgureAyarla(fgr_pl_axis_3)
    # fgr_pl_4, fgr_pl_axis_4 = plt.subplots(f_S_Limite, 2)
    # GoruntuIcinFgureAyarla(fgr_pl_axis_4)
    batch_size
    if not (model is None):
        pred = model.predict(X_train, batch_size=batch_size)
        pred_test = model.predict(X_test, batch_size=batch_size)
        # pred_images_x_false = autoencoder.predict(images_x_false[0:f_S_Limite], batch_size=batch_size)
        # catogarial_toimage(pred)
        # pred = pred*(255/10)
        # pred_test = (pred_test)*(255/10)
        # pred_images_x_false = (pred_images_x_false > 0.5).astype(np.uint8)

        if ismask_line:
            pred = pred.reshape(pred.shape[0], img_width, img_height,img_channels)
            pred_test=pred_test.reshape((pred_test.shape[0],img_width, img_height,img_channels))
            # y_train=y_train.reshape((y_train.shape[0],y_train.shape[1]*y_train.shape[2],y_train.shape[3]))


    else:
        pred = None
        pred_valid = None
        pred_images_x_false = None
        img_width, img_height, img_channels

    name = '_X_t_' + str(model_idx) + '.png'
    DravİmageFgure(X_train[0:f_S_Limite], y_train[0:f_S_Limite], fgr_pl, fgr_pl_axis, pred=pred, name=name,
                   kayitYeri=kayitYeri, pause=None)
    name = '_X_test_' + str(model_idx) + '.png'
    DravİmageFgure(X_test[0:f_S_Limite], y_test[0:f_S_Limite], fgr_pl_2, fgr_pl_axis_2, pred=pred_test, name=name,
                   kayitYeri=kayitYeri, pause=None)
    plt.show()
    # plt.pause(1)
import pickle
def AnaVeriKayit(kayitYeri,kfold_index,images_img, images_mask):
    with open(kayitYeri, "wb") as fp:  # Pickling
        pickle.dump((kfold_index, images_img, images_mask), fp)
    return None

def AnaVeriKayitGetir(path):
    try:
        with open(path, "rb") as fp:  # Unpickling
            tumveri = pickle.load(fp)

        return tumveri[0],tumveri[1],tumveri[2]
    except:
        return None,None,None

def digerKAyitlaryeni(kayitYeri,pred_test_id,pred_test_list,X_test,y_test,fold_no):

    np.savez(kayitYeri + '_Veri_H_.npz',
             X_test=X_test,
             y_test=y_test,
             pred_test_id=pred_test_id,
             pred_test_list=pred_test_list,
             fold_no=fold_no)

    return None

def digerKAyitlaryeni_Getir(path ):

    a = np.load(path)
    X_test=a['X_test']
    y_test =a['y_test']
    fold_no = a['fold_no']
    pred_test_id = a['pred_test_id']
    pred_test_list = a['pred_test_list']

    return pred_test_id,pred_test_list,X_test,y_test,fold_no


def digerKAyitlarGetir(path):
    a = np.load(path)
    # print(a.files)
    pred_test=a['pred_test']

    X_train =a['X_train']
    y_train = a['y_train']
    X_test = a['X_test']
    y_test = a['y_test']
    pred_test = a['pred_test']


    return X_train,y_train  , X_test, y_test,pred_test

