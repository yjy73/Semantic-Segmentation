import tensorflow as tf
import cv2
from tensorflow.keras.layers import Input
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
from keras.utils.data_utils import get_file
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image
import os
from models import fcn_vgg_16
from keras.utils.data_utils import get_file
from tensorflow.python.framework.ops import disable_eager_execution
import matplotlib.pyplot as plt
#disable_eager_execution()


def show_train_history(train_history,train,validation,loss,val_loss):
    plt.subplot(121)
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.subplot(122)
    plt.plot(train_history.history[loss])
    plt.plot(train_history.history[val_loss])
    plt.ylabel("loss")
    plt.xlabel('Epoch')
    plt.legend(['train','val'],loc='upper right')
    plt.show()

def generate_arrays_from_file(trainData_path_root,trainMask_path_root,batch_size):
    all_train_img = []
    for image_name in os.listdir(trainData_path_root):
        all_train_img.append(image_name)
    all_train_mask = []
    for mask_name in os.listdir(trainMask_path_root):
        all_train_mask.append(mask_name)
    n = len(all_train_img)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        seed = np.random.randint(1,1000)
        for _ in range(batch_size):
            if i==0:
                np.random.seed(seed=seed)
                np.random.shuffle(all_train_img)
                np.random.seed(seed=seed)
                np.random.shuffle(all_train_mask)
            #-------------------------------------#
            #   读取输入图片并进行归一化和resize
            #-------------------------------------#
            img_name = all_train_img[i]
            img = Image.open(os.path.join(trainData_path_root, img_name)).convert('RGB')
            img = img.resize((WIDTH,HEIGHT), Image.BICUBIC)
            img = np.array(img)/255.
            X_train.append(img)
            #-------------------------------------#
            #   读取标签图片并进行归一化和resize
            #-------------------------------------#
            mask_name = all_train_mask[i]
            label = Image.open(os.path.join(trainMask_path_root, mask_name))
            label = label.resize((int(WIDTH),int(HEIGHT)), Image.NEAREST)
            label = np.array(label,dtype=np.uint8)
            #label = np.expand_dims(label,-1)
            label[label == 255] = NCLASSES-1
            if len(np.shape(label)) == 3:
                label = np.array(label)[:,:,0]  #mask三通道转单通道
            label = np.reshape(np.array(label), [-1])
            #print(np.unique(label)) 
            label = np.eye(NCLASSES)[np.array(label, np.int32)] 
            Y_train.append(label)
            i = (i+1) % n
        #print(Y_train)
        yield (np.array(X_train), np.array(Y_train))

def loss(y_true, y_pred):
    cross_loss = categorical_crossentropy(y_true=y_true,y_pred=y_pred)
    loss = K.sum(cross_loss)/(HEIGHT*WIDTH) #归一化，loss值缩小一点
    return loss

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    NCLASSES = 20
    HEIGHT,WIDTH = int(1024),int(2048)
    lr = 1e-3
    batch = 1
    TRAIN_NUM = 2975
    VAL_NUM = 500

    trainData_path_root= r"F:\STUDY\python_code\semanticSegmentation\create_dataset\dataset_ready4train\raw_image\cityscapes_train"
    trainMask_path_root = r"semanticSegmentation\create_dataset\dataset_ready4train\mask\cityscapes_19classes_train"
    
    valData_path_root = r"F:\STUDY\python_code\semanticSegmentation\create_dataset\dataset_ready4train\raw_image\cityscapes_val"
    valMask_path_root = r"F:\STUDY\python_code\semanticSegmentation\create_dataset\dataset_ready4train\mask\cityscapes_19classes_val"

    #print(generate_arrays_from_file(trainData_path_root=trainData_path_root,trainMask_path_root=trainMask_path_root, batch_size=batch))


    fcn = fcn_vgg_16.FCN_8(n_class=NCLASSES,inputshape=(HEIGHT,WIDTH,3))

    weight_path = r'semanticSegmentation\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    #weight = get_file(weight_path)
    fcn.load_weights(weight_path,by_name=True,skip_mismatch=True)
    for i in range(19):
        fcn.layers[i].trainable = False
    print(fcn.summary())
    #print(fcn.layers[0])

   
    #inputs_ = Input(shape=(1024,2048,3))
    #
    #fcn.call(inputs=inputs_)
    #fcn.build(input_shape=(None,1024,2048,3))
    sgd = SGD(learning_rate=lr, momentum=0.9)
    adam = Adam(learning_rate=lr)
    log_dir = r'F:\STUDY\python_code\semanticSegmentation\logs'
    checkpoint = ModelCheckpoint(os.path.join(log_dir,'FULL_SIZE_SGD0.9_ep{epoch:03d}-loss{loss:.3f}-train_acc{accuracy:.3f}-val_loss{val_loss:.3f}-val_acc{val_accuracy:.3f}.h5'),
                                    monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, mode='auto', verbose=True, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    fcn.compile(loss = categorical_crossentropy,
                optimizer = sgd,
                metrics=['accuracy'],
                #experimental_run_tf_function=False
                )
    train_history = fcn.fit_generator(generate_arrays_from_file(trainData_path_root,trainMask_path_root, batch_size=batch),
                steps_per_epoch=TRAIN_NUM//batch,
                validation_data=generate_arrays_from_file(valData_path_root,valMask_path_root, batch_size=batch),
                validation_steps=VAL_NUM//batch,
                epochs=100,
                initial_epoch=0,
                callbacks=[checkpoint, reduce_lr,early_stopping]
                )

    show_train_history(train_history,'accuracy','val_accuracy','loss','val_loss')