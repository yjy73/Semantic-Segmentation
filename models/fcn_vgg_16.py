from tensorflow.keras import layers
from keras.models import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16

def get_vgg(inputshape = (1024,2048,3)):
    input_img = layers.Input(inputshape)
    o = (layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1), padding='same',activation='relu',input_shape=inputshape,name='block1_conv1'))(input_img)
    o = (layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1), padding='same',activation='relu',name='block1_conv2'))(o)
    o = (layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),name='block1_pool'))(o)
    o = (layers.Conv2D(filters=128,kernel_size=(3,3),strides=(1,1), padding='same',activation='relu',name='block2_conv1'))(o)
    o = (layers.Conv2D(filters=128,kernel_size=(3,3),strides=(1,1), padding='same',activation='relu',name='block2_conv2'))(o)
    o = (layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),name='block2_pool'))(o)
    o = (layers.Conv2D(filters=256,kernel_size=(3,3),strides=(1,1), padding='same',activation='relu',name='block3_conv1'))(o)
    o = (layers.Conv2D(filters=256,kernel_size=(3,3),strides=(1,1), padding='same',activation='relu',name='block3_conv2'))(o)
    o = (layers.Conv2D(filters=256,kernel_size=(3,3),strides=(1,1), padding='same',activation='relu',name='block3_conv3'))(o)
    o = (layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),name='block3_pool'))(o)
    b3_pool = o
    o = (layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1), padding='same',activation='relu',name='block4_conv1'))(o)
    o = (layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1), padding='same',activation='relu',name='block4_conv2'))(o)
    o = (layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1), padding='same',activation='relu',name='block4_conv3'))(o)
    o = (layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),name='block4_pool'))(o)
    b4_pool = o
    o = (layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1), padding='same',activation='relu',name='block5_conv1'))(o)
    o = (layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1), padding='same',activation='relu',name='block5_conv2'))(o)
    o = (layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1), padding='same',activation='relu',name='block5_conv3'))(o)
    o = (layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),name='block5_pool'))(o)
    b5_pool = o
    #print(net.summary())
    return input_img, [b3_pool,b4_pool,b5_pool]

def FCN_8(base=get_vgg, inputshape = (1024,2048,3), n_class = 19):
    layers_get = ['block3_pool','block4_pool','block5_pool']
    #conv_base = VGG16(weights='imagenet',
    #                            include_top=False,
    #                            input_shape=inputshape
    #                            )
    #if pretrain:
    #        conv_base.trainable = False
    input_img, conv_base = base(inputshape)
    [b3_p, b4_p, b5_p] = conv_base

    o = b5_p
    o = layers.Conv2D(filters=4096,kernel_size=(7,7),activation='relu',padding='same',name='conv_fc_1')(o)
    o = layers.Dropout(0.5)(o)
    o = layers.Conv2D(filters=4096,kernel_size=(1,1),activation='relu',padding='same',name='conv_fc_2')(o)
    o = layers.Dropout(0.5)(o)
    score_fr = layers.Conv2D(filters=n_class,kernel_size=(1,1),activation='relu',padding='same',name='conv_classify')(o)
    upscore_fr = layers.Conv2DTranspose(filters=n_class, activation='relu', kernel_size=(4,4), strides=(2,2), padding='same',name='trans_conv_base')(score_fr)

    pool4 = b4_p
    score_pool4 = layers.Conv2D(filters=n_class,kernel_size=(1,1),activation='relu',padding='same',name='conv_classify_pool4')(pool4)
    fuse_pool4 = layers.add([upscore_fr, score_pool4])
    upscore_pool4 = layers.Conv2DTranspose(filters=n_class, activation='relu', kernel_size=(4,4), strides=(2,2), padding='same',name='trans_conv_pool4')(fuse_pool4)

    pool3 = b3_p
    score_pool3 = layers.Conv2D(filters=n_class,kernel_size=(1,1),activation='relu',padding='same',name='conv_calssify_pool3')(pool3)
    fuse_pool3 = layers.add([upscore_pool4, score_pool3])
    upscore_pool3 = layers.Conv2DTranspose(filters=n_class, activation='relu', kernel_size=(16,16), strides=(8,8), padding='same',name='trans_conv_pool3')(fuse_pool3)

    o = layers.Reshape((int(inputshape[0])*int(inputshape[1]), -1))(upscore_pool3)
    o = layers.Softmax()(o)
    model = Model(input_img, o)
    return model

#if __name__ == '__main__':
#    net = FCN_8(get_vgg)
#    print(net.summary())