import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.applications.vgg16 import VGG16
#from models.vgg_16 import get_vgg





class FCN(Model):
    def __init__(self, n_class, pretrain=True, input_shape=(227,227,3)):
        super().__init__()
        self.n_class = n_class
        self.pretrain = pretrain
        self.layers_get = ['block3_pool','block4_pool','block5_pool']
        #self.conv_base = get_vgg(inputshape=input_shape)
        self.conv_base = VGG16(weights='imagenet',
                                include_top=False,
                                input_shape=input_shape
                                )
        if self.pretrain:
            self.conv_base.trainable = False
        self.conv_fc1 = layers.Conv2D(filters=4096,kernel_size=(7,7),activation='relu',padding='same',name='conv_fc_1')
        self.conv_fc2 = layers.Conv2D(filters=4096,kernel_size=(1,1),activation='relu',padding='same',name='conv_fc_2')
        self.conv_fc3 = layers.Conv2D(filters=self.n_class,kernel_size=(1,1),activation='relu',padding='same',name='conv_classify')
        self.conv_fc_pool4 = layers.Conv2D(filters=self.n_class,kernel_size=(1,1),activation='relu',padding='same',name='conv_classify_pool4')
        self.conv_fc_pool3 = layers.Conv2D(filters=self.n_class,kernel_size=(1,1),activation='relu',padding='same',name='conv_calssify_pool3')
        self.dropout = layers.Dropout(0.3)
        self.trans_conv1 = layers.Conv2DTranspose(filters=self.n_class, activation='relu', kernel_size=(4,4), strides=(2,2), padding='same',name='trans_conv_base')
        self.trans_conv2 = layers.Conv2DTranspose(filters=self.n_class, activation='relu', kernel_size=(4,4), strides=(2,2), padding='same',name='trans_conv_pool4')
        self.trans_conv3 = layers.Conv2DTranspose(filters=self.n_class, activation='relu', kernel_size=(16,16), strides=(8,8), padding='same',name='trans_conv_pool3')
        #self.crop = layers.Cropping2D(cropping=((1,1)))
        #self.crop2 = layers.Cropping2D(cropping=((2,3),(3,2)))

    def call(self, inputs,):
        out_block5_pool = self.conv_base.get_layer("block5_pool").output
        out_block4_pool = self.conv_base.get_layer("block4_pool").output
        out_block3_pool = self.conv_base.get_layer("block3_pool").output

        
        x = self.conv_base(inputs)
        x = self.conv_fc1(x)
        x = self.dropout(x)
        x = self.conv_fc2(x)
        x = self.dropout(x)
        score_fr = self.conv_fc3(x)
        upscore_fr = self.trans_conv1(score_fr)
        #upscore_fr = self.crop(upscore_fr) #一定要裁剪，不然和pool4_conv的特征图大小不一样

        pool4_conv = self.conv_fc_pool4(out_block4_pool)
        #pool4_conv_size = 
        fuse_pool4 = layers.add([upscore_fr, pool4_conv])
        upscore_pool4 = self.trans_conv2(fuse_pool4)
        #upscore_pool4 = self.crop(upscore_pool4)

        pool3_conv = self.conv_fc_pool3(out_block3_pool)
        fuse_pool3 = layers.add([upscore_pool4, pool3_conv])
        upscore_pool3 = self.trans_conv3(fuse_pool3)
        #upscore_pool3 = self.crop2(upscore_pool3)
        #upscore_pool3 = layers.Flatten()(upscore_pool3)###########

        return upscore_pool3

    #def get_size(self,):
        out_block4_pool = self.conf_base.get_layer("block4_pool").output_shape[1:3]
        print(out_block4_pool)
        return 0

fcn = FCN(19,input_shape=(1024,2048,3))
fcn.build(input_shape=(None, 1024,2048,3))
###fcn.compile(loss='categorical_crossentropy')
print(fcn.summary())