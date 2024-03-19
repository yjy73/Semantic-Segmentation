from models.fcn_vgg_16 import FCN_8
from PIL import Image
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(threshold=np.inf)
Label = namedtuple('Label', [
                   'name',
                   'id',
                   'trainId',
                   'category',
                   'categoryId',
                   'hasInstances',
                   'ignoreInEval',
                   'color'])
 
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

 

def matrix(pre, mask_path):
    #pred = pre.reshape((int(HEIGHT),int(WIDTH), NCLASS)).argmax(axis=2)
    pred = np.reshape(pre,[int(HEIGHT)*int(WIDTH)])
    #print(pred.shape)
    print(f'predict_classes:{np.unique(pred)}')
    mask = Image.open(mask_path)
    mask = mask.resize((int(WIDTH),int(HEIGHT)), Image.Resampling.NEAREST)
    mask = np.array(mask,dtype=np.uint8)
    #print(np.unique(mask))
    mask[mask == 255] = NCLASS-1
    if len(np.shape(mask)) == 3:
        mask = np.array(mask)[:,:,0]  #mask三通道转单通道
    mask = np.reshape(np.array(mask), [-1])
    mask = np.reshape(mask,[int(HEIGHT)*int(WIDTH)])
    #print(mask.shape)
    print(f'actual_classes{np.unique(mask)}')
    len_ = NCLASS

    confusion = np.zeros((len_, len_), dtype=np.int64)
    confusion += np.bincount(
        len_ * mask.astype(int) + pred,   
        minlength=len_ ** 2).reshape((len_, len_))
    plt.figure()
    plt.grid(False)
    sns.heatmap(confusion,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='',xticklabels = LABEL_,yticklabels = LABEL_)
    #plt.imshow(confusion)
    #plt.colorbar()
    plt.show()
    return confusion

def get_index(pred):
    confusion = matrix(pred, MASK_PATH)
    index = {}
    index['pixel_acc'] = np.diag(confusion).sum() / confusion.sum()
    iou_denominator = (confusion.sum(axis=1) + confusion.sum(axis=0) - np.diag(confusion)) 
    iou = np.diag(confusion) / iou_denominator
    miou = np.nanmean(iou[:-1])
    index['miou'] = miou
    freq = np.sum(confusion, axis=1) / np.sum(confusion)
    FWIoU = (freq[freq > 0] * iou[freq > 0]).sum()
    index['FWIoU'] = FWIoU
    print(index)



if __name__ =='__main__':
    NCLASS=20
    HEIGHT=int(1024)
    WIDTH=int(2048)
    IMAGE_PATH = r'semanticSegmentation\create_dataset\dataset_ready4train\raw_image\cityscapes_val\frankfurt_000000_007365_leftImg8bit.png'
    MASK_PATH = r'semanticSegmentation\create_dataset\dataset_ready4train\mask\cityscapes_19classes_val\frankfurt_000000_007365_gtFine_labelTrainIds.png'
    index2color = []
    IDX_ = []
    LABEL_ = []
    for obj in labels:
        if obj.ignoreInEval:
            continue
        idx = obj.trainId
        label = obj.name
        color = obj.color
        IDX_.append(idx)
        LABEL_.append(label)
        index2color.append(color)
    IDX_.append(NCLASS)
    LABEL_.append('void')
    index2color.append((0,0,0))   # add an void class
    print(index2color)

    
    fcn = FCN_8(n_class=NCLASS,inputshape=(HEIGHT,WIDTH,3))

    weight_path = r'semanticSegmentation\logs\FULL_SIZE_SGD0.9_ep055-loss0.605-train_acc0.825-val_loss0.623-val_acc0.821.h5'
    fcn.load_weights(weight_path)

    img_ori = Image.open(IMAGE_PATH).convert('RGB')
    orininal_h = np.array(img_ori).shape[0]
    orininal_w = np.array(img_ori).shape[1]

    img = img_ori.resize((int(WIDTH),int(HEIGHT)), Image.BICUBIC)
    
    img = np.array(img)/255.0
    img = img.reshape(-1, HEIGHT, WIDTH, 3)
    #print(img.shape)
    
    pred = fcn.predict(img)[0]
    #print(np.array(pred).shape)
    pred = pred.reshape((int(HEIGHT), int(WIDTH), NCLASS)).argmax(axis=2)
    #print(pred.shape)
    get_index(pred)
    

    seg_img = np.zeros((HEIGHT,WIDTH,3))

    #验证傻逼
    #c = 255
    #label = Image.open(r'semanticSegmentation\create_dataset\dataset_ready4train\mask\cityscapes_19classes_val\frankfurt_000000_000294_gtFine_labelTrainIds.png')
    #label = label.resize((int(WIDTH),int(HEIGHT)), Image.NEAREST)
    #label = np.array(label,dtype=np.uint8)
    #print(np.unique(label))
    #label[label == 255] = NCLASS-1
    #if len(np.shape(label)) == 3:
    #    label = np.array(label)[:,:,0]  #mask三通道转单通道
    ##label = np.reshape(np.array(label), [-1]) 
    #print(label.shape)
    #seg_img[:, :, 0] += ((label[:,: ] == c) * index2color[c][0]).astype('uint8')
    #seg_img[:, :, 1] += ((label[:,: ] == c) * index2color[c][1]).astype('uint8')
    #seg_img[:, :, 2] += ((label[:,: ] == c) * index2color[c][2]).astype('uint8')
    for c in range(NCLASS):
            seg_img[:, :, 0] += ((pred[:,: ] == c) * index2color[c][0]).astype('uint8')
            seg_img[:, :, 1] += ((pred[:,: ] == c) * index2color[c][1]).astype('uint8')
            seg_img[:, :, 2] += ((pred[:,: ] == c) * index2color[c][2]).astype('uint8')
    
    img_ori.show()
    seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h))
    seg_img.show()
    image = Image.blend(img_ori,seg_img,0.5)
    image.show()

    