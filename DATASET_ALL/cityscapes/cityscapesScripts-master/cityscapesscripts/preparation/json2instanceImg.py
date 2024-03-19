#!/usr/bin/python
#
# Reads labels as polygons in JSON format and converts them to instance images,
# where each pixel has an ID that represents the ground truth class and the
# individual instance of that class.
#
# The pixel values encode both, class and the individual instance.
# The integer part of a division by 1000 of each ID provides the class ID,
# as described in labels.py. The remainder is the instance ID. If a certain
# annotation describes multiple instances, then the pixels have the regular
# ID of that class.
#
# Example:
# Let's say your labels.py assigns the ID 26 to the class 'car'.
# Then, the individual cars in an image get the IDs 26000, 26001, 26002, ... .
# A group of cars, where our annotators could not identify the individual
# instances anymore, is assigned to the ID 26.
#
# Note that not all classes distinguish instances (see labels.py for a full list).
# The classes without instance annotations are always directly encoded with
# their regular ID, e.g. 11 for 'building'.
#
# Usage: json2instanceImg.py [OPTIONS] <input json> <output image>
# Options:
#   -h   print a little help text
#   -t   use train IDs
#
# Can also be used by including as a module.
#
# Uses the mapping defined in 'labels.py'.
#
# See also createTrainIdInstanceImgs.py to apply the mapping to all annotations in Cityscapes.
#

# python imports
from __future__ import print_function, absolute_import, division
import os, sys, getopt

# Image processing
from PIL import Image
from PIL import ImageDraw

# cityscapes imports
from cityscapesscripts.helpers.annotation import Annotation
from cityscapesscripts.helpers.labels     import labels, name2label

# Print the information
def printHelp():
    print('{} [OPTIONS] inputJson outputImg'.format(os.path.basename(sys.argv[0])))
    print('')
    print(' Reads labels as polygons in JSON format and converts them to instance images,')
    print(' where each pixel has an ID that represents the ground truth class and the')
    print(' individual instance of that class.')
    print('')
    print(' The pixel values encode both, class and the individual instance.')
    print(' The integer part of a division by 1000 of each ID provides the class ID,')
    print(' as described in labels.py. The remainder is the instance ID. If a certain')
    print(' annotation describes multiple instances, then the pixels have the regular')
    print(' ID of that class.')
    print('')
    print(' Example:')
    print(' Let\'s say your labels.py assigns the ID 26 to the class "car".')
    print(' Then, the individual cars in an image get the IDs 26000, 26001, 26002, ... .')
    print(' A group of cars, where our annotators could not identify the individual')
    print(' instances anymore, is assigned to the ID 26.')
    print('')
    print(' Note that not all classes distinguish instances (see labels.py for a full list).')
    print(' The classes without instance annotations are always directly encoded with')
    print(' their regular ID, e.g. 11 for "building".')
    print('')
    print('Options:')
    print(' -h                 Print this help')
    print(' -t                 Use the "trainIDs" instead of the regular mapping. See "labels.py" for details.')

# Print an error message and quit
def printError(message):
    print('ERROR: {}'.format(message))
    print('')
    print('USAGE:')
    printHelp()
    sys.exit(-1)

# Convert the given annotation to a label image
def createInstanceImage(annotation, encoding):
    # the size of the image
    size = ( annotation.imgWidth , annotation.imgHeight )

    # the background
    if encoding == "ids":
        backgroundId = name2label['unlabeled'].id
    elif encoding == "trainIds":
        backgroundId = name2label['unlabeled'].trainId
    else:
        print("Unknown encoding '{}'".format(encoding))
        return None

    # this is the image that we want to create
    instanceImg = Image.new("I", size, backgroundId)

    # a drawer to draw into the image
    drawer = ImageDraw.Draw( instanceImg )

    # a dict where we keep track of the number of instances that
    # we already saw of each class
    nbInstances = {}
    for labelTuple in labels:
        if labelTuple.hasInstances:
            nbInstances[labelTuple.name] = 0

    # loop over all objects
    for obj in annotation.objects:
        label   = obj.label
        polygon = obj.polygon

        # If the object is deleted, skip it
        if obj.deleted:
            continue

        # if the label is not known, but ends with a 'group' (e.g. cargroup)
        # try to remove the s and see if that works
        # also we know that this polygon describes a group
        isGroup = False
        if ( not label in name2label ) and label.endswith('group'):
            label = label[:-len('group')]
            isGroup = True

        if not label in name2label:
            printError( "Label '{}' not known.".format(label) )

        # the label tuple
        labelTuple = name2label[label]

        # get the class ID
        if encoding == "ids":
            id = labelTuple.id
        elif encoding == "trainIds":
            id = labelTuple.trainId

        # if this label distinguishs between individual instances,
        # make the id a instance ID
        if labelTuple.hasInstances and not isGroup and id != 255:
            id = id * 1000 + nbInstances[label]
            nbInstances[label] += 1

        # If the ID is negative that polygon should not be drawn
        if id < 0:
            continue

        try:
            drawer.polygon( polygon, fill=id )
        except:
            print("Failed to draw polygon with label {} and id {}: {}".format(label,id,polygon))
            raise

    return instanceImg

# A method that does all the work
# inJson is the filename of the json file
# outImg is the filename of the instance image that is generated
# encoding can be set to
#     - "ids"      : classes are encoded using the regular label IDs
#     - "trainIds" : classes are encoded using the training IDs
def json2instanceImg(inJson,outImg,encoding="ids"):
    annotation = Annotation()
    annotation.fromJsonFile(inJson)
    instanceImg = createInstanceImage( annotation , encoding )
    instanceImg.save( outImg )

# The main method, if you execute this script directly
# Reads the command line arguments and calls the method 'json2instanceImg'
def main(argv):
    trainIds = False
    try:
        opts, args = getopt.getopt(argv,"ht")
    except getopt.GetoptError:
        printError( 'Invalid arguments' )
    for opt, arg in opts:
        if opt == '-h':
            printHelp()
            sys.exit(0)
        elif opt == '-t':
            trainIds = True
        else:
            printError( "Handling of argument '{}' not implementend".format(opt) )

    if len(args) == 0:
        printError( "Missing input json file" )
    elif len(args) == 1:
        printError( "Missing output image filename" )
    elif len(args) > 2:
        printError( "Too many arguments" )

    inJson = args[0]
    outImg = args[1]

    if trainIds:
        json2instanceImg( inJson , outImg , 'trainIds' )
    else:
        json2instanceImg( inJson , outImg )

# call the main method
if __name__ == "__main__":
    main(sys.argv[1:])


##!/usr/bin/python
##
## Reads labels as polygons in JSON format and converts them to instance images,
## where each pixel has an ID that represents the ground truth class and the
## individual instance of that class.
##
## The pixel values encode both, class and the individual instance.
## The integer part of a division by 1000 of each ID provides the class ID,
## as described in labels.py. The remainder is the instance ID. If a certain
## annotation describes multiple instances, then the pixels have the regular
## ID of that class.
##
## Example:
## Let's say your labels.py assigns the ID 26 to the class 'car'.
## Then, the individual cars in an image get the IDs 26000, 26001, 26002, ... .
## A group of cars, where our annotators could not identify the individual
## instances anymore, is assigned to the ID 26.
##
## Note that not all classes distinguish instances (see labels.py for a full list).
## The classes without instance annotations are always directly encoded with
## their regular ID, e.g. 11 for 'building'.
##
## Usage: json2instanceImg.py [OPTIONS] <input json> <output image>
## Options:
##   -h   print a little help text
##   -t   use train IDs
##
## Can also be used by including as a module.
##
## Uses the mapping defined in 'labels.py'.
##
## See also createTrainIdInstanceImgs.py to apply the mapping to all annotations in Cityscapes.
##
#
## python imports
#from __future__ import print_function, absolute_import, division
#import os, sys, getopt
#
## Image processing
#from PIL import Image
#from PIL import ImageDraw
#from collections import namedtuple
#
## cityscapes imports
#from cityscapesscripts.helpers.annotation import Annotation
##from cityscapesscripts.helpers.labels     import name2label
#
#Label = namedtuple( 'Label' , [
#
#    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
#                    # We use them to uniquely name a class
#
#    'id'          , # An integer ID that is associated with this label.
#                    # The IDs are used to represent the label in ground truth images
#                    # An ID of -1 means that this label does not have an ID and thus
#                    # is ignored when creating ground truth images (e.g. license plate).
#                    # Do not modify these IDs, since exactly these IDs are expected by the
#                    # evaluation server.
#
#    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
#                    # ground truth images with train IDs, using the tools provided in the
#                    # 'preparation' folder. However, make sure to validate or submit results
#                    # to our evaluation server using the regular IDs above!
#                    # For trainIds, multiple labels might have the same ID. Then, these labels
#                    # are mapped to the same class in the ground truth images. For the inverse
#                    # mapping, we use the label that is defined first in the list below.
#                    # For example, mapping all void-type classes to the same ID in training,
#                    # might make sense for some approaches.
#                    # Max value is 255!
#
#    'category'    , # The name of the category that this label belongs to
#
#    'categoryId'  , # The ID of this category. Used to create ground truth images
#                    # on category level.
#
#    'hasInstances', # Whether this label distinguishes between single instances or not
#
#    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
#                    # during evaluations or not
#
#    'color'       , # The color of this label
#    ] )
#
#labels = [
#    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
#    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
#    Label(  'road'                 ,  7 ,      255 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
#    Label(  'sidewalk'             ,  8 ,      255 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
#    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
#    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
#    Label(  'building'             , 11 ,      255 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
#    Label(  'wall'                 , 12 ,      255 , 'construction'    , 2       , False        , False        , (102,102,156) ),
#    Label(  'fence'                , 13 ,      255 , 'construction'    , 2       , False        , False        , (190,153,153) ),
#    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
#    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
#    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
#    Label(  'pole'                 , 17 ,      255 , 'object'          , 3       , False        , False        , (153,153,153) ),
#    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
#    Label(  'traffic light'        , 19 ,      255 , 'object'          , 3       , False        , False        , (250,170, 30) ),
#    Label(  'traffic sign'         , 20 ,      255 , 'object'          , 3       , False        , False        , (220,220,  0) ),
#    Label(  'vegetation'           , 21 ,      255 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
#    Label(  'terrain'              , 22 ,      255 , 'nature'          , 4       , False        , False        , (152,251,152) ),
#    Label(  'sky'                  , 23 ,      255 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
#    Label(  'person'               , 24 ,        5 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
#    Label(  'rider'                , 25 ,        6 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
#    Label(  'car'                  , 26 ,        3 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
#    Label(  'truck'                , 27 ,        8 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
#    Label(  'bus'                  , 28 ,        2 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
#    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0, 90) ),
#    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,110) ),
#    Label(  'train'                , 31 ,        7 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
#    Label(  'motorcycle'           , 32 ,        4 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
#    Label(  'bicycle'              , 33 ,        1 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
#    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
#]
#
## name to label object
#name2label      = { label.name    : label for label in labels           }
## id to label object
#id2label        = { label.id      : label for label in labels           }
## trainId to label object
#trainId2label   = { label.trainId : label for label in reversed(labels) }
## category to list of label objects
#category2labels = {}
#for label in labels:
#    category = label.category
#    if category in category2labels:
#        category2labels[category].append(label)
#    else:
#        category2labels[category] = [label]
#
## Print the information
#def printHelp():
#    print('{} [OPTIONS] inputJson outputImg'.format(os.path.basename(sys.argv[0])))
#    print('')
#    print(' Reads labels as polygons in JSON format and converts them to instance images,')
#    print(' where each pixel has an ID that represents the ground truth class and the')
#    print(' individual instance of that class.')
#    print('')
#    print(' The pixel values encode both, class and the individual instance.')
#    print(' The integer part of a division by 1000 of each ID provides the class ID,')
#    print(' as described in labels.py. The remainder is the instance ID. If a certain')
#    print(' annotation describes multiple instances, then the pixels have the regular')
#    print(' ID of that class.')
#    print('')
#    print(' Example:')
#    print(' Let\'s say your labels.py assigns the ID 26 to the class "car".')
#    print(' Then, the individual cars in an image get the IDs 26000, 26001, 26002, ... .')
#    print(' A group of cars, where our annotators could not identify the individual')
#    print(' instances anymore, is assigned to the ID 26.')
#    print('')
#    print(' Note that not all classes distinguish instances (see labels.py for a full list).')
#    print(' The classes without instance annotations are always directly encoded with')
#    print(' their regular ID, e.g. 11 for "building".')
#    print('')
#    print('Options:')
#    print(' -h                 Print this help')
#    print(' -t                 Use the "trainIDs" instead of the regular mapping. See "labels.py" for details.')
#
## Print an error message and quit
#def printError(message):
#    print('ERROR: {}'.format(message))
#    print('')
#    print('USAGE:')
#    printHelp()
#    sys.exit(-1)
#
## Convert the given annotation to a label image
#def createInstanceImage(annotation, encoding):
#    # the size of the image
#    size = ( annotation.imgWidth , annotation.imgHeight )
#
#    # the background
#    if encoding == "ids":
#        backgroundId = name2label['unlabeled'].id
#    elif encoding == "trainIds":
#        backgroundId = name2label['unlabeled'].trainId
#    else:
#        print("Unknown encoding '{}'".format(encoding))
#        return None
#
#    # this is the image that we want to create
#    instanceImg = Image.new("I", size, backgroundId)
#
#    # a drawer to draw into the image
#    drawer = ImageDraw.Draw( instanceImg )
#
#    # a dict where we keep track of the number of instances that
#    # we already saw of each class
#    nbInstances = {}
#    for labelTuple in labels:
#        has = labelTuple.hasInstances
#        if has:
#            nbInstances[labelTuple.name] = 0
#
#    # loop over all objects
#    id_count = 0
#    for obj in annotation.objects:
#        label   = obj.label
#        polygon = obj.polygon
#
#        # If the object is deleted, skip it
#        if obj.deleted:
#            continue
#
#        # if the label is not known, but ends with a 'group' (e.g. cargroup)
#        # try to remove the s and see if that works
#        # also we know that this polygon describes a group
#        isGroup = False
#        if ( not label in name2label ) and label.endswith('group'):
#            label = label[:-len('group')]
#            isGroup = True
#
#        if not label in name2label:
#            printError( "Label '{}' not known.".format(label) )
#
#        # the label tuple
#        labelTuple = name2label[label]
#
#        # get the class ID
#        if encoding == "ids":
#            id = labelTuple.id
#        elif encoding == "trainIds":
#            id = labelTuple.trainId
#
#        # if this label distinguishs between individual instances,
#        # make the id a instance ID
#        if labelTuple.hasInstances and not isGroup and id != 255:
#            #id = id * 1000 + nbInstances[label]
#            #nbInstances[label] += 1
#            id_count += 1
#            id = id_count 
#
#        # If the ID is negative that polygon should not be drawn
#        if id < 0:
#            continue
#
#        try:
#            drawer.polygon( polygon, fill=id )
#        except:
#            print("Failed to draw polygon with label {} and id {}: {}".format(label,id,polygon))
#            raise
#
#    return instanceImg
#
## A method that does all the work
## inJson is the filename of the json file
## outImg is the filename of the instance image that is generated
## encoding can be set to
##     - "ids"      : classes are encoded using the regular label IDs
##     - "trainIds" : classes are encoded using the training IDs
#def json2instanceImg(inJson,outImg,encoding="ids"):
#    annotation = Annotation()
#    annotation.fromJsonFile(inJson)
#    instanceImg = createInstanceImage( annotation , encoding )
#    instanceImg.save( outImg )
#
## The main method, if you execute this script directly
## Reads the command line arguments and calls the method 'json2instanceImg'
#def main(argv):
#    trainIds = False
#    try:
#        opts, args = getopt.getopt(argv,"ht")
#    except getopt.GetoptError:
#        printError( 'Invalid arguments' )
#    for opt, arg in opts:
#        if opt == '-h':
#            printHelp()
#            sys.exit(0)
#        elif opt == '-t':
#            trainIds = True
#        else:
#            printError( "Handling of argument '{}' not implementend".format(opt) )
#
#    if len(args) == 0:
#        printError( "Missing input json file" )
#    elif len(args) == 1:
#        printError( "Missing output image filename" )
#    elif len(args) > 2:
#        printError( "Too many arguments" )
#
#    inJson = args[0]
#    outImg = args[1]
#
#    if trainIds:
#        json2instanceImg( inJson , outImg , 'trainIds' )
#    else:
#        json2instanceImg( inJson , outImg )
#
## call the main method
#if __name__ == "__main__":
#    main(sys.argv[1:])
#    #print(labels)
#    #f = r'semanticSegmentation\DATASET_ALL\cityscapes\cityscapesScripts-master\gtFine\train\aachen\aachen_000000_000019_gtFine_polygons.json'
#    #dst = f.replace( "_polygons.json" , "_instanceTrainIds.png" )
#    #json2instanceImg(f , dst , "trainIds")
