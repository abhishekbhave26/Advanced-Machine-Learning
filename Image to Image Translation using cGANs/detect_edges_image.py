#https://github.com/s9xie/hed
#https://www.pyimagesearch.com/2019/03/04/holistically-nested-edge-detection-with-opencv-and-deep-learning/

import cv2
import os
import glob
import random
import numpy as np
from PIL import Image

def doHED(train_data,val_data,test_data):
    
    class CropLayer(object):
    	def __init__(self, params, blobs):
    		# initialize our starting and ending (x, y)-coordinates of
    		# the crop
    		self.startX = 0
    		self.startY = 0
    		self.endX = 0
    		self.endY = 0
    
    	def getMemoryShapes(self, inputs):
    		# the crop layer will receive two inputs -- we need to crop
    		# the first input blob to match the shape of the second one,
    		# keeping the batch size and number of channels
    		(inputShape, targetShape) = (inputs[0], inputs[1])
    		(batchSize, numChannels) = (inputShape[0], inputShape[1])
    		(H, W) = (targetShape[2], targetShape[3])
    
    		# compute the starting and ending crop coordinates
    		self.startX = int((inputShape[3] - targetShape[3]) / 2)
    		self.startY = int((inputShape[2] - targetShape[2]) / 2)
    		self.endX = self.startX + W
    		self.endY = self.startY + H
    
    		# return the shape of the volume (we'll perform the actual
    		# crop during the forward pass
    		return [[batchSize, numChannels, H, W]]
    
    	def forward(self, inputs):
    		# use the derived (x, y)-coordinates to perform the crop
    		return [inputs[0][:, :, self.startY:self.endY,
    				self.startX:self.endX]]
    
    
    x={}
    x["edge_detector"]="hed_model"
    protoPath = os.path.sep.join([x["edge_detector"],"deploy.prototxt"])
    modelPath = os.path.sep.join([x["edge_detector"],"hed_pretrained_bsds.caffemodel"])
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    
    # register our new layer with the model
    cv2.dnn_registerLayer("Crop", CropLayer)
    
    for i in train_data:
        # load the input image and grab its dimensions
        image_path="zips/building/"+i
        image = cv2.imread(image_path)
        (H, W) = image.shape[:2]
        print("HED for {}".format(image_path))
        # construct a blob out of the input image for the Holistically-Nested
        # Edge Detector
        blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
        	mean=(104.00698793, 116.66876762, 122.67891434),
        	swapRB=False, crop=False)
        
        # set the blob as the input to the network and perform a forward pass
        # to compute the edges
        #print("[INFO] performing holistically-nested edge detection...")
        net.setInput(blob)
        hed = net.forward()
        hed = cv2.resize(hed[0, 0], (W, H))
        hed = (255 * hed).astype("uint8")
        
        path1="Buildingdata/A/train/"
        path2="Buildingdata/B/train/"
        path3="BuildingfinalData/train/"
        
        cv2.imwrite(os.path.join(path1,i),image)
        
        ret,thresh1 = cv2.threshold(hed,50.0,255.0,cv2.THRESH_BINARY)
        x=thresh1
        x[x == 0] =254
        x[x==255]=0
        x[x==254]=255
        
        cv2.imwrite(os.path.join(path2,i),x)
        hed2=cv2.imread(os.path.join(path2,i))
        
        im_AB = np.concatenate([image, hed2], 1)
        cv2.imwrite(os.path.join(path3,i),im_AB)

    
    for i in val_data:
        # load the input image and grab its dimensions
        image_path="zips/building/"+i
        image = cv2.imread(image_path)
        (H, W) = image.shape[:2]
        print("HED for {}".format(image_path))
        # construct a blob out of the input image for the Holistically-Nested
        # Edge Detector
        blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
        	mean=(104.00698793, 116.66876762, 122.67891434),
        	swapRB=False, crop=False)
        
        # set the blob as the input to the network and perform a forward pass
        # to compute the edges
        #print("[INFO] performing holistically-nested edge detection...")
        net.setInput(blob)
        hed = net.forward()
        hed = cv2.resize(hed[0, 0], (W, H))
        hed = (255 * hed).astype("uint8")
        path1="Buildingdata/A/val/"
        path2="Buildingdata/B/val/"
        path3="BuildingfinalData/val/"
        
        cv2.imwrite(os.path.join(path1,i),image)
        
        ret,thresh1 = cv2.threshold(hed,50.0,255.0,cv2.THRESH_BINARY)
        x=thresh1
        x[x == 0] =254
        x[x==255]=0
        x[x==254]=255
        
        cv2.imwrite(os.path.join(path2,i),x)
        hed2=cv2.imread(os.path.join(path2,i))
        
        im_AB = np.concatenate([image, hed2], 1)
        cv2.imwrite(os.path.join(path3,i),im_AB)

    
    
    for i in test_data:
        # load the input image and grab its dimensions
        image_path="zips/building/"+i
        image = cv2.imread(image_path)
        (H, W) = image.shape[:2]
        print("HED for {}".format(image_path))
        # construct a blob out of the input image for the Holistically-Nested
        # Edge Detector
        blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
        	mean=(104.00698793, 116.66876762, 122.67891434),
        	swapRB=False, crop=False)
        
        # set the blob as the input to the network and perform a forward pass
        # to compute the edges
        #print("[INFO] performing holistically-nested edge detection...")
        net.setInput(blob)
        hed = net.forward()
        hed = cv2.resize(hed[0, 0], (W, H))
        hed = (255 * hed).astype("uint8")
        path1="Buildingdata/A/test/"
        path2="Buildingdata/B/test/"
        path3="BuildingfinalData/test/"
        
        cv2.imwrite(os.path.join(path1,i),image)
        
        ret,thresh1 = cv2.threshold(hed,50.0,255.0,cv2.THRESH_BINARY)
        x=thresh1
        x[x == 0] =254
        x[x==255]=0
        x[x==254]=255
        
        cv2.imwrite(os.path.join(path2,i),x)
        hed2=cv2.imread(os.path.join(path2,i))
        
        im_AB = np.concatenate([image, hed2], 1)
        cv2.imwrite(os.path.join(path3,i),im_AB)

        
    return 0



list_AllFiles=glob.glob("D:/us/Courses/AML 674/P3/zips/building/*.jpg")
random.shuffle(list_AllFiles)
length=len(list_AllFiles)

train_data = list_AllFiles[:int(0.8*length)]
test_data = list_AllFiles[int(0.8*length):int(0.8*length+0.1*length)]
val_data = list_AllFiles[int(0.8*length+0.1*length):]

def createList(new):
    final_list=[]
    for i in range(0,len(new)):
        if("\\" in new[i]):
            x=new[i].split("\\")
            if(x[1] not in final_list):
                final_list.append(x[1])
    return final_list

train_data=createList(train_data)
val_data=createList(val_data)
test_data=createList(test_data)


doHED(train_data,val_data,test_data)
