import cv2
import os
import glob
import random
import numpy as np

def doCanny(train_data,val_data,test_data):
    
    
    for i in train_data:
        # load the input image and grab its dimensions
        image_path="zips/org data/"+i
        image = cv2.imread(image_path)
        (H, W) = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        hed = cv2.Canny(blur,50,200)
        print("Canny for {}".format(image_path))
   
        path1="dataCanny/A/train/"
        path2="dataCanny/B/train/"
        path3="finalDataCanny/train/"
        
        cv2.imwrite(os.path.join(path1,i),image)
        
        cv2.imwrite(os.path.join(path2,i),hed)
        hed2=cv2.imread(os.path.join(path2,i))
        
        im_AB = np.concatenate([image, hed2], 1)
        cv2.imwrite(os.path.join(path3,i),im_AB)

    
    for i in val_data:
        # load the input image and grab its dimensions
        image_path="zips/org data/"+i
        image = cv2.imread(image_path)
        (H, W) = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        hed = cv2.Canny(blur,50,200)
        print("Canny for {}".format(image_path))
       
        path1="dataCanny/A/val/"
        path2="dataCanny/B/val/"
        path3="finalDataCanny/val/"
        
        cv2.imwrite(os.path.join(path1,i),image)
    
        
        cv2.imwrite(os.path.join(path2,i),hed)
        hed2=cv2.imread(os.path.join(path2,i))
        
        im_AB = np.concatenate([image, hed2], 1)
        cv2.imwrite(os.path.join(path3,i),im_AB)

    
    
    for i in test_data:
        # load the input image and grab its dimensions
        image_path="zips/org data/"+i
        image = cv2.imread(image_path)
        (H, W) = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        hed = cv2.Canny(blur,50,200)
        print("Canny for {}".format(image_path))
        
        path1="dataCanny/A/test/"
        path2="dataCanny/B/test/"
        path3="finalDataCanny/test/"
        
        cv2.imwrite(os.path.join(path1,i),image)
       
        cv2.imwrite(os.path.join(path2,i),hed)
        hed2=cv2.imread(os.path.join(path2,i))
        
        im_AB = np.concatenate([image, hed2], 1)
        cv2.imwrite(os.path.join(path3,i),im_AB)

        
    return 0



list_AllFiles=glob.glob("D:/us/Courses/AML 674/P3/zips/org data/*.jpg")
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


doCanny(train_data,val_data,test_data)
