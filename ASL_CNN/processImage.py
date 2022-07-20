import numpy as np
import cv2
import os
import csv

minValue = 70

def func(path):    
    frame = cv2.imread(path)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.GaussianBlur(gray,(5,5),2)
    # #blur = cv2.bilateralFilter(roi,9,75,75)
    
    roi_gray = cv2.adaptiveThreshold(roi_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, roi_gray = cv2.threshold(roi_gray, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    return roi_gray



if not os.path.exists("data_inv"):
    os.makedirs("data_inv")
if not os.path.exists("data_inv/train"):
    os.makedirs("data_inv/train")
if not os.path.exists("data_inv/test"):
    os.makedirs("data_inv/test")
path="data/train"
path1 = "data_inv"
a=['label']


for i in range(64*64):
    a.append("pixel"+str(i))
    

label=0
var = 0
c1 = 0
c2 = 0

for (dirpath,dirnames,filenames) in os.walk(path):
    for dirname in dirnames:
        print(dirname)
        for(direcpath,direcnames,files) in os.walk(path+"/"+dirname):
            if not os.path.exists(path1+"/train/"+dirname):
                os.makedirs(path1+"/train/"+dirname)
            if not os.path.exists(path1+"/test/"+dirname):
                os.makedirs(path1+"/test/"+dirname)
            
            # num=0.75*len(files)
            num = len(files)
            i=0
            for file in files:
                var+=1
                actual_path=path+"/"+dirname+"/"+file
                actual_path1=path1+"/"+"train/"+dirname+"/"+file
                actual_path2=path1+"/"+"test/"+dirname+"/"+file
                img = cv2.imread(actual_path, 0)
                bw_image = func(actual_path)
                if i<num:
                    c1 += 1
                    cv2.imwrite(actual_path1 , bw_image)
                else:
                    c2 += 1
                    cv2.imwrite(actual_path2 , bw_image)
                    
                i=i+1
                
        label=label+1
print(var)
print(c1)
print(c2)





