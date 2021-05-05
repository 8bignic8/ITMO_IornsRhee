#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numba import jit
import cv2
import os
import time


# In[ ]:


def mathiTMO(pic):
    pic = pic/((2**8)-1)
    x = len(pic[0,:,:])-1
    y = len(pic[:,0,:])-1
    c = len(pic[0,0,:])-1
    new_pic=pic#.astype(np.float64)
    while(x>=0): #y
        y = len(pic[:,0,:])-1
        while(y>=0): #x
            c = len(pic[0,0,:])-1
            while(c>=0): #x
                pixelInfloat = ((10*((0.11*pic[y,x,0]+0.59*pic[y,x,1]+0.3*pic[y,x,2])**10)+1.8)*pic[y,x,c])
                new_pic[y,x,c] = pixelInfloat
                c = c -1
            y = y -1
        x = x -1
    return new_pic


# In[ ]:


#Read Picture and return it

def readThePicture(picturepath):
    #  open ImageObject
    try:
        print('Reading <==== '+picturepath)
        img = cv2.imread(picturepath, cv2.IMREAD_UNCHANGED)# | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    except:
        print('There was an error while reading the picture')
        img = 0
    return img


# In[ ]:


def savePic(picture,fileName,extention,outPath): #saves the given array as a pictures to the given output path
    outPath = outPath+fileName+'.'+extention
    try:

        #print(picture.shape)
        print('Writing picture to ====> '+outPath)
        cv2.imwrite(outPath,picture)
        
    except:
        print('Failed while saving picture: '+fileName+' to '+ outPath+' sorry :(')
        print('--------------------')


# In[ ]:


###https://link.springer.com/content/pdf/10.1007%2F978-3-319-30285-0_12.pdf

sdrPic = './SDR/'
sdrPic = input('Pleace put in your inputpath for your SDR Pictures:def = '+sdrPic) or sdrPic
if not os.path.exists(sdrPic):
        os.mkdir(sdrPic)
allDir = os.listdir(sdrPic)


# In[ ]:


start_time = time.time() #start the timeing of the Prgramm
allDir = os.listdir(sdrPic)
if(len(allDir)>= 0):
    sumDir = len(allDir)-1
    while(sumDir>=0):
        if(allDir[sumDir]!='.DS_Store'):
            akDir = sdrPic+allDir[sumDir]+'/SDR/'
            print(akDir)
            picnum= len(os.listdir(akDir))-1
            while(picnum>=0):   
                name = os.listdir(akDir)[picnum]
                if(name!='.DS_Store'):
                    pic = readThePicture(akDir+name)
                    jutfunk = jit()(mathiTMO)
                    c = jutfunk(pic)
                    c = np.clip(c*((2**16)-1),0,((2**16)-1)).astype(np.uint16)
                    outP = sdrPic+allDir[sumDir]+'/HDRGen/'
                    if not os.path.exists(outP):
                        os.mkdir(outP)

                    savePic(c,name.split('.')[0],'png',outP)
                picnum = picnum-1
        sumDir = sumDir-1
print('Finished and it took: '+str((time.time() - start_time)/60)+'minutes')        


# In[ ]:


exit()


# In[ ]:





# In[ ]:





# In[ ]:




