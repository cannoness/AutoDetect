# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 18:14:47 2017

@author: cattius
"""
import cv2 as cv2
import numpy as np
from itertools import compress
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from matplotlib import pyplot as plt
import math

from sklearn.preprocessing import StandardScaler

font = cv2.FONT_HERSHEY_SIMPLEX
#people finding helper function
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(11,11))
def PersonObject(frame):
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    f = np.copy(frame)
    min_YCrCb = np.array([75,138,111],np.uint8)
    max_YCrCb = np.array([172,155,124],np.uint8)
    min_YCrCb1 = np.array([56, 142, 110],np.uint8)
#    max_YCrCb1 = np.array([254, 175, 125],np.uint8)
#    min_YCrCb = np.array([0,133,77],np.uint8)
#    max_YCrCb = np.array([254,175,125],np.uint8)
#    # Convert image to YCrCb
    image_YCrCb = cv2.cvtColor(f,cv2.COLOR_BGR2YCrCb)
    
    imgYCC = cv2.GaussianBlur(image_YCrCb, (11, 11), 0)
        # Find region with skin tone in YCrCb image
    skin_region = cv2.inRange(imgYCC,min_YCrCb,max_YCrCb)
    skin_region2 = cv2.inRange(imgYCC,min_YCrCb1,max_YCrCb)
    skin_region = cv2.bitwise_and(skin_region,skin_region, mask = skin_region2)
    
    skin_region = cv2.morphologyEx(skin_region, cv2.MORPH_OPEN, kernel)
    skin_region = cv2.dilate(skin_region, None, iterations=10)
    skin_region = cv2.morphologyEx(skin_region, cv2.MORPH_CLOSE, kernel)
#    skin_region = cv2.morphologyEx(skin_region, cv2.MORPH_OPEN, kernel)
    skin_region = cv2.GaussianBlur(skin_region, (29,29), 0)
    skin_region = cv2.erode(skin_region, kernel2, iterations=2)
    skin_region = cv2.morphologyEx(skin_region, cv2.MORPH_OPEN, kernel,iterations=2)
    res = cv2.bitwise_and(f,f,mask=skin_region)
    cv2.imshow("output skin region", res)
        
    return res

def FindObject(hsv,frame,an_object):
    #doing it this way for customization down the road
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    if an_object=="paper":
        #ideally these will be defined by a file that has these values stored,
        #or more ideally the file will define all the code.[(12, 16, 175), (27, 36, 
        mask1 = cv2.inRange(hsv, (0, 0, 160), (188, 20, 240))
        mask = cv2.inRange(hsv, (0, 0, 174), (180, 255, 255))
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        mask1 = cv2.erode(mask1, None, iterations=2)
        mask1 = cv2.dilate(mask1, None, iterations=2)
        
        res1 = cv2.bitwise_and(frame,frame, mask= mask1)
        res2 = cv2.bitwise_and(frame,frame, mask= mask)
        res2 = res2 - res1
        greyfr2 = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
        _,thr2 = cv2.threshold(greyfr2,210,255,cv2.THRESH_BINARY)
        thr2 = cv2.morphologyEx(thr2, cv2.MORPH_CLOSE, kernel,iterations=2)
        thr2 = cv2.morphologyEx(thr2, cv2.MORPH_OPEN, kernel)
#        cv2.imshow('papel1',thr2)
#        cv2.imshow('papel',res2)
        return thr2
    
    elif an_object=="pencil":
#       Lower bounds: [ 69 125   0]
#       Upper bounds: [254 199 105]
#        Lower bounds: [ 15   0 120]
#        Upper bounds: [ 38 255 255]
       t = cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)
       mask2 = cv2.inRange(hsv, (13, 45, 60), (30, 176, 255))
       mask2 = cv2.erode(mask2, None, iterations=1)
       mask2 = cv2.dilate(mask2, kernel, iterations=3)
       mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=4)
       mask2 = cv2.GaussianBlur(mask2, (11, 11), 0)
#       mask3 = cv2.inRange(t, (14, 128, 74), (255, 174, 102))
       mask3 = cv2.inRange(t, (69, 125, 0), (254, 199, 105))
       mask3 = cv2.erode(mask3, None, iterations=1)
       mask3 = cv2.dilate(mask3, kernel, iterations=3)
       mask3 = cv2.morphologyEx(mask3, cv2.MORPH_CLOSE, kernel, iterations=4)
       mask3 = cv2.GaussianBlur(mask3, (11, 11), 0)
#       mask3 = mask2*mask3
       res = cv2.bitwise_and(frame,frame, mask= mask3)
       greyfr = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
       _,thr = cv2.threshold(greyfr,130,240,cv2.THRESH_BINARY)
       th2 = PersonObject(frame)
       th2 = cv2.GaussianBlur(th2, (11, 11), 0)
       th2 = cv2.dilate(th2, kernel, iterations=3)
       _,th2 = cv2.threshold(cv2.cvtColor(th2,cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY)
#       thr = thr - th2
#       cv2.imshow('pencdil',thr)
#       cv2.imshow('pencil',res)
       return thr
    
    elif an_object =="keyboard":
#        t = cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)
#        mask2 = cv2.inRange(t, (45, 115, 132), (116, 122, 146))
#        mask2 = cv2.inRange(hsv, (94, 0, 9), (151, 124, 159))
        min_YCrCb = np.array([0, 120,  127],np.uint8)
        max_YCrCb = np.array([144, 255, 177],np.uint8)
        # Convert image to YCrCb
        image_YCrCb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)
        
            # Find region with skin tone in YCrCb image
        dark_region = cv2.inRange(image_YCrCb,min_YCrCb,max_YCrCb)
        res3 = cv2.erode(dark_region, None, iterations=2)
        res3 = cv2.dilate(res3, None, iterations=2)
        res3 = cv2.bitwise_and(frame,frame, mask = res3)
        
        greyfr3 = cv2.cvtColor(res3,cv2.COLOR_BGR2GRAY)
#        thr3 = cv2.morphologyEx(thr3, cv2.MORPH_OPEN, kernel, iterations = 2)
        _,thr3 = cv2.threshold(greyfr3,0,255,cv2.THRESH_BINARY)
        return thr3,greyfr3
    
    elif an_object =="table":
#        mask2 = cv2.inRange(hsv, (60, 0, 205), (122, 120, 255))
#        mask = cv2.inRange(hsv, (0, 0, 248), (108, 246, 255))
#        mask = cv2.erode(mask, None, iterations=2)
#         mask = cv2.inRange(hsv, (87, 0, 188), (169, 87, 255))
         #mask2 = cv2.erode(mask, None, iterations=1)
#         mask = cv2.dilate(mask, None, iterations=4)
         min_YCrCb = np.array([181, 109, 127],np.uint8)
         max_YCrCb = np.array([255, 135, 160],np.uint8)
        # Convert image to YCrCb
         image_YCrCb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)
        
            # Find region with skin tone in YCrCb image
         mask = cv2.inRange(image_YCrCb,min_YCrCb,max_YCrCb)
         res2 = cv2.bitwise_and(frame,frame, mask= mask)
         greyfr3 = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
         _,thr3 = cv2.threshold(greyfr3,0,255,cv2.THRESH_BINARY)
         thr3 = cv2.morphologyEx(thr3, cv2.MORPH_CLOSE, kernel, iterations = 3)
         thr3 = cv2.morphologyEx(thr3, cv2.MORPH_OPEN, kernel,iterations = 2)
         return thr3
    
def InteractObject(frame,thresh1,thresh2,hsv,ang,obj_names,centers,hist_list, threh3 = None, KNN = None):
    fr = np.copy(frame)
    padding = 50
    
    if obj_names == "pencil,paper": 
        box_size = 100
        temp = np.copy(threh3)
        _, contour, _ = cv2.findContours(temp.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for c in contour:
            hull = cv2.convexHull(c)
            cv2.drawContours(temp,[hull],-1,(255,255,255),cv2.FILLED)
        mask3 = np.empty_like(thresh1)
        #needs to depend on the height of the table
        mask3[70:] = 255
        mask3[:70] = 0
        
        t= cv2.bitwise_and(temp,temp,mask=mask3)
        th = cv2.dilate(t, kernel2, iterations=2)
        _,th = cv2.threshold(th,0,255,cv2.THRESH_BINARY)
        #mask everything outside of the table
        thrsh11 = cv2.bitwise_and(th,thresh1)
        _, contour, _ = cv2.findContours(thrsh11.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for c in contour:
            if cv2.contourArea(c) > 2000 or cv2.contourArea(c) < 40:
                continue   
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        		# compute the bounding box of the contour and use the
    			# bounding box to determine the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)            
            angle = int(np.math.atan((y-y+h)/(x-x+w))*180/np.math.pi) 
#            cv2.rectangle(fr, (cX-int(box_size/2)-10, cY-int(box_size/2)-),
#                              (cX +int(box_size/2)+10, cY + int(box_size/2)+100), (0, 255, 0), 2)
#                               
            
            if ar > 0.20 and ar < 0.95 or ar > 1.05 and ar < 2.7: 
                #since the centroids vary very little we'll just compare for the closest one
                #OR we'll append, this gives us the index of the histogram stack
                c = np.array(centers)
                if c.size != 0:
                    c1 =  np.array([item[0] for item in c])
                    index = (c1<cX+padding)*(c1>cX-padding)*(c[:,1]<cY+padding)*(c[:,1]>cY-padding)
                    ex=np.where(index) [0]
                    if ex.size==0:
                        if np.sum(thresh2[cY-int(box_size/2):cY+int(box_size/2)+70,cX-int(box_size/2)-10:cX+int(box_size/2)+10])>0:
                            tocalc=hsv[cY-int(box_size/2):cY+int(box_size/2),cX-int(box_size/2):cX+int(box_size/2)]
                            antocalc=ang[cY-int(box_size/2):cY+int(box_size/2),cX-int(box_size/2):cX+int(box_size/2)]
                            if np.sum(tocalc) > 0:
                                bins=(9, 9)
                                hi,_ = np.histogram(tocalc, bins = 10, normed = True)
                                an,_ = np.histogram(antocalc, bins = 10, normed = True)  
                                
#                                hi[0] = hi[0]*np.cos(an[0])
#                                hi[1] = hi[1]*np.sin(an[1])
#                                
#                                if (np.mean(hi[1:]) <= 0.05):
#                                    return fr,centers,hist_list
                                
#                                hi = hi/hi.ravel().sum()
#                                an = an/an.ravel().sum()  
#                                print(hi)                              
                                hi = hi.flatten()
                                an=an.flatten()
#                                hi = np.array(cv2.normalize(hi,None,0,255,cv2.NORM_MINMAX))      
#                                an = np.array(cv2.normalize(an,None,0,255,cv2.NORM_MINMAX))
                                hi = np.append(hi,an)
                                
                                hi[np.isnan(hi)] = 10**-10
                                
                                cv2.rectangle(fr, (cX-int(box_size/2), cY-int(box_size/2)),
                                              (cX +int(box_size/2), cY + int(box_size/2)), (0, 255, 0), 2)
                                cv2.rectangle(fr, (cX-int(box_size/2), cY-int(box_size/2)),
                                              (cX-int(box_size/2) + len("pencil?") * 6, cY-int(box_size/2) - 10), (0, 0, 255), -1, cv2.LINE_AA)
                                cv2.putText(fr, "pencil?", (cX-int(box_size/2) , cY-int(box_size/2)), font, 0.3, (255, 255, 255), 1)
                                hist_list.append([hi])
                                centers.append([cX,cY])
                    else:         
                        ex = np.where(index) [0][0] 
                        cY = centers[ex][1] 
                        cX = centers[ex][0]
                        if np.sum(thresh2[cY-int(box_size/2):cY+int(box_size/2)+70,cX-int(box_size/2)-10:cX+int(box_size/2)+10])>0:
                            tocalc=hsv[cY-int(box_size/2):cY+int(box_size/2),cX-int(box_size/2):cX+int(box_size/2)]
                            antocalc=ang[cY-int(box_size/2):cY+int(box_size/2),cX-int(box_size/2):cX+int(box_size/2)]
                            if np.sum(tocalc) > 0:
                                bins=(9, 9)
#                                hi = cv2.calcHist(tocalc, [0, 1], None, bins, [ 0, 30, 0, 30])
#                                an = cv2.calcHist(antocalc, [0, 1], None, bins, [ 0, 2*np.pi, 0, 2*np.pi])  
#                                
#                                hi[0] = hi[0]*np.cos(an[0])
#                                hi[1] = hi[1]*np.sin(an[1])
#                                
#                                if (np.mean(hi[1:]) <= 0.05):
#                                    return fr,centers,hist_list
#                                
#                                hi = hi/hi.ravel().sum()
#                                an = an/an.ravel().sum()  
#                                hi = hi.flatten()
#                                
#                                hi[np.isnan(hi)] = 10**-10
                                hi,_ = np.histogram(tocalc, bins = 10, normed = True)
                                an,_ = np.histogram(antocalc, bins = 10, normed = True)  
                                
#                                hi[0] = hi[0]*np.cos(an[0])
#                                hi[1] = hi[1]*np.sin(an[1])
#                                
#                                if (np.mean(hi[1:]) <= 0.05):
#                                    return fr,centers,hist_list
                                
#                                hi = hi/hi.ravel().sum()
#                                an = an/an.ravel().sum()  
#                                print(hi)                              
                                hi = hi.flatten()
                                an=an.flatten()
#                                hi = np.array(cv2.normalize(hi,None,0,255,cv2.NORM_MINMAX))      
#                                an = np.array(cv2.normalize(an,None,0,255,cv2.NORM_MINMAX))
                                hi = np.append(hi,an)
                                cv2.rectangle(fr, (cX-int(box_size/2), cY-int(box_size/2)), 
                                              (cX +int(box_size/2), cY + int(box_size/2)), (0, 255, 0), 2)
                                cv2.rectangle(fr, (cX-int(box_size/2), cY-int(box_size/2)),
                                              (cX-int(box_size/2) + len("pencil?") * 6, cY-int(box_size/2) - 10), (0, 0, 255), -1, cv2.LINE_AA)
                                cv2.putText(fr, "pencil?", (cX-int(box_size/2) , cY-int(box_size/2)), font, 0.3, (255, 255, 255), 1)

                               #keep track of feature using the centroid, don't want to mix up the training features
                                hist_list[ex].append(hi)
                else:                            
                           #keep track of feature using the centroid, don't want to mix up the training features
                    if np.sum(thresh2[cY-int(box_size/2):cY+int(box_size/2)+70,cX-int(box_size/2)-10:cX+int(box_size/2)+10])>0:
                            tocalc=hsv[cY-int(box_size/2):cY+int(box_size/2),cX-int(box_size/2):cX+int(box_size/2)]
                            antocalc=ang[cY-int(box_size/2):cY+int(box_size/2),cX-int(box_size/2):cX+int(box_size/2)]
                            if np.sum(tocalc) > 0:
                                bins=(9, 9)
#                                hi = cv2.calcHist(tocalc, [0, 1], None, bins, [ 0, 30, 0, 30])
#                                an = cv2.calcHist(antocalc, [0, 1], None, bins, [ 0, 2*np.pi, 0, 2*np.pi])  
#                                
#                                hi[0] = hi[0]*np.cos(an[0])
#                                hi[1] = hi[1]*np.sin(an[1])
#                                
#                                if (np.mean(hi[1:]) <= 0.05):
#                                    return fr,centers,hist_list
#                                
#                                hi = hi/hi.ravel().sum()
#                                an = an/an.ravel().sum()  
#                                hi = hi.flatten()
#                                
#                                hi[np.isnan(hi)] = 10**-10
                                hi,_ = np.histogram(tocalc, bins = 10, normed = True)
                                an,_ = np.histogram(antocalc, bins = 10, normed = True)  
                                
#                                hi[0] = hi[0]*np.cos(an[0])
#                                hi[1] = hi[1]*np.sin(an[1])
#                                
#                                if (np.mean(hi[1:]) <= 0.05):
#                                    return fr,centers,hist_list
                                
#                                hi = hi/hi.ravel().sum()
#                                an = an/an.ravel().sum()  
#                                print(hi)                              
                                hi = hi.flatten()
                                an = an.flatten()
#                                hi = np.array(cv2.normalize(hi,None,0,255,cv2.NORM_MINMAX))      
#                                an = np.array(cv2.normalize(an,None,0,255,cv2.NORM_MINMAX))
                                hi = np.append(hi,an)
                                cv2.rectangle(fr, (cX-int(box_size/2), cY-int(box_size/2)), 
                                              (cX +int(box_size/2), cY + int(box_size/2)), (0, 255, 0), 2)
                                cv2.rectangle(fr, (cX-int(box_size/2), cY-int(box_size/2)),
                                              (cX-int(box_size/2) + len("pencil?") * 6, cY-int(box_size/2) - 10), (0, 0, 255), -1, cv2.LINE_AA)
                                cv2.putText(fr, "pencil?", (cX-int(box_size/2) , cY-int(box_size/2)), font, 0.3, (255, 255, 255), 1)
                                
                                hist_list.append([hi])
                                centers.append([cX,cY])

        return fr,centers,hist_list
    
    
    elif obj_names == "keyboard,table": 
        #need to mask anything outside of the table...
        fr = np.copy(frame)
        thrtemp = np.copy(thresh2)
        mask3 = np.empty_like(thresh1)
        #needs to depend on the height of the table
        mask3[100:] = 255
        mask3[:100] = 0
        thr = np.copy(thresh1)
        t= cv2.bitwise_and(thrtemp,thrtemp,mask=mask3)
        _,th = cv2.threshold(t,0,255,cv2.THRESH_BINARY)
        th = cv2.erode(th, kernel2, iterations = 1)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel2, iterations = 13)
        _, contour, _ = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for c in contour:
            hull = cv2.convexHull(c)
            cv2.drawContours(th,[hull],-1,(255,255,255),cv2.FILLED)

        #fill in the holes so we don't lose information
#        cv2.imshow('keyboard',thr)
#        cv2.imshow('keyboard',th)
        #mask everything outside of the table
        thrsh11 = cv2.bitwise_and(th,thr)
        thrsh11 = cv2.morphologyEx(thrsh11, cv2.MORPH_CLOSE, None, iterations = 2)
        thrsh11 = cv2.morphologyEx(thrsh11, cv2.MORPH_OPEN, None, iterations = 2)
#        thrsh11 = cv2.dilate(thrsh11, None, iterations=1)
        thrsh112 = cv2.bitwise_and(fr,fr,mask=thrsh11)
        thrsh11 = cv2.morphologyEx(thrsh11, cv2.MORPH_OPEN, None, iterations = 2)
        thrsh11 = cv2.morphologyEx(thrsh11, cv2.MORPH_CLOSE, None, iterations = 2)
#        cv2.imshow('pls be right', thrsh11)
        keyboard = cv2.cvtColor(thrsh112, cv2.COLOR_BGR2GRAY)
        person = PersonObject(fr ) 
#        person = cv2.GaussianBlur(person, (11, 11), 0)
        _,p1 = cv2.threshold(person,0,255,cv2.THRESH_BINARY)
        _, contour, _ = cv2.findContours(thrsh11.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for c in contour:
            if cv2.contourArea(c) > 100000 or cv2.contourArea(c) < 200:
                continue   
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            
        		# compute the bounding box of the contour and use the
    			# bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            angle = int(np.math.atan((y-y+h)/(x-x+w))*180/np.math.pi) 
            #trying something new here...
            #look for hands inside the keyboard
            hand_check = np.sum(p1[y-10:y+h+10,x-10:x+w+10])
            if len(approx) >= 3 and hand_check > 50:    
#                person2 = cv2.GaussianBlur(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY),(11,11),0)
                gray = cv2.cvtColor(frame[y:y+h,x:x+w],cv2.COLOR_BGR2GRAY)
                max_height = gray.shape[0]
                max_width = gray.shape[1]
                height = 128
                width = 128
                if max_height < height or max_width < width:
                    # get scaling factor
                    scaling_factor = max_height / float(height)
                    max_height = 128
                    if max_width/float(width) < scaling_factor:
                        scaling_factor = max_width / float(width)
                        max_width = 128
                    # resize image
                    gray = cv2.resize(gray, None, fx=scaling_factor, fy=scaling_factor)
                # fill excess by zero padding the image
                zero_padded = np.zeros((height,width,3), np.uint8)
                zero_padded[:max_height,:max_width] = gray
#                person1  = cv2.GaussianBlur(gray,(3,3),0)
#                person2 = cv2.adaptiveThreshold(person1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                               cv2.THRESH_BINARY,21,2)
                person2 = cv2.medianBlur(zero_padded,5)
#                hist = cv2.calcHist(person2[y:y+h,x:x+w],[0,1],None,[256,256],[0,256,0,256])
        
#                hist = hist/hist.ravel().sum()
                person2 = person2.reshape(1, -1)
                p = KNN.predict(person2)
                
                if p[0] == 0:
                    if np.sum(hsv[y-30:y+h+30,x-30:x+w+30]) > 0:
                        tocalc=hsv[y-30:y+h+30,x-30:x+w+30] 
                        antocalc =ang[y-30:y+h+30,x-30:x+w+30]
                        if np.sum(tocalc) > 0:
                            bins=(9, 9)
#                           hi = cv2.calcHist(tocalc, [0, 1], None, bins, [ 0, 30, 0, 30])
#                           an = cv2.calcHist(antocalc, [0, 1], None, bins, [ 0, 2*np.pi, 0, 2*np.pi])  
#                                
#                           hi[0] = hi[0]*np.cos(an[0])
#                           hi[1] = hi[1]*np.sin(an[1])
#                                
#                           if (np.mean(hi[1:]) <= 0.05):
#                                return fr,centers,hist_list
#                                
#                           hi = hi/hi.ravel().sum()
#                           an = an/an.ravel().sum()  
#                           hi = hi.flatten()
#                                
#                           hi[np.isnan(hi)] = 10**-10
                            hi,_ = np.histogram(tocalc, bins = 10, normed = True)
                            an,_ = np.histogram(antocalc, bins = 10, normed = True)  
                            
#                             
                            hi = hi.flatten()
                            an=an.flatten()
                            hi = np.append(hi,an)
                            hist_list.append(hi)
                            cv2.rectangle(fr, (x-30, y-30), 
                             (x + w+30, y + h+30), (255, 255, 0), 2)
                            cv2.rectangle(fr, (x-30,y-30),
                                     (x-30 + len("typing?") * 6, y-30 - 10), (0, 0, 255), -1, cv2.LINE_AA)
                            cv2.putText(fr, "typing?", (x-30, y-30), font, 0.3, (255, 255, 255), 1)
                            
                elif p[0] == 1:
                    cv2.rectangle(fr, (x-30, y-30), 
                             (x + w+30, y + h+30), (255, 255, 0), 2)
                    cv2.rectangle(fr, (x-30,y-30),
                             (x-30 + len("monitor?") * 6, y-30 - 10), (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.putText(fr, "monitor?", (x-30, y-30), font, 0.3, (255, 255, 255), 1)

                    pass
                elif p[0] == 2:
                    cv2.rectangle(fr, (x-30, y-30), 
                             (x + w+30, y + h+30), (255, 255, 0), 2)
                    cv2.rectangle(fr, (x-30,y-30),
                             (x-30 + len("mouse?") * 6, y-30 - 10), (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.putText(fr, "mouse?", (x-30, y-30), font, 0.3, (255, 255, 255), 1)
                    pass
    
                    '''
                    todo: combine overlapping boxes
                    '''
                        #since the centroids vary very little we'll just compare for the closest one
                        #OR we'll append, this gives us the index of the histogram stack
#                        c = np.array(centers)
#                        if c.size != 0:
#                            c1 =  np.array([item[0] for item in c])
#                            index = (c1<cX+padding)*(c1>cX-padding)*(c[:,1]<cY+padding)*(c[:,1]>cY-padding)
#                            ex=np.where(index) [0]
#                            if ex.size==0:
#                                centers.append([cX,cY])
#                                hist_list.append([h_normalized])
#                            else:         
#                                ex = np.where(index) [0][0]  
#                                   #keep track of feature using the centroid, don't want to mix up the training features
#                                hist_list[ex].append(h_normalized)
#                        else:                            
#                                   #keep track of feature using the centroid, don't want to mix up the training features
#                            hist_list.append([h_normalized])
#                            centers.append([cX,cY])

        return fr,centers,hist_list
        
    elif obj_names == "talking":   
        fr = np.copy(frame)
        person = PersonObject(fr )
        person = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
        _,person = cv2.threshold(person,0,255,cv2.THRESH_BINARY)
        #do a knn and find the face, then subdivide it and check the bottom half for motion.
        _, contour, _ = cv2.findContours(person.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        p=[-1]
        for c in contour:
            if cv2.contourArea(c) > 100000 or cv2.contourArea(c) < 1200:
                continue   
            p = [-1]
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
             
        		# compute the bounding box of the contour and use the
    			# bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            angle = int(np.math.atan((y-y+h)/(x-x+w))*180/np.math.pi) 
            #trying something new here...
            
            if ar <= 1.5 and ar >= 0.50:    
                
                gray = cv2.cvtColor(frame[y:y+h,x:x+w] ,cv2.COLOR_BGR2GRAY)
                max_height = gray.shape[0]
                max_width = gray.shape[1]
                height = 128
                width = 128
                if max_height < height or max_width < width:
                    # get scaling factor
                    scaling_factor = max_height / float(height)
                    max_height = 128
                    if max_width/float(width) < scaling_factor:
                        scaling_factor = max_width / float(width)
                        max_width = 128
                    # resize image
                    gray = cv2.resize(gray, None, fx=scaling_factor, fy=scaling_factor)
                # fill excess by zero padding the image
                zero_padded = np.zeros((height,width,3), np.uint8)
                zero_padded[:max_height,:max_width] = gray
                
#                person1  = cv2.GaussianBlur(gray,(3,3),0)
#                person2 = cv2.adaptiveThreshold(person1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                               cv2.THRESH_BINARY,21,2)
                person2 = cv2.medianBlur(zero_padded,5)
#                hist = cv2.calcHist(person2[y:y+h,x:x+w],[0,1],None,[256,256],[0,256,0,256])
        
#                hist = hist/hist.ravel().sum()
                person2 = person2.reshape(1, -1)
                p = KNN.predict(person2)
                
                if p[0] == 0:
                    cv2.rectangle(fr, (x, y), 
                              (x + w, y + h), (255, 0, 0), 2)
                    cv2.rectangle(fr, (x,y),
                              (x + len("face") * 6, y  - 10), (0, 255, 0), -1, cv2.LINE_AA)
                    cv2.putText(fr, "face", (x , y ), font, 0.3, (255, 255, 255), 1)
                    #Mouth is 2*1.618 down the face (golden ratio)
                    tocalc=hsv[y+int(h/1.614):y+h,x:x+w] 
                    antocalc=ang[y+int(h/1.614):y+h,x:x+w]
                    if np.sum(tocalc) > 0:
                        bins=(9, 9)
#                        hi = cv2.calcHist(tocalc, [0, 1], None, bins, [ 0, 30, 0, 30])
#                        an = cv2.calcHist(antocalc, [0, 1], None, bins, [ 0, 2*np.pi, 0, 2*np.pi])  
                        cv2.rectangle(fr, (x, y+int(h/1.614)), 
                              (x + w, y + h), (0, 0, 255), 2)       
#                        hi[0] = hi[0]*np.cos(an[0])
#                        hi[1] = hi[1]*np.sin(an[1])
#                                
#                        if (np.mean(hi[1:]) <= 0.05):
#                            return fr,centers,hist_list
#                                
#                        hi = hi/hi.ravel().sum()
#                        an = an/an.ravel().sum()  
#                        hi = hi.flatten()
#                                
#                        hi[np.isnan(hi)] = 10**-10
                        hi,_ = np.histogram(tocalc, bins = 10, normed = True)
                        an,_ = np.histogram(antocalc, bins = 10, normed = True)  
                            
#                             
                        hi = hi.flatten()
                        an=an.flatten()
                        hi = np.append(hi,an)
                        hist_list.append(hi)
                elif p[0] == 1:
                    continue
#                    cv2.rectangle(fr, (x, y), 
#                              (x + w, y + h), (255, 0, 255), 2)
#                    cv2.rectangle(fr, (x,y),
#                              (x + len("noface") * 6, y  - 10), (0, 0, 255), -1, cv2.LINE_AA)
#                    cv2.putText(fr, "noface", (x , y ), font, 0.3, (255, 255, 255), 1)
            
        return fr,centers,hist_list

def LabelObject(frame,thresh1,thresh2,hsv,ang,obj_names,kmeans, hist, p, centers, threh3 = None, KNN = None, t = None, tester = None):
    fr = np.copy(frame)
    padding = 50
    
    if obj_names == "pencil,paper":
        box_size = 75
        temp = np.copy(threh3)
        _, contour, _ = cv2.findContours(temp.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for c in contour:
            hull = cv2.convexHull(c)
            cv2.drawContours(temp,[hull],-1,(255,255,255),cv2.FILLED)
        mask3 = np.empty_like(thresh1)
        #needs to depend on the height of the table
        mask3[70:] = 255
        mask3[:70] = 0
        
        t= cv2.bitwise_and(temp,temp,mask=mask3)
        th = cv2.dilate(t, kernel2, iterations=2)
        _,th = cv2.threshold(th,0,255,cv2.THRESH_BINARY)
        #mask everything outside of the table
        thrsh11 = cv2.bitwise_and(th,thresh1)
        _, contour, _ = cv2.findContours(thrsh11.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for c in contour:
            if cv2.contourArea(c) > 400 or cv2.contourArea(c) < 40:
                continue   
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        		# compute the bounding box of the contour and use the
    			# bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            angle = int(np.math.atan((y-y+h)/(x-x+w))*180/np.math.pi) 
            #trying something new here...
            
            if ar > 0.20 and ar < 0.95 or ar > 1.05 and ar < 2.7:    
                #cv2.rectangle(fr, (x-20, y-20), (x + w+20, y + h+20), (0, 255, 255), 2)
                #since the centroids vary very little we'll just compare for the closest one
                #OR we'll append, this gives us the index of the histogram stack     
                ex = -1 # just in case                   
                c = np.array(centers)
                if c.size != 0:
                    c1 =  np.array([item[0] for item in c])
                    index = (c1<cX+padding)*(c1>cX-padding)*(c[:,1]<cY+padding)*(c[:,1]>cY-padding)
                    ex=np.where(index) [0]
                    if ex.size==0:
                        if np.sum(thresh2[cY-int(box_size/2):cY+int(box_size/2)+70,cX-int(box_size/2)-10:cX+int(box_size/2)+10])>0:
                            tocalc=hsv[cY-int(box_size/2):cY+int(box_size/2),cX-int(box_size/2):cX+int(box_size/2)]
                            antocalc=ang[cY-int(box_size/2):cY+int(box_size/2),cX-int(box_size/2):cX+int(box_size/2)]
                                     
                            if np.sum(tocalc) > 0:
                                bins=(9, 9)
        #                        hi = cv2.calcHist(tocalc, [0, 1], None, bins, [ 0, 30, 0, 30])
        #                        an = cv2.calcHist(antocalc, [0, 1], None, bins, [ 0, 2*np.pi, 0, 2*np.pi])  
        #                        
        #                        hi[0] = hi[0]*np.cos(an[0])
        #                        hi[1] = hi[1]*np.sin(an[1])
        #                        
        #                        if (np.mean(hi[1:]) <= 0.05):
        #                            return fr, hist, p,centers
        #                        
        #                        hi = hi/hi.ravel().sum()
        #                        an = an/an.ravel().sum()   
                                hi,_ = np.histogram(tocalc, bins = 10, normed = True)
                                an,_ = np.histogram(antocalc, bins = 10, normed = True)  
                                hi = hi.flatten()
                                an = an.flatten()
                                hi = np.append(hi,an)
#                        
#                               hi[np.isnan(hi)] = 10**-10
                                hist.append([hi])
                                centers.append([cX,cY])
                        ex = len(centers)-1
                        p.append(-1)
                    else:         
                        ex = np.where(index) [0][0] 
                        cY = centers[ex][1] 
                        cX = centers[ex][0] 
                           #keep track of feature using the centroid, don't want to mix up the training features
                        if np.sum(thresh2[cY-int(box_size/2):cY+int(box_size/2)+70,cX-int(box_size/2)-10:cX+int(box_size/2)+10])>0:
                            tocalc=hsv[cY-int(box_size/2):cY+int(box_size/2),cX-int(box_size/2):cX+int(box_size/2)]
                            antocalc=ang[cY-int(box_size/2):cY+int(box_size/2),cX-int(box_size/2):cX+int(box_size/2)]
                                     
                            if np.sum(tocalc) > 0:
                                bins=(9, 9)
        #                        hi = cv2.calcHist(tocalc, [0, 1], None, bins, [ 0, 30, 0, 30])
        #                        an = cv2.calcHist(antocalc, [0, 1], None, bins, [ 0, 2*np.pi, 0, 2*np.pi])  
        #                        
        #                        hi[0] = hi[0]*np.cos(an[0])
        #                        hi[1] = hi[1]*np.sin(an[1])
        #                        
        #                        if (np.mean(hi[1:]) <= 0.05):
        #                            return fr, hist, p,centers
        #                        
        #                        hi = hi/hi.ravel().sum()
        #                        an = an/an.ravel().sum()   
                                hi,_ = np.histogram(tocalc, bins = 10, normed = True)
                                an,_ = np.histogram(antocalc, bins = 10, normed = True)  
                                hi = hi.flatten()
                                an = an.flatten()
                                hi = np.append(hi,an)
        #                        
    #                        hi[np.isnan(hi)] = 10**-10
                        
                                if len(hist[ex]) < 3: 
                                    cv2.rectangle(fr, (cX-int(box_size/2), cY-int(box_size/2)), 
                                                  (cX +int(box_size/2), cY + int(box_size/2)), (0, 255, 0), 2)
                                    cv2.rectangle(fr, (cX-int(box_size/2), cY-int(box_size/2)),
                                                  (cX-int(box_size/2) + len("??") * 6, cY-int(box_size/2) - 10), (0, 0, 255), -1, cv2.LINE_AA)
                                    cv2.putText(fr, "??", (cX-int(box_size/2) , cY-int(box_size/2)), font, 0.3, (255, 255, 255), 1)
                
                                    hist[ex].append(hi)
                                else:
                                    
                                    histy= np.array(np.mean(hist[ex], axis = 0))
                                    X =  np.vstack([histy]) 
#                                    X =  np.vstack([hist[ex]]) 
                                        
                                    if tester == "knn":    
#                                        scaler = StandardScaler()
#                                        data = scaler.fit_transform(X.astype(np.float64))
                                        pred = kmeans.predict(X)
                                    else:
                                        pred = kmeans.predict(X)
                                    p[ex] = int(np.around(np.sum(pred)/len(pred)))
                                    hist[ex].clear()                         
                                    hist[ex].append(hi)
                else:                            
                    #keep track of feature using the centroid, don't want to mix up the training features
                    if np.sum(thresh2[cY-int(box_size/2):cY+int(box_size/2)+70,cX-int(box_size/2)-10:cX+int(box_size/2)+10])>0:
            
                        tocalc=hsv[cY-int(box_size/2):cY+int(box_size/2),cX-int(box_size/2):cX+int(box_size/2)]
                        antocalc=ang[cY-int(box_size/2):cY+int(box_size/2),cX-int(box_size/2):cX+int(box_size/2)]
                                 
                        if np.sum(tocalc) > 0:
                            bins=(9, 9)
    #                        hi = cv2.calcHist(tocalc, [0, 1], None, bins, [ 0, 30, 0, 30])
    #                        an = cv2.calcHist(antocalc, [0, 1], None, bins, [ 0, 2*np.pi, 0, 2*np.pi])  
    #                        
    #                        hi[0] = hi[0]*np.cos(an[0])
    #                        hi[1] = hi[1]*np.sin(an[1])
    #                        
    #                        if (np.mean(hi[1:]) <= 0.05):
    #                            return fr, hist, p,centers
    #                        
    #                        hi = hi/hi.ravel().sum()
    #                        an = an/an.ravel().sum()   
                            hi,_ = np.histogram(tocalc, bins = 10, normed = True)
                            an,_ = np.histogram(antocalc, bins = 10, normed = True)  
                            hi = hi.flatten()
                            an = an.flatten()
                            hi = np.append(hi,an)
#                        
    #                        hi[np.isnan(hi)] = 10**-10
                            hist.append([hi])
                            centers.append([cX,cY])
                    p.append(-1)
                    ex = len(centers)-1
                if p[ex] == 0:                     
                    cv2.rectangle(fr, (cX-int(box_size/2), cY-int(box_size/2)),
                                       (cX +int(box_size/2), cY + int(box_size/2)), (0, 255, 255), 2)
                    cv2.rectangle(fr, (cX-int(box_size/2), cY-int(box_size/2)),
                                      (cX-int(box_size/2) + len("Writing") * 6, cY-int(box_size/2) - 10), (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.putText(fr, "Writing", (cX-int(box_size/2) , cY-int(box_size/2)), font, 0.3, (255, 255, 255), 1)
                elif p[ex] == 1:
                    cv2.rectangle(fr, (cX-int(box_size/2), cY-int(box_size/2)),
                                       (cX +int(box_size/2), cY + int(box_size/2)), (0, 0, 255), 2)
                    cv2.rectangle(fr, (cX-int(box_size/2), cY-int(box_size/2)),
                                      (cX-int(box_size/2) + len("Not Writing") * 6, cY-int(box_size/2) - 10), (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.putText(fr, "Not Writing", (cX-int(box_size/2) , cY-int(box_size/2)), font, 0.3, (255, 255, 255), 1)
                            
        return fr, hist, p, centers
    
    elif obj_names == "keyboard,table": 
        fr = np.copy(frame)
        thrtemp = np.copy(thresh2)
        mask3 = np.empty_like(thresh1)
        #needs to depend on the height of the table
        mask3[100:] = 255
        mask3[:100] = 0
        thr = np.copy(thresh1)
        thhh= cv2.bitwise_and(thrtemp,thrtemp,mask=mask3)
        _,th = cv2.threshold(thhh,0,255,cv2.THRESH_BINARY)
        th = cv2.erode(th, kernel2, iterations = 1)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel2, iterations = 13)
        _, contour, _ = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for c in contour:
            hull = cv2.convexHull(c)
            cv2.drawContours(th,[hull],-1,(255,255,255),cv2.FILLED)

        #fill in the holes so we don't lose information
#        cv2.imshow('keyboard',thr)
#        cv2.imshow('keyboard',th)
        #mask everything outside of the table
        thrsh11 = cv2.bitwise_and(th,thr)
        thrsh11 = cv2.morphologyEx(thrsh11, cv2.MORPH_CLOSE, None, iterations = 2)
        thrsh11 = cv2.morphologyEx(thrsh11, cv2.MORPH_OPEN, None, iterations = 2)
#        thrsh11 = cv2.dilate(thrsh11, None, iterations=1)
        thrsh112 = cv2.bitwise_and(fr,fr,mask=thrsh11)
        thrsh11 = cv2.morphologyEx(thrsh11, cv2.MORPH_OPEN, None, iterations = 2)
        thrsh11 = cv2.morphologyEx(thrsh11, cv2.MORPH_CLOSE, None, iterations = 5)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel2, iterations = 13)
        cv2.imshow('pls be right', thrsh11)
        keyboard = cv2.cvtColor(thrsh112, cv2.COLOR_BGR2GRAY)
        person = PersonObject(fr ) 
#        person = cv2.GaussianBlur(person, (11, 11), 0)
        _,p1 = cv2.threshold(person,0,255,cv2.THRESH_BINARY)
        _, contour, _ = cv2.findContours(thrsh11.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for c in contour:
            if cv2.contourArea(c) > 8000 or cv2.contourArea(c) < 200:
                continue   
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            
        		# compute the bounding box of the contour and use the
    			# bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            angle = int(np.math.atan((y-y+h)/(x-x+w))*180/np.math.pi) 
            #trying something new here...
            #look for hands inside the keyboard
            hand_check = np.sum(p1[y-10:y+h+10,x-10:x+w+10])
            if len(approx) >= 3 and hand_check > 50:    
#                person2 = cv2.GaussianBlur(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY),(11,11),0)
                gray = cv2.cvtColor(frame[y:y+h,x:x+w],cv2.COLOR_BGR2GRAY)
                person2 = cv2.resize(gray, (128,128))
#                person1  = cv2.GaussianBlur(gray,(3,3),0)
#                person2 = cv2.adaptiveThreshold(person1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                               cv2.THRESH_BINARY,21,2)
                person2 = cv2.medianBlur(person2,5)
#                hist = cv2.calcHist(person2[y:y+h,x:x+w],[0,1],None,[256,256],[0,256,0,256])
        
#                hist = hist/hist.ravel().sum()
                person2 = person2.reshape(1, -1)
                p = KNN.predict(person2)
                if p[0] == 0:                    
                    if np.sum(hsv[y-20:y+h+20,x-20:x+w+20]) > 0:
                        tocalc=hsv[y-30:y+h+30,x-30:x+w+30] 
                        antocalc =ang[y-30:y+h+30,x-30:x+w+30]
                        if np.sum(tocalc) > 0 :
                            bins=(9, 9)

#                           hi = cv2.calcHist(tocalc, [0, 1], None, bins, [ 0, 30, 0, 30])
#                           an = cv2.calcHist(antocalc, [0, 1], None, bins, [ 0, 2*np.pi, 0, 2*np.pi])  
#                                
#                           hi[0] = hi[0]*np.cos(an[0])
#                           hi[1] = hi[1]*np.sin(an[1])
#                                
#                           if (np.mean(hi[1:]) <= 0.05):
#                                return fr,centers,hist_list
#                                
#                           hi = hi/hi.ravel().sum()
#                           an = an/an.ravel().sum()  
#                           hi = hi.flatten()
#                                
#                           hi[np.isnan(hi)] = 10**-10
                            hi,_ = np.histogram(tocalc, bins = 10, normed = True)
                            an,_ = np.histogram(antocalc, bins = 10, normed = True)  
                            
#                             
                            hi = hi.flatten()
                            an=an.flatten()
                            hi = np.append(hi,an)
                        
                            c = np.array(centers)
                            if c.size != 0:
                                c1 =  np.array([item[0] for item in c])
                                index = (c1<cX+padding)*(c1>cX-padding)*(c[:,1]<cY+padding)*(c[:,1]>cY-padding)
                                ex=np.where(index) [0]
                                if ex.size==0:
                                    centers.append([cX,cY])
                                    hist.append([hi])
                                    ex = len(centers)-1
                                    t.append(1)
                                else:         
                                    ex = np.where(index) [0][0]  
                                       #keep track of feature using the centroid, don't want to mix up the training features
                                    if len(hist[ex]) < 3:                            
                                        hist[ex].append(hi)
                                    else:
                                        
                                        X =  np.vstack([hist[ex]]) 
                                        if tester == "knn":    
                                            scaler = StandardScaler()
                                            data = scaler.fit_transform(X.astype(np.float64))
                                            pred = kmeans.predict(data)
                                        else:
                                            pred = kmeans.predict(X)
                                        t[ex] = int(np.around(np.sum(pred)/len(pred)))
                                        hist[ex].pop()                          
                                        hist[ex].append(hi)
                            else:                            
                                #keep track of feature using the centroid, don't want to mix up the training features
                                hist.append([hi])
                                t.append(1)
                                centers.append([cX,cY])
                                ex = len(centers)-1
                            if t[ex] == 0:
                                cv2.rectangle(fr, (x, y), 
                                  (x + w, y + h), (255, 0, 0), 2)
                                cv2.rectangle(fr, (x,y),
                                  (x + len("typing") * 6, y  - 10), (0, 255, 0), -1, cv2.LINE_AA)
                                cv2.putText(fr, "typing", (x , y ), font, 0.3, (255, 255, 255), 1)
                            elif t[ex] == 1:
                                cv2.rectangle(fr, (x, y), 
                                  (x + w, y + h), (255, 0, 0), 2)
                                cv2.rectangle(fr, (x,y),
                                  (x + len("not typing") * 6, y  - 10), (0, 255, 0), -1, cv2.LINE_AA)
                                cv2.putText(fr, "not typing", (x , y ), font, 0.3, (255, 255, 255), 1)
                elif p[0] == 1:
                    #continue
                    cv2.rectangle(fr, (x, y), 
                              (x + w, y + h), (255, 0, 255), 2)
                    cv2.rectangle(fr, (x,y),
                              (x + len("monitor") * 6, y  - 10), (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.putText(fr, "monitor", (x , y ), font, 0.3, (255, 255, 255), 1)
                elif p[0] == 2:
                    
                    #continue
                    cv2.rectangle(fr, (x, y), 
                              (x + w, y + h), (255, 0, 255), 2)
                    cv2.rectangle(fr, (x,y),
                              (x + len("mouse") * 6, y  - 10), (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.putText(fr, "mouse", (x , y ), font, 0.3, (255, 255, 255), 1)
                
        return fr,hist,p,centers,t
        
    elif obj_names == "talking":   
        fr = np.copy(frame)
        person = PersonObject(fr)
        person = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
        _,person = cv2.threshold(person,0,255,cv2.THRESH_BINARY)
        #do a knn and find the face, then subdivide it and check the bottom half for motion.
        _, contour, _ = cv2.findContours(person.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for c in contour:
            if cv2.contourArea(c) < 1000: #cv2.contourArea(c) > 10000 or 
                continue   
            p = [-1]
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
             
        		# compute the bounding box of the contour and use the
    			# bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            angle = int(np.math.atan((y-y+h)/(x-x+w))*180/np.math.pi) 
            #trying something new here...
            
            if ar <= 1.5 and ar >= 0.5:    
#                person2 = cv2.GaussianBlur(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY),(11,11),0)
                gray = cv2.cvtColor(frame[y:y+h,x:x+w],cv2.COLOR_BGR2GRAY)
                person2 = cv2.resize(gray, (128,128))
#                person1  = cv2.GaussianBlur(gray,(3,3),0)
#                person2 = cv2.adaptiveThreshold(person1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                               cv2.THRESH_BINARY,21,2)
                person2 = cv2.medianBlur(person2,5)
#                hist = cv2.calcHist(person2[y:y+h,x:x+w],[0,1],None,[256,256],[0,256,0,256])
        
#                hist = hist/hist.ravel().sum()
                person2 = person2.reshape(1, -1)
                p = KNN.predict(person2)
                
                if p[0] == 0:                    
                    #Mouth is 1.618 down the face (golden ratio)
                    tocalc=hsv[y+int(h/1.614):y+h,x:x+w] 
                    antocalc=ang[y+int(h/1.614):y+h,x:x+w]
                    if np.sum(tocalc) > 0:
                        bins=(9, 9)
#                        hi = cv2.calcHist(tocalc, [0, 1], None, bins, [ 0, 30, 0, 30])
#                        an = cv2.calcHist(antocalc, [0, 1], None, bins, [ 0, 2*np.pi, 0, 2*np.pi])  
#                                
#                        hi[0] = hi[0]*np.cos(an[0])
#                        hi[1] = hi[1]*np.sin(an[1])
##                                
##                        if (np.mean(hi[1:]) <= 0.05):
##                            return fr,hist,p,centers,t
#                                
#                        hi = hi/hi.ravel().sum()
#                        an = an/an.ravel().sum()  
#                        hi = hi.flatten()
#                                
#                        hi[np.isnan(hi)] = 10**-10
                        hi,_ = np.histogram(tocalc, bins = 10, normed = True)
                        an,_ = np.histogram(antocalc, bins = 10, normed = True)  
                            
#                             
                        hi = hi.flatten()
                        an=an.flatten()
                        hi = np.append(hi,an)
                        c = np.array(centers)
                        if c.size != 0:
                            c1 =  np.array([item[0] for item in c])
                            index = (c1<cX+padding)*(c1>cX-padding)*(c[:,1]<cY+padding)*(c[:,1]>cY-padding)
                            ex=np.where(index) [0]
                            if ex.size==0:
                                centers.append([cX,cY])
                                hist.append([hi])
                                ex = len(centers)-1
                                t.append(1)
                            else:         
                                ex = np.where(index) [0][0]  
                                   #keep track of feature using the centroid, don't want to mix up the training features
                                if len(hist[ex]) < 3:                            
                                    hist[ex].append(hi)
                                else:
                                    histy= np.array(np.mean(hist[ex], axis = 0))
                                    X =  np.vstack([histy]) 
                                    if tester == "knn":    
                                        scaler = StandardScaler()
                                        data = scaler.fit_transform(X.astype(np.float64))
                                        pred = kmeans.predict(data)
                                    else:
                                        pred = kmeans.predict(X)
                                    t[ex] = int(np.around(np.sum(pred)/len(pred)))
                                    hist[ex].pop()  
                                    hist[ex].pop()
                                    hist[ex].pop()                       
                                    hist[ex].append(hi)
                        else:                            
                            #keep track of feature using the centroid, don't want to mix up the training features
                            hist.append([hi])
                            t.append(1)
                            centers.append([cX,cY])
                            ex = len(centers)-1
                        if t[ex] == 0:
                            cv2.rectangle(fr, (x, y), 
                              (x + w, y+5 + h), (255, 0, 0), 2)
                            cv2.rectangle(fr, (x,y),
                              (x + len("talking") * 6, y  - 10), (0, 255, 0), -1, cv2.LINE_AA)
                            cv2.putText(fr, "talking", (x , y ), font, 0.3, (255, 255, 255), 1)
                        elif t[ex] == 1:
                            cv2.rectangle(fr, (x, y), 
                              (x + w, y + h), (255, 0, 0), 2)
                            cv2.rectangle(fr, (x,y),
                              (x + len("not talking") * 6, y  - 10), (0, 255, 0), -1, cv2.LINE_AA)
                            cv2.putText(fr, "not talking", (x , y ), font, 0.3, (255, 255, 255), 1)
                elif p[0] == 1:
                    pass
                    #continue
#                    cv2.rectangle(fr, (x, y), 
#                              (x + w, y + h), (255, 0, 255), 2)
#                    cv2.rectangle(fr, (x,y),
#                              (x + len("noface") * 6, y  - 10), (0, 0, 255), -1, cv2.LINE_AA)
#                    cv2.putText(fr, "noface", (x , y ), font, 0.3, (255, 255, 255), 1)
                
        return fr,hist,p,centers,t