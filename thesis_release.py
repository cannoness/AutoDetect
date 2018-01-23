'''
Usage:
 
    Import video from any location, split into positive, negative, training and
    testing, run through produce_histograms()
    
    Train and test with model of your choosing.
    
    Results can be played back using play_with_labels()

'''
import cv2 as cv2
import time as time
import imutils
import numpy as np
from matplotlib import pyplot as plt
import define_object_test
from faceknn import *

# Function from: http://programmingcomputervision.com/
def draw_flow(im,flow,step=8):
    """ Plot optical flow at sample points
        spaced step pixels apart. """
        
    h,w = im.shape[:2]
    y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1)
    fx,fy = flow[y.astype(int),x.astype(int)].T
        
    # create line endpoints
    lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines = np.int32(lines)
    
    # create image and draw
    vis = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    for (x1,y1),(x2,y2) in lines:
        cv2.line(vis,(x1,y1),(x2,y2),(0,255,0),1)
        cv2.circle(vis,(x1,y1),1,(0,255,0), -1)
    
    return vis

def play_with_labels(kmeans,video,display,tester):

    camera = cv2.VideoCapture(video)
    time.sleep(0.25)
    hist = []
    ret, frame1 = camera.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    prvs  = imutils.resize(prvs, width=1000)
    frame1  = imutils.resize(frame1, width=1000)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    camera.set(cv2.CAP_PROP_FPS, 10)
    p = [] 
    p.append(-1)
    tt = []
    centers = []
    while True:
            #slow down the fps
        grabbed, frame = camera.read()
        if grabbed:
            frame = imutils.resize(frame, width=1000)
            pass
        else:
            break 
        # blur the frame and convert it to the HSV color space
        blurred = cv2.GaussianBlur(frame, (11,11), 0)
        blurred = cv2.medianBlur(blurred, 15)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        thr = define_object_test.FindObject(hsv,frame,"pencil")
        thr2 = define_object_test.FindObject(hsv,frame,"paper")
        thr3,_ = define_object_test.FindObject(hsv,frame,"keyboard")
        thr4 = define_object_test.FindObject(hsv,frame,"table")
        
        grayframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
        grayframe = cv2.medianBlur(grayframe, 5)              
        current_frame = grayframe
    
        flow = cv2.calcOpticalFlowFarneback(prvs, grayframe, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        mag[mag == -np.inf] = 0
        hsv[...,0] = ang*180/np.pi/2
        hsv2 = np.array(cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX))
                
        prvs = current_frame.astype(np.uint8)
        
           
    #    motion_vectors = draw_flow(current_frame, flow)
    #    cv2.imshow('Motion vector plot', motion_vectors)
        
        #run each test data through this and stack all the histograms
        if display=="writing":
            frame, hist, p, centers = define_object_test.LabelObject(frame, thr, thr2, hsv2,ang,"pencil,paper",kmeans,hist, p,centers,threh3 = thr4, tester = tester)
            cv2.imshow('Boxes',frame) 
        elif display == "typing":
            frame, hist, p, centers, tt = define_object_test.LabelObject(frame, thr3, thr4, hsv2, ang,"keyboard,table",kmeans,hist,p,centers, KNN = keyboard, t = tt, tester = tester)
            cv2.imshow('Boxes',frame)
        elif display == "talking":
            frame, hist, p, centers, tt = define_object_test.LabelObject(frame, thr3, thr4, hsv2, ang,"talking",kmeans,hist, p,  centers, KNN = faces, t = tt, tester = tester)
            cv2.imshow('Boxes',frame)
            
        
        #todo: break up into tensor flow- GPU distributed to attempt google cloud                     
        cv2.imshow('Boxes',frame)    
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
             
        
    camera.release()
    cv2.destroyAllWindows()



def ProduceHistograms(video,display):
    centers = []    
    hist_list = []    
    hist_mean = []
    
    camera = cv2.VideoCapture(video)
    time.sleep(0.25)
    
    ret, frame1 = camera.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    prvs  = imutils.resize(prvs, width=700)
    frame1  = imutils.resize(frame1, width=700)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    
    camera.set(cv2.CAP_PROP_FPS, 10)
    while True:
            #slow down the fps
        grabbed, frame = camera.read()
        if grabbed:
            frame = imutils.resize(frame, width=700)
            pass
        else:
            break

        # blur the frame and convert it to the HSV color space
        
        blurred = cv2.GaussianBlur(frame, (11,11), 0)
        kernel2 = np.array([[0, -1, 0],
                           [-1, 5, -1],
                            [0, -1, 0]])
        blurred = cv2.filter2D(blurred, -1, kernel2)
        blurred = cv2.medianBlur(blurred, 15)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        thr = define_object_test.FindObject(hsv,frame,"pencil")
        thr2 = define_object_test.FindObject(hsv,frame,"paper")
        thr3,_ = define_object_test.FindObject(hsv,frame,"keyboard")
        thr4 = define_object_test.FindObject(hsv,frame,"table")
        
        grayframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
        grayframe = cv2.medianBlur(grayframe, 5)       
        current_frame = grayframe
    
        flow = cv2.calcOpticalFlowFarneback(prvs, grayframe, None, 0.5, 3, 18, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        mag[mag == -np.inf] = 0.001
        hsv[...,0] = ang*180/np.pi/2
        ang = ang*180/np.pi/2
        hsv2 = np.array(cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX))
                
        prvs = current_frame.astype(np.uint8)
       
        
    #    motion_vectors = draw_flow(current_frame, flow)
    #    cv2.imshow('Motion vector plot', motion_vectors)
    
        #run each test data through this and stack all the histograms
        if display=="writing":
            frame,centers,hist_list = define_object_test.InteractObject(frame, thr, thr2, hsv2, ang,"pencil,paper",centers,hist_list, threh3 = thr4)
            cv2.imshow('Boxes',frame) 
        elif display == "typing":
            frame,centers,hist_list = define_object_test.InteractObject(frame, thr3, thr4, hsv2, ang,"keyboard,table",centers,hist_list, KNN = keyboard)
            cv2.imshow('Boxes',frame)
        elif display == "talking":
            frame,centers,hist_list = define_object_test.InteractObject(frame, thr3, thr4, hsv2, ang,"talking",centers,hist_list, KNN= faces)
            cv2.imshow('Boxes',frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        #Currently only works for writing, comment out for talking or typing
        
#     for each centroid...
    for histo in hist_list:    
#    # if we have more than 3 frames with the same feature, average them, then add them to the final list
        if len(histo) == 1:
            hist_mean.append(histo[0])
        elif len(histo) == 2:
            hist = np.array(np.mean(histo[0:1], axis = 0))
            hist_mean.append(hist)
        else:
            for i in range(0,len(histo)-3,3):
                hist = np.array(np.mean(histo[i:i+2], axis = 0))
                hist_mean.append(hist)
                
        #Uncomment for Talking or Typing, comment above.
#    if len(hist_list) == 1:
#            hist_mean.append(hist_list[0])
#    elif len(hist_list) == 2:
#            hist = np.array(np.mean(hist_list[0:1], axis = 0))
#            hist_mean.append(hist)
#    else:
#            for i in range(0,len(hist_list)-3,3):
#                hist = np.array(np.mean(hist_list[i:i+2], axis = 0))
#                hist_mean.append(hist)
#        
    camera.release()
    cv2.destroyAllWindows()
    return hist_mean    