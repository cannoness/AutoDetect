# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 01:35:20 2017

@author: cattius
"""
import os
from define_object_test import *
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

from sklearn import neighbors, datasets

def KNN(which, path):
    if which == "faces":
        y = []
        X = []
        training_set = os.listdir(path) 
        for img in training_set:
            fil = cv2.imread(path+img,0)
            fil = cv2.resize(fil, (128,128))

            person = cv2.medianBlur(fil,5)

            y.append(0)
            X.append(person)
        training_set3 = os.listdir(path) 
        for img in training_set3:
            fil = cv2.imread(path+img,0)
            max_height = fil.shape[0]
            max_width = fil.shape[1]
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
                fil = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor)
            # fill excess by zero padding the image
            zero_padded = np.zeros((height,width,3), np.uint8)
            zero_padded[:max_height,:max_width] = fil

            fil = cv2.dilate(zero_padded,None,1)
#            hist = cv2.calcHist(fil, [0,1],None,[256,256],[0,256,0,256])
#            hist = hist/hist.ravel().sum()
            y.append(0)
            X.append(fil)
        training_set2 = os.listdir(path) 
        for img in training_set2:
            fil = cv2.imread(path+img,0)
            max_height = fil.shape[0]
            max_width = fil.shape[1]
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
                fil = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor)
            # fill excess by zero padding the image
            zero_padded = np.zeros((height,width,3), np.uint8)
            zero_padded[:max_height,:max_width] = fil

#            person = PersonObject(fil)
#            person = cv2.GaussianBlur(gray,(3,3),0)
#            person = cv2.adaptiveThreshold(person,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                cv2.THRESH_BINARY,21,2)
            person = cv2.medianBlur(zero_padded,5)
#            hist = cv2.calcHist(person, [0,1],None,[256,256],[0,256,0,256])
#            
#            hist = hist/hist.ravel().sum()
            y.append(1)
            X.append(person)
        
        data = np.array(X).reshape((np.array(X).shape[0], -1))
        
#        X = np.array(X)[:,:2]
        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2,random_state=42)    
#        scaler = StandardScaler()
#        X_train_scaled = scaler.fit_transform(np.array(X_train).astype(np.float64))
#        X_test_scaled = scaler.fit_transform(np.array(X_test).astype(np.float64))
    
        knn_clf = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=7)
        
        #fit training data to classifier
        knn_clf.fit(X_train, y_train)
        y_knn_pred = knn_clf.predict(X_test)
        
        # compute and store accuracy
        acc =  accuracy_score(y_test, y_knn_pred)
        cm = confusion_matrix(y_test, y_knn_pred, labels=np.unique(y))           
        print(acc,cm)
#        neighbors = range(2,10)
#        cv_scores = []
#        for k in neighbors:
#             knn = KNeighborsClassifier( weights='distance', n_neighbors=k)
#             scores = cross_val_score(knn, X_train_scaled, y_train, cv=10, scoring="accuracy")
#             cv_scores.append(scores.mean())
#        MSE = [1 - x for x in cv_scores]
#        optimal_k = neighbors[MSE.index(min(MSE))]
    #    plt.plot(neighbors, MSE)
    #    plt.xlabel('Number of Neighbors K')
    #    plt.ylabel('Misclassification Error')
    #    y_train_pred = cross_val_predict(knn_clf, X_train_scaled, y_train, cv=5)
    #    conf_mx = confusion_matrix(y_train, y_train_pred)
    #    plt.matshow(conf_mx, cmap=plt.cm.gray)
    #    row_sums = conf_mx.sum(axis=1, keepdims=True)
    #    norm_conf_mx = conf_mx / row_sums
    #    np.fill_diagonal(norm_conf_mx, 0)
    #    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
#        print (optimal_k)
    
    #    plt.show()
        cv2.destroyAllWindows()
        return knn_clf
    
    elif which=="keyboards":
        y = []
        X = []
        training_set = os.listdir(path) 
        for img in training_set:
            fil = cv2.imread(path+img,0)
            max_height = fil.shape[0]
            max_width = fil.shape[1]
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
                fil = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor)
            # fill excess by zero padding the image
            zero_padded = np.zeros((height,width,3), np.uint8)
            zero_padded[:max_height,:max_width] = fil

#            blurred = cv2.GaussianBlur(fil, (11,11), 0)
            kernel2 = np.array([[0, -1, 0],
                               [-1, 5, -1],
                                [0, -1, 0]])
            blurred = cv2.filter2D(zero_padded, -1, kernel2)
            blurred = cv2.medianBlur(blurred, 5)

            y.append(0)
            X.append(blurred)
        training_set3 = os.listdir(path) 
        for img in training_set3:
            fil = cv2.imread(path+img,0)
            max_height = fil.shape[0]
            max_width = fil.shape[1]
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
                fil = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor)
            # fill excess by zero padding the image
            zero_padded = np.zeros((height,width,3), np.uint8)
            zero_padded[:max_height,:max_width] = fil
                
#            blurred = cv2.GaussianBlur(fil, (11,11), 0)
            kernel2 = np.array([[0, -1, 0],
                               [-1, 5, -1],
                                [0, -1, 0]])
            blurred = cv2.filter2D(zero_padded, -1, kernel2)
            blurred = cv2.medianBlur(blurred, 5)
            y.append(1)
            X.append(blurred)
        training_set2 = os.listdir(path) 
        for img in training_set2:
            fil = cv2.imread(path+img,0)
            max_height = fil.shape[0]
            max_width = fil.shape[1]
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
                fil = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor)
            # fill excess by zero padding the image
            zero_padded = np.zeros((height,width,3), np.uint8)
            zero_padded[:max_height,:max_width] = fil

#            blurred = cv2.GaussianBlur(fil, (11,11), 0)
            kernel2 = np.array([[0, -1, 0],
                               [-1, 5, -1],
                                [0, -1, 0]])
            blurred = cv2.filter2D(zero_padded, -1, kernel2)
            blurred = cv2.medianBlur(blurred, 5)
            y.append(2)
            X.append(blurred)
        
        data = np.array(X).reshape((np.array(X).shape[0], -1))
        
#        X = np.array(X)[:,:2]
        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2,random_state=42)    
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(np.array(X_train).astype(np.float64))
        X_test_scaled = scaler.fit_transform(np.array(X_test).astype(np.float64))
    
        knn_clf = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=5)
        
        #fit training data to classifier
        knn_clf.fit(X_train, y_train)
        y_knn_pred = knn_clf.predict(X_test)
        
        # compute and store accuracy
        acc =  accuracy_score(y_test, y_knn_pred)
        cm = confusion_matrix(y_test, y_knn_pred, labels=np.unique(y))           
        print(acc,cm)
    #    neighbors = range(2,10)
    #    cv_scores = []
    #    for k in neighbors:
    #         knn = KNeighborsClassifier( weights='distance', n_neighbors=k)
    #         scores = cross_val_score(knn, X_train_scaled, y_train, cv=10, scoring="accuracy")
    #         cv_scores.append(scores.mean())
    #    MSE = [1 - x for x in cv_scores]
    #    optimal_k = neighbors[MSE.index(min(MSE))]
    #    plt.plot(neighbors, MSE)
    #    plt.xlabel('Number of Neighbors K')
    #    plt.ylabel('Misclassification Error')
    #    y_train_pred = cross_val_predict(knn_clf, X_train_scaled, y_train, cv=5)
    #    conf_mx = confusion_matrix(y_train, y_train_pred)
    #    plt.matshow(conf_mx, cmap=plt.cm.gray)
    #    row_sums = conf_mx.sum(axis=1, keepdims=True)
    #    norm_conf_mx = conf_mx / row_sums
    #    np.fill_diagonal(norm_conf_mx, 0)
    #    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
        #f.DNN_train(X,y)  
    
    #    plt.show()
        cv2.destroyAllWindows()
        return knn_clf