# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 13:51:34 2018

@author: admin
"""


from sklearn.cluster import KMeans
import cv2
import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
import os.path
import sys
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QDialog, QLabel,QFileDialog
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap
import matplotlib.image as mpimg


class KMean(QDialog):
    fileOriginalImage =""
    img = [[]]
    def __init__(self):
        super(KMean,self).__init__()
        loadUi('Kmeans.ui',self)
        self.setWindowTitle('Kmeans Cluster')
        
        try:
           self.pushButton.disconnect()
           self.pushButton_2.disconnect()
        except:
            pass
        self.pushButton.clicked.connect(self.on_pushButton_clicked)
        self.pushButton_2.clicked.connect(self.on_pushButton_2_clicked)
             
    def openFileNameDialog(self):  
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        self.fileOriginalImage = fileName
        
    def Kmean_Clusters(self):
        
        pathImage = self.fileOriginalImage
        img_ = cv2.imread(pathImage,1)
        self.img = img_
        if self.img is None:
            print("Could not open file or find the image")
        else:
            img_g = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            #reshape 1 column and rows unknown
            X = np.reshape(img_g, (-1,1))
            print("Please wait a moment!")
            clusters = self.determineCluster(X)
            self.clustering(clusters,img_g,X)
            print("Ok") 
            
    @pyqtSlot()
    def on_pushButton_2_clicked(self):
         self.openFileNameDialog()
         self.displayImageOriginal(self.fileOriginalImage, 5, 100,200,200)
            
    @pyqtSlot()
    def on_pushButton_clicked(self):
        self.Kmean_Clusters() 
        self.displayImageSegmentation('image_segmentation.jpeg', 310, 100,200,200)
        
    def displayImageSegmentation(self, fileImage, x, y, weight, height):
        Segmentation_img = QPixmap(fileImage)
        Segmentation_img = Segmentation_img.scaled(weight, height)
        self.label_2.setGeometry(x,y,weight,height)
        self.label_2.setPixmap(Segmentation_img)
        
    def displayImageOriginal(self, fileImage, x, y, weight, height):
        original_img = QPixmap(fileImage)
        original_img = original_img.scaled(weight, height)
        self.label.setGeometry(x,y,weight,height)
        self.label.setPixmap(original_img)
        
    # determing number of clusters that apply Elbow method
    def determineCluster(self,X):
        K = range(1,11)
        cost = []
        for k  in K:
            kmeans = KMeans(n_clusters = k, random_state=0).fit(X)
            inertia = kmeans.inertia_
            cost.append(inertia)
        # caculate knee
        kn = KneeLocator(list(K), cost, S=1.0, invert=False, direction='decreasing')
        clusters = kn.knee
        return clusters
    
    #clustering algorithm
    def clustering(self,clusters,img_g,X):
        kmeans = KMeans(n_clusters = clusters, random_state=0).fit(X)
        label = kmeans.labels_
        kmeans.predict(X)
        matrix_label = np.reshape(label,(img_g.shape[0],img_g.shape[1]))
        mpimg.imsave("image_segmentation.jpeg", matrix_label)
        
        
if __name__=="__main__":
    app = QApplication(sys.argv)
    widget = KMean()
    widget.show()
    sys.exit(app.exec_())





