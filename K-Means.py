#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 15:40:50 2018

@author: varshath
"""

import numpy as np 
from matplotlib import pyplot as plt
from copy import deepcopy
import cv2

UBIT = 'vharisha' 
np.random.seed(sum([ord(c) for c in UBIT]))
def plot_points(points,centroids,cluster_points,num_of_clusters,name):
    colors = ['r', 'g', 'b']
    fig, ax = plt.subplots()
    for k in range(num_of_clusters):
        new_points=np.asarray([points[j] for j in range(len(points)) if int(cluster_points[j])==k])
        ax.scatter(new_points[:, 0], new_points[:, 1], marker='^', edgecolors=colors[k], s=50, c=colors[k])
        for i in range(len(centroids)):
            ax.scatter(centroids[i][0], centroids[i][1], marker='o', s=50, c=colors[i])
            x = centroids[i][0]
            y = centroids[i][1]
            text = '(' + str(round(centroids[i][0],2)) + ', ' + str(round(centroids[i][1],2)) + ')'
            ax.text(x, y, text,fontsize=7)
        for i in range(len(new_points)):
            x = new_points[i][0]
            y = new_points[i][1]
            text = '(' + str(round(new_points[i][0],2)) + ', ' + str(round(new_points[i][1],2)) + ')'
            ax.text(x, y, text,fontsize=7)
    fig.savefig(name)


def compute_euclidean_distance(a,b,axis_no):
    return np.linalg.norm(a-b,axis=axis_no)



def recompute_centroids(points,cluster,num_of_clusters,centroids):
    new_centroids=deepcopy(centroids)
    for c in range(num_of_clusters):
        clustered=np.asarray([points[k] for k in range(len(points)) if cluster[k]==c])
        if clustered is None:
            new_centroids[c]=centroids[c]
        else:
            new_centroids[c]=np.mean(clustered,axis=0)
    return new_centroids


def do_clustering(points,clusters):
    new_clusters=np.zeros(points.shape[0])
    count=0
    for point in points:
        distances_from_clusters=compute_euclidean_distance(point,clusters,1)
        assigned_cluster = np.argmin(distances_from_clusters)
        new_clusters[count]=assigned_cluster
        count+=1
    return new_clusters
        
def get_random_centroids(img,num_of_clusters):
    mean = np.mean(img, axis = 0)
    std = np.std(img, axis = 0)
    return np.random.randn(num_of_clusters,img.shape[1]) * std + mean

def do_quantiztation(clusters,centroids,height,width):
    
    depth = centroids.shape[1]
    quantized_image = np.zeros((width,height,depth))
    count = 0
    for i in range(width):
        for j in range(height):
            val=centroids[int(clusters[count])]
            quantized_image[i][j] = val
            count+= 1
    return quantized_image

def normalise(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))*255




def get_quantized_image(num_of_clusters):
    image=cv2.imread('baboon.jpg')
    image = np.array(image, dtype=np.float64)/255
    width=image.shape[0]
    height=image.shape[1]
    depth=image.shape[2]
    img_pixels=np.reshape(image,(width*height,depth))
    centroids=get_random_centroids(img_pixels,num_of_clusters)
    old_centroids=np.zeros(centroids.shape)
    new_centroids=centroids
    clusters = np.zeros(len(img_pixels))
    same=np.array_equal(old_centroids,new_centroids)
    while True:
        old_centroids = deepcopy(new_centroids)
        clusters=do_clustering(img_pixels,old_centroids)
        new_centroids = recompute_centroids(img_pixels,clusters,num_of_clusters,old_centroids)
        same = np.array_equal(old_centroids,new_centroids)
        if same:
            break
    new_image = do_quantiztation(clusters,new_centroids,height,width)
    imgs.append(new_image)
    plt.imshow(new_image)
    new_image=normalise(new_image)
    cv2.imwrite('task3_baboon_'+str(num_of_clusters)+'.jpg',new_image)
    imgs.append(new_image)







x_points=np.asarray([5.9,4.6 ,6.2 ,4.7 ,5.5 ,5.0 ,4.9 ,6.7 ,5.1 ,6.0 ])
y_points=np.asarray([3.2, 2.9, 2.8, 3.2, 4.2, 3.0, 3.1, 3.1, 3.8, 3.0])
count=0
points=[[0.0 for j in range(2)] for i in range(y_points.shape[0])]
for i in range(len(points)):
    points[i][0]=x_points[i]
    points[i][1]=y_points[i]    
points=np.float64(points)
num_of_clusters=3
centroid_x = [6.2, 6.6,6.5]
centroids_y = [3.2, 3.7, 3.0]
centroids=[[0.0 for j in range(2)] for i in range(len(centroid_x))]
for i in range(len(centroid_x)):
    centroids[i][0]=centroid_x[i]
    centroids[i][1]=centroids_y[i]
centroids=np.float64(centroids)

clustered_points=np.asarray(do_clustering(points,centroids))

#plot_points(points,centroids,clustered_points,num_of_clusters)



new_clusters = np.zeros(len(points))

iterations=3

for i in range(1,iterations):
    new_clusters=np.asarray(do_clustering(points,centroids))
    name='task3_iter'+str(i)+'_a.jpg'
    plot_points(points,centroids,new_clusters,num_of_clusters,name)
    centroids = recompute_centroids(points,new_clusters,num_of_clusters,centroids)
    name='task3_iter'+str(i)+'_b.jpg'
    plot_points(points,centroids,new_clusters,num_of_clusters,name)









imgs=[]
d=[3,5,20,10]

for num_of_clusters in d:
    get_quantized_image(num_of_clusters)

    

