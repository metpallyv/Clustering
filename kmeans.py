import sys, getopt
import math
import csv
import math
import copy
import time
import numpy as np
from collections import Counter
from numpy import *
import matplotlib.pyplot as plt
import re

#normalize the data
def normalize(a):
    b = np.apply_along_axis(lambda x: (x - np.mean(x)), 0, a)
    # z_scores_np = (a - a.mean()) / a.std()
    return b

#load the file and normalize the data
def load_csv(file):
    X = genfromtxt(file, delimiter=",", dtype=str)
    X = X.astype(np.float)
    Y = X[:, -1]
    X = X[:, :-1]
    X = normalize(X)
    return X, Y

#cluster assignment for each point
def cluster_assignment(x, y, initial_centroids):
    cluster_dict = {}
    labelsDict = {}
    for i, row in enumerate(x):
        smallest_dist = 9999
        cluster_id = -1
        label = y[i]
        for k, centroid in enumerate(initial_centroids):
            dist = np.sqrt(sum((centroid - row) ** 2))
            if dist < smallest_dist:
                smallest_dist = dist
                cluster_id = k
        if cluster_dict.has_key(cluster_id):
            l = cluster_dict[cluster_id]
            l = np.vstack([l, row])
            cluster_dict[cluster_id] = l
            labelsDict[cluster_id].append(label)
        else:
            cluster_dict[cluster_id] = row
            labelsDict[cluster_id] = [label]
    return cluster_dict, labelsDict

#return the centroid value for each cluster
def get_centroids(cluster_dict):
    centroid_dict = {}
    for k in cluster_dict.keys():
        l = cluster_dict[k]
        li = np.mean(l, axis=0)
        centroid_dict[k] = li
    return centroid_dict

#calculate sum squared error
def cal_sse(centroid_dict, cluster_dict):
    s = 0
    for k in cluster_dict.keys():
        x = cluster_dict[k]
        centroid = centroid_dict[k]
        for i in range(x.shape[0]):
            dist = np.sqrt(sum((centroid - x[i]) ** 2))
            s += dist
    return s

def getLabelsGroupDict(labels):
    labelsDict = {}
    for i in labels:
        labelsDict[i] = labelsDict.get(i, 0) + 1
    return labelsDict

#get the entropy of the class labels
def entropyOfClassLabels(labels):
    totalLabels = len(labels)
    labelsDict = getLabelsGroupDict(labels)
    hy = 0
    for key, val in labelsDict.iteritems():
        t = float(val / float(totalLabels))
        hy += t * (math.log(t, 2)) * (-1.0)
    return hy

#NMI value
def getNMIValue(clusterLabelDict, hy):
    # Calculate entropy of cluster labels
    hcList = []
    labelsSumList = [len(l) for l in clusterLabelDict.itervalues()]
    totalLabels = sum(labelsSumList)
    #Calculating entropy of cluster labels here itself to improve performance rather
    #than calling entropyOfClassLabels method
    #Iterate to get the total number of labels
    hc = 0
    for val in labelsSumList:
        #totalLabels += len(val)
        #hcList.extend([key for k in val])
        t = float(val / float(totalLabels))
        hc += t * (math.log(t, 2)) * (-1.0)
    #hc = entropyOfClassLabels(hcList)
    hyc = 0
    for key, val in clusterLabelDict.iteritems():
        p = float(len(val) / float(totalLabels))
        hyc += p * entropyOfClassLabels(val)
    nmi = float(2.0 * (hy - hyc) / (hy + hc))
    return nmi

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "f:e:")
    except getopt.GetoptError as error:
        print "Unknown input argument provided : ", " Error : ", str(error)
        sys.exit(2)
    newfile = ""
    for opt, value in opts:
        if opt == "-f":
            newfile = value
        if opt == "-e":
            no_clusters = int(value)
    X, Y = load_csv(newfile)
    hy = entropyOfClassLabels(Y)
    cost_func_list = []
    for i in range(1, no_clusters):
        best_sse = 9999
        bestClusterLabelDict = {}
        for j in range(75):
            x = X
            y = Y
            centroids = np.empty(shape=[0, X.shape[1]])
            # initialize centroids to random points
            for k in range(i + 1):
                ran = random.randint(1, x.shape[0])
                centroids = np.vstack([centroids, x[ran, :]])
                x = np.delete(x, ran, axis=0)
                y = np.delete(y, ran, axis=0)
            #k-means algorithm
            cluster_dict = {}
            centroid_dict = {}
            max_ite = 50
            ite = 0
            labelsDict = {}
            while ite < max_ite:
                #cluster assignment step
                cluster_dict, labelsDict = cluster_assignment(x, y, centroids)
                #move centroids step
                centroid_dict = get_centroids(cluster_dict)
                centroids = centroid_dict.values()
                ite += 1
            sse = cal_sse(centroid_dict, cluster_dict)
            sse = sse / float(X.shape[0])
            if sse < best_sse:
                best_sse = sse
                bestClusterLabelDict = labelsDict
                bestClusterDict = cluster_dict
        nmi = getNMIValue(bestClusterLabelDict, hy)
        print "*****************************"
        print "Number Of Clusters = ", i + 1
        print "SSE value : ", best_sse
        print "NMI value : ", nmi
        for key, val in bestClusterLabelDict.iteritems():
            print "ClusterID: ", key, " Labels in Cluster : ", getLabelsGroupDict(val)
        print "*****************************"
        cost_func_list.append(best_sse)
    print "CostFunc List : ", cost_func_list
    print "K values : ", range(no_clusters)
    cost_func_list = np.array(cost_func_list)
    plt.suptitle("K means algorithm Data plot")
    plt.plot(range(2, no_clusters + 1), cost_func_list)
    plt.ylabel("SSE")
    plt.xlabel("K value ")
    tokens = re.split('[. /]', newfile)
    fileName = "Kmean_yeastData" + tokens[1]
    plt.savefig(fileName)
    
if __name__ == "__main__":
    main(sys.argv[1:])
