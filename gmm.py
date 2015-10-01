import sys, getopt
import math
import csv
import math
import copy
import time
import numpy as np
from decimal import Decimal
from collections import Counter
from numpy import *
import matplotlib.pyplot as plt
import re

#normalize the data
def normalize(a):
    b = np.apply_along_axis(lambda x: (x-np.mean(x)),0,a)
    col_ind = []
    for col in range(b.shape[1]):
        le = sum(b[:,col])
        if le == 0:
            col_ind.append(col)
    b = np.delete(b,np.s_[col_ind],axis =1)
    return b

#load the file and normalize the data
def load_csv(file):
    X = genfromtxt(file, delimiter=",",dtype=str)
    X = X.astype(np.float)
    Y = X[:,-1]
    X = X[:,:-1]
    X = normalize(X)
    return X, Y

#cluster assignment for each point
def cluster_assignment(x,initial_centroids):
    cluster_dict = {}
    for row in x:
        smallest_dist = 9999
        cluster_id = -1
        for k,centroid in enumerate(initial_centroids):
            dist = np.sqrt(sum((centroid - row) ** 2))
            if dist < smallest_dist:
                smallest_dist = dist
                cluster_id = k
        if cluster_dict.has_key(cluster_id):
            l = cluster_dict[cluster_id]
            l = np.vstack([l, row])
            cluster_dict[cluster_id] = l
        else:
            cluster_dict[cluster_id] = row
    return cluster_dict

#calculate sum squared error 
def cal_sse(x,label_dict,centroids):
    sse = 0
    for i in range(x.shape[0]):
        label_val = label_dict[i]
        centroid_val = centroids[label_val]
        dist = np.sqrt(sum((centroid_val - x[i]) ** 2))
        sse += dist
    return sse

# Implementing the Multivariate Gaussian Density Function
def Gausssian_normal(x, mu, cov):
    part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)) )
    part2 = np.exp(-.5 * np.einsum('ij, ij -> i',x - mu, np.dot(np.linalg.inv(cov),(x - mu).T).T ) )
    return part1 * np.exp(part2)

#Gaussian cluster estimation
def cluster_estimation(R,x,centroids,covar,pi):
    for k in range(R.shape[1]):
        tmp = pi[k] * Gausssian_normal(x,centroids[k], covar[k])
        R[:,k] = tmp
    # Normalize the responsibility matrix
    R = (R.T / np.sum(R, axis = 1)).T
    return R

def getLabelsGroupDict(labels):
    labelsDict = {}
    for i in labels:
        labelsDict[i] = labelsDict.get(i,0)+1
    return labelsDict

#Calculation of entropy of class labels
def entropyOfClassLabels(labels):
    totalLabels = len(labels)
    labelsDict = getLabelsGroupDict(labels) # Get the no of instances belonging to a particular class
    hy = 0
    for key,val in labelsDict.iteritems():
        t = float(val/float(totalLabels))
        hy += t*(math.log(t,2))*(-1.0)
    return hy

def cal_hyc_each_cluster(cluster_values,y):
    class_dict = {}
    len_of_cluster = len(cluster_values)
    #build a dict with class_values and count
    for x in cluster_values:
        class_dict[y[x]] = class_dict.get(y[x], 0) + 1
    #cal the hyc value now
    hyc_cluster = 0
    for key in class_dict.keys():
        val = float(class_dict[key])/len_of_cluster
        hyc_cluster += val*(math.log(val,2))*(-1.0)
    return hyc_cluster

def getNMIValue(clusterDict,hy,y):
    #Calculate entropy of cluster labels
    hc = 0
    total_no_of_instances = len(clusterDict.values())
    for val in clusterDict.keys():
        t = float(len(clusterDict[val]))/total_no_of_instances
        hc += t*(math.log(t,2))*(-1)
    #calculate the hyc
    hyc = 0
    for k in clusterDict.keys():
        x_of_cur_cluster = clusterDict[k]
        hyc_cluster = cal_hyc_each_cluster(x_of_cur_cluster,y)
        hyc_cluster = float(len(x_of_cur_cluster)/total_no_of_instances)* hyc_cluster
        hyc += hyc_cluster
    #calculate NMI now
    nmi = float(2.0*(hy - hyc)/(hy+hc))
    return nmi

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"f:t:e:")
    except getopt.GetoptError as error:
        print "Unknown input argument provided : "," Error : ",str(error)
        sys.exit(2)
    newfile = ""
    for opt,value in opts:
        if opt == "-f":
            newfile = value
        if opt == "-e":
            no_clusters = int(value)
    X,Y = load_csv(newfile)
    hy = entropyOfClassLabels(Y)
    #print X
    log_likelihoods = []
    cost_func_list = []
    nnmi_list = []
    for i in range(1,no_clusters):
        best_sse = 999999
        best_cluster_dict = {}
        for j in range(100):
            x = X
            centroids = np.empty(shape=[0, X.shape[1]])
            #initialize centroids to random points
            for k in range(i+1):
                ran=random.randint(1, x.shape[0])
                centroids = np.vstack([centroids, x[ran,:]])
                x = np.delete(x,ran, axis=0)
            #initial covariance matrix
            covar = [np.cov(x.T)] * i
            # initialize the probabilities/weights for each gaussians
            pi = [1./i] * i
            # responsibility matrix is initialized to all zeros
            R = np.zeros((x.shape[0], i))
            cluster_dict = {}
            max_ite = 50
            ite = 0
            while ite < max_ite:
                #cluster assignment step
                R= cluster_estimation(R,x,centroids,covar,pi)
                # The number of data points belonging to each gaussian
                no_data_points = np.sum(R, axis = 0)
                #update means,covariance,pi
                for k in range(i):
                    # means
                    centroids[k] = 1. / no_data_points[k] * np.sum(R[:, k] * x.T, axis = 1).T
                    x_mu = np.matrix(x - centroids[k])
                    ## covariances
                    covar[k] = np.array(1 / no_data_points[k] * np.dot(np.multiply(x_mu.T,  R[:, k]), x_mu))
                    ## and finally the probabilities
                    pi[k] = 1. / x.shape[0] * no_data_points[k]
                # Likelihood computation
                log_likelihood = np.sum(np.log(np.sum(R, axis = 1)))
                cluster_label_dict = {}
                label_dict = {}
                for a,val in enumerate(R):
                    l = val
                    max = 0
                    best_cluster_ind = 0
                    for cluster_ind,val in enumerate(l):
                        if val > max:
                            max = val
                            best_cluster_ind = cluster_ind
                    label_dict[a] = best_cluster_ind
                    if cluster_dict.has_key(best_cluster_ind):
                        li = cluster_dict[best_cluster_ind]
                        li.append(a)
                        cluster_dict[best_cluster_ind] = li
                    else:
                        l = []
                        l.append(a)
                        cluster_dict[best_cluster_ind] = l
                    cluster_label_dict[best_cluster_ind] = cluster_label_dict.get(best_cluster_ind,0)+1
                sse = cal_sse(x,label_dict,centroids)
                sse = sse/float(X.shape[0])
                if sse < best_sse:
                    best_sse = sse
                    best_cluster_dict = cluster_dict
                ite +=1
        nmi = getNMIValue(best_cluster_dict,hy,Y)
        print "*****************************"
        print "Number Of Clusters = ",i+1
        print "SSE value : ", best_sse
        print "NMI value : ",nmi
        #for key,val in best_cluster_dict.iteritems():
        #    print "ClusterID: ",key+1," Labels in Cluster : ",val
        #print "*****************************"
        cost_func_list.append(best_sse)
        nnmi_list.append(nmi)
    print "CostFunc List : ",cost_func_list
    print "K values : ",range(no_clusters)
    print " NMI list",nnmi_list
    cost_func_list = np.array(cost_func_list)
    plt.suptitle("GMM plot")
    plt.plot(range(2,no_clusters+1),cost_func_list)
    plt.ylabel("SSE")
    plt.xlabel("K value ")
    tokens = re.split('[. /]',newfile)
    fileName = "GMM_Ecoli"+tokens[1]
    plt.savefig(fileName)

if __name__ == "__main__":
    main(sys.argv[1:])
