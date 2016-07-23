__author__ = 'zhf'
#coding=utf-8

from numpy import *
import operator

# Creating training data
def createdata():
    group = array([[1,1],[1,1.1],[0.9,1],[0,0.1],[0.1,0],[0.1,0.1],[0,1],[0.1,0.9],[0,0.8]])
    labels = ['A','A','A','B','B','B','C','C','C']
    return group,labels

# testing process
def knnclassify(testing_data,group,labels,k):
    datasize = group.shape[0]           #the rows of array
    # calculate the distance
    diffMat = tile (testing_data,(datasize,1))-group
    sqdiffMat = diffMat**2
    sqdiffMatsum=sqdiffMat.sum(axis=1)
    distances = sqdiffMatsum**2
    # majority voting rule
    classcount={}
    sortdistances = distances.argsort()
    for i in range(k):
        votelabel = labels[sortdistances[i]]
        classcount[votelabel] = classcount.get(votelabel,0)+1
    sortclass = sorted(classcount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortclass[0][0]

# main
if __name__ == "__main__":
    string = raw_input("please enter two numbers, split by comma:")
    input_data = string.split(",")
    testing_data = []
    for i in range(len(input_data)):
        testing_data.append(float(input_data[i]))
    string1 = raw_input("please enter the k:")
    k = int(string1)
    group, labels = createdata()
    label = knnclassify(testing_data, group, labels, k)
    print "the label of input data is:" + str(label)