import json
from scipy.cluster.hierarchy import dendrogram, linkage
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.spatial.distance as distance

#LOAD JSON FILE
f = open('C:\\Users\\sahil\\Desktop\\titanic.json', 'r')
d = json.load(f)

#Ceate empty table to store the json values
data = [[0 for i in range(0, 6)] for j in range(0, len(d))]

#Add json data to python table 'data' and encode categorical variables
for i in range(0, len(d)):
    #Age
    if d[i]['Age'] == '':
        data[i][0] = None
    else:
        data[i][0] = float(d[i]['Age'])
    #Fare
    data[i][1] = float(d[i]['Fare'])
    #Add siblings/spouse and parent/child into family members
    data[i][2] = float(d[i]['SiblingsAndSpouses']) + float(d[i]['ParentsAndChildren'])
    #Encode embarked location
    if d[i]['Embarked'] == 'C':
        data[i][3] = 1.
    elif d[i]['Embarked'] == 'Q':
        data[i][3] = 2.
    elif d[i]['Embarked'] == 'S':
        data[i][3] = 3.
    else:
        data[i][3] = None
    #Encode Gender
    if d[i]['Sex'] == 'male':
        data[i][4] = 0
    elif d[i]['Sex'] == 'female':
        data[i][4] = 1
    else:
        data[i][4] = None
    #Survivors
    data[i][5] = float(d[i]['Survived'])
print 'Original data has a total of ',len(data), 'rows'


#Now we remove the rows with missing values
truedata = []
for i in range(0,len(data)):
    if None not in data[i]:
        truedata.append(data[i])
print 'After removing rows with missing values, we have a total of',len(truedata),'rows left!\n\n'

#Normalizing Data Values
print 'Normalizing data... \nValues for "Survived or not" have not been normalized as these are already binary....'
#Find min and max values amongst all the columns
minvals={}
maxvals={}
for j in range(0,5):
	minD=truedata[0][j]
	maxD=truedata[0][j]
	for i in range(0,len(truedata)):
		if truedata[i][j] < minD:
			minD=truedata[i][j]
		if truedata[i][j] > maxD:
			maxD=truedata[i][j]
		minvals[j]=minD
		maxvals[j]=maxD
# print minvals, maxvals-- verified from json file itself
#copy the normalized values to new table - 'data' that will be used from now on
truedata=np.array(truedata)
data = truedata.copy()
for j in range(0,5):
	for i in range(0, len(truedata)):
		 data[i][j]= (data[i][j] - minvals[j]) / (maxvals[j]-minvals[j])
print 'Normalizing Successful.....\n\n'

#CALCULATE AND PLOT THE DENDROGRAM----------------------------------------------
#perform hierarchial clustering
from scipy.cluster.hierarchy import dendrogram, linkage
#save the clustering results in variable z
#making sure not to take 'Survived' column while plotting the dendrogram
#It is the target variable: only used while considering and coloring the plots
Z = linkage(data[:, 0:5], method='ward', metric='euclidean')
#create a dendrogram from the result of the hierarchial clustering
dendrogram(Z, leaf_rotation = 90, leaf_font_size = 8) #rotate the x axis labels, set the font size for x axis labels
#add title and axis labels
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.axhline(y=8, color='black')
plt.show()




#USE 3-MEANS CLUSTERING
#initialize cluster centers
c1 = [0.2, 0.4, 0.3, 0.6, 0.2]
c2 = [0.9, 0.8, 0.0, 0.2, 0.3]
c3 = [0.5, 0.6, 0.2, 0.1, 0.6]
centroids = [c1, c2, c3]

prev1 = []
prev2 = []
prev3 = []
cluster1 = []
cluster2 = []
cluster3 = []

# create function to calculate cluster mean
def clusterMean(cluster):
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    for instance in cluster:
        a += instance[0]
        b += instance[1]
        c += instance[2]
        d += instance[3]
        e += instance[4]
    sums = [a,b,c,d,e]
    if len(cluster) != 0:
        mean = [float(Sum)/len(cluster) for Sum in sums]
    else:
        mean=0
    return mean


t = 0 #use to check whether centroids stop reassigning
p=0 #use to mention max iterations
while p<100 and t<1:
    cluster1 = []
    cluster2 = []
    cluster3 = []
    for instance in data:
        d1 = distance.euclidean(instance[0:5], c1)
        d2 = distance.euclidean(instance[0:5], c2)
        d3 = distance.euclidean(instance[0:5], c3)
        if d1<d2 and d1<d3:
            cluster1.append(instance)
            # print 'cluster1'
        elif d2<d3 and d2<d1:
            cluster2.append(instance)
            # print 'cluster2'
        else:
            cluster3.append(instance)
            # print 'cluster3'

    cluster1=np.array(cluster1)
    cluster2=np.array(cluster2)
    cluster3=np.array(cluster3)

    #check if old cluster is equal to new cluster
    #if yes, stop the loop
    if np.array_equal(cluster1, prev1) and np.array_equal(cluster2, prev2) and np.array_equal(cluster3, prev3):
        t += 1

    c1 = clusterMean(cluster1)
    c2 = clusterMean(cluster2)
    c3 = clusterMean(cluster3)
    prev1 = cluster1[:]
    prev2 = cluster2[:]
    prev3 = cluster3[:]

    centroids=[c1,c2,c3]
    centroids=np.array(centroids)

    if p in (0,1,2,3,4,5,6,7,8,9,10):
        plt.figure(2)
        # introduce labels for each feature
        names = ['Age', 'Fare', 'Family', 'Location', 'Gender']

        k = 1
        for i in range(5):
            for j in range(5):
                if i < j:
                    plt.subplot(5, 2, k)  # subplots creates panels
                    plt.scatter(cluster1[:, i], cluster1[:, j], c='b', marker='D')
                    plt.scatter(cluster2[:, i], cluster2[:, j], c='g', marker='>')
                    plt.scatter(cluster3[:, i], cluster3[:, j], c='r', marker='+')

                    # plot the cluster centers
                    plt.plot(c1[i], c1[j], 'go', label='centroid 1', color='black')
                    plt.plot(c2[i], c2[j], 'go', label='centroid 2', color='black')
                    plt.plot(c3[i], c3[j], 'go', label='centroid 3', color='black')
                    plt.xlabel(names[i])
                    plt.ylabel(names[j])
                    # plt.show()
                    k += 1
        plt.savefig('%i.png' % p)
        p += 1

#check percentage of people survived in each cluster
t1 = 0
t2 = 0
t3 = 0
for instance in cluster1:
    if instance[5] == 1:
        t1 += 1
for instance in cluster2:
    if instance[5] == 1:
        t2 += 1
for instance in cluster3:
    if instance[5] == 1:
        t3 += 1

print (t1/float(len(cluster1))*100), 'survived in cluster 1'
print (t2/float(len(cluster2))*100), 'survived in cluster 2'
print (t3/float(len(cluster3))*100), 'survived in cluster 3'

print 'Check the file directory for plotted graphs...'