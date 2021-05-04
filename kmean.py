#!/usr/bin/env python3
import os,random,math,csv
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool

def moving_average(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window,'same')

def data_processing_verify(data,window_size):
	sample_size = 500
	a = random.sample(data,sample_size)
	b = np.array(a,dtype=float)
	smooth_data = np.zeros((len(a),len(a[0])))	
	for i in range(len(b)):
		smooth_data[i,:] = moving_average(b[i,:],window_size)
	return smooth_data

def data_processing(data,window_size):
	a = data
	b = np.array(a,dtype=float)
	smooth_data = np.zeros((len(a),len(a[0])))	
	for i in range(len(b)):
		smooth_data[i,:] = moving_average(b[i,:],window_size)
	return smooth_data

def getMin(L):
	m=L[0]
	for i in L:
		if i <= m:
			m=i
	return L.index(m),m

def getMax(L):
	m=L[0]
	for i in L:
		if i >= m:
			m=i
	return L.index(m),m

def dist(vecA,vecB):
	#input pure data,no name
	c=[]
	for i in range(len(vecA)):
		a=float(vecA[i])-float(vecB[i])
		c.append(a)
	sq=0
	for i in range(len(c)):
		sq+=c[i]**2
	dist=sq**0.5
	return dist

def meanlist(lists,data):
	c=[]
	col=len(data[0])
	l=len(lists)
	mean=[]
	for j in range(1,col):
		s=0
		for i in range(len(lists)):
			s+=data[int(lists[i])][j]
		m=s/l
		mean.append(m)
	return mean

def quickSort(L,start,end):
	i,j=start,end
	if i>j:
		return L
	key=L[i][1]
	while i<j:
		while i<j and key<=L[j][1]:
			j=j-1
		L[i],L[j]=L[j],L[i]
		while i<j and key>=L[i][1]:
			i=i+1
		L[j],L[i]=L[i],L[j]
	L[i][1]=key
	quickSort(L,start,i-1)
	quickSort(L,j+1,end)
	return L

def equal(listA,listB):
	listA=quickSort(listA,0,len(listA)-1)
	listB=quickSort(listB,0,len(listB)-1)
	for i in range(len(listA)):
		if listA[i]!=listB[i]:
			change=True
		else:
			change=False
	return change

def initcent(data,k):
	#output pure number data
	row=len(data)
	col=len(data[0])
	rang=[i for i in range(1,row)]
	lists=random.sample(rang,k)  
	centroids=[]
	for i in range(k):
		centroids.append(data[lists[i]][1:])
	return centroids

def kmean(data,k):
	row=len(data)
	col=len(data[0])
	Cluster=[]
	Wcss=[]
	for i in range(100*k):  #using dynamic iterTime
		centroids=initcent(data,k)#giving initial centroids
		change=True
		while change:     # running unitl the centroids do not change
			change=False
			cluster=[[] for i in range(k)]
			optcluster=[[] for i in range(k)]
			distance=[]
			for i in range(1,row):
				dis=[]
				for j in range(k):	
					d=dist(data[i][1:],centroids[j])  #calculate the distance between data ponits and each centroid
					dis.append(d)
				
				min_index,min_dist=getMin(dis)  # finding the nearest centroid and the distance
				cluster[min_index].append(i)    # put row position into the cluster
				optcluster[min_index].append(data[i][0]) #put the name of the datapoint into optcluster
				distance.append(min_dist) #store min distances 
			lists=list(range(1,row))   
			for i in range(k):
				if len(cluster[i])==0:    # giving every empty cluster with a random datapoint
					cluster[i]=random.sample(lists,1)
					optcluster[i]=data[cluster[i][0]][0]
			new_centroids=[]
			for i in range(k):		#calculate new centroids
				new_centroids.append(meanlist(cluster[i],data))

			change=equal(new_centroids,centroids) # judging the centroids changes
			centroids=new_centroids
			wcss=0
			for i in range(len(distance)):
				wcss+=(distance[i])**2
		Wcss.append(wcss)
		Cluster.append(optcluster)
		index,min_wcss=getMin(Wcss)
	return Cluster[index]


def two_step_clustering_verify(data1,data2,k,A,B):
	#inital phase
	final_data1 = data1[:,A:B] # remove first 10h unstable data
	final_data2 = data2[:,A:B]
	final_data = np.vstack((final_data1,final_data2))
	#plot_show(final_data)
	##### cluster first time
	d= []
	for i in range(len(final_data1)):
		a,m = getMin(list(final_data1[i,:]))
		d.append([i,a,0])
	for i in range(len(final_data2)):
		a,m = getMin(list(final_data2[i,:]))
		d.append([i+len(final_data1),a,0])
	cluster = kmean(d,k)
	#print(cluster)
	##### sub-cluster 
	group1 = cluster[0]
	group2 = cluster[1]

	
	data3 = final_data[group1,:]
	data4 = final_data[group2,:]

	x = []
	for i in range(len(data3)):
		a,m = getMax(list(data3[i,:]))
		b,m = getMin(list(data3[i,:]))
		x.append([group1[i],a,b])
	cluster_1 = kmean(x,k)
	y = []
	for i in range(len(data4)):
		a,m = getMax(list(data4[i,:]))
		b,m = getMin(list(data4[i,:]))
		y.append([group2[i],a,b])
	cluster_2 = kmean(y,k)
	#print(cluster_1)
	#print(cluster_2)
	index1,L1 = getMax([len(cluster_1[0]),len(cluster_1[1])])
	index2,L2 = getMax([len(cluster_2[0]),len(cluster_2[1])])
	final_cluster1 = cluster_1[index1]
	final_cluster2 = cluster_2[index1]
	#print(final_cluster1)
	#print(final_cluster2)
	return final_cluster1,final_cluster2

def accuracy(cluster1,cluster2,data1,data2):
	L1 = [i for i in range(len(data1))]
	L2 = [i for i in range(len(data1),len(data1)+len(data2))]
	a1 = list(set(cluster1).intersection(set(L1)))	
	a2 = list(set(cluster1).intersection(set(L2)))
	a3 = list(set(cluster2).intersection(set(L1)))
	a4 = list(set(cluster2).intersection(set(L2)))
	#print(cluster1)
	#print(L1)
	acc1 = len(a1)/len(cluster1)
	acc2 = len(a2)/len(cluster1)
	acc3 = len(a3)/len(cluster2)
	acc4 = len(a4)/len(cluster2)
	#print(acc1,acc2,acc3,acc4)
	ACC1 = max(acc1,acc2)
	ACC2 = max(acc3,acc4)
	ACC = (ACC1+ACC2)/2
	return ACC

def verify(P0,P1,k,A,B,w):
	data1 = data_processing_verify(P0,w)

	data2 = data_processing_verify(P1,w)

	cluster1,cluster2 = two_step_clustering_verify(data1,data2,k,A,B)
	acc = accuracy(cluster1,cluster2,data1,data2)
	return acc

def parameter():
	k=2
	########################################################
	#verification of the method
	D0 = []
	D1 = []
	f1 = csv.reader(open("cell1.csv",encoding='utf-8-sig'))
	f2 = csv.reader(open("cell2.csv",encoding='utf-8-sig'))	
	for row in f1:
		D0.append(row[:100])
	for row in f2:
		D1.append(row[:100])
	Acc = np.zeros((25,25,10))
	for i in range(0,25):
		a = int(i+5)
		for j in range(0,25):
			b = int(j + a + 5) 
			for q in range(1,11):
				Acc[i,j,q-1] = verify(D0,D1,k,a,b,q)
	pos = np.unravel_index(np.argmax(Acc),Acc.shape)
	x,y,z = pos[0],pos[1],pos[2]
	max_acc = Acc[x,y,z]
	print(max_acc,x,y,z)

	
if __name__ == '__main__':
	parameter()
