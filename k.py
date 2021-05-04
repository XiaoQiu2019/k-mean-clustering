#!/usr/bin/env python3
import os,random,math,csv
import numpy as np
import kmean
from multiprocessing.dummy import Pool as ThreadPool
def two_step_clustering(data1,k,A,B,w):
	data = data_processing(data1,w)
	final_data = data[:,A:B] # remove first 10h unstable data
	##### cluster first time
	d= []
	for i in range(len(final_data)):
		a,m = getMin(list(final_data[i,:]))
		d.append([i,a,0])
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
	index1,L1 = getMax([len(cluster_1[0]),len(cluster_1[1])])
	index2,L2 = getMax([len(cluster_2[0]),len(cluster_2[1])])
	final_cluster1 = cluster_1[index1]
	final_cluster2 = cluster_2[index1]
	return final_cluster1,final_cluster2
def main():
	#cluster experimental data
	k = 2
	x,y,z = kmean.parameter()
	D2 = []
	cell = []
	f3 = csv.reader(open("cell.csv",encoding='utf-8-sig'))
	for row in f3:
		D2.append(row[:100])
		cell.append(row)
	cluster1,cluster2 = two_step_clustering(D2,k,x,y,z)
	cell1 = []
	cell2 = []
	for x in cluster1:
		cell1.append(cell[x])
	for x in cluster2:
		cell2.append(cell[x])
	output_1=open('cell_group1.xls',"w",encoding="gbk")
	for i in range(len(cell1[0])):
		for j in range(len(cell1)):
			output_1.write(str(cell1[j][i]))
			output_1.write("\t")
		output_1.write("\n")
	output_1.close()

	output_2=open('cell_group2.xls',"w",encoding="gbk")
	for i in range(len(cell2[0])):
		for j in range(len(cell2)):
			output_2.write(str(cell2[j][i]))
			output_2.write("\t")
		output_2.write("\n")
	output_2.close()

	

main()
