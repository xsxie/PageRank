# -*- coding: utf-8 -*-

"""
	Page Rank System
	Author: Xie xueshuo
	Date: 2019.01.09
"""

import os
import sys
import math
import time

#定义读取和保存文件路径
#file_path = os.path.dirname(os.path.realpath(sys.executable))+'\\' #获取可执行程序的当前路径，用于编译为可执行文件时使用
file_path = os.getcwd()+'\\' #获取文件所在的当前路径，运行.py文件时使用
save_path = file_path + 'data\\'#定义文件保存的路径

'''
    定义Graph类，由给定的数据文件WikiData.txt，生成数据的Directed graph，
    并考虑spider traps and dead ends等情况
    
    生成相应的Sparse Matrix
    
    对Sparse Matrix进行分块处理，进行性能优化
'''

class Graph(object):

    '''
    Graph类的初始化，节点和边的数量
    '''
    def __init__(self, block_cap):
        self.block_cap = block_cap
        self.node_number = {}
        self.edge_number = 0

    '''
    Graph类添加节点的属性
    '''
    def add_node_list(self, node_list):
        for i in node_list:
            self.add_node(i)

    '''
    Graph类添加节点的属性
    '''
    def add_node(self, node):
        if not node in self.nodes():
            self.node_number[node] = []

    '''
    Graph类添加边的属性
    '''
    def add_edge(self, edge):
        u, v = edge
        self.add_node_list(edge)  #
        if (v not in self.node_number[u]):
            self.node_number[u].append(v)
            self.edge_number += 1

    '''
    Graph类节点的属性
    '''
    def nodes(self):
        return self.node_number.keys()

    '''
    生成Sparse Matrix
    '''
    def generate_block_matrix(self):
        nodes = list(self.nodes())
        length_nodes = len(nodes)
        self.blocks = int(math.ceil(length_nodes / self.block_cap))
        self.R = {i: {} for i in range(self.blocks + 1)}
        self.add_edge_number = 0

        for i in range (length_nodes):
            k = math.floor(i / self.block_cap)
            node = nodes[i]
            self.R[k][node] = 1 / length_nodes
            '''
           对spider traps and dead ends等情况进行处理
            '''
            if len(self.node_number[node]) == 0:
                self.node_number[node].append(node)
                self.add_edge_number += 1
        self.Matric = {i: [] for i in range(self.blocks)}

        '''
       对Sparse Matrix进行分块处理，优化性能 
        '''
        for i in range (len(nodes)):
            node = nodes[i]
            block_number = math.floor(i / self.block_cap)
            degree = len(self.node_number[node])
            src = node
            for k in range(self.blocks):
                destination = [out_dgree for out_dgree in self.node_number[node] if out_dgree in self.R[k].keys()]
                if len(destination) > 0:
                    tmp = [src, block_number, degree]
                    tmp.append(destination)
                    self.Matric[k].append(tmp)
        print("\tThe number of edges are = %d,\tThe number of nodes are = %d,\tThe number of deadnodes are = %d" % (
        self.edge_number, len(self.node_number), self.add_edge_number))
        return self.Matric, self.R, self.blocks, length_nodes

'''
   读取给定的数据文件WikiData.txt，调用Graph类生成数据的Directed graph，并考虑spider traps and dead ends等情况 
'''
def generate_graph(data_file, block_cap):
	graph = Graph(block_cap)
	open_data = open(data_file)
	for line in open_data:
		data = line.strip().split()
		if len(data) != 2:
			print('The input data read error!')
		node_a = data[0]
		node_b = data[1]
		graph.add_edge((node_a, node_b))
	open_data.close()
	return graph

'''
    生成pagerank的结果并保存
    选取top100的pagerank结果并保存
'''
def page_rank(beta, srcfile, block_cap, dstpath):

    thresold = 1e-6

    '''
    读取WikiData.txt数据，并生成相应的Sparse Matrix
    
    考虑spider traps and dead ends等情况
    '''
    print("Reading input file WikiData.txt\n\tThe input file path is: %s"%(srcfile))
    start = time.time()
    g = generate_graph(srcfile,block_cap)
    M,R,blocks,N = g.generate_block_matrix()
    print('\tReading the input file into a sparse matrix, the time is: %fs\n'%(time.time()-start))

    '''
    多轮迭代求pagerank
    '''
    print("Iterative results Pagerank, teleport parameter = %f"%(beta))
    total_time = 0
    iteration = 0
    new_thresold = 1
    while(new_thresold > thresold):
        iteration += 1
        new_thresold = 0
        newR = {}
        start = time.time()
        new_rank = 0
        rS = 0

        for i in range(blocks):
            m = M[i]
            r = R[i]
            newR[i] = {node:0 for node in r.keys()}

            for line in m:
                src,blocknum,degree,outlink = line
                for node in outlink:
                    newR[i][node] += beta*R[blocknum][src]/degree
            new_rank += sum(newR[i].values())
            rS += sum(R[i].values())
        total_time += time.time()-start
        rank_number = 0
        p = (1 - new_rank) / N
        for i in range(blocks):
            start = time.time()
            for node in newR[i].keys():
                newR[i][node] += p
            new_thresold += sum([abs(newR[i][key] - R[i][key]) for key in R[i].keys()])
            total_time += time.time() - start
            rank_number += sum(newR[i].values())

        print('\tThe number of interation is: %d ,The number of Pagerank is: %f,\tThe new pagerank parametser = %f, \tThe new_thresold = %e'%(iteration, rank_number, new_rank,new_thresold))
        R = newR
    iteration_time = total_time/iteration
    print('\tThe average time of each iteration is: %f,The total time of iteration is: %f\n'%(iteration_time,total_time))

    '''
    生成全部的pagerank结果报保存到data/all_pagerank_result.txt文件
    '''
    print("Saving result file")
    top100dict = []
    print("\tThe final page rank is saving...")
    if not os.path.exists(dstpath):
        os.mkdir(dstpath)
    filename = "data/all_pagerank_result.txt"
    with open(filename, "w") as f_w:
	    for b in range(blocks):
		    keys = list(R[b].keys())
		    values = list(R[b].values())
		    top100dict.extend((sorted(R[b].items(),key = lambda x:x[1],reverse = True))[:100])
		    for key, value in zip(keys,values):
			    f_w.write(key + '\t\t' + str(value) + '\n')
    f_w.close()
    print("\tThe all page rank is save to %s"%(filename))

    '''
    生成top100的pagerank结果报保存到data/top100_pagerank_result.txt文件
    '''
    top100 = (sorted(top100dict,key = lambda x:x[1],reverse = True))[:100]
    top100name = "data/top100_pagerank_result.txt"
    with open(top100name, "w") as f_w:
	    for key, value in top100:
		    f_w.write(key + '\t\t' + str(value) + '\n')
    f_w.close()
    print("\tThe top100 page rank is save to %s" % (top100name))

if __name__ == '__main__':
    teleport_parameter = 0.85
    input_file = file_path + 'input\\WikiData.txt'
    block_cap = 2000
    page_rank(teleport_parameter, input_file, block_cap, save_path)