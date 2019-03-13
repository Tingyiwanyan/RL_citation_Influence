import numpy as np
import json
import random
from numpy.linalg import inv
from numpy import linalg as LA
import matplotlib.pyplot as plt
import networkx as nx

#with open('/home/tingyi/Dropbox/RL_Citation_Influence/data/aminer_author2paper.json') as data_file:
#    aminer_author2paper = json.load(data_file)
#with open('/home/tingyi/Dropbox/RL_Citation_Influence/data/aminer_coauthor.json') as data_file2:
#    aminer_coauthor = json.load(data_file2)
"""
Read Data
"""
rowcount = 0
with open('/Users/tingyiwanyan/RL_citation_Influence/data/aminer_reference.json') as data_file:
    data = json.load(data_file)
    #aminer_reference = json.load(data_file3)
    #for line in data_file3:
    #    data.append(json.load(line))
    #for line in data_file:
    #    output = json.dumps(line)
    #    rowcount +=1
    #    print(output)
    #    if rowcount == 1000:
    #        break

"""
Construct Citation Network, using node2vec
See reference:https://cs.stanford.edu/people/jure/pubs/node2vec-kdd16.pdf
"""
def count_ref_num(data,index):
    """
    Count how many citations in one paper
    """
    cpid_origin = data['RECORDS'][index]['cpid']
    cpid = data['RECORDS'][index]['cpid']
    index_count = index
    count = 1
    while cpid == cpid_origin:
        index_count += 1
        cpid = data['RECORDS'][index_count]['cpid']
        count += 1
    return count

#def one_level_walk_bfs(G,node,level_count,data):
#    """
#    generate one level of graph, add graph weight as transition probability,
#    uniformly initialize transititon probability
#    """
#    cpid = node['cpid']
#    G.add_weighted_edges_from()


def policy_evaluation(G,gamma):
    """
    perform policy evaluation
    """
    error_max = 0
    for i in G.node.keys():
        value_old = G.node[i]['value']
        G.node[i]['value'] = 0
        for j in G.neighbors(i):
            G.node[i]['value'] += G.edge[i][j]['weight']*(G.node[j]['cite_num']+gamma*G.node[j]['value'])
        error = np.abs(value_old - G.node[i]['value'])
        if error > error_max:
            error_max = error
    while(error_max > 10):
        for i in G.node.keys():
            value_old = G.node[i]['value']
            G.node[i]['value'] = 0
            for j in G.neighbors(i):
                G.node[i]['value'] += G.edge[i][j]['weight']*(G.node[j]['cite_num']+gamma*G.node[j]['value'])
            error = np.abs(value_old - G.node[i]['value'])
            if error > error_max:
                error_max = error


"""
Initiate Parameters
"""
walk_len = 10
num_origin = 2000
G = nx.DiGraph()
index = 0
num_count = 0
gamma = 0.9
cpid = data['RECORDS'][0]['cpid']
citation_num = count_ref_num(data,index)
random_init_prob = 1/citation_num

while(num_count != num_origin):
    if cpid != data['RECORDS'][index]['cpid']:
        cpid = data['RECORDS'][index]['cpid']
        citation_num = count_ref_num(data,index)
        random_init_prob = 1/citation_num
        pid = data['RECORDS'][index]['pid']
        G.add_weighted_edges_from([(cpid,pid,random_init_prob)])
        G.add_node(pid,cite_num=0)
        G.add_node(cpid,value=0)
        G.add_node(cpid,cite_num=citation_num)
        G.add_node(pid,value=0)
        #num_count += 1
    else:
        one_level_count = 0
        while one_level_count != citation_num:
            pid = data['RECORDS'][index]['pid']
            G.add_weighted_edges_from([(cpid,pid,random_init_prob)])
            G.add_node(pid,cite_num=0)
            G.add_node(cpid,value=0)
            G.add_node(cpid,cite_num=citation_num)
            G.add_node(pid,value=0)
            index += 1
            one_level_count += 1
        num_count += 1
