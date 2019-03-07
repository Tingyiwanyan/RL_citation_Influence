import numpy as np
from numpy import loadtxt
import json
from numpy.linalg import inv
from numpy import linalg as LA
import matplotlib.pyplot as plt

with open('/home/tingyi/Dropbox/RL_Citation_Influence/data/aminer_author2paper.json') as data_file:
    aminer_author2paper = json.load(data_file)
#with open('/home/tingyi/Dropbox/RL_Citation_Influence/data/aminer_coauthor.json') as data_file2:
#    aminer_coauthor = json.load(data_file2)
with open('/home/tingyi/Dropbox/RL_Citation_Influence/data/aminer_reference.json') as data_file3:
    aminer_reference = json.load(data_file3)
