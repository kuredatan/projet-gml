# coding: utf-8

import models
import argparse
import pandas as pd
import numpy as np
import random

random.seed(123456789)

parser = argparse.ArgumentParser(description='Recommender system')
parser.add_argument('--data', type=str, default='ml-1m', metavar='D',
                    help='folder where data is located.')
parser.add_argument('--user', type=int, default=1, metavar='U',
                    help='userID to which the object should be recommended.')
parser.add_argument('--thres', type=float, default=0., metavar='T',
                    help='serendipity threshold for the recommendation.')
parser.add_argument('--eps', type=float, default=0.6, metavar='T',
                    help='epsilon threshold to build epsilon neighbourhood similarity graph.')
parser.add_argument('--k', type=int, default=0, metavar='T',
                    help='k threshold to build k-nn similarity graph.')
parser.add_argument('--var', type=float, default=100, metavar='T',
                    help='variance value to build similarity graph.')
args = parser.parse_args()

##################
## LOADING DATA ##
##################

path = "../Datasets/"
dataset = path + args.data + "/"
if (dataset == "../Datasets/ml-1m/"):
	n_objects = 3706//2
rn = dataset + "ratings_u="+str(args.user)+"_no="+str(n_objects)+".dat"
X = pd.read_csv(rn, sep=',', names = ['UserID', 'MovieID', 'Rating', 'Timestamp'], 
	encoding = 'latin1', engine = 'python')
X = X.iloc[range(1, X.size//4), :]
on = dataset + "objects_u="+str(args.user)+"_no="+str(n_objects)+".dat"
X_objects = pd.read_csv(on,  sep=',', 
	names = ['MovieID', 'Title', 'Genres'],  
	encoding = 'latin1', engine ='python')
X_objects = X_objects.iloc[range(1, X_objects.size//3), :]
gn = dataset + "graph_u="+str(args.user)+"_no="+str(n_objects)+"_eps="+str(args.eps)+"_k="+str(args.k)+"_var="+str(args.var)+".dat"
labels = [X_objects["MovieID"].iloc[i] for i in range(X_objects.MovieID.size)]
W = pd.read_csv(gn, sep=',', names = labels, encoding = 'latin1', engine = 'python')
W = W.iloc[range(1, W.size//len(labels)), :]
fn = dataset + "features_u="+str(args.user)+"_no="+str(n_objects)+".dat"
features = np.loadtxt(fn, delimiter=',')

##################
## BENCHMARK    ##
##################

random_rec = models.Random(X, W, features)
own_rec = models.Recommender(X, W, features)

n_iter = 30
print("Number of iterations: " + str(n_iter))
random_rec.reinitialize()
try:
	for i in range(n_iter):
		random_rec.run()
	random_rec.plot_results()
except AssertionError:
	pass


