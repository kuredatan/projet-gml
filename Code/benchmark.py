# coding: utf-8

## python3.6 benchmark.py --user 2 --eps 0.6 --var 100

import models
import argparse
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

# For replication
#random.seed(123456789)

parser = argparse.ArgumentParser(description='Recommender system')
parser.add_argument('--data', type=str, default='ml-1m', metavar='D',
                    help='folder where data is located.')
parser.add_argument('--user', type=int, default=1, metavar='U',
                    help='userID to which the object should be recommended.')
parser.add_argument('--s', type=int, default=1, metavar='S',
                    help='serendipity threshold for the recommendation.')
parser.add_argument('--eps', type=float, default=0.6, metavar='E',
                    help='epsilon threshold to build epsilon neighbourhood similarity graph.')
parser.add_argument('--k', type=int, default=0, metavar='T',
                    help='k threshold to build k-nn similarity graph.')
parser.add_argument('--var', type=float, default=100, metavar='V',
                    help='variance value to build similarity graph.')
parser.add_argument('--method', type=str, default="random", metavar='M',
                    help='Method for recommender: \"random\", \"lagree\" or \"greedy\".')
parser.add_argument('--horizon', type=int, default=30, metavar='H',
                    help='Horizon.')
parser.add_argument('--niter', type=int, default=10, metavar='N',
                    help='Number of simulations.')
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

switch = {
     "random": models.Random(X, W, features),
     "lagree": models.Recommender(X, W, features, s=1, K=5, N=10),
     "greedy": models.Greedy(X, W, features),
}

if (not switch.get(args.method, None)):
	raise ValueError

horizon = args.horizon
n_iter = args.niter
method_name = args.method
print("Number of iterations: " + str(n_iter) + ", horizon: " + str(horizon))
recommender = switch[method_name]
regret = np.zeros(horizon)
volume = np.zeros(horizon)
nerr = 0
for _ in tqdm(range(n_iter), "Benchmark " + method_name):
	recommender.reinitialize()
	try:
		for __ in range(horizon):
			recommender.run(verbose=False)
		regret += np.array(recommender.regret_arr)
		volume += np.array(recommender.volume_arr)
	except AssertionError:
		print("Assertion error!")
		nerr += 1
		pass
## To smooth the regret curve
print("There have been " + str(nerr) + " errors.")
n_iter -= nerr
regret *= 1/float(n_iter)
recommender.regret_arr = regret.tolist()
volume *= 1/float(n_iter)
recommender.volume_arr = volume.tolist()
recommender.plot_results()
