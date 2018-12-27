# coding: utf-8

import models
import argparse
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time
import os

## python3.6 benchmark.py --data ml-20m --user 1 --eps 0.6 --var 100 --niter 100 --horizon 100 --method random

# For replication
random.seed(123456789)

parser = argparse.ArgumentParser(description='Recommender system')
parser.add_argument('--data', type=str, default='ml-1m', metavar='D',
                    help='folder where data is located.')
parser.add_argument('--user', type=int, default=1, metavar='U',
                    help='userID to which the object should be recommended.')
parser.add_argument('--s', type=int, default=1, metavar='S',
                    help='serendipity threshold (>1) for the recommendation.')
parser.add_argument('--K', type=int, default=5, metavar='K',
                    help='candidate number for the [Lagrée et al.] recommendation.')
parser.add_argument('--alpha', type=float, default=0.1, metavar='A',
                    help='alpha parameter for the LinUCB recommendation.')
parser.add_argument('--lambda_', type=float, default=0.0001, metavar='L',
                    help='lambda parameter for the LinUCB recommendation.')
parser.add_argument('--epsilon', type=float, default=0.2, metavar='P',
                    help='epsilon parameter (between 0 and 1) for the epsilon-greedy recommendation.')
parser.add_argument('--N', type=int, default=10, metavar='N',
                    help='time budget for the [Lagrée et al.] recommendation.')
parser.add_argument('--eps', type=float, default=0.6, metavar='E',
                    help='epsilon threshold to build epsilon neighbourhood similarity graph.')
parser.add_argument('--k', type=int, default=0, metavar='T',
                    help='k threshold to build k-nn similarity graph.')
parser.add_argument('--var', type=float, default=100, metavar='V',
                    help='variance value to build similarity graph.')
parser.add_argument('--method', type=str, default="random", metavar='M',
                    help='Method for recommender: \"random\", \"lagree\" or \"greedy\".')
parser.add_argument('--horizon', type=int, default=100, metavar='H',
                    help='Horizon.')
parser.add_argument('--niter', type=int, default=100, metavar='N',
                    help='Number of simulations.')
args = parser.parse_args()

##################
## LOADING DATA ##
##################

path = "../Datasets/"
dataset = path + args.data + "/"

if not os.path.exists(dataset):
	os.mkdir(dataset)

if (dataset == "../Datasets/ml-1m/"):
	n_objects = 3706//2
if (dataset == "../Datasets/ml-20m/"):
	n_objects = 20000264//8

rn = dataset + "ratings_u="+str(args.user)+"_no="+str(n_objects)+".dat"
on = dataset + "objects_u="+str(args.user)+"_no="+str(n_objects)+".dat"
gn = dataset + "graph_u="+str(args.user)+"_no="+str(n_objects)+"_eps="+str(args.eps)+"_k="+str(args.k)+"_var="+str(args.var)+".dat"
fn = dataset + "features_u="+str(args.user)+"_no="+str(n_objects)+".dat"

X = pd.read_csv(rn, sep=',',  encoding = 'latin1', engine = 'python')
X_objects = pd.read_csv(on,  sep=',', encoding = 'latin1', engine ='python')
features = np.loadtxt(fn, delimiter=',')
labels = [X_objects["MovieID"].iloc[i] for i in range(X_objects.MovieID.size)]
W = pd.read_csv(gn, sep=',', names = labels, encoding = 'latin1', engine = 'python')
W = W.iloc[range(1, W.size//len(labels)), :]

##################
## BENCHMARK    ##
##################

switch = {
     "random": models.Random(X, W, features),
     "lagree": models.Recommender(X, W, features, s=args.s, K=args.K, N=args.N),
     "greedy": models.Greedy(X, W, features, epsilon=args.epsilon),
     "linUCB": models.LinUCB(X, W, features, lambda_=args.lambda_, alpha=args.alpha),
     "rotting": models.Rotting(X, W, features),
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
serendipity = np.zeros(horizon)
nerr = 0
times = []
for _ in tqdm(range(n_iter), "Benchmark " + method_name):
	recommender.reinitialize()
	try:
		tic = time()
		for __ in range(horizon):
			recommender.run(verbose=False)
		toc = time()
		times.append(toc-tic)
		regret += np.array(recommender.regret_arr)
		volume += np.array(recommender.volume_arr)
		serendipity += np.array(recommender.serendipity_arr)
	except AssertionError:
		print("Assertion error!")
		nerr += 1
		pass
runtime = round(sum(times)/n_iter, 2)
u = runtime/3600.0
h = int(u)
u = (u-h)*60
m = int(u)
s = int(round((u-m)*60))
h = str(h)+"h" if (h>0) else ""
m = str(m)+"min" if (m>0) else ""
s = str(s)+"sec" if (s>0) else ""
rt = h+m+s
## To smooth the regret/volume curve
print("There have been " + str(nerr) + " errors/" + str(n_iter) + " iterations.")
n_iter -= nerr
if (n_iter > 0):
	regret *= 1/float(n_iter)
	recommender.regret_arr = regret.tolist()
	volume *= 1/float(n_iter)
	recommender.volume_arr = volume.tolist()
	serendipity *= 1/float(n_iter)
	recommender.serendipity_arr = serendipity.tolist()
	recommender.plot_results()
	#plt.show()
	plt.savefig("../Results/"+args.data+"/"+args.method+"-"+rt+".png", bbox_inches="tight")

## Compare different values of serendipity threshold, ...
if (method_name == "lagree"):
	s_values = range(1, 6)
	## TODO test K and N 
	m = len(s_values)
	regret = np.zeros((horizon, m))
	volume = np.zeros((horizon, m))
	serendipity = np.zeros((horizon, m))
	colors = ["k","r","b","g","c","p","m"][:m+1]
	for i in range(m):
		recommender = models.Recommender(X, W, features, s=s_values[i], K=args.K, N=args.N)
		nerr = 0
		for _ in tqdm(range(n_iter), "Benchmark " + method_name + " s = " + str(s_values[i])):
			recommender.reinitialize()
			try:
				for __ in range(horizon):
					recommender.run(verbose=False)
				regret[:, i] += np.array(recommender.regret_arr)
				volume[:, i] += np.array(recommender.volume_arr)
				## Normalized values
				serendipity[:, i] += np.array(recommender.serendipity_arr)
			except AssertionError:
				print("Assertion error!")
				nerr += 1
				pass
		## To smooth the regret/volume curve
		print("There have been " + str(nerr) + " errors/" + str(n_iter) + " iterations.")
		n_iter -= nerr
		if (n_iter > 0):
			regret[:, i] *= 1/float(n_iter)
			volume[:, i] *= 1/float(n_iter)
			serendipity[:, i] *= 1/float(n_iter)
	plt.figure(figsize=(20, 4.19))
	plt.title("Regret and diversity variation depending on the value of s parameter")
	plt.subplot(131)
	for i in range(m):
		plt.plot(np.array(regret[:, i]).cumsum(), colors[i] + "-", label="Regret with s = " + str(s_values[i]))
	plt.ylabel('(Expected) cumulative regret')
	plt.xlabel('Rounds')
	plt.legend()
	plt.subplot(132)
	for i in range(m):
		plt.plot(volume[:, i], colors[i] + "-", label="Volume with s = " + str(s_values[i]))
	plt.ylabel('Diversity of the recommendations')
	plt.xlabel('Rounds')
	plt.legend()
	plt.subplot(133)
	for i in range(m):
		plt.plot(np.array(serendipity[:, i]).cumsum(), colors[i] + "-", label="Serendipity with s = " + str(s_values[i]))
	plt.ylabel('Serendipity value')
	plt.xlabel('Rounds')
	plt.legend()
	#plt.show()
	plt.savefig("../Results/"+args.data+"/lagree.png", bbox_inches="tight")
