# coding: utf-8

import numpy as np
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import scipy.spatial.distance as sd
import seaborn as sns

## python3.6 process-data.py --data ml-1m --user 1 --eps 0.6 --var 100

parser = argparse.ArgumentParser(description='Recommender system')
parser.add_argument('--data', type=str, default='ml-1m', metavar='D',
                    help='folder where data is located.')
parser.add_argument('--user', type=int, default=1, metavar='U',
                    help='userID to which the object should be recommended.')
parser.add_argument('--thres', type=float, default=0., metavar='T',
                    help='serendipity threshold for the recommendation.')
parser.add_argument('--eps', type=float, default=0, metavar='T',
                    help='epsilon threshold to build epsilon neighbourhood similarity graph.')
parser.add_argument('--k', type=int, default=0, metavar='T',
                    help='k threshold to build k-nn similarity graph.')
parser.add_argument('--var', type=float, default=1, metavar='T',
                    help='variance value to build similarity graph.')
args = parser.parse_args()

path = "../Datasets/"
dataset = path + args.data + "/"
print("Dataset path is: \'" + dataset + "\'")
usern = 1

############################
## LOAD AND PROCESS DATA  ##
############################

if (dataset == "../Datasets/ml-1m/"):
	n_objects = 3706//2
	nrows = None
	ext = ".dat"
	sep = "::"
	names_r = ['UserID', 'MovieID', 'Rating', 'Timestamp']
	names_u = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip']
	names_o = ['MovieID', 'Title', 'Genres']
	columns_r = ['UserID', 'MovieID', 'Rating', 'Timestamp']
	columns_u = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip']
	columns_o = ['MovieID', 'Title', 'Genres']
if (dataset == "../Datasets/ml-20m/"):
	n_objects = 20000264//8
	## OK if we use the first users
	nrows = 10000
	ext = ".csv"
	sep = ","
	names_r, names_o, names_u = None, None, ['UserID', 'Gender', 'Age', 'Occupation', 'Zip']
	columns_r = ['UserID', 'MovieID', 'Rating', 'Timestamp']
	columns_u = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip']
	columns_o = ['MovieID', 'Title', 'Genres']

rn = dataset + "ratings_u="+str(args.user)+"_no="+str(n_objects)+".dat"
un = dataset + "users_u="+str(args.user)+"_no="+str(n_objects)+".dat"
on = dataset + "objects_u="+str(args.user)+"_no="+str(n_objects)+".dat"
if (not os.path.exists(rn) or not os.path.exists(un) or not os.path.exists(on)):
	ratings = pd.read_table(dataset + 'ratings'+ext, sep=sep,
		encoding = 'latin1', engine = 'python', names=names_r, nrows=nrows)
	print("Ratings loaded! Size = " + str(ratings.size))
	objects = pd.read_table(dataset + 'movies'+ext,  sep=sep, 
		encoding = 'latin1', engine ='python', names=names_o, nrows=None)
	print("Objects loaded! Size = " + str(objects.size))
	users = pd.read_table(dataset + 'users'+ext,  sep=sep, 
		encoding = 'latin1', engine = 'python', names=names_u, nrows=args.user+1)
	print("Users loaded! Size = " + str(users.size))
	ratings.columns = columns_r
	users.columns = columns_u
	objects.columns = columns_o
	m = len(ratings.MovieID.unique())
	n = len(ratings.UserID.unique())
	n_objects = m//2
	print("#objects = " + str(m) + " -- #users = " + str(n))
	id_users = [args.user]
	X_users = users.loc[users.UserID.isin(id_users)]
	## Select less than m//2 objects with at least N=1 ratings with the user
	## to lessen the effect of missing data in the dataset
	id_objects = ratings.loc[ratings.UserID.isin(id_users)].MovieID.unique()[:n_objects]
	X_objects = objects.loc[objects.MovieID.isin(id_objects)]
	X = ratings.loc[ratings.MovieID.isin(id_objects)]
	X = X.loc[X.UserID.isin(id_users)]
	print("Dataframes are built")
	print("|X_objects| = " + str(X_objects.size//3) + " x 3")
	print("|X_users| = " + str(X_users.size//5) + " x 5")
	print("|X| = " + str(X.size//4) + " x 4")
	if (not os.path.exists(rn)):
		X.to_csv(rn, sep=',')
	if (not os.path.exists(un)):
		X_users.to_csv(un, sep=',')
	if (not os.path.exists(on)):
		X_objects.to_csv(on, sep=',')
	print("Done!")
else:
	X = pd.read_csv(rn, sep=',',  encoding = 'latin1', engine = 'python')
	X_users = pd.read_csv(un, sep=',', encoding = 'latin1', engine = 'python')
	X_objects = pd.read_csv(on,  sep=',', encoding = 'latin1', engine ='python')

#######################
# BUILD OBJECT GRAPH  #
#######################

## Building object feature matrix
if (True):
	fn = dataset + "features_u="+str(args.user)+"_no="+str(n_objects)+".dat"
	if (not os.path.exists(fn)):
		## Similarity on movies is using genres
		genres = X_objects["Genres"]
		n_obj = genres.size
		genres = [genres.iloc[i].split('|') for i in range(n_obj)]
		genres_u = list(set([y for x in genres for y in x]))
		genres_dict = {k: v for v, k in enumerate(genres_u)}
		n_genres = len(genres_u)
		## One-hot encoding of the movie genre
		features = np.zeros((n_obj, n_genres))
		for i in range(n_obj):
			is_present = [genres_dict[g] for g in genres[i]]
			features[i, np.ix_(is_present)] = 1
		np.savetxt(fn, features, delimiter=',')
		print("Done! Shape = " + str(np.shape(features)))
	else:
		features = np.loadtxt(fn, delimiter=',')

## Use Gaussian similarity function: s(x, y) = exp(-||x-y||^2_2/var) with fixed var
## graph will be an epsilon-neighbourhood 
## Helper function from TD's (source: P. Perrault)

## Ensures that the graph is connected
def is_connected(adj):
	n = np.shape(adj)[0]
	adjn=np.zeros((n,n))
	adji=adj.copy()
	for i in range(n):
		adjn+=adji
		adji=adji.dot(adj)
	return len(np.where(adjn == 0)[0])==0

if (True):
	gn = dataset + "graph_u="+str(args.user)+"_no="+str(n_objects)+"_eps="+str(args.eps)+"_k="+str(args.k)+"_var="+str(args.var)+".dat"
	if (not os.path.exists(gn)):
		assert args.eps + args.k != 0, "Choose either epsilon graph or k-nn graph"
		dists = sd.squareform(sd.pdist(features, "sqeuclidean"))
		W = np.exp(-dists / args.var)
		## Values between 0 and 1
		W = W-np.min(W)
		W = W/np.max(W)
		if args.eps:
			W[W < args.eps] = 0
		elif args.k:
			sort = np.argsort(W)[:, ::-1]
			mask = sort[:, args.k + 1:]
			for i, row in enumerate(mask):
				W[i, row] = 0
		## Remove autosimilarity and ensure symmetry
		np.fill_diagonal(W, 0)
		W = (W + W.T)/2
		## Remove weights
		W[W > 0] = 1
		assert is_connected(W), "Graph is not connected!"
		np.savetxt(gn, W, delimiter=',')
		print("Done! Shape = " + str(np.shape(W)))
		labels = [X_objects["MovieID"].iloc[i] for i in range(X_objects.MovieID.size)]
		W = pd.DataFrame(W, columns=labels)
		W.index = labels
		W.to_csv(gn, sep=',')
		print("Done!")
	else:
		labels = [X_objects["MovieID"].iloc[i] for i in range(X_objects.MovieID.size)]
		W = pd.read_csv(gn, sep=',', names = labels, 
			encoding = 'latin1', engine = 'python')
		W = W.iloc[range(1, W.size//len(labels)), :]
		assert is_connected(W), "Graph is not connected!"
		## Heatmap
		## Take subset of 20 samples to make it visible
		n = min(20, W.size//len(labels))
		x = W.iloc[range(n), range(n)]
		ax = sns.heatmap(x, linewidth=0.1, cbar=False)
		plt.title("Heatmap of the graph matrix")
		plt.ylabel("Movies")
		plt.xlabel("movies")
		plt.show()
