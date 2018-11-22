# coding: utf-8

import numpy as np
import pandas as pd
import scipy.spatial.distance as sd
import argparse

parser = argparse.ArgumentParser(description='Recommender system')
parser.add_argument('--data', type=str, default='ml-1m', metavar='D',
                    help='folder where data is located.')
parser.add_argument('--user', type=str, default='', metavar='U',
                    help='user(name) to which the object should be recommended.')
parser.add_argument('--thres', type=int, default=0, metavar='T',
                    help='serendipity threshold for the recommendation.')
args = parser.parse_args()

dataset = "../Datasets/" + args.data + "/"
print("Dataset path is: \'" + dataset + "\'")

################
# LOAD DATA   ##
################

if (dataset == "../Datasets/ml-1m/"):
	ratings = pd.read_table(dataset + 'ratings.dat', sep='::', 
		names = ['UserID', 'MovieID', 'Rating', 'Timestamp'], 
		encoding = 'latin1', engine = 'python')
	objects = pd.read_table(dataset + 'movies.dat',  sep='::', 
		names = ['MovieID', 'Title', 'Genres'],  
		encoding = 'latin1', engine ='python')
	users = pd.read_table(dataset + 'users.dat',  sep='::', 
		names = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip'], 
		encoding = 'latin1', engine = 'python')
	## For testing pre-processing functions
	n_objects = 10
	n_users = 10
	idx_objects = ratings.MovieID.unique()[:n_objects]
	idx_users = ratings.UserID.unique()[:n_users]
	#users.set_index("UserID", inplace=True)
	#objects.set_index("MovieID", inplace=True)
	X_users = users.iloc[idx_users, :]
	X_objects = objects.iloc[idx_objects, :]
	print(ratings.UserID)
	print(ratings[ratings.UserID == idx_users[0] and ratings.MovieID == idx_objects[0]])
	raise ValueError
	X = ratings.loc[idx_users]
	X.set_index("MovieID", inplace=True)
	X = X.loc[idx_objects]
	print(X.columns)
	print(X) ##TODO


raise ValueError

#######################
# DATA PREPROCESSING  #
#######################

## Adapted from M. Pirotti's code for Reinforcement Learning recitation #2
ratings_count = ratings.groupby(by='MovieID', as_index=True).size()
# top_ratings = ratings_count.sort_values(ascending=False)[:N]
top_ratings = ratings_count[ratings_count>=N]
print(top_ratings.head(10))

#######################
# BUILD OBJECT GRAPH  #
#######################

## use Gaussian similarity function: s(x, y) = exp(- ||x-y||²_2/var) with fixed var
## graph will be an epsilon-neighbourhood 
## Helper function from TD's (source P. Perrault)
def build_graph(X, var=1, eps=0.5):
	dists = sd.squareform(sd.pdist(X, "sqeuclidean"))
	W = np.exp(-dists / var)
	W[W < eps] = 0
	return W

## Helper function from TD's (source P. Perrault)
def plot_edges_and_points(X, W, title='Object similarity graph'):
	n = len(X)
	G=nx.from_numpy_matrix(W)
	nx.draw_networkx_edges(G,X)
	for i in range(n):
		plt.plot(X[i,0], X[i,1], "ro")
	plt.title(title)
	plt.axis('equal')
       
## Helper function from TD's (source P. Perrault)
def plot_graph_matrix(X, W):
	plt.figure()
	plt.clf()
	plt.subplot(1,2,1)
	plot_edges_and_points(X,W)
	plt.subplot(1,2,2)
	plt.imshow(W, extent=[0, 1, 0, 1])
	plt.show() 

W = build_graph(X)
plot_graph_matrix(X, W)

raise ValueError

# from scipy.linalg import clarkson_woodruff_transform
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF

ratings = pd.read_table('ml-1m/ratings.dat', sep='::', 
                        names = ['UserID', 'MovieID', 'Rating', 'Timestamp'],
                        encoding = 'latin1',
                        engine = 'python')
movies  = pd.read_table('ml-1m/movies.dat',  sep='::',
                        names = ['MovieID', 'Title', 'Genres'], 
                        encoding = 'latin1',
                        engine ='python')
users   = pd.read_table('ml-1m/users.dat',  sep='::', 
                        names = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip'], 
                        encoding = 'latin1',
                        engine = 'python')

N = 1000
ratings_count = ratings.groupby(by='MovieID', as_index=True).size()
# top_ratings = ratings_count.sort_values(ascending=False)[:N]
top_ratings = ratings_count[ratings_count>=N]
top_ratings.head(10)

# movies_topN = movies[movies.MovieID.isin(top_ratings.index)]
# print('Shape: {}'.format(movies_topN.shape))
# movies_topN
ratings_topN = ratings[ratings.MovieID.isin(top_ratings.index)]
print('Shape: {}'.format(ratings_topN.shape))
ratings_topN.head(10)

n_users = ratings_topN.UserID.unique().shape[0]
n_movies = ratings_topN.MovieID.unique().shape[0]
print('Number of users = {} | Number of movies = {}'.format(n_users, n_movies))

R_df = ratings_topN.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
R_df.head()

M = R_df.as_matrix()
sparsity=round(1.0-np.count_nonzero(M)/float(n_users*n_movies),3)
print('Number of users = {} | Number of movies = {}'.format(n_users, n_movies))
print('The sparsity level is {}%'.format(sparsity*100))

K = 30

U, s, Vt = svds(M, k = K)
s=np.diag(s)
U = np.dot(U,s)
print('U: {}'.format(U.shape))
print('Vt: {}'.format(Vt.shape))

model = NMF(n_components=K, init='random', random_state=0)
W = model.fit_transform(M)
H = model.components_
print('W: {}'.format(W.shape))
print('H: {}'.format(H.shape))

np.savetxt('U.csv', W, delimiter=',') 
np.savetxt('Vt.csv', H, delimiter=',') 
