# coding: utf-8

import numpy as np
import pandas as pd
from random import sample
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, KMeans
import scipy.spatial.distance as sd

#####################
## Recommenders    ##
#####################

#' ratings: data frame of ratings: at least with the 
# following columns ['UserID', 'MovieID', 'Rating'] each rating belongs to [|1,5|]
#' features: feature NumPy matrix with which the graph was built
#' graph: data frame of undirected, unweighted similarity graph on objects
#' perc_revealed: percentage of the revealed labels in the rating matrix
# at the beginning of the run
#' eps: array of the parameters of the Gaussian distribution from which the noise 
# is drawn
class Bandit(object):
	def __init__(self, ratings, graph, features, perc_masked=0.8, eps=[0,1]):
		self.n_ratings = ratings.size//len(ratings.columns)
		self.R = ratings[1:]
		self.indexes = self.R.MovieID
		self.n_obj = len(self.indexes)
		self.graph = graph
		self.eps = eps
		self.features = features
		## Binary function which indicates whether a node has been 
		## explored by the user
		self.n_masked = int(perc_masked*self.n_ratings)
		self.masked_labels = list(map(str, sample(list(map(int, self.indexes)), n_masked)))
		for i in self.masked_labels:
			assert i in self.indexes.values
		self.f = pd.DataFrame(np.ones((1, self.n_obj)), columns=self.indexes)
		for i in self.masked_labels:
			self.f[i] = 0
		self.regret_arr = []
		self.volume_arr = []

	def get_unexplored_items(self):
		candidates = []
		for l in self.indexes:
			if (not self.f[l][0]):
				candidates.append(l)
		return candidates

	def get_explored_items(self):
		candidates = self.get_unexplored_items()
		return list(set(candidates).symmetric_difference(self.indexes))

	def reinitialize(self):
		self.f = pd.DataFrame(np.ones((1, self.n_obj)), columns=self.indexes)
		for i in self.masked_labels:
			self.f[i] = 0
		self.regret_arr = []
		self.volume_arr = []

	def plot_results(self):
		n_iter = len(self.regret_arr)
		plt.figure(1)
		plt.subplot(121)
		plt.plot(np.array(self.regret_arr).cumsum(), "r-", label="Regret")
		plt.ylabel('(Expected) cumulative regret')
		plt.xlabel('Rounds')
		plt.legend()
		plt.subplot(122)
		plt.plot(np.array(self.volume_arr), "g-", label="Diversity")
		plt.ylabel('Diversity of the recommendations')
		plt.xlabel('Rounds')
		plt.legend()
		plt.show()

	def reward(self, action):
		assert action in self.indexes.values and not self.f[action][0], "{} not in the set of objects".format(action)
		## Draw noise from a Gaussian distribution
		sigma = np.random.normal(self.eps[0], self.eps[1])
		mu = float(self.R.iloc[np.where(action == self.indexes.values)[0].tolist()[0]].Rating)
		reward = round(mu + sigma, 3)
		self.f[action] = 1
		return reward

	## True optimal value among the actions that have not been played yet
	def best_reward(self):
		candidates = self.get_unexplored_items()
		n = len(candidates)
		assert n > 0, "All {} actions have been explored!".format(self.f.size)
		r = np.matrix(self.R[self.indexes.isin(candidates)].Rating.values, dtype=int)[0]
		best_reward = np.max(r)
		return best_reward
	## Quantifies the diversity of the recommendations
	## lbda is for regularization, 0 < lbda < 1
	def parallelotope_volume(self, lbda=0.5):
		explored = self.get_explored_items()
		explored = [np.where(e == self.indexes.values)[0].tolist()[0] for e in explored]
		V = self.features[explored, :]
		return np.sqrt(np.linalg.det(np.dot(V, V.T)+lbda*np.eye(len(explored))))

	def recommend(self):
		pass

	def update(self):
		pass

	def run(self, verbose=True):
		action, best_reward = self.recommend()
		reward = self.reward(action)
		if (verbose):
			print("- Action " + str(action) + " has been recommended, yielding a reward = " + str(reward))
		regret = best_reward-reward
		self.regret_arr.append(regret)
		volume = self.parallelotope_volume()
		self.volume_arr.append(volume)
		self.update()

## Random, uniform, recommendations			
class Random(Bandit):
	def recommend(self):
		super(Random, self).recommend()
		candidates = self.get_unexplored_items()
		n = len(candidates)
		assert n > 0, "All {} actions have been explored!".format(self.f.size)
		best_reward = self.best_reward()
		action = np.random.choice(candidates, p=[1/float(n)]*n)
		return action, best_reward

## Greedy recommendations (maximize diversity of the recommendation)		
class Greedy(Bandit):
	def recommend(self):
		super(Greedy, self).recommend()
		candidates = self.get_unexplored_items()
		n = len(candidates)
		assert n > 0, "All {} actions have been explored!".format(self.f.size)
		vol_ = np.zeros(n)
		for i in range(n):
			action = candidates[i]
			self.f[action] = 1
			v = self.parallelotope_volume()
			vol_[i] = v
			self.f[action] = 0
		action = candidates[np.argmax(vol_)]
		best_reward = self.best_reward()
		return action, best_reward

## Adaptation of [Lagrée et al.]'s method			
class Recommender(Bandit):
	#' s: serendipity threshold >= 1
	#' K: number of candidates at each step
	#' N: time budget
	def __init__(self, ratings, graph, features, s, K, N, perc_masked=0.8, eps=[0,1]):
		super(Recommender, self).__init__(ratings, graph, features, perc_masked=0.8, eps=[0,1])
		self.s = s
		self.K = K
		self.N = N
	## K: number of "centroids" to select
	## In practice, we perform K-means on the set of unexplored elements
	## at distance less than s edges of explored elements
	## We have ensured that the graph is connected in process-data.py
	def select_candidates(self):
		candidates = self.get_unexplored_items()
		n = len(candidates)
		assert n > 0, "All {} actions have been explored!".format(self.f.size)
		W = self.graph
		W[W > 0] = 1
		## Get elements at distance less than s of explored elements
		## that is, min_{y explored} ||x-y||_{edges} <= s
		W = np.linalg.matrix_power(W, self.s)
		candidates = [np.where(c == self.indexes.values)[0].tolist()[0] for c in candidates]
		W = np.delete(W[candidates, :], candidates, 1)
		candidates = np.array(candidates)[np.where(np.sum(W, 1) > 0)[0].tolist()].tolist()
		## K-means + whiten?
		## Use MiniBatchKMeans when |candidates| is too high
		F = self.features[candidates, :]
		km = KMeans(n_clusters=self.K, random_state=0).fit(F)
		centroids = km.cluster_centers_
		dists = sd.squareform(sd.pdist(np.concatenate((F, centroids), 0), "sqeuclidean"))
		dists = dists[-5:, :-5]
		## Select the node closest to each centroid
		res = []
		for i in range(self.K):
			res.append(candidates[np.argmin(dists[i, :])])
		candidates = [self.indexes.values[k] for k in res]
		supports = []
		for k in res:
			support = np.where(self.graph.values[k, :] == 1)[0].tolist()
			support_k = []
			for a in support:
				id_a = self.indexes.values[a]
				support_k.append([id_a, self.graph.values[k, a]])
			supports.append(support_k)
		return candidates, supports

	def recommend(self):
		super(Recommender, self).recommend()
		candidates, supports = self.select_candidates(self.s, self.K)
		nk = np.ones(self.K)
		sk = np.zeros((self.K, self.N))
		bk = np.zeros(self.K)
		for t in range(self.K):
			for k in candidates:
				## TODO observe the spread for candidate k
				pass
			pass
		for t in range(self.K+1, self.N):
			## TODO compute bk(t)
			## choose q = argmax bk(t)
			## nk(q) += 1
			## sk(q, t) = spread for q
			pass
		best_reward = self.best_reward(candidates)
		return action, best_reward

	def update(self):
		## Update weights on graph edges
		## TODO
		pass
		return action

	def best_reward(self, candidates):
		super(Recommender, self).best_reward()
		for c in candidates:
			assert c in self.indexes and not self.f[c][0], "{} not action".format(c)
		r = np.matrix(self.R[self.indexes.isin(candidates)].Rating.values, dtype=int)[0]
		best_reward = np.max(r)
		return best_reward
