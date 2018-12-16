# coding: utf-8

import numpy as np
import pandas as pd
from random import sample
import matplotlib.pyplot as plt

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
	def __init__(self, ratings, graph, features, perc_masked=0.33, eps=[0,1]):
		self.n_ratings = ratings.size//len(ratings.columns)
		self.R = ratings[1:]
		self.R.set_index("MovieID")
		self.indexes = list(map(int, self.R.MovieID.unique()))
		self.n_movies = len(self.indexes)
		self.graph = graph
		self.eps = eps
		self.features = features
		## Binary function which indicates whether a node has been 
		## explored by the user
		n_masked = int(perc_masked*self.n_ratings)
		self.masked_labels = sample(list(map(int, self.indexes)), n_masked)
		self.f = pd.DataFrame(np.ones((1, self.n_movies)), columns=self.R.MovieID)
		for i in self.masked_labels:
			self.f[i] = 0
		self.regret_arr = []
		self.volume_arr = []

	def get_unexplored_items(self):
		candidates = []
		for l in self.f.columns:
			if (not self.f[l][0]):
				candidates.append(l)
		return candidates

	def get_explored_items(self):
		candidates = []
		for l in self.f.columns:
			if (self.f[l][0]):
				candidates.append(l)
		return candidates

	def reinitialize(self):
		self.f = pd.DataFrame(np.ones((1, self.n_movies)), columns=self.R.MovieID)
		for i in self.masked_labels:
			self.f[i] = 0
		self.regret_arr = []
		self.volume_arr = []
		print("* Reinitialized! *")

	def plot_results(self):
		n_iter = len(self.regret_arr)
		plt.figure(1)
		plt.subplot(121)
		plt.plot(np.array(self.regret_arr).cumsum(), "r-", label="Regret")
		plt.ylabel('(Expected) cumulative regret')
		plt.xlabel('Rounds')
		plt.legend()
		plt.subplot(122)
		plt.plot(np.array(self.volume_arr), "g+", label="Diversity")
		plt.ylabel('Diversity of the recommendations')
		plt.xlabel('Rounds')
		plt.legend()
		plt.show()

	def reward(self, action):
		assert action in self.indexes and not self.f[action][0], "{} not in the set of objects".format(action)
		## Draw noise from a Gaussian distribution
		sigma = np.random.normal(self.eps[0], self.eps[1])
		mu = float(self.R.iloc[self.indexes.index(action)].Rating)
		reward = round(mu + sigma, 3)
		self.f[action] = 1
		return reward

	## True optimal value among the actions that have not been played yet
	def best_reward(self):
		candidates = list(map(str, self.get_unexplored_items()))
		r = np.matrix(self.R[self.R.MovieID.isin(candidates)].Rating.values, dtype=int)[0]
		best_reward = np.max(r)
		return best_reward

	def parallelotope_volume(self):
		explored = list(map(int, self.get_explored_items()))
		explored = [self.indexes.index(e) for e in explored]
		V = self.features[explored, :]
		return np.sqrt(np.linalg.det(np.dot(V, V.T)))

	def recommend(self):
		pass

	def update(self):
		pass

	def run(self):
		action, best_reward = self.recommend()
		reward = self.reward(action)
		print("- Action " + str(action) + " has been recommended, yielding a reward = " + str(reward))
		regret = best_reward-reward
		self.regret_arr.append(regret)
		volume = self.parallelotope_volume()
		self.volume_arr.append(volume)

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

## [Lagrée et al. method]			
class Recommender(Bandit):
	## K: number of "centroids" to select
	def select_candidates(K=5):
		pass

	def recommend(thres):
		super(Recommender, self).recommend()
		candidates = self.select_candidates()
		best_reward = self.best_reward(candidates)
		return action, best_reward

	def update(self):
		pass
		return action

	def best_reward(self, candidates):
		super(Recommender, self).best_reward()
		for c in candidates:
			assert c in self.indexes and not self.f[c][0], "{} not action".format(action)
		r = np.matrix(self.R[self.R.MovieID.isin(candidates)].Rating.values, dtype=int)[0]
		best_reward = np.max(r)
		return best_reward
	#' thres: serendipity threshold
	def run(thres=0.):
		super(Recommender, self).run()
		action, best_reward = self.recommend(thres)
		reward = self.reward(action)
		print("- Action " + str(action) + " has been recommended, yielding a reward = " + str(reward))
		regret = best_reward-reward
		self.regret_arr.append(regret)
		volume = self.parallelotope_volume()
		self.volume_arr.append(volume)
		self.update()
