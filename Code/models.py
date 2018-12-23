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

########################################################################################
########################################################################################
## General model

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
		self.masked_labels = list(map(str, sample(list(map(int, self.indexes)), self.n_masked)))
		for i in self.masked_labels:
			assert i in self.indexes.values
		self.f = pd.DataFrame(np.ones((1, self.n_obj)), columns=self.indexes)
		for i in self.masked_labels:
			self.f[i] = 0
		self.regret_arr = []
		self.volume_arr = []
		self.serendipity_arr = []
		#self.W = self.graph.values	
		#for i in range(self.n_obj):
		#	action = self.indexes.values[i]
		#	rew = np.array([self.draw_reward(action) for _ in range(self.n_obj)])
		#	self.W[:, i] = np.multiply(self.W[:, i], rew)

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
		self.reward_arr = []
		self.serendipity_arr = []
		#self.W = self.graph.values	
		#for i in range(self.n_obj):
		#	action = self.indexes.values[i]
		#	rew = np.array([self.draw_reward(action) for _ in range(self.n_obj)])
		#	self.W[:, i] = np.multiply(self.W[:, i], rew)

	def plot_results(self):
		n_iter = len(self.regret_arr)
		plt.figure(1)
		plt.title("Regret and diversity of the recommendations")
		plt.subplot(131)
		plt.plot(np.array(self.regret_arr).cumsum(), "r-", label="Regret")
		plt.ylabel('(Expected) cumulative regret')
		plt.xlabel('Rounds')
		plt.legend()
		plt.subplot(132)
		plt.plot(np.array(self.volume_arr), "g-", label="Diversity")
		plt.ylabel('Diversity of the recommendations')
		plt.xlabel('Rounds')
		plt.legend()
		plt.subplot(133)
		plt.plot(np.array(self.serendipity_arr).cumsum(), "b-", label="Serendipity (MC estimate)")
		plt.ylabel('Serendipity value')
		plt.xlabel('Rounds')
		plt.legend()
		plt.show()

	def draw_reward(self, action):
		assert action in self.indexes.values, "{} not in the set of objects".format(action)
		## Draw noise from a Gaussian distribution
		sigma = np.random.normal(self.eps[0], self.eps[1])
		mu = float(self.R.iloc[np.where(action == self.indexes.values)[0].tolist()[0]].Rating)
		reward = round(mu + sigma, 3)
		return reward

	def reward(self, action):
		assert not self.f[action][0], "{} already explored!".format(action)
		reward = self.draw_reward(action)
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

	def serendipity_value(self):
		#f = np.where(self.f.values[0] == 1)[0]
		#nf = np.where(self.f.values[0] == 0)[0]
		## do not yield any more reward
		#W = self.W+1
		#W[f, f] = 0
		#W -= 1
		#W = (W+W.T)/2
		#D = np.diag(np.sum(W > 0, 1))
		#L = np.dot(np.linalg.inv(D), D-W)
		#svalue = np.dot(self.f.values, np.dot(L, self.f.values.T))[0][0]
		return 0#svalue

	def recommend(self):
		pass

	def update(self, reward, action):
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
		serendipity = self.serendipity_value()
		self.serendipity_arr.append(serendipity)
		self.reward_arr.append(reward)
		self.update(reward, action)

########################################################################################
########################################################################################

## Random, uniform, recommendations			
class Random(Bandit):
	def recommend(self):
		candidates = self.get_unexplored_items()
		n = len(candidates)
		assert n > 0, "All {} actions have been explored!".format(self.f.size)
		best_reward = self.best_reward()
		action = np.random.choice(candidates, p=[1/float(n)]*n)
		return action, best_reward

########################################################################################
########################################################################################

## Greedy recommendations (maximize diversity of the recommendation)
## - with probability 1-epsilon the action maximizing the diversity measure is selected
## - with probability epsilon a random action is selected		
class Greedy(Bandit):
	def __init__(self, ratings, graph, features, epsilon, perc_masked=0.8, eps=[0,1]):
		super(Greedy, self).__init__(ratings, graph, features, perc_masked=perc_masked, eps=eps)
		self.epsilon = epsilon

	def recommend(self):
		candidates = self.get_unexplored_items()
		n = len(candidates)
		assert n > 0, "All {} actions have been explored!".format(self.f.size)
		vol_ = np.zeros(n)
		if (np.random.choice([0, 1], p=[self.epsilon, 1-self.epsilon])):
			action = np.random.choice(candidates, p=[1/float(n)]*n)
		else:
			for i in range(n):
				action = candidates[i]
				self.f[action] = 1
				v = self.parallelotope_volume()
				vol_[i] = v
				self.f[action] = 0
			action = candidates[np.argmax(vol_)]
		best_reward = self.best_reward()
		return action, best_reward

########################################################################################
########################################################################################

## Linear UCB
## Code adapted from Matteo Pirotti for RL classes		
class LinUCB(Bandit):
	def __init__(self, ratings, graph, features, lambda_, alpha, random_state=0, noise=0., perc_masked=0.8, eps=[0,1]):
		super(LinUCB, self).__init__(ratings, graph, features, perc_masked=perc_masked, eps=eps)
		self.noise = noise
		self.local_random = np.random.RandomState(random_state)
		self.real_theta = self.local_random.randn(self.n_features)
		self.theta_hat = np.zeros((self.n_features))
		## Estimate of np.dot(Z_t^T, Z_t) = sum_{1 <= u <= t-1} np.dot(phi_{a_u}^T, phi_{a_u})
		self.ZZ = np.zeros((self.n_features, self.n_features))
		## Estimate of np.dot(Z_t^T, y_t) = sum_{1 <= u <= t-1} r_{u}*phi_{a_u}^T
		self.Zy = np.zeros((self.n_features))
		self.lambda_ = lambda_
		self.I = np.eye(self.n_features)
		self.alpha = alpha

	def reward(self, action):
		assert action in self.indexes.values, "{} not in the set of objects".format(action)
		a = np.where(action == self.indexes.values)[0].tolist()[0]
		reward = np.dot(self.features[a, :], self.real_theta) + self.noise * self.local_random.randn(1)[0]
		return reward

	def reinitialize(self):
		super(LinUCB, self).reinitialize()
		self.theta_hat = self.theta_hat*0+1
		self.ZZ = self.ZZ*0
		self.Zy = self.Zy*0

	def update(self, reward, action):
		a = np.where(action == self.indexes.values)[0].tolist()[0]
		## Update on parameters
		self.Zy += reward*self.features[a, :]
		z_t = np.reshape(self.features[a, :], (1, self.n_features))
		self.ZZ += np.multiply(np.transpose(z_t), z_t)
		## lambda_ is kept constant
		self.theta_hat = np.dot(np.linalg.inv(self.ZZ+self.lambda_*self.I), self.Zy)

	def best_reward(self):
		candidates = self.get_unexplored_items()
		n = len(candidates)
		assert n > 0, "All {} actions have been explored!".format(self.f.size)
		candidates = [np.where(c == self.indexes.values)[0].tolist()[0] for c in candidates]
		D = np.dot(self.features, self.real_theta)[candidates]
		return np.max(D)

	def recommend(self):
		candidates = self.get_unexplored_items()
		n = len(candidates)
		assert n > 0, "All {} actions have been explored!".format(self.f.size)
		unexplored = [np.where(c == self.indexes.values)[0].tolist()[0] for c in candidates]
		self.theta_hat = np.reshape(self.theta_hat, (self.n_features, 1))
		phi = lambda a : np.reshape(self.features[a, :], (self.n_features, 1))
		phiT = lambda a : np.transpose(phi(a))
		est = np.linalg.inv(self.ZZ+self.lambda_*self.I)
		beta = lambda a : self.alpha*np.sqrt(np.dot(phiT(a), np.dot(est, phi(a))))
		arm_values = [np.dot(phiT(a), self.theta_hat)[0, 0]+beta(a)[0,0] for a in unexplored]
		action = candidates[arm_values.index(max(arm_values))]
		best_reward = self.best_reward()
		return action, best_reward

	@property
	def n_features(self):
		return self.features.shape[1]

	@property
	def n_actions(self):
		return self.features.shape[0]

########################################################################################
########################################################################################

## Rotting Bandit			
class Rotting(Bandit):
	pass

########################################################################################
########################################################################################

## Ad-Hoc method from [Abbassi et al.]			
class AdHoc(Bandit):
	pass

########################################################################################
########################################################################################

## Adaptation of [Lagrée et al.]'s method			
class Recommender(Bandit):
	#' s: serendipity threshold >= 1
	#' K: number of candidates at each step
	#' N: time budget
	def __init__(self, ratings, graph, features, s, K, N, perc_masked=0.8, eps=[0,1]):
		super(Recommender, self).__init__(ratings, graph, features, perc_masked=perc_masked, eps=eps)
		self.s = s
		self.K = K
		self.N = N
	## In practice, we perform K-means on the set of unexplored elements
	## at distance less than s edges of explored elements
	## We have ensured that the graph is connected in process-data.py
	def select_candidates(self):
		unexplored = self.get_unexplored_items()
		n = len(unexplored)
		assert n > 0, "All {} actions have been explored!".format(self.f.size)
		W = self.graph
		W[W > 0] = 1
		## Get elements at distance less than s of explored elements
		## that is, min_{y explored} ||x-y||_{edges} <= s
		## (W^s)_{i,j} gives the number of walks of length s between vertices i and j
		## The number of walks of length at most s between i and j is thus
		## sum_{k \leq s} (W^k)_{i,j}
		WW = W
		for k in range(1, self.s):
			W = np.dot(W, W)
			WW += W
		W = np.asarray(WW > 0, dtype=int)
		candidates = [np.where(c == self.indexes.values)[0].tolist()[0] for c in unexplored]
		W = np.delete(W[candidates, :], candidates, 1)
		candidates = np.array(candidates)[np.where(np.sum(W, 1) > 0)[0].tolist()].tolist()
		## K-means + whiten?
		## Use MiniBatchKMeans when |F| is too high
		## Not using centroid graph with Floyd-Warshall
		nclusters = min(self.K, len(candidates))
		F = self.features[candidates, :]
		if (np.prod(np.shape(F)) > 500):
			km = MiniBatchKMeans(n_clusters=nclusters, random_state=0).fit(F)
		else:
			km = KMeans(n_clusters=nclusters, random_state=0).fit(F)
		centroids = km.cluster_centers_
		dists = sd.squareform(sd.pdist(np.concatenate((F, centroids), 0), "sqeuclidean"))
		dists = dists[-nclusters:, :-nclusters]
		## Select the node closest to each centroid
		res = []
		for i in range(nclusters):
			#print((F[np.argmin(dists[i, :]), :] == np.asarray(centroids[i,:] > 0.5, dtype=float)).all())
			res.append(candidates[np.argmin(dists[i, :])])
		candidates = [self.indexes.values[k] for k in res]
		supports = []
		## Support of unexplored items for each candidates
		for k in res:
			support = np.where(self.graph.values[k, :] == 1)[0].tolist()
			support_k = []
			for a in support:
				id_a = self.indexes.values[a]
				## if unexplored
				if (not self.f[id_a][0]):
					#support_k.append([id_a, self.graph.values[k, a]])
					support_k.append(id_a)
			supports.append(support_k)
		return candidates, supports
	## activated means reward > 3 (where max reward = 5, min reward = 1)
	def is_active_node(self, reward):
		return(int(reward > 3))
	## The recommend will return the candidate with the 
	## highest expected spread, as described in [Lagrée et al.]
	def recommend(self):
		candidates, supports = self.select_candidates()
		## Need to reinitialize because the setting changes
		nk = np.ones(self.K)
		## sk[k, t, i] = 1 iff. node i is activated by candidate k at time t
		sk = np.zeros((self.K, self.N, self.n_obj))
		## uk[k, i] = 1 iff. node i has been activated exactly once by candidate k
		uk = np.zeros((self.K, self.n_obj))
		## zk[k, i] = 1 iff. node i has never been activated by candidate k
		zk = np.ones((self.K, self.n_obj))
		## Good-Turing estimator
		rk = np.zeros(self.K)
		## Compute bk(t), as defined in [Lagrée et al.]
		bk = np.zeros(self.K)
		lbdk = np.zeros(self.K)
		for t in range(self.K):
			for i_k in range(len(candidates)):
				## Observe the "expected spread" for candidate k
				#spread = [self.is_active_node(self.reward(s[0])) if (s[0] in supports[i_k]) else 0 for s in range(self.n_obj)]
				spread = [self.is_active_node(self.reward(s)) if (s in supports[i_k]) else 0 for s in range(self.n_obj)]
				sk[i_k, 0, :] = spread
				for i in range(len(spread)):
					if (spread[i]):
						#idx = np.where(supports[i_k][i][0] == self.indexes.values)[0].tolist()[0]
						idx = np.where(supports[i_k][i] == self.indexes.values)[0].tolist()[0]
						if (not uk[i_k, idx]):
							uk[i_k, idx] = 1
						if (uk[i_k, idx]):
							uk[i_k, idx] = 0
						if (zk[i_k, idx]):
							zk[i_k, idx] = 0
		for t in range(self.K+1, self.N):
			## Good-Turing estimator
			for i_k in range(self.K):
				hapaxes = [uk[i_k, i]*np.prod(zk[i_k, list(set([i]).symmetric_difference(range(self.n_obj)))]) for i in range(self.n_obj)]
				rk[i_k] = 1/float(nk[i_k])*sum(hapaxes)
				lbdk[i_k] = sum([np.sum(sk[i_k, s, :]) for s in range(int(nk[i_k]))])/float(nk[i_k])
				## Compute bk(t), as defined in [Lagrée et al.]
				bk[i_k] = rk[i_k]+(1+np.sqrt(2))*np.sqrt((lbdk[i_k]*np.log(4*t))/nk[i_k])+np.log(4*t)/float(3*nk[i_k])
			## choose q = argmax bk(t)
			q = np.argmax(bk)
			## Update
			nk[q] += 1
			#spread = [self.is_active_node(self.reward(s[0])) if (s[0] in supports[q]) else 0 for s in range(self.n_obj)]
			spread = [self.is_active_node(self.reward(s)) if (s in supports[q]) else 0 for s in range(self.n_obj)]
			sk[q, t, :] = spread
		action = candidates[q]#candidates[np.argmax(bk)]
		best_reward = self.best_reward(candidates)
		return action, best_reward

	def best_reward(self, candidates):
		for c in candidates:
			assert c in self.indexes.values and not self.f[c][0], "{} not action".format(c)
		r = np.matrix(self.R[self.indexes.isin(candidates)].Rating.values, dtype=int)[0]
		best_reward = np.max(r)
		return best_reward
