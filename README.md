# projet-gml

Project for GML class for the MVA. Clémence Réda, 2018. (All rights reserved)

## Project proposition

**Name:** Recommender system with serendipity

**Topic:**  online region exploration, recommendation

**Category:** review, implementation, theory

**Description:** A lacking feature of current recommender systems is that they usually do not allow discovery of new elements, that is, an element that might be interesting, but which is different of what the user is used to. [1] has introduced a method that might recommend novel items. 

The goal of this project is to formalize the problem (in particular, the modelling of the users' behaviour), to extend it to an online setting, to offer a review of the cases that might be correctly processed by this type of clustering and not by regular ones, and to design a method of region exploration that might solve this issue, by determining which criteria should be taken into account and possibly maximized.

[1] Abbassi, Z., Amer-Yahia, S., Lakshmanan, L. V., Vassilvitskii, S., & Yu, C. (2009, October). Getting recommender systems to think outside the box. In Proceedings of the third ACM conference on Recommender systems (pp. 285-288). ACM.

## Documentation about the code

The */code* folder contains three **Python** files:

- *process-data.py* contains the code related to the **data processing** (for the MovieLens database "ml-1m")

It builds the data frame of the ratings by the one user in input (denoted X), along the data frame describing the movies (denoted Xobjects) and the data frame describing the features of the unique user (denoted Xusers) to which I want to recommend movies. Then it builds the similarity graph on objects (features are the one-hot encoding of the genres to which each movie belongs). For two feature vectors *x* and *y* associated with two movies, the corresponding similarity coefficient is *s(x, y) = exp(-||x-y||^2_2/var)*, where *var* is a user-selected parameter and *||.||_2* is the L2-norm, and we build out of these similarity coefficients an *epsilon*-neighbourhood graph (*epsilon* is one of the inputs). Graph matrix, rating data frame and feature matrix are then stored in *.dat* files.

- *models.py* contains the code related to the different **bandit models** that I will benchmark.

The **Bandit** class contains the true rating data frame, the object similarity graph matrix, and a data frame *f* which indicates whether a movie has already been seen by the user whom I target. It also contains two arrays meant to store the regret and "diversity" measure from time 0 to the horizon. This class has three public functions: *Bandit.run* which performs **one** round of recommendation, *Bandit.plot_results* which plots the cumulative regret and the diversity at each round (using arrays stored in the object), *Bandit.reinitialize* which allows to reset the content of the regret and diversity arrays, and of the data frame *f* of explored movies.

Four bandit models are implemented so far:
+ **Random**, which selects a movie at random in the set of unexplored items;
+ *epsilon*-**Greedy**, which, with probability *epsilon*, selects a movie at random in the set of unexplored items, and, with probability *1-epsilon*, selects a movie which maximizes the value of the diversity measure in the set of unexplored items.
+ **Linear UCB**, which assumes that the reward function is linear with respect to the feature vector of the explored movies.
+ An adaptation of [[Lagrée et al.](https://ieeexplore.ieee.org/abstract/document/8215581/)] **method** for online influence maximization, which uses Good-Turing estimators.

**Regret** is computed as follows: *R_{T} = \sum_{t \leq T} max_a{r(a)}-r(a_{t})* (the difference between the best reward obtained using the best reward of the round and the reward obtained by playing the recommended action *a_{t}* at round *t*).

**Diversity coefficient** is computed as follows: *d_{T} = \sqrt{|V_{T}.t(V_{T})|}*, where *t(A)* is the transpose of the matrix *A*, and *V_{t}* is the matrix which lines are the features vectors of all explored movies up to time *t*.

- *benchmark.py* contains the code related to the benchmark of the different bandit models.

Example of command:

```bash
python3.6 benchmark.py --user 2 --eps 0.6 --var 100 --niter 100 --horizon 100 --method {random|greedy|lagree|linUCB}
```

It runs the selected method for *horizon* rounds, and plots the cumulative regret along with the diversity measure for each round (both **averaged** over *niter* trajectories). For the adaptation of the [[Lagrée et al.]]'s method, it also performs the same test, but for several different values of *serendipity threshold*.
