from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
# --
from scipy.spatial.distance import pdist
import numpy as np

def average_precision(pos_distances, neg_distances):
	"""
	pos_distances : (k x 1, float) : distances from reference
	example to same class examples
	neg_distances : (num_examples - k x 1, float) : distances from reference
	example to other class examples
	-------------------------------------------------------------------
	Calculate average precision and precision-recall breakeven, and return
	the average precision / precision-recall breakeven calculated
	using `pos_distances` and `neg_distances`.
	-------------------------------------------------------------------
	returns average_precision, precision-recall break even : (float, float)
	"""
	distances = np.concatenate([pos_distances, neg_distances])
	matches = np.concatenate([np.ones(len(pos_distances)), np.zeros(len(neg_distances))])

	# Sort from shortest to longest distance
	sorted_i = np.argsort(distances)
	distances = distances[sorted_i]
	matches = matches[sorted_i]

	# Calculate precision
	precision = np.cumsum(matches)/np.arange(1, len(matches) + 1)

	# Calculate average precision: the multiplication with matches and division
	# by the number of positive examples is to not count precisions at the same
	# recall point multiple times.
	average_precision = np.sum(precision * matches) / len(pos_distances)

	# Calculate recall
	recall = np.cumsum(matches)/len(pos_distances)

	# More than one precision can be at a single recall point, take the max one
	for n in range(len(recall) - 2, -1, -1):
		precision[n] = max(precision[n], precision[n + 1])

	# Calculate precision-recall breakeven
	prb_i = np.argmin(np.abs(recall - precision))
	prb = (recall[prb_i] + precision[prb_i])/2.

	return average_precision, prb

def generate_matches_array(labels):
	"""
	labels : (num_examples x 1, int) : array of dataset ids
	-------------------------------------------------------------------
	Calculates an array of bool in the same order as the distances from
	`scipy.spatial.distance.pdist` indicating whether a distance is for
	matching or non-matching labels.
	-------------------------------------------------------------------
	returns matches : (num_examples x 1, bool)
	"""
	N = len(labels)
	assert( N*(N-1) % 2 == 0)
	matches = np.zeros(N*(N-1) // 2, dtype=np.bool)

	# For every distance, mark whether it is a true match or not
	cur_matches_i = 0
	for n in range(N):
		cur_label = labels[n]
		matches[cur_matches_i:cur_matches_i + (N - n) - 1] = np.asarray(labels[n + 1:]) == cur_label
		cur_matches_i += N - n - 1
	return matches

def calculate_average_precision(X, labels, metric="cosine"):
	"""
	X : (num_examples x embedding dimension, float) : matrix of embeddings
	labels : (num_examples x 1, int) : array of class ids
	-------------------------------------------------------------------
	Find average precision and precision-recall breakeven calculated on
	fixed-dimensional set `X`.
	-------------------------------------------------------------------
	returns average_precision, precision-recall break even : (float, float)
	"""
	N, D = X.shape
	matches = generate_matches_array(labels)
	distances = pdist(X, metric)
	return average_precision(distances[matches == True], distances[matches == False])
