# The important technique here is to speed up the double and triple integrals by breaking
# up the functions to be integrated into separate sub-functions, each sub-function
# corresponding to a discontinuity in the larger function. Thus for calculating the
# normalization factor for receiver 2 signal 0, we have to integrate over h, theta 1 and
# theta 2. But either h < theta 1 & theta 2, h > theta & theta 2, theta 1 < h < theta 2,
# or theta 2 < h < theta 1. If the first then the denominator of sender 1 signal 0 will
# only include the non-normalized sending probabilities for signal 0 and signal 2, if the
# second then it will include only include the non-normalized sending probabilities for
# signal 0, and signal 1, if the third then it will include the non-normalized sending
# probabilities for signal 0, 1, and 2, and if the fourth then it will only include the
# non-normalized sending probabilities for signal 0. So we break up the normalized sending
# function into 4 sub-functions. For the denominator of sender 1 signal 1 we only have to
# consider that h >= theta 1, since otherwise sender 1 signal 1 is 0 anyway. So there we
# only have 2 sub-functions. Similar remarks and techniques apply to sender 1 signal 2,
# and to the numerators and denominators of receiver 2 marginalizations.

import time
import numpy
numpy.set_printoptions(linewidth = 120)
numpy.set_printoptions(precision = 15)
numpy.set_printoptions(threshold = numpy.nan)
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot
from scipy.stats import norm
from scipy.stats import beta
from scipy.stats import uniform
from scipy import integrate

###################

def theta_1_on_theta_2_distribution(theta_distribution, n):
	temp_array_0 = numpy.empty([0, n])
	for theta_2_num in range(n):
		temp_array_1 = numpy.empty(0)
		for theta_1_num in range(n):
			if theta_1_num >= theta_2_num:
				temp_value = theta_distribution.pdf(array_0[theta_1_num]) / (1. - theta_distribution.cdf(array_0[theta_2_num]))
			else:
				temp_value = 0.
			temp_array_1 = numpy.append(temp_array_1, temp_value)
		temp_array_0 = numpy.insert(temp_array_0, theta_2_num, temp_array_1, axis = 0)
	return temp_array_0

def theta_2_on_theta_1_distribution(theta_distribution, n):
	temp_array_0 = numpy.empty([n, 0])
	for theta_1_num in range(n):
		temp_array_1 = numpy.empty(0)
		for theta_2_num in range(n):
			if theta_2_num <= theta_1_num:
				temp_value = theta_distribution.pdf(array_0[theta_2_num]) / (theta_distribution.cdf(array_0[theta_1_num]))
			else:
				temp_value = 0.
			temp_array_1 = numpy.append(temp_array_1, temp_value)
		temp_array_0 = numpy.insert(temp_array_0, theta_1_num, temp_array_1, axis = 1)
	return temp_array_0

###################

def receiver_0_signal_0(h):
	return state_distribution.pdf(h)

def receiver_0_signal_1(h, theta_1):
	if h < theta_1:
		return 0.
	else:
		return state_distribution.pdf(h) / (1. - state_distribution.cdf(theta_1))
		
def receiver_0_signal_2(h, theta_2):
	if h > theta_2:
		return 0.
	else:
		return state_distribution.pdf(h) / (state_distribution.cdf(theta_2))

def sender_1_signal_0_non_normalized(h):
	return numpy.exp(choice_parameter * (numpy.log(state_distribution.pdf(h)) - 0))

def sender_1_signal_1_non_normalized_full(h, theta_1):
	if h < theta_1:
		return 0
	else:
		return numpy.exp(choice_parameter * (numpy.log(state_distribution.pdf(h) / (1. - state_distribution.cdf(theta_1))) - cost))

def sender_1_signal_1_non_normalized(h, theta_1):
	return numpy.exp(choice_parameter * (numpy.log(state_distribution.pdf(h) / (1. - state_distribution.cdf(theta_1))) - cost))

def sender_1_signal_2_non_normalized_full(h, theta_2):
	if h > theta_2:
		return 0
	else:
		return numpy.exp(choice_parameter * (numpy.log(state_distribution.pdf(h) / (state_distribution.cdf(theta_2))) - cost))

def sender_1_signal_2_non_normalized(h, theta_2):
	return numpy.exp(choice_parameter * (numpy.log(state_distribution.pdf(h) / (state_distribution.cdf(theta_2))) - cost))
		
###################

def sender_1_signal_0_normalized_full(h, theta_1, theta_2):
	return sender_1_signal_0_non_normalized(h) / (sender_1_signal_0_non_normalized(h) + sender_1_signal_1_non_normalized_full(h, theta_1) + sender_1_signal_2_non_normalized_full(h, theta_2))

def sender_1_signal_0_normalized_h_lt_theta_1_lte_theta_2(h, theta_1, theta_2):
	return sender_1_signal_0_non_normalized(h) / (sender_1_signal_0_non_normalized(h) + sender_1_signal_2_non_normalized(h, theta_2))

def sender_1_signal_0_normalized_h_lt_theta_1_gt_theta_2(h, theta_1, theta_2):
	return sender_1_signal_0_non_normalized(h) / (sender_1_signal_0_non_normalized(h))

def sender_1_signal_0_normalized_h_gte_theta_1_lte_theta_2(h, theta_1, theta_2):
	return sender_1_signal_0_non_normalized(h) / (sender_1_signal_0_non_normalized(h) + sender_1_signal_1_non_normalized(h, theta_1) + sender_1_signal_2_non_normalized(h, theta_2))

def sender_1_signal_0_normalized_h_gte_theta_1_gt_theta_2(h, theta_1, theta_2):
	return sender_1_signal_0_non_normalized(h) / (sender_1_signal_0_non_normalized(h) + sender_1_signal_1_non_normalized(h, theta_1))

def receiver_2_signal_0_non_normalized_full(h, theta_1, theta_2):
	if theta_distribution_type == 'unrelated+uniform':
		return sender_1_signal_0_normalized_full(h, theta_1, theta_2) * state_distribution(h) * theta_distribution.pdf(theta_1) * theta_distribution.pdf(theta_2)

def receiver_2_signal_0_non_normalized_h_lt_theta_1_lte_theta_2(h, theta_1, theta_2):
	value = sender_1_signal_0_normalized_h_lt_theta_1_lte_theta_2(h, theta_1, theta_2) * state_distribution.pdf(h) * theta_distribution.pdf(theta_1) * theta_distribution.pdf(theta_2)
# 	print 'receiver_2_signal_0_non_normalized_h_lt_theta_1_lte_theta_2(%s, %s, %s) = %s' % (h, theta_1, theta_2, value)
	return value
	
def receiver_2_signal_0_non_normalized_h_lt_theta_1_gt_theta_2(h, theta_1, theta_2):
	value = sender_1_signal_0_normalized_h_lt_theta_1_gt_theta_2(h, theta_1, theta_2) * state_distribution.pdf(h) * theta_distribution.pdf(theta_1) * theta_distribution.pdf(theta_2)
# 	print 'receiver_2_signal_0_non_normalized_h_lt_theta_1_gt_theta_2(%s, %s, %s) = %s' % (h, theta_1, theta_2, value)
	return value

def receiver_2_signal_0_non_normalized_h_gte_theta_1_lte_theta_2(h, theta_1, theta_2):
	value = sender_1_signal_0_normalized_h_gte_theta_1_lte_theta_2(h, theta_1, theta_2) * state_distribution.pdf(h) * theta_distribution.pdf(theta_1) * theta_distribution.pdf(theta_2)
# 	print 'receiver_2_signal_0_non_normalized_h_gte_theta_1_lte_theta_2(%s, %s, %s) = %s' % (h, theta_1, theta_2, value)
	return value
	
def receiver_2_signal_0_non_normalized_h_gte_theta_1_gt_theta_2(h, theta_1, theta_2):
	value = sender_1_signal_0_normalized_h_gte_theta_1_gt_theta_2(h, theta_1, theta_2) * state_distribution.pdf(h) * theta_distribution.pdf(theta_1) * theta_distribution.pdf(theta_2)
# 	print 'receiver_2_signal_0_non_normalized_h_gte_theta_1_gt_theta_2(%s, %s, %s) = %s' % (h, theta_1, theta_2, value)
	return value
	
###################

def sender_1_signal_1_normalized_full(h, theta_1, theta_2):
	return sender_1_signal_1_non_normalized_full(h, theta_1) / (sender_1_signal_0_non_normalized(h) + sender_1_signal_1_non_normalized_full(h, theta_1) + sender_1_signal_2_non_normalized_full(h, theta_2))

# def sender_1_signal_1_normalized_h_lt_theta_1_lte_theta_2(h, theta_1, theta_2):
# 	return sender_1_signal_1_non_normalized(h, theta_1) / (sender_1_signal_0_non_normalized(h) + sender_1_signal_2_non_normalized(h, theta_2))

# def sender_1_signal_1_normalized_h_lt_theta_1_gt_theta_2(h, theta_1, theta_2):
# 	return sender_1_signal_1_non_normalized(h, theta_1) / (sender_1_signal_0_non_normalized(h))

def sender_1_signal_1_normalized_h_gte_theta_1_lte_theta_2(h, theta_1, theta_2):
	return sender_1_signal_1_non_normalized(h, theta_1) / (sender_1_signal_0_non_normalized(h) + sender_1_signal_1_non_normalized(h, theta_1) + sender_1_signal_2_non_normalized(h, theta_2))

def sender_1_signal_1_normalized_h_gte_theta_1_gt_theta_2(h, theta_1, theta_2):
	return sender_1_signal_1_non_normalized(h, theta_1) / (sender_1_signal_0_non_normalized(h) + sender_1_signal_1_non_normalized(h, theta_1))

def receiver_2_signal_1_non_normalized_full(h, theta_1, theta_2):
	if theta_distribution_type == 'unrelated':
		return sender_1_signal_1_normalized_full(h, theta_1, theta_2) * state_distribution.pdf(h) * theta_distribution.pdf(theta_1) * theta_distribution.pdf(theta_2)

# def receiver_2_signal_1_non_normalized_h_lt_theta_1_lte_theta_2(h, theta_1, theta_2):
# 	return sender_1_signal_1_normalized_h_lt_theta_1_lte_theta_2(h, theta_1, theta_2) * state_distribution.pdf(h) * theta_distribution.pdf(theta_1) * theta_distribution.pdf(theta_2)
	
# def receiver_2_signal_1_non_normalized_h_lt_theta_1_gt_theta_2(h, theta_1, theta_2):
# 	return sender_1_signal_1_normalized_h_lt_theta_1_gt_theta_2(h, theta_1, theta_2) * state_distribution.pdf(h) * theta_distribution.pdf(theta_1) * theta_distribution.pdf(theta_2)

def receiver_2_signal_1_non_normalized_h_gte_theta_1_lte_theta_2(h, theta_1, theta_2):
	return sender_1_signal_1_normalized_h_gte_theta_1_lte_theta_2(h, theta_1, theta_2) * state_distribution.pdf(h) * theta_distribution.pdf(theta_1) * theta_distribution.pdf(theta_2)

def receiver_2_signal_1_non_normalized_h_gte_theta_1_gt_theta_2(h, theta_1, theta_2):
	return sender_1_signal_1_normalized_h_gte_theta_1_gt_theta_2(h, theta_1, theta_2) * state_distribution.pdf(h) * theta_distribution.pdf(theta_1) * theta_distribution.pdf(theta_2)

###################

def sender_1_signal_2_normalized_full(h, theta_1, theta_2):
	return sender_1_signal_2_non_normalized(h, theta_2) / (sender_1_signal_0_non_normalized(h) + sender_1_signal_1_non_normalized_full(h, theta_1) + sender_1_signal_2_non_normalized_full(h, theta_2))

def sender_1_signal_2_normalized_h_lt_theta_1_lte_theta_2(h, theta_1, theta_2):
	return sender_1_signal_2_non_normalized(h, theta_2) / (sender_1_signal_0_non_normalized(h) + sender_1_signal_2_non_normalized(h, theta_2))

# def sender_1_signal_2_normalized_h_lt_theta_1_gt_theta_2(h, theta_1, theta_2):
# 	return sender_1_signal_2_non_normalized(h, theta_2) / (sender_1_signal_0_non_normalized(h))

def sender_1_signal_2_normalized_h_gte_theta_1_lte_theta_2(h, theta_1, theta_2):
	return sender_1_signal_2_non_normalized(h, theta_2) / (sender_1_signal_0_non_normalized(h) + sender_1_signal_1_non_normalized(h, theta_1) + sender_1_signal_2_non_normalized(h, theta_2))

# def sender_1_signal_2_normalized_h_gte_theta_1_gt_theta_2(h, theta_1, theta_2):
# 	return sender_1_signal_2_non_normalized(h, theta_2) / (sender_1_signal_0_non_normalized(h) + sender_1_signal_1_non_normalized(h, theta_1))

def receiver_2_signal_2_non_normalized_full(h, theta_1, theta_2):
	if theta_distribution_type == 'unrelated':
		return sender_1_signal_2_normalized_full(h, theta_1, theta_2) * state_distribution.pdf(h) * theta_distribution.pdf(theta_1) * theta_distribution.pdf(theta_2)

def receiver_2_signal_2_non_normalized_h_lt_theta_1_lte_theta_2(h, theta_1, theta_2):
	value = sender_1_signal_2_normalized_h_lt_theta_1_lte_theta_2(h, theta_1, theta_2) * state_distribution.pdf(h) * theta_distribution.pdf(theta_1) * theta_distribution.pdf(theta_2)
# 	print 'receiver_2_signal_2_non_normalized_h_lt_theta_1_lte_theta_2(%s, %s, %s) = %s' % (h, theta_1, theta_2, value)
	return value
	
# def receiver_2_signal_2_non_normalized_h_lt_theta_1_gt_theta_2(h, theta_1, theta_2):
# 	return sender_1_signal_2_normalized_h_lt_theta_1_gt_theta_2(h, theta_1, theta_2) * state_distribution.pdf(h) * theta_distribution.pdf(theta_1) * theta_distribution.pdf(theta_2)

def receiver_2_signal_2_non_normalized_h_gte_theta_1_lte_theta_2(h, theta_1, theta_2):
	value = sender_1_signal_2_normalized_h_gte_theta_1_lte_theta_2(h, theta_1, theta_2) * state_distribution.pdf(h) * theta_distribution.pdf(theta_1) * theta_distribution.pdf(theta_2)
# 	print 'receiver_2_signal_2_non_normalized_h_gte_theta_1_lte_theta_2(%s, %s, %s) = %s' % (h, theta_1, theta_2, value)
	return value

# def receiver_2_signal_2_non_normalized_h_gte_theta_1_gt_theta_2(h, theta_1, theta_2):
# 	return sender_1_signal_2_normalized_h_gte_theta_1_gt_theta_2(h, theta_1, theta_2) * state_distribution.pdf(h) * theta_distribution.pdf(theta_1) * theta_distribution.pdf(theta_2)

###################

# def receiver_2_signal_0_normalized_h(h, receiver_2_signal_0_normalization_factor):
# 	return integrate.dblquad(lambda theta_1, theta_2, h: receiver_2_signal_0_non_normalized_full(h, theta_1, theta_2), lower_bound, upper_bound, lambda x: lower_bound, lambda x: upper_bound, args = (h)) [0] / receiver_2_signal_0_normalization_factor

def receiver_2_signal_0_normalized_h(h_1, receiver_2_signal_0_normalization_factor):
	value_1 = integrate.dblquad(lambda theta_1, theta_2, h: receiver_2_signal_0_non_normalized_h_lt_theta_1_lte_theta_2(h, theta_1, theta_2), h_1, upper_bound, lambda x: h_1, lambda x: upper_bound, args = (h_1,)) [0]
	value_2 = integrate.dblquad(lambda theta_1, theta_2, h: receiver_2_signal_0_non_normalized_h_lt_theta_1_gt_theta_2(h, theta_1, theta_2), lower_bound, h_1, lambda x: h_1, lambda x: upper_bound, args = (h_1,)) [0]
	value_3 = integrate.dblquad(lambda theta_1, theta_2, h: receiver_2_signal_0_non_normalized_h_gte_theta_1_lte_theta_2(h, theta_1, theta_2), h_1, upper_bound, lambda x: lower_bound, lambda x: h_1, args = (h_1,)) [0]
	value_4 = integrate.dblquad(lambda theta_1, theta_2, h: receiver_2_signal_0_non_normalized_h_gte_theta_1_gt_theta_2(h, theta_1, theta_2), lower_bound, h_1, lambda x: lower_bound, lambda x: h_1, args = (h_1,)) [0]
	return (value_1 + value_2 + value_3 + value_4) / receiver_2_signal_0_normalization_factor

###################

# def receiver_2_signal_1_normalized_h(h, receiver_2_signal_1_normalization_factor):
# 	return (integrate.dblquad(lambda theta_1, theta_2, h: receiver_2_signal_1_non_normalized_full(h, theta_1, theta_2), h, upper_bound, lambda x: lower_bound, lambda x: h, args = (h))[0] + integrate.dblquad(lambda theta_1, theta_2, h : receiver_2_signal_1_non_normalized_full(h, theta_1, theta_2), lower_bound, h, lambda x: lower_bound, lambda x: h, args = (h))[0]) / receiver_2_signal_1_normalization_factor

def receiver_2_signal_1_normalized_h(h, receiver_2_signal_1_normalization_factor):
	value_3 = integrate.dblquad(lambda theta_1, theta_2, h: receiver_2_signal_1_non_normalized_h_gte_theta_1_lte_theta_2(h, theta_1, theta_2), h, upper_bound, lambda x: lower_bound, lambda x: h, args = (h,)) [0]
	value_4 = integrate.dblquad(lambda theta_1, theta_2, h: receiver_2_signal_1_non_normalized_h_gte_theta_1_gt_theta_2(h, theta_1, theta_2), lower_bound, h, lambda x: lower_bound, lambda x: h, args = (h,)) [0]
	return (value_3 + value_4) / receiver_2_signal_1_normalization_factor

# def receiver_2_signal_1_normalized_theta_1(theta_1, receiver_2_signal_1_normalization_factor):
# 	return (integrate.dblquad(lambda theta_2, h, theta_1: receiver_2_signal_1_non_normalized_full(h, theta_1, theta_2), theta_1, upper_bound, lambda x: x, lambda x: upper_bound, args = (theta_1))[0] + integrate.dblquad(lambda theta_2, h, theta_1: receiver_2_signal_1_non_normalized_full(h, theta_1, theta_2), theta_1, upper_bound, lambda x: lower_bound, lambda x: x, args = (theta_1))[0]) / receiver_2_signal_1_normalization_factor

def receiver_2_signal_1_normalized_theta_1(theta_1, receiver_2_signal_1_normalization_factor):
	value_3 = integrate.dblquad(lambda theta_2, h, theta_1: receiver_2_signal_1_non_normalized_h_gte_theta_1_lte_theta_2(h, theta_1, theta_2), theta_1, upper_bound, lambda x: x, lambda x: upper_bound, args = (theta_1,)) [0]
	value_4 = integrate.dblquad(lambda theta_2, h, theta_1: receiver_2_signal_1_non_normalized_h_gte_theta_1_gt_theta_2(h, theta_1, theta_2), theta_1, upper_bound, lambda x: lower_bound, lambda x: x, args = (theta_1,)) [0]
	return (value_3 + value_4) / receiver_2_signal_1_normalization_factor

###################

# def receiver_2_signal_2_normalized_h(h, receiver_2_signal_2_normalization_factor):
# 	return (integrate.dblquad(lambda theta_1, theta_2, h: receiver_2_signal_2_non_normalized_full(h, theta_1, theta_2), h, upper_bound, lambda x: h, lambda x: upper_bound, args = (h))[0] + integrate.dblquad(lambda theta_1, theta_2, h : receiver_2_signal_2_non_normalized_full(h, theta_1, theta_2), h, upper_bound, lambda x: lower_bound, lambda x: h, args = (h))[0]) / receiver_2_signal_2_normalization_factor

def receiver_2_signal_2_normalized_h(h, receiver_2_signal_2_normalization_factor):
	value_1 = integrate.dblquad(lambda theta_1, theta_2, h: receiver_2_signal_2_non_normalized_h_lt_theta_1_lte_theta_2(h, theta_1, theta_2), h, upper_bound, lambda x: h, lambda x: upper_bound, args = (h,)) [0]
	value_3 = integrate.dblquad(lambda theta_1, theta_2, h: receiver_2_signal_2_non_normalized_h_gte_theta_1_lte_theta_2(h, theta_1, theta_2), h, upper_bound, lambda x: lower_bound, lambda x: h, args = (h,)) [0]
	return (value_1 + value_3) / receiver_2_signal_2_normalization_factor

# def receiver_2_signal_2_normalized_theta_2(theta_2, receiver_2_signal_2_normalization_factor):
# 	return (integrate.dblquad(lambda theta_1, h, theta_2: receiver_2_signal_2_non_normalized_full(h, theta_1, theta_2), lower_bound, theta_2, lambda x: x, lambda x: upper_bound, args = (theta_2))[0] + integrate.dblquad(lambda theta_1, h, theta_2: receiver_2_signal_2_non_normalized_full(h, theta_1, theta_2), lower_bound, theta_2, lambda x: lower_bound, lambda x: x, args = (theta_2))[0])/ receiver_2_signal_1_normalization_factor

def receiver_2_signal_2_normalized_theta_2(theta_2, receiver_2_signal_2_normalization_factor):
	value_1 = integrate.dblquad(lambda theta_1, h, theta_2: receiver_2_signal_2_non_normalized_h_lt_theta_1_lte_theta_2(h, theta_1, theta_2), lower_bound, theta_2, lambda x: x, lambda x: upper_bound, args = (theta_2,)) [0]
	value_3 = integrate.dblquad(lambda theta_1, h, theta_2: receiver_2_signal_2_non_normalized_h_gte_theta_1_lte_theta_2(h, theta_1, theta_2), lower_bound, theta_2, lambda x: lower_bound, lambda x: x, args = (theta_2,)) [0]
	return (value_1 + value_3) / receiver_2_signal_2_normalization_factor

##########################################################################################

# Here we have the settings for a level 0 receiver decoding probabilities, given a fixed
# theta. This forms the common basis for both Lassiter and Goodman's original model and
# our modified model.

cost = 2.
choice_parameter = 4.
lower_bound = 0.
upper_bound = 1.
num_states = 320

# mu = 0.
# sigma = 1.
# state_distribution = norm(mu,sigma)

alpha_parameter = 1.
beta_parameter = 9.
location_parameter = lower_bound
scale_parameter = upper_bound - lower_bound
state_distribution = beta(alpha_parameter, beta_parameter, loc = location_parameter, scale = scale_parameter)

# state_distribution = uniform(lower_bound, upper_bound - lower_bound)

theta_distribution_type = 'unrelated+uniform'

if theta_distribution_type == 'normal':
	theta_distribution = norm(mu, sigma)
elif theta_distribution_type == 'Beta':
	theta_distribution = beta(3, 3, loc = lower_bound, scale = upper_bound - lower_bound)
elif theta_distribution_type == 'uniform':
	theta_distribution = uniform(lower_bound, upper_bound - lower_bound)
elif theta_distribution_type == 'unrelated+uniform':
	theta_distribution = uniform(lower_bound, upper_bound - lower_bound)

array_0 = numpy.flipud(numpy.linspace(upper_bound, lower_bound, num_states, endpoint = False)) - ((numpy.flipud(numpy.linspace(upper_bound, lower_bound, num_states, endpoint = False)) - numpy.linspace(lower_bound, upper_bound, num_states, endpoint = False))/2)

#########################

if theta_distribution_type == 'normal' or theta_distribution_type == 'Beta' or theta_distribution_type == 'uniform':

	theta_1_on_theta_2_distribution_array = theta_1_on_theta_2_distribution(theta_distribution, len(array_0))
	theta_2_on_theta_1_distribution_array = theta_2_on_theta_1_distribution(theta_distribution, len(array_0))

	theta_1_on_theta_2_distribution_array = theta_1_on_theta_2_distribution_array / numpy.sum(theta_1_on_theta_2_distribution_array, axis = 1)[numpy.newaxis].T
	theta_2_on_theta_1_distribution_array = theta_2_on_theta_1_distribution_array / numpy.sum(theta_2_on_theta_1_distribution_array, axis = 0)

	theta_1_on_theta_2_distribution_array = theta_1_on_theta_2_distribution_array * numpy.transpose(theta_distribution.pdf(array_0)[numpy.newaxis])
	theta_2_on_theta_1_distribution_array = theta_2_on_theta_1_distribution_array * theta_distribution.pdf(array_0)

	theta_1_on_theta_2_distribution_array = theta_1_on_theta_2_distribution_array / numpy.sum(theta_1_on_theta_2_distribution_array)
	theta_2_on_theta_1_distribution_array = theta_2_on_theta_1_distribution_array / numpy.sum(theta_2_on_theta_1_distribution_array)

	theta_2_by_theta_1_distribution_array = (theta_1_on_theta_2_distribution_array + theta_2_on_theta_1_distribution_array)/2.

	# theta_2_by_theta_1_distribution_array = theta_2_by_theta_1_distribution_array / ((upper_bound - lower_bound) / len(array_0))

	print 'theta_2_by_theta_1_distribution_array = \n%s' % theta_2_by_theta_1_distribution_array

	print numpy.sum(theta_2_by_theta_1_distribution_array)

elif theta_distribution_type == 'unrelated+uniform':
	
	theta_2_by_theta_1_distribution_array = numpy.full([num_states, num_states], 1./(num_states**2))


theta_1_distribution = numpy.sum(theta_2_by_theta_1_distribution_array, axis = 0)
theta_2_distribution = numpy.sum(theta_2_by_theta_1_distribution_array, axis = 1)

# fig, ax = pyplot.subplots(1,1)
# pyplot.plot(array_0, theta_1_distribution)
# pyplot.plot(array_0, theta_2_distribution)
# pyplot.show()
	
#########################

sender_1_signal_0_non_normalized_array = numpy.empty(0)
for h_num in range(len(array_0)):
	value = sender_1_signal_0_non_normalized(array_0[h_num])
	sender_1_signal_0_non_normalized_array = numpy.append(sender_1_signal_0_non_normalized_array, value)

# print 'sender_1_signal_0_non_normalized_array = \n%s' % sender_1_signal_0_non_normalized_array

sender_1_signal_1_non_normalized_array = numpy.empty([0, len(array_0)])
for theta_num in range(len(array_0)):
	temp_array = numpy.empty(0)
	for h_num in range(len(array_0)):
		value = sender_1_signal_1_non_normalized_full(array_0[h_num], array_0[theta_num])
		temp_array = numpy.append(temp_array, value)
	sender_1_signal_1_non_normalized_array = numpy.insert(sender_1_signal_1_non_normalized_array, theta_num, temp_array, axis = 0)

# print 'sender_1_signal_1_non_normalized_array = \n%s' % sender_1_signal_1_non_normalized_array

sender_1_signal_2_non_normalized_array = numpy.empty([0, len(array_0)])
for theta_num in range(len(array_0)):
	temp_array = numpy.empty(0)
	for h_num in range(len(array_0)):
		value = sender_1_signal_2_non_normalized_full(array_0[h_num], array_0[theta_num])
		temp_array = numpy.append(temp_array, value)
	sender_1_signal_2_non_normalized_array = numpy.insert(sender_1_signal_2_non_normalized_array, theta_num, temp_array, axis = 0)

# print 'sender_1_signal_2_non_normalized_array = \n%s' % sender_1_signal_2_non_normalized_array
	
#########################

denominator_array = numpy.empty([0, len(array_0), len(array_0)])
for theta_2_num in range(len(array_0)):
	temp_array_0 = (numpy.tile(sender_1_signal_0_non_normalized_array, (len(array_0), 1)) + sender_1_signal_1_non_normalized_array + numpy.tile(sender_1_signal_2_non_normalized_array[theta_2_num], [len(array_0), 1]))
	denominator_array = numpy.insert(denominator_array, theta_2_num, temp_array_0, axis = 0)

sender_1_signal_0_normalized_array = numpy.tile(sender_1_signal_0_non_normalized_array, (len(array_0), 1)) / denominator_array
sender_1_signal_1_normalized_array = sender_1_signal_1_non_normalized_array / denominator_array
sender_1_signal_2_normalized_array = numpy.reshape(sender_1_signal_2_non_normalized_array, (len(array_0), 1, len(array_0))) / denominator_array

sender_1_signal_0_normalized_array = sender_1_signal_0_normalized_array * numpy.reshape(theta_2_by_theta_1_distribution_array, (len(array_0), len(array_0), 1))
sender_1_signal_1_normalized_array = sender_1_signal_1_normalized_array * numpy.reshape(theta_2_by_theta_1_distribution_array, (len(array_0), len(array_0), 1))
sender_1_signal_2_normalized_array = sender_1_signal_2_normalized_array * numpy.reshape(theta_2_by_theta_1_distribution_array, (len(array_0), len(array_0), 1))

# print sender_1_signal_0_normalized_array + sender_1_signal_1_normalized_array + sender_1_signal_2_normalized_array
# print numpy.sum(sender_1_signal_0_normalized_array + sender_1_signal_1_normalized_array + sender_1_signal_2_normalized_array)

#########################

sender_1_signal_0_h_array = numpy.sum(numpy.sum(sender_1_signal_0_normalized_array, axis = 0), axis = 0)
sender_1_signal_1_h_array = numpy.sum(numpy.sum(sender_1_signal_1_normalized_array, axis = 0), axis = 0)
sender_1_signal_2_h_array = numpy.sum(numpy.sum(sender_1_signal_2_normalized_array, axis = 0), axis = 0)

print 'sender_1_signal_0_h_array = \n%s' % sender_1_signal_0_h_array
print 'sender_1_signal_1_h_array = \n%s' % sender_1_signal_0_h_array
print 'sender_1_signal_2_h_array = \n%s' % sender_1_signal_0_h_array

#########################

fixed_theta_1_num = numpy.int(numpy.ceil(len(array_0)*(8./12.)))
fixed_theta_2_num = numpy.int(numpy.ceil(len(array_0)*(4./12.)))

print 'fixed_theta_1 = %s' % array_0[fixed_theta_1_num]
print 'fixed_theta_2 = %s' % array_0[fixed_theta_2_num]

if theta_distribution_type == 'normal' or theta_distribution_type == 'Beta' or theta_distribution_type == 'uniform':

	sender_1_signal_0_fixed_theta_1_fixed_theta_2_h_array = sender_1_signal_0_normalized_array[fixed_theta_2_num, fixed_theta_1_num]
	sender_1_signal_0_fixed_theta_1_fixed_theta_2_h_array = sender_1_signal_0_fixed_theta_1_fixed_theta_2_h_array / theta_2_by_theta_1_distribution_array[fixed_theta_2_num, fixed_theta_1_num]

	print 'sender_1_signal_0_fixed_theta_1_fixed_theta_2_h_array = \n%s' % sender_1_signal_0_fixed_theta_1_fixed_theta_2_h_array

	sender_1_signal_1_fixed_theta_1_h_array = numpy.sum(sender_1_signal_1_normalized_array[:fixed_theta_1_num + 1, fixed_theta_1_num], axis = 0)
	sender_1_signal_1_fixed_theta_1_h_array = (sender_1_signal_1_fixed_theta_1_h_array / numpy.sum(theta_2_by_theta_1_distribution_array[:fixed_theta_1_num + 1, fixed_theta_1_num], axis = 0))

	print 'sender_1_signal_1_fixed_theta_1_h_array = \n%s' % sender_1_signal_1_fixed_theta_1_h_array

	sender_1_signal_2_fixed_theta_2_h_array = numpy.sum(sender_1_signal_2_normalized_array[fixed_theta_2_num, fixed_theta_2_num:], axis = 0)
	sender_1_signal_2_fixed_theta_2_h_array = (sender_1_signal_2_fixed_theta_2_h_array / numpy.sum(theta_2_by_theta_1_distribution_array[fixed_theta_2_num, fixed_theta_2_num:]))

	print 'sender_1_signal_2_fixed_theta_2_h_array = \n%s' % sender_1_signal_2_fixed_theta_2_h_array

elif theta_distribution_type == 'unrelated+uniform':

	sender_1_signal_0_fixed_theta_1_fixed_theta_2_h_array = sender_1_signal_0_normalized_array[fixed_theta_2_num, fixed_theta_1_num]
	sender_1_signal_0_fixed_theta_1_fixed_theta_2_h_array = sender_1_signal_0_fixed_theta_1_fixed_theta_2_h_array / theta_2_by_theta_1_distribution_array[fixed_theta_2_num, fixed_theta_1_num]

	print 'sender_1_signal_0_fixed_theta_1_fixed_theta_2_h_array = \n%s' % sender_1_signal_0_fixed_theta_1_fixed_theta_2_h_array

	sender_1_signal_1_fixed_theta_1_h_array = numpy.sum(sender_1_signal_1_normalized_array[:, fixed_theta_1_num], axis = 0)
	sender_1_signal_1_fixed_theta_1_h_array = (sender_1_signal_1_fixed_theta_1_h_array / numpy.sum(theta_2_by_theta_1_distribution_array[:, fixed_theta_1_num], axis = 0))

	print 'sender_1_signal_1_fixed_theta_1_h_array = \n%s' % sender_1_signal_1_fixed_theta_1_h_array

	sender_1_signal_2_fixed_theta_2_h_array = numpy.sum(sender_1_signal_2_normalized_array[fixed_theta_2_num, :], axis = 0)
	sender_1_signal_2_fixed_theta_2_h_array = (sender_1_signal_2_fixed_theta_2_h_array / numpy.sum(theta_2_by_theta_1_distribution_array[fixed_theta_2_num, :]))

	print 'sender_1_signal_2_fixed_theta_2_h_array = \n%s' % sender_1_signal_2_fixed_theta_2_h_array

#########################

print 'here now'

time_0 = time.time()
print time_0

receiver_2_signal_0_normalization_factor_h_gte_theta_1_gt_theta_2 = integrate.tplquad(lambda theta_1, theta_2, h: receiver_2_signal_0_non_normalized_h_gte_theta_1_gt_theta_2(h, theta_1, theta_2), lower_bound, upper_bound, lambda x: lower_bound, lambda x: x, lambda x, y: lower_bound, lambda x, y: x) [0]
# receiver_2_signal_0_normalization_factor_h_gte_theta_1_gt_theta_2_beta_1_9 = 0.014157417045957481
print 'receiver_2_signal_0_normalization_factor_h_gte_theta_1_gt_theta_2 = %s' % repr(receiver_2_signal_0_normalization_factor_h_gte_theta_1_gt_theta_2)

time_1 = time.time()
print time_1

receiver_2_signal_0_normalization_factor_h_lt_theta_1_lte_theta_2 = integrate.tplquad(lambda theta_1, theta_2, h: receiver_2_signal_0_non_normalized_h_lt_theta_1_lte_theta_2(h, theta_1, theta_2), lower_bound, upper_bound, lambda x: x, lambda x: upper_bound, lambda x, y: x, lambda x, y: upper_bound)[0]
# receiver_2_signal_0_normalization_factor_h_lt_theta_1_lte_theta_2_beta_1_9 = 0.8160334535892759
print 'receiver_2_signal_0_normalization_factor_h_lt_theta_1_lte_theta_2 = %s' % repr(receiver_2_signal_0_normalization_factor_h_lt_theta_1_lte_theta_2)

time_2 = time.time()
print time_2

receiver_2_signal_0_normalization_factor_h_lt_theta_1_gt_theta_2 = integrate.tplquad(lambda theta_1, theta_2, h: receiver_2_signal_0_non_normalized_h_lt_theta_1_gt_theta_2(h, theta_1, theta_2), lower_bound, upper_bound, lambda x: lower_bound, lambda x: x, lambda x, y: x, lambda x, y: upper_bound) [0]
# receiver_2_signal_0_normalization_factor_h_lt_theta_1_gt_theta_2_beta_1_9 = 0.0.08181818181818182
print 'receiver_2_signal_0_normalization_factor_h_lt_theta_1_gt_theta_2 = %s' % repr(receiver_2_signal_0_normalization_factor_h_lt_theta_1_gt_theta_2)

time_3 = time.time()
print time_3

receiver_2_signal_0_normalization_factor_h_gte_theta_1_lte_theta_2 = integrate.tplquad(lambda theta_1, theta_2, h: receiver_2_signal_0_non_normalized_h_gte_theta_1_lte_theta_2(h, theta_1, theta_2), lower_bound, upper_bound, lambda x: x, lambda x: upper_bound, lambda x, y: lower_bound, lambda x, y: x) [0]
# receiver_2_signal_0_normalization_factor_h_gte_theta_1_lte_theta_2_beta_1_9 = 0.07346331723644262
print 'receiver_2_signal_0_normalization_factor_h_gte_theta_1_lte_theta_2 = %s' % repr(receiver_2_signal_0_normalization_factor_h_gte_theta_1_lte_theta_2)

time_4 = time.time()
print time_4

receiver_2_signal_0_normalization_factor = receiver_2_signal_0_normalization_factor_h_lt_theta_1_lte_theta_2 + receiver_2_signal_0_normalization_factor_h_lt_theta_1_gt_theta_2 + receiver_2_signal_0_normalization_factor_h_gte_theta_1_lte_theta_2 + receiver_2_signal_0_normalization_factor_h_gte_theta_1_gt_theta_2
# receiver_2_signal_0_normalization_factor_beta_1_9 = 0.9854723696898579
print 'receiver_2_signal_0_normalization_factor = %s' % repr(receiver_2_signal_0_normalization_factor)

time_5 = time.time()
print time_5

receiver_2_signal_1_normalization_factor_h_lte_theta_2 = integrate.tplquad(lambda theta_1, theta_2, h: receiver_2_signal_1_non_normalized_h_gte_theta_1_lte_theta_2(h, theta_1, theta_2), lower_bound, upper_bound, lambda x: x, lambda x: upper_bound, lambda x, y: lower_bound, lambda x, y: x)[0]
# receiver_2_signal_1_normalization_factor_h_lte_theta_2_beta_1_9 = 0.008306891315719619
print 'receiver_2_signal_1_normalization_factor_h_lte_theta_2 = %s' % repr(receiver_2_signal_1_normalization_factor_h_lte_theta_2)

time_6 = time.time()
print time_6

receiver_2_signal_1_normalization_factor_h_gt_theta_2 = integrate.tplquad(lambda theta_1, theta_2, h: receiver_2_signal_1_non_normalized_h_gte_theta_1_gt_theta_2(h, theta_1, theta_2), lower_bound, upper_bound, lambda x: lower_bound, lambda x: x, lambda x, y: lower_bound, lambda x, y: x)[0]
# receiver_2_signal_1_normalization_factor_h_gt_theta_2_beta_1_9 = 0.004024401135860705
print 'receiver_2_signal_1_normalization_factor_h_gt_theta_2 = %s' % repr(receiver_2_signal_1_normalization_factor_h_gt_theta_2)

time_7 = time.time()
print time_7

receiver_2_signal_1_normalization_factor = receiver_2_signal_1_normalization_factor_h_lte_theta_2 + receiver_2_signal_1_normalization_factor_h_gt_theta_2
# receiver_2_signal_1_normalization_factor_beta_1_9 = 0.012331292451580324
print 'receiver_2_signal_1_normalization_factor = %s' % repr(receiver_2_signal_1_normalization_factor)

time_8 = time.time()
print time_8

receiver_2_signal_2_normalization_factor_h_lt_theta_1 = integrate.tplquad(lambda theta_1, theta_2, h: receiver_2_signal_2_non_normalized_h_lt_theta_1_lte_theta_2(h, theta_1, theta_2), lower_bound, upper_bound, lambda x: x, lambda x: upper_bound, lambda x, y: x, lambda x, y: upper_bound)[0]
# receiver_2_signal_2_normalization_factor_h_lt_theta_1_beta_1_9 = 0.0021483645925423622
print 'receiver_2_signal_2_normalization_factor_h_lt_theta_1 = %s' % repr(receiver_2_signal_2_normalization_factor_h_lt_theta_1)

time_9 = time.time()
print time_9

receiver_2_signal_2_normalization_factor_h_gte_theta_1 = integrate.tplquad(lambda theta_1, theta_2, h: receiver_2_signal_2_non_normalized_h_gte_theta_1_lte_theta_2(h, theta_1, theta_2), lower_bound, upper_bound, lambda x: x, lambda x: upper_bound, lambda x, y: lower_bound, lambda x, y: x)[0]
# receiver_2_signal_2_normalization_factor_h_gte_theta_1_beta_1_9 = 4.797327618788977e-05
print 'receiver_2_signal_2_normalization_factor_h_gte_theta_1 = %s' % repr(receiver_2_signal_2_normalization_factor_h_gte_theta_1)

time_10 = time.time()
print time_10

receiver_2_signal_2_normalization_factor = receiver_2_signal_2_normalization_factor_h_lt_theta_1 + receiver_2_signal_2_normalization_factor_h_gte_theta_1
# receiver_2_signal_2_normalization_factor_beta_1_9 = 0.002196337868730252
print 'receiver_2_signal_2_normalization_factor = %s' % repr(receiver_2_signal_2_normalization_factor)

time_11 = time.time()
print time_11

receiver_2_signal_0_h_array = numpy.empty(0)
for h in array_0:
	receiver_2_signal_0_h_array = numpy.append(receiver_2_signal_0_h_array, receiver_2_signal_0_normalized_h(h, receiver_2_signal_0_normalization_factor))
print 'receiver_2_signal_0_h_array = \n%s' % receiver_2_signal_0_h_array

receiver_2_signal_1_h_array = numpy.empty(0)
for h in array_0:
	receiver_2_signal_1_h_array = numpy.append(receiver_2_signal_1_h_array, receiver_2_signal_1_normalized_h(h, receiver_2_signal_1_normalization_factor))
print 'receiver_2_signal_1_h_array = \n%s' % receiver_2_signal_1_h_array

receiver_2_signal_1_theta_1_array = numpy.empty(0)
for h in array_0:
	receiver_2_signal_1_theta_1_array = numpy.append(receiver_2_signal_1_theta_1_array, receiver_2_signal_1_normalized_theta_1(h, receiver_2_signal_1_normalization_factor))
print 'receiver_2_signal_1_theta_1_array = \n%s' % receiver_2_signal_1_theta_1_array

receiver_2_signal_2_h_array = numpy.empty(0)
for h in array_0:
	receiver_2_signal_2_h_array = numpy.append(receiver_2_signal_2_h_array, receiver_2_signal_2_normalized_h(h, receiver_2_signal_2_normalization_factor))
print 'receiver_2_signal_2_h_array = \n%s' % receiver_2_signal_2_h_array

receiver_2_signal_2_theta_2_array = numpy.empty(0)
for h in array_0:
	receiver_2_signal_2_theta_2_array = numpy.append(receiver_2_signal_2_theta_2_array, receiver_2_signal_2_normalized_theta_2(h, receiver_2_signal_2_normalization_factor))
print 'receiver_2_signal_2_theta_2_array = \n%s' % receiver_2_signal_2_theta_2_array

#########################

fig, ax = pyplot.subplots(1, 2, figsize = (12,5))

pyplot.subplot(1, 2, 1)
line = pyplot.plot(array_0, sender_1_signal_0_h_array, lw = 2, color = 'k')
line = pyplot.plot(array_0, sender_1_signal_1_h_array, lw = 2, color = 'b')
line = pyplot.plot(array_0, sender_1_signal_2_h_array, lw = 2, color = 'r')

line = pyplot.plot(array_0, sender_1_signal_0_fixed_theta_1_fixed_theta_2_h_array, lw = 2, linestyle = '--', color = 'k')
line = pyplot.plot(array_0, sender_1_signal_1_fixed_theta_1_h_array, lw = 2, linestyle = '--', color = 'b')
line = pyplot.plot(array_0, sender_1_signal_2_fixed_theta_2_h_array, lw = 2, linestyle = '--', color = 'r')

line = pyplot.plot(array_0, numpy.sum(sender_1_signal_1_normalized_array[:,:,fixed_theta_1_num], axis = 0)/numpy.sum(theta_2_by_theta_1_distribution_array, axis = 0), lw = 5, linestyle = ':', color = 'b')
line = pyplot.plot(array_0, numpy.sum(sender_1_signal_2_normalized_array[:,:,fixed_theta_2_num], axis = 1)/numpy.sum(theta_2_by_theta_1_distribution_array, axis = 1), lw = 5, linestyle = ':', color = 'r')

pyplot.subplot(1, 2, 2)
line = pyplot.plot(array_0, receiver_2_signal_0_h_array, lw = 2, color = 'k')
line = pyplot.plot(array_0, receiver_2_signal_1_h_array, lw = 2, color = 'b')
line = pyplot.plot(array_0, receiver_2_signal_1_theta_1_array, lw = 2, linestyle = '--', color = 'b')
line = pyplot.plot(array_0, receiver_2_signal_2_h_array, lw = 2, color = 'r')
line = pyplot.plot(array_0, receiver_2_signal_2_theta_2_array, lw = 2, linestyle = '--', color = 'r')

pyplot.subplot(1, 2, 1)
pyplot.legend([r'$\sigma_{1}(u_{0}|h)$', r'$\sigma_{1}(u_{1}|h)$', r'$\sigma_{1}(u_{2}|h)$', r'$\sigma_{1}(u_{0}|h, \theta_{1} \approx %s, \theta_{2} \approx %s)$' % (numpy.around(array_0[fixed_theta_1_num], decimals = 2), numpy.around(array_0[fixed_theta_2_num], decimals = 2)), r'$\sigma_{1}(u_{1}|h, \theta_{1} \approx %s)$' % numpy.around(array_0[fixed_theta_1_num], decimals = 2), r'$\sigma_{1}(u_{2}|h, \theta_{2} \approx %s)$' % numpy.around(array_0[fixed_theta_2_num], decimals = 2)], loc = 0, fontsize = 14)

pyplot.subplot(1, 2, 2)
pyplot.legend([r'$\rho_{2}(h|u_{0})$', r'$\rho_{2}(h|u_{1})$', r'$\rho_{2}(\theta_{1}|u_{1})$', r'$\rho_{2}(h|u_{2})$', r'$\rho_{2}(\theta_{2}|u_{2})$'], loc = 0, fontsize = 14)

fig.text(.4, 0, r'$Lassiter\ and\ Goodman\ Three\ Signals\ Integrated' + '\n', fontsize = 10)

# fig.text(.4, 0, r'$\lambda = %s, C(u_{1}), C(u_{2}) = %s, \mu = %s, \sigma = %s, num\ states = %s, theta\ distribution\ type = %s$' % (choice_parameter, cost, mu, sigma, num_states, theta_distribution_type), fontsize = 10)
fig.text(.4, 0, r'$\lambda = %s, C(u_{1}), C(u_{2}) = %s, \alpha = %s, \beta = %s, num\ states = %s, theta\ distribution\ type = %s$' % (choice_parameter, cost, alpha_parameter, beta_parameter, num_states, theta_distribution_type), fontsize = 10)
# fig.text(.4, 0, r'$\lambda = %s, C(u_{1}), C(u_{2}) = %s, Uniform distribution, num\ states = %s, theta\ distribution\ type = %s$' % (choice_parameter, cost, num_states, theta_distribution_type), fontsize = 10)

# pyplot.savefig('Lassiter and Goodman Model Three Signals Integrated Normal Distribution.pdf')
pyplot.savefig('Lassiter and Goodman Model Three Signals Integrated Beta Distribution.pdf')
# pyplot.savefig('Lassiter and Goodman Model Three Signals Integrated Uniform Distribution.pdf')

pyplot.show()
pyplot.close()


#########################

# Below we have the output for a level 2 receiver for the following parameters:

# cost = 2.
# choice_parameter = 4.
# lower_bound = 0.
# upper_bound = 1.
# num_states = 320
# 
# alpha_parameter = 1.
# beta_parameter = 9.
# location_parameter = lower_bound
# scale_parameter = upper_bound - lower_bound
# state_distribution = beta(alpha_parameter, beta_parameter, loc = location_parameter, scale = scale_parameter)
# 
# theta_distribution_type = 'unrelated+uniform'

# We save them here to avoid having to run the program again.


# receiver_2_signal_0_h_array = 
# [  4.432302413846746e+00   4.336231692784414e+00   4.241511653555366e+00   4.147321968819439e+00   4.052780062362539e+00
#    3.957664361750149e+00   3.862456605567820e+00   3.767832338736853e+00   3.674326810445477e+00   3.582281853483929e+00
#    3.491894821310003e+00   3.403271900886365e+00   3.316464661196289e+00   3.231492195474783e+00   3.148354098015342e+00
#    3.067038078092493e+00   2.987524495970455e+00   2.909789121944501e+00   2.833804852380817e+00   2.759542800675562e+00
#    2.686973005206494e+00   2.616064897372188e+00   2.546787616121124e+00   2.479110222249791e+00   2.413001845995863e+00
#    2.348431789432101e+00   2.285369597707815e+00   2.223785108470012e+00   2.163648485763688e+00   2.104930242728491e+00
#    2.047601256093892e+00   1.991632774591439e+00   1.936996422802357e+00   1.883664201548349e+00   1.831608485652314e+00
#    1.780802019704556e+00   1.731217912343545e+00   1.682829629481122e+00   1.635610986859622e+00   1.589536142315106e+00
#    1.544579588132146e+00   1.500716143908171e+00   1.457920950396390e+00   1.416169464863745e+00   1.375437458580196e+00
#    1.335701017143867e+00   1.296936544435684e+00   1.259120771077142e+00   1.222230768321252e+00   1.186243968321180e+00
#    1.151138191669216e+00   1.116891682953631e+00   1.083483154812551e+00   1.050891840544855e+00   1.019097554749070e+00
#    9.880807606990062e-01   9.578226422531870e-01   9.283051770919816e-01   8.995112070818005e-01   8.714245007189307e-01
#    8.440298020735699e-01   8.173128606071729e-01   7.912604368094237e-01   7.658602798570011e-01   7.411010753878340e-01
#    7.169723638399884e-01   6.934644323414939e-01   6.705681855082309e-01   6.482750023665576e-01   6.265765876971797e-01
#    6.054648262623340e-01   5.849316476533054e-01   5.649689080469723e-01   5.455682932516202e-01   5.267212453525790e-01
#    5.084189133085150e-01   4.906521262039957e-01   4.734113866491172e-01   4.566868810645685e-01   4.404685032637415e-01
#    4.247458877666553e-01   4.095084495547072e-01   3.947454274072504e-01   3.804459284670201e-01   3.665989721983522e-01
#    3.531935323862327e-01   3.402185762495158e-01   3.276631000963915e-01   3.155161612326357e-01   3.037669060481373e-01
#    2.924045943630623e-01   2.814186202215996e-01   2.707985293884394e-01   2.605340338400341e-01   2.506150235571402e-01
#    2.410315759236490e-01   2.317739630244363e-01   2.228326571159192e-01   2.141983345201437e-01   2.058618781687414e-01
#    1.978143789984285e-01   1.900471363759417e-01   1.825516577079752e-01   1.753196573711964e-01   1.683430550789046e-01
#    1.616139737843879e-01   1.551247372064600e-01   1.488678670499105e-01   1.428360799825152e-01   1.370222844206847e-01
#    1.314195771675971e-01   1.260212399406188e-01   1.208207358188160e-01   1.158117056362521e-01   1.109879643424489e-01
#    1.063434973477307e-01   1.018724568680944e-01   9.756915828165398e-02   9.342807650653411e-02   8.944384240826389e-02
#    8.561123924319609e-02   8.192519914320369e-02   7.838079964584381e-02   7.497326027329411e-02   7.169793916263395e-02
#    6.855032974943143e-02   6.552605750609361e-02   6.262087673602063e-02   5.983066742425736e-02   5.715143214505759e-02
#    5.457929302653829e-02   5.211048877241307e-02   4.974137174063874e-02   4.746840507867951e-02   4.528815991499296e-02
#    4.319731260625840e-02   4.119264203979958e-02   3.927102699060393e-02   3.742944353229553e-02   3.566496250138806e-02
#    3.397474701411823e-02   3.235605003514074e-02   3.080621199735285e-02   2.932265847210565e-02   2.790289788905375e-02
#    2.654451930489108e-02   2.524519022021909e-02   2.400265444379434e-02   2.281473000340414e-02   2.167930710262250e-02
#    2.059434612270261e-02   1.955787566886694e-02   1.856799066026175e-02   1.762285046284869e-02   1.672067706451311e-02
#    1.585975329167471e-02   1.503842106672128e-02   1.425507970541098e-02   1.350818425391188e-02   1.279624386434690e-02
#    1.211782020847725e-02   1.147152592873978e-02   1.085602312599343e-02   1.027002188332083e-02   9.712278825237676e-03
#    9.181595711670765e-03   8.676818066073040e-03   8.196833837051668e-03   7.740572092892467e-03   7.307001748371847e-03
#    6.895130323254757e-03   6.504002732040990e-03   6.132700103369191e-03   5.780338631150294e-03   5.446068453283710e-03
#    5.129072559701403e-03   4.828565729061964e-03   4.543793492741333e-03   4.274031126053591e-03   4.018582666075531e-03
#    3.776779955546414e-03   3.547981712321220e-03   3.331572623862715e-03   3.126962466264589e-03   2.933585247304772e-03
#    2.750898373035035e-03   2.578381837419537e-03   2.415537434541980e-03   2.261887992907579e-03   2.116976631372757e-03
#    1.980366036242124e-03   1.851637759078785e-03   1.730391534780682e-03   1.616244619481946e-03   1.508831148147214e-03
#    1.407801509314244e-03   1.312821742912596e-03   1.223572950160341e-03   1.139750725712096e-03   1.061064605305737e-03
#    9.872375306274690e-04   9.180053307019702e-04   8.531162194229137e-04   7.923303088450455e-04   7.354191378650298e-04
#    6.821652159241093e-04   6.323615813715901e-04   5.858113741338761e-04   5.423274223396029e-04   5.017318425571254e-04
#    4.638556533062777e-04   4.285384015119800e-04   3.956278015728073e-04   3.649793867232042e-04   3.364561723734844e-04
#    3.099283311172130e-04   2.852728791009544e-04   2.623733734567162e-04   2.411196205027059e-04   2.214073944232800e-04
#    2.031381661441685e-04   1.862188421241959e-04   1.705615127328778e-04   1.560832103441758e-04   1.427056756861598e-04
#    1.303551341824328e-04   1.189620800371635e-04   1.084610690115908e-04   9.879051924911935e-05   8.989251996678188e-05
#    8.171264777859568e-05   7.419979042102097e-05   6.730597765537531e-05   6.098621912664338e-05   5.519834896266778e-05
#    4.990287690220709e-05   4.506284574479555e-05   4.064369491974330e-05   3.661312997597279e-05   3.294099779869443e-05
#    2.959916736319172e-05   2.656141584019588e-05   2.380331987150381e-05   2.130215183861031e-05   1.903678095120185e-05
#    1.698757898638623e-05   1.513633051351250e-05   1.346614744336868e-05   1.196138774443098e-05   1.060757817267641e-05
#    9.391340865264490e-06   8.300323652136711e-06   7.323133943281595e-06   6.449276053063607e-06   5.669091826617116e-06
#    4.973705277611369e-06   4.354965744524393e-06   3.805403683623713e-06   3.318178847231301e-06   2.887036700639036e-06
#    2.506266221960831e-06   2.170660156084484e-06   1.875477614841576e-06   1.616409632679121e-06   1.389543320464190e-06
#    1.191335060119405e-06   1.018578763034443e-06   8.683793459529657e-07   7.381273175825338e-07   6.254750427474168e-07
#    5.283146003458428e-07   4.447571541768741e-07   3.731137584587582e-07   3.118775225701074e-07   2.597068432746841e-07
#    2.154110024525512e-07   1.779344984773932e-07   1.463449993785893e-07   1.198209830877520e-07   9.764061840167147e-08
#    7.917154880290108e-08   6.386152365459742e-08   5.122982357127738e-08   4.085942900512336e-08   3.238988327843460e-08
#    2.551080343682354e-08   1.995599439477533e-08   1.549812389528615e-08   1.194391780843371e-08   9.129837249933652e-09
#    6.918200909971610e-09   5.193717844856317e-09   3.860397799304819e-09   2.838807895412327e-09   2.064138617675268e-09
#    1.481916162103797e-09   1.049682636883317e-09   7.327178246153117e-10   5.033682499184447e-10   3.398202356431894e-10
#    2.250467446612032e-10   1.459093138723232e-10   9.239772553528283e-11   5.699135307923075e-11   3.412736428187980e-11
#    1.976216415191273e-11   1.101650346321317e-11   5.874519892858939e-12   2.975214707180839e-12   1.417803170996556e-12
#    6.280249515448286e-13   2.544159426507004e-13   9.217304978352254e-14   2.892379884406018e-14   7.491966706492105e-15
#    1.482685025455964e-15   1.956381442279786e-16   1.305848309054164e-17   2.160062675334117e-19   3.241581726727483e-23]
# receiver_2_signal_1_h_array = 
# [  1.909232692727424e-04   5.934647048625604e-04   1.025882558443840e-03   1.490857322203663e-03   1.990901378278722e-03
#    2.528670176576563e-03   3.107411446305809e-03   3.731057577387919e-03   4.404128928852163e-03   5.131661504795571e-03
#    5.919203531206843e-03   6.772851172524239e-03   7.699302418475018e-03   8.705920746126628e-03   9.800806148531528e-03
#    1.099287332994634e-02   1.229193763413205e-02   1.370880951844122e-02   1.525539850048009e-02   1.694482756366620e-02
#    1.879155906147781e-02   2.081153321100417e-02   2.302232031385832e-02   2.544328788085068e-02   2.809578387211520e-02
#    3.100333723756210e-02   3.419187696771286e-02   3.768997073563275e-02   4.152908411362200e-02   4.574386112220484e-02
#    5.037242655270698e-02   5.545671005081641e-02   6.104279132129595e-02   6.718126496775868e-02   7.392762236146817e-02
#    8.134264647517651e-02   8.949281374799352e-02   9.845069468222248e-02   1.082953419235062e-01   1.191126509502501e-01
#    1.309956741122600e-01   1.440448635462691e-01   1.583682124304873e-01   1.740812571602186e-01   1.913068954733360e-01
#    2.101749676193368e-01   2.308215398472662e-01   2.533878225551650e-01   2.780186504989224e-01   3.048604509828365e-01
#    3.340586298170179e-01   3.657543162235229e-01   4.000804291566399e-01   4.371570608658731e-01   4.770862205137329e-01
#    5.199460414900713e-01   5.657846288799033e-01   6.146138036601579e-01   6.664030796553062e-01   7.210742769270904e-01
#    7.784972177595456e-01   8.384869551627564e-01   9.008029379665771e-01   9.651504161577004e-01   1.031184238891023e+00
#    1.098515009295541e+00   1.166717357370675e+00   1.235339902742066e+00   1.303916330394385e+00   1.371976916180442e+00
#    1.439059825672629e+00   1.504721567885527e+00   1.568546101176761e+00   1.630152241192210e+00   1.689199186167971e+00
#    1.745390131554228e+00   1.798474077437932e+00   1.848246029346610e+00   1.894545853155756e+00   1.937256070939706e+00
#    1.976298882746912e+00   2.011632677355722e+00   2.043248260529615e+00   2.071164988838673e+00   2.095426955789323e+00
#    2.116099338306753e+00   2.133264977614844e+00   2.147021240201108e+00   2.157477181975406e+00   2.164751021547162e+00
#    2.168967916086996e+00   2.170258024718300e+00   2.168754839012256e+00   2.164593757210633e+00   2.157910877645585e+00
#    2.148841986945594e+00   2.137521719597737e+00   2.124082866959400e+00   2.108655815639697e+00   2.091368097129453e+00
#    2.072344032529019e+00   2.051704458125434e+00   2.029566519354544e+00   2.006043522322062e+00   1.981244833537322e+00
#    1.955275819833408e+00   1.928237821612137e+00   1.900228153571843e+00   1.871340127961934e+00   1.841663096173474e+00
#    1.811282505132991e+00   1.780279965529694e+00   1.748733329386380e+00   1.716716774892204e+00   1.684300896761266e+00
#    1.651552800673026e+00   1.618536200596938e+00   1.585311518010931e+00   1.551935982197373e+00   1.518463730946044e+00
#    1.484945911115648e+00   1.451430778607344e+00   1.417963797388764e+00   1.384587737277789e+00   1.351342770254163e+00
#    1.318266565115865e+00   1.285394380337642e+00   1.252759155022529e+00   1.220391597864862e+00   1.188320274066055e+00
#    1.156571690163136e+00   1.125170376745400e+00   1.094138969047160e+00   1.063498285414760e+00   1.033267403654459e+00
#    1.003463735274491e+00   9.741030976400643e-01   9.451997840644837e-01   9.167666318629963e-01   8.888150883987516e-01
#    8.613552751524033e-01   8.343960498485162e-01   8.079450666731994e-01   7.820088346182892e-01   7.565927739880525e-01
#    7.317012711047913e-01   7.073377312499753e-01   6.835046298776084e-01   6.602035621365175e-01   6.374352907381237e-01
#    6.151997922060498e-01   5.934963015436647e-01   5.723233553553340e-01   5.516788334568082e-01   5.315599990097917e-01
#    5.119635372153123e-01   4.928855925979166e-01   4.743218049268069e-01   4.562673437783622e-01   4.387169418085990e-01
#    4.216649267431365e-01   4.051052521257336e-01   3.890315268553954e-01   3.734370435429965e-01   3.583148057178743e-01
#    3.436575539143761e-01   3.294577906678722e-01   3.157078044492679e-01   3.023996925665707e-01   2.895253830616125e-01
#    2.770766556295444e-01   2.650451615757740e-01   2.534224429173018e-01   2.421999504428231e-01   2.313690610430934e-01
#    2.209210941523151e-01   2.108473273350598e-01   2.011390111073860e-01   1.917873829780938e-01   1.827836807409118e-01
#    1.741191550407573e-01   1.657850812367805e-01   1.577727705844801e-01   1.500735807587539e-01   1.426789257393297e-01
#    1.355802850796152e-01   1.287692125795828e-01   1.222373443829115e-01   1.159764065181990e-01   1.099782219036549e-01
#    1.042347168343009e-01   9.873792697030100e-02   9.348000284466787e-02   8.845321490819727e-02   8.364995810494487e-02
#    7.906275616450953e-02   7.468426512035337e-02   7.050727691634953e-02   6.652472227172047e-02   6.272967332749545e-02
#    5.911534592060122e-02   5.567510152466900e-02   5.240244887213305e-02   4.929104527184376e-02   4.633469763607972e-02
#    4.352736323049986e-02   4.086315016024895e-02   3.833631760509094e-02   3.594127581612228e-02   3.367258588629174e-02
#    3.152495930663271e-02   2.949325731980071e-02   2.757249008219259e-02   2.575781564561874e-02   2.404453876919218e-02
#    2.242810957179890e-02   2.090412203521625e-02   1.946831236765220e-02   1.811655723718868e-02   1.684487188432618e-02
#    1.564940812254510e-02   1.452645223551934e-02   1.347242278389960e-02   1.248386829787520e-02   1.155746495898119e-02
#    1.069001411930629e-02   9.878439824806760e-03   9.119786254127826e-03   8.411215111606399e-03   7.750002976444051e-03
#    7.133538614350390e-03   6.559320257714735e-03   6.024952860128061e-03   5.528145330843196e-03   5.066707754531594e-03
#    4.638548601469776e-03   4.241671933066011e-03   3.874174607419755e-03   3.534243489391816e-03   3.220152669452281e-03
#    2.930260695366680e-03   2.663007820577124e-03   2.416913272936218e-03   2.190572547255431e-03   1.982654724937874e-03
#    1.791899823777294e-03   1.617116180820625e-03   1.457177871010835e-03   1.311022164149927e-03   1.177647022548780e-03
#    1.056108641561323e-03   9.455190350347891e-04   8.450436675460986e-04   7.538991351363118e-04   6.713508961007532e-04
#    5.967109860480840e-04   5.293361472634606e-04   4.686252351933972e-04   4.140175118863868e-04   3.649905507959984e-04
#    3.210582924386903e-04   2.817691570014878e-04   2.467042143599254e-04   2.154753547837172e-04   1.877238006818462e-04
#    1.631181170343325e-04   1.413528299464601e-04   1.221468343370293e-04   1.052419116180297e-04   9.040130483945807e-05
#    7.740835090017185e-05   6.606516933535634e-05   5.619140710464826e-05   4.762303872217641e-05   4.021123848793467e-05
#    3.382121465957074e-05   2.833128953478360e-05   2.363182862801137e-05   1.962432738231394e-05   1.622052536220643e-05
#    1.334156917543725e-05   1.091722293796207e-05   8.885125047533362e-06   7.190089986250540e-06   5.783453831101849e-06
#    4.622462113973692e-06   3.669698638797123e-06   2.892553833453294e-06   2.262731187761539e-06   1.755790306331363e-06
#    1.350725086271000e-06   1.029575514711746e-06   7.770715698323999e-07   5.803077015510580e-07   4.284070146414248e-07
#    3.124241843595992e-07   2.248184503955799e-07   1.594463418429928e-07   1.113063072081023e-07   7.636476426882379e-08
#    5.140216764024557e-08   3.387764401878547e-08   2.181077191860804e-08   1.367911326554820e-08   8.330138468377744e-09
#    4.906224541122539e-09   2.781216506831028e-09   1.508787183103374e-09   7.775110111412472e-10   3.770569333728264e-10
#    1.699976512620449e-10   7.010676193863547e-11   2.586103927297275e-11   8.264243000184381e-12   2.180374778755229e-12
#    4.395985545315455e-13   5.910444843127185e-14   4.020765652713343e-15   6.779925222067584e-17   1.037419289593419e-20]
# receiver_2_signal_1_theta_1_array = 
# [  1.415560920032450e-02   1.541125505201969e-02   1.678137675510134e-02   1.827674035568735e-02   1.990929135628267e-02
#    2.169228855529634e-02   2.364036744456351e-02   2.576960172157282e-02   2.809762007027849e-02   3.064376619014945e-02
#    3.342928361272661e-02   3.647751928654690e-02   3.981414255487500e-02   4.346738635383991e-02   4.746830934917976e-02
#    5.185108327443348e-02   5.665330765606817e-02   6.191635440577323e-02   6.768574474173439e-02   7.401156086313267e-02
#    8.094889471487116e-02   8.855833601472865e-02   9.690650143132483e-02   1.060666063627562e-01   1.161190801162289e-01
#    1.271522243667153e-01   1.392629134979586e-01   1.525573337047564e-01   1.671517554421814e-01   1.831733308011050e-01
#    2.007609034965617e-01   2.200658141740479e-01   2.412526774302356e-01   2.645000990453395e-01   2.900012921417054e-01
#    3.179645390168219e-01   3.486134309581836e-01   3.821868011951750e-01   4.189382461512595e-01   4.591351074002355e-01
#    5.030567615976602e-01   5.509920390387044e-01   6.032355649810052e-01   6.600827940345870e-01   7.218234906066834e-01
#    7.887334030264390e-01   8.610638928209242e-01   9.390293228344355e-01   1.022792089297752e+00   1.112445315238178e+00
#    1.207993416702389e+00   1.309331016667967e+00   1.416221014543708e+00   1.528273010025031e+00   1.644923699701793e+00
#    1.765421262576029e+00   1.888816053132057e+00   2.013960036093618e+00   2.139517227980533e+00   2.263986875204143e+00
#    2.385740158933808e+00   2.503069908574922e+00   2.614251256142866e+00   2.717609591709206e+00   2.811590864380124e+00
#    2.894828494671513e+00   2.966201131701175e+00   3.024876274153332e+00   3.070336279229242e+00   3.102385255607661e+00
#    3.121137427853717e+00   3.126989417275562e+00   3.120580231930223e+00   3.102743452170520e+00   3.074456137511155e+00
#    3.036788482562989e+00   2.990857398761732e+00   2.937786193194411e+00   2.878671526833879e+00   2.814557981670526e+00
#    2.746419913094481e+00   2.675149825587392e+00   2.601552267434674e+00   2.526342156010803e+00   2.450146475215289e+00
#    2.373508389218207e+00   2.296892957290570e+00   2.220693787402543e+00   2.145240114182464e+00   2.070803919536518e+00
#    1.997606826880731e+00   1.925826591365650e+00   1.855603079803055e+00   1.787043687590586e+00   1.720228178634214e+00
#    1.655212960982946e+00   1.592034828275643e+00   1.530714207454850e+00   1.471257958426226e+00   1.413661772948766e+00
#    1.357912219201668e+00   1.303988476070247e+00   1.251863797871900e+00   1.201506746464379e+00   1.152882223760584e+00
#    1.105952333827586e+00   1.060677100105306e+00   1.017015059918047e+00   9.749237554070335e-01   9.343601372940857e-01
#    8.952808954882931e-01   8.576427284511348e-01   8.214025614166901e-01   7.865177219957474e-01   7.529460803482627e-01
#    7.206461599611136e-01   6.895772240925019e-01   6.596993421176580e-01   6.309734393121410e-01   6.033613330205937e-01
#    5.768257576641775e-01   5.513303806250393e-01   5.268398106988158e-01   5.033196005159497e-01   4.807362440904224e-01
#    4.590571704528206e-01   4.382507341568563e-01   4.182862033091276e-01   3.991337456562835e-01   3.807644131680233e-01
#    3.631501254751099e-01   3.462636524561051e-01   3.300785962125048e-01   3.145693726273877e-01   2.997111926660286e-01
#    2.854800435467576e-01   2.718526698855643e-01   2.588065548976402e-01   2.463199017223791e-01   2.343716149247526e-01
#    2.229412822148281e-01   2.120091564181241e-01   2.015561377220915e-01   1.915637562179849e-01   1.820141547525080e-01
#    1.728900720996420e-01   1.641748264598690e-01   1.558522992914049e-01   1.479069194759735e-01   1.403236478199849e-01
#    1.330879618906304e-01   1.261858411853458e-01   1.196037526322438e-01   1.133286364184399e-01   1.073478921426834e-01
#    1.016493652882813e-01   9.622133401200068e-02   9.105249624438880e-02   8.613195709677486e-02   8.144921657009059e-02
#    7.699415756055443e-02   7.275703415721134e-02   6.872846022628072e-02   6.489939827726160e-02   6.126114860573954e-02
#    5.780533870786232e-02   5.452391296147332e-02   5.140912256892949e-02   4.845351575666825e-02   4.564992822663395e-02
#    4.299147385472430e-02   4.047153563146707e-02   3.808375684019350e-02   3.582203246802840e-02   3.368050084507507e-02
#    3.165353550723013e-02   2.973573727812176e-02   2.792192656572340e-02   2.620713586925369e-02   2.458660249203184e-02
#    2.305576145601722e-02   2.161023861381934e-02   2.024584395402448e-02   1.895856509574159e-02   1.774456096832917e-02
#    1.660015567232215e-02   1.552183251763474e-02   1.450622823517263e-02   1.355012735804377e-02   1.265045676861373e-02
#    1.180428040770693e-02   1.100879414231051e-02   1.026132078819281e-02   9.559305283902524e-03   8.900310012668946e-03
#    8.282010268777545e-03   7.702189865048134e-03   7.158736878096136e-03   6.649639528109604e-03   6.172982189926811e-03
#    5.726941532251008e-03   5.309782781889983e-03   4.919856109958966e-03   4.555593137035830e-03   4.215503554307326e-03
#    3.898171857794981e-03   3.602254192798080e-03   3.326475305740084e-03   3.069625600653102e-03   2.830558297582841e-03
#    2.608186690243961e-03   2.401481500302890e-03   2.209468325711640e-03   2.031225180562518e-03   1.865880123979254e-03
#    1.712608975605638e-03   1.570633115297504e-03   1.439217364668654e-03   1.317667948185227e-03   1.205330531546874e-03
#    1.101588335136281e-03   1.005860320361532e-03   9.175994467581676e-04   8.362909977598990e-04   7.614509730884656e-04
#    6.926245457544173e-04   6.293845817012554e-04   5.713302201658542e-04   5.180855128678992e-04   4.692981201806857e-04
#    4.246380624746724e-04   3.837965248638847e-04   3.464847136235439e-04   3.124327625851020e-04   2.813886878523342e-04
#    2.531173892191135e-04   2.273996967061254e-04   2.040314606699066e-04   1.828226839733904e-04   1.635966947424442e-04
#    1.461893582678262e-04   1.304483266464821e-04   1.162323247901792e-04   1.034104714631520e-04   9.186163404366236e-05
#    8.147381573722631e-05   7.214357400165104e-05   6.377546897602687e-05   5.628154073739296e-05   4.958081423994097e-05
#    4.359883082237347e-05   3.826720519933651e-05   3.352320688276009e-05   2.930936500841284e-05   2.557309557204631e-05
#    2.226635010815147e-05   1.934528487257169e-05   1.676994961803498e-05   1.450399507905190e-05   1.251439830960305e-05
#    1.077120504359477e-05   9.247288274197019e-06   7.918122273898247e-06   6.761571302410989e-06   5.757692274444533e-06
#    4.888550683826484e-06   4.138049104498999e-06   3.491767612545130e-06   2.936815496608901e-06   2.461693646865067e-06
#    2.056167035067272e-06   1.711146720158459e-06   1.418580835464255e-06   1.171354034608689e-06   9.631948939919198e-07
#    7.885907899521655e-07   6.427097885978299e-07   5.213291057419227e-07   4.207697133983901e-07   3.378366879095624e-07
#    2.697649129652302e-07   2.141697685469086e-07   1.690024541858724e-07   1.325096118601882e-07   1.031969303745643e-07
#    7.979642916717518e-08   6.123713516978500e-08   4.661888161144968e-08   3.518897250187545e-08   2.632147095808036e-08
#    1.949888354742080e-08   1.429602641123477e-08   1.036587210638144e-08   7.427188856771089e-09   5.253796243618908e-09
#    3.665273281595821e-09   2.518966328300621e-09   1.703155356539144e-09   1.131247782764381e-09   7.368792907103492e-10
#    4.698109167401796e-10   2.925210727249172e-10   1.774001733752529e-10   1.044641078629909e-10   5.951095029257076e-11
#    3.265304011572554e-11   1.716388192397327e-11   8.586214204433821e-12   4.054078604235828e-12   1.787785017623089e-12
#    7.263133877576869e-13   2.669205021398661e-13   8.652925654316477e-14   2.386886738506171e-14   5.307326518467099e-15
#    8.720155425480270e-16   9.082888876813062e-17   4.396141672669973e-18   4.430297935009674e-20   2.250824536405997e-24]
# receiver_2_signal_2_h_array = 
# [  3.466804518419183e+01   2.765918298142103e+01   2.113347076111662e+01   1.543860422355403e+01   1.095052852665722e+01
#    7.748917104252501e+00   5.598182454747952e+00   4.176080742397396e+00   3.223616370112498e+00   2.568873192097600e+00
#    2.105053212870579e+00   1.766551742906958e+00   1.512554453104801e+00   1.317098349971877e+00   1.163245169205296e+00
#    1.039661030998826e+00   9.385771129822813e-01   8.545476156890255e-01   7.836756928385785e-01   7.231198243141129e-01
#    6.707720101194112e-01   6.250435738442116e-01   5.847198012591294e-01   5.488595014544710e-01   5.167244407744449e-01
#    4.877289931862642e-01   4.614036976037679e-01   4.373685284955858e-01   4.153130476094469e-01   3.949814952361857e-01
#    3.761614712638817e-01   3.586752553447494e-01   3.423730882707000e-01   3.271279256223970e-01   3.128313071739595e-01
#    2.993900794583416e-01   2.867237762126062e-01   2.747625101672334e-01   2.634452652826300e-01   2.527185048315567e-01
#    2.425350302964016e-01   2.328530407372068e-01   2.236353533791289e-01   2.148487546716886e-01   2.064634574924448e-01
#    1.984526452609673e-01   1.907920875808623e-01   1.834598150817343e-01   1.764358435618720e-01   1.697019393558048e-01
#    1.632414194103996e-01   1.570389806864449e-01   1.510805544427778e-01   1.453531816776524e-01   1.398449065548982e-01
#    1.345446850591567e-01   1.294423064329806e-01   1.245283251759418e-01   1.197940015601566e-01   1.152312487670446e-01
#    1.108325849059437e-01   1.065910883628468e-01   1.025003551664153e-01   9.855445735651862e-02   9.474790169092724e-02
#    9.107558840585145e-02   8.753277012049777e-02   8.411501130322196e-02   8.081814895854844e-02   7.763825532326733e-02
#    7.457160336686523e-02   7.161463578716254e-02   6.876393800317360e-02   6.601621541065736e-02   6.336827492006379e-02
#    6.081701057460034e-02   5.835939287085330e-02   5.599246128698677e-02   5.371331946465376e-02   5.151913248320476e-02
#    4.940712569692212e-02   4.737458466479709e-02   4.541885577558457e-02   4.353734724849323e-02   4.172753026456743e-02
#    3.998694005090826e-02   3.831317679691412e-02   3.670390632796010e-02   3.515686049781091e-02   3.366983728769463e-02
#    3.224070061879981e-02   3.086737989749317e-02   2.954786932016928e-02   2.828022696853877e-02   2.706257372732379e-02
#    2.589309205557650e-02   2.477002464083062e-02   2.369167296257776e-02   2.265639578854680e-02   2.166260762426279e-02
#    2.070877717646762e-02   1.979342559241251e-02   1.891512509635269e-02   1.807249727071858e-02   1.726421152106722e-02
#    1.648898353600406e-02   1.574557377719548e-02   1.503278600343427e-02   1.434946583166786e-02   1.369449933702552e-02
#    1.306681175691932e-02   1.246536582631886e-02   1.188916127451665e-02   1.133723267835482e-02   1.080864885855374e-02
#    1.030251152405872e-02   9.817954183152041e-03   9.354141065508663e-03   8.910266081426166e-03   8.485551817099302e-03
#    8.079248564770295e-03   7.690633386567934e-03   7.319009210845199e-03   6.963703959834198e-03   6.624069707456363e-03
#    6.299481866150505e-03   5.989338401613120e-03   5.693059074379381e-03   5.410084952765781e-03   5.139876364289316e-03
#    4.881915232277012e-03   4.635700829292528e-03   4.400751495965168e-03   4.176603212695647e-03   3.962809115336281e-03
#    3.758938917519915e-03   3.564578351906322e-03   3.379328629648195e-03   3.202805917410510e-03   3.034640831307688e-03
#    2.874477947152219e-03   2.721975326435894e-03   2.576804057491102e-03   2.438647811304579e-03   2.307202411479417e-03
#    2.182175417863670e-03   2.063285723384704e-03   1.950263163648636e-03   1.842848138882851e-03   1.740791247817570e-03
#    1.643852933119159e-03   1.551803138003867e-03   1.464421244072025e-03   1.381494397246182e-03   1.302819899807991e-03
#    1.228202204345639e-03   1.157453973180349e-03   1.090395524656199e-03   1.026854558786143e-03   9.666658915866199e-04
#    9.096711978392144e-04   8.557187620270407e-04   8.046632372023610e-04   7.563654115502837e-04   7.106919824212911e-04
#    6.675153376129645e-04   6.267133436884219e-04   5.881691411259049e-04   5.517709461004232e-04   5.174118587046873e-04
#    4.849896774224786e-04   4.544067196733410e-04   4.255696482529368e-04   3.983893034986411e-04   3.727805410149581e-04
#    3.486620747981810e-04   3.259563256043127e-04   3.045892744087282e-04   2.844903208103209e-04   2.655921462369853e-04
#    2.478305818132540e-04   2.311444807547075e-04   2.154755951574651e-04   2.007691502390725e-04   1.869705785740178e-04
#    1.740307141332289e-04   1.619017663903364e-04   1.505390008013585e-04   1.398980642382858e-04   1.299382920838181e-04
#    1.206206785477513e-04   1.119081855375519e-04   1.037656568795128e-04   9.615973569645415e-05   8.905878484608031e-05
#    8.243281032654532e-05   7.625338755817067e-05   7.049359045258147e-05   6.512792318278819e-05   6.013225456993986e-05
#    5.548375500463374e-05   5.116083582274365e-05   4.714309105778230e-05   4.341124149379473e-05   3.994708094472835e-05
#    3.673342468811770e-05   3.375405998278123e-05   3.099369860203583e-05   2.843793131570837e-05   2.607318425596031e-05
#    2.388667710363668e-05   2.186638303351620e-05   2.000099035846419e-05   1.827986581408749e-05   1.669301942704923e-05
#    1.523107091173256e-05   1.388521754143818e-05   1.264720344176892e-05   1.150929025528647e-05   1.046422912793418e-05
#    9.505233969093021e-06   8.625955938484240e-06   7.820459114451693e-06   7.083542811516764e-06   6.409285964609019e-06
#    5.793254710582070e-06   5.230944501945832e-06   4.718158099980464e-06   4.250987451534258e-06   3.825796467253735e-06
#    3.439204664604289e-06   3.088071640155297e-06   2.769482336714627e-06   2.480733072000032e-06   2.219318296624916e-06
#    1.982918050250963e-06   1.769386085816284e-06   1.576738632782892e-06   1.403238452674172e-06   1.247007808636144e-06
#    1.106579721104658e-06   9.805201267724544e-07   8.675098785973893e-07   7.663367819330362e-07   6.758880945924308e-07
#    5.951434690391916e-07   5.231683156763420e-07   4.591075669603888e-07   4.021798228068727e-07   3.516718584739372e-07
#    3.069334768128063e-07   2.673726874585099e-07   2.324511962012753e-07   2.016801884287393e-07   1.746163911619067e-07
#    1.508583988238848e-07   1.300432484799158e-07   1.118432308705971e-07   9.596292412743783e-08   8.213643761142156e-08
#    7.012485385119410e-08   5.971385707808953e-08   5.071153736069781e-08   4.294635983224032e-08   3.626528897990783e-08
#    3.053205842672223e-08   2.562557708362413e-08   2.143846298259632e-08   1.787569652090200e-08   1.485338525218886e-08
#    1.229763275248830e-08   1.014350446824998e-08   8.334083819723006e-09   6.819612186438782e-09   5.556706742500798e-09
#    4.507650438063421e-09   3.639748740007864e-09   2.924748049629060e-09   2.338311008350981e-09   1.859544184319688e-09
#    1.470573903403223e-09   1.156166247881095e-09   9.033874951562099e-10   7.013015073909729e-10   5.407008112795466e-10
#    4.138683254417939e-10   3.143669013821160e-10   2.368540428199505e-10   1.769193576886870e-10   1.309424774372068e-10
#    9.596934967377539e-11   6.960497288624990e-11   4.992079617646493e-11   3.537515288087019e-11   2.475114741723365e-11
#    1.707283834186314e-11   1.159956224890551e-11   7.752374010676287e-12   5.089081924166284e-12   3.275804579686412e-12
#    2.063583099373537e-12   1.269314060174423e-12   7.603497568333951e-13   4.421877760813991e-13   2.487413626200812e-13
#    1.347459788023764e-13   6.993432899608483e-14   3.452987593765162e-14   1.608898112007900e-14   7.000312071487994e-15
#    2.805513985549849e-15   1.016892311566447e-15   3.250698754724639e-16   8.840564322523339e-17   1.937624556570667e-17
#    3.137417587429485e-18   3.219825298365993e-19   1.535123412204997e-20   1.523589608144065e-22   7.621443136566413e-27]
# receiver_2_signal_2_theta_2_array = 
# [  3.181054361441562   9.344068344093909  14.577915144764972  17.369778634288657  16.906712716890752  14.30853722046059
#   11.254817938506326   8.633024691974455   6.628193201716633   5.153374840835566   4.074337644868435   3.277868254415177
#    2.681380005402792   2.227339056957993   1.876073396345825   1.600094994130776   1.380139083558266   1.202512017271507
#    1.057333553181284   0.93736827343226    0.837241337668599   0.752905654960817   0.681274960915674   0.619967643206591
#    0.567125448905561   0.521283495202324   0.481275893173805   0.446166410386576   0.415196955186327   0.387748895293259
#    0.363313722560688   0.341470596027885   0.321868997909075   0.304215226375599   0.288261793384668   0.273799040784744
#    0.260648463919353   0.248657359583816   0.237694508587578   0.227646672109483   0.218415732329296   0.209916346287373
#    0.202074010988247   0.194823459878029   0.188107327765908   0.181875034321001   0.176081846408004   0.170688087431874
#    0.165658468066627   0.160961517639207   0.156569099322189   0.152455995383876   0.1485995512227     0.144979368906389
#    0.141577042547018   0.13837592914946    0.135360949634643   0.132518415608474   0.12983587816049    0.127301995563286
#    0.124906417228749   0.122639681679335   0.12049312662771    0.118458809538602   0.116529437282848   0.114698303693573
#    0.112959234004568   0.111306535296419   0.109734952200532   0.108239627218078   0.106816065102191   0.105460100829589
#    0.104167870753643   0.102935786586428   0.101760511903758   0.100638940906138   0.09956817920108    0.098545526399634
#    0.097568460343196   0.09663462279655    0.095741806460371   0.09488794317158    0.09407109317337    0.093289435348754
#    0.092541258322281   0.091824952344294   0.091139001880921   0.090481978840861   0.089852536377228   0.089249403209074
#    0.088671378413031   0.088117326640616   0.087586173721398   0.087076902616332   0.086588549689235   0.08612020126767
#    0.0856709904674     0.085240094257186   0.084826730742985   0.084430156652698   0.084049665004513   0.083684582943093
#    0.083334269729395   0.082998114874481   0.08267553639878    0.082365979216469   0.082068913626025   0.081783833904857
#    0.081510256993528   0.081247721273816   0.080995785420475   0.080754027329257   0.080522043112487   0.080299446161482
#    0.080085866265972   0.079880948777107   0.079684353850178   0.079495755697804   0.079314841915784   0.079141312829285
#    0.078974880895373   0.078815270133667   0.078662215567756   0.078515462745748   0.078374767234695   0.0782398941937
#    0.07811061793617    0.077986721515431   0.077867996363803   0.077754241910025   0.077645265262835   0.077540880859527
#    0.077440910183968   0.077345181455641   0.077253529360231   0.077165794800498   0.077081824607745   0.077001471354722
#    0.076924593061389   0.07685105335886    0.07678072001727    0.076713466596537   0.076649171032946   0.076587715782874
#    0.076528987632522   0.076472877533887   0.076419280499335   0.076368095366014   0.076319224731525   0.076272574785928
#    0.076228055159043   0.076185578854451   0.076145062074796   0.07610642412957    0.076069587334142   0.07603447686545
#    0.076001020698779   0.07596914934714    0.075938796104772   0.075909896877907   0.07588238984906    0.075856215550265
#    0.075831318431142   0.075807640328988   0.07578512967307    0.075763735566169   0.075743408980208   0.075724102867615
#    0.075705771941212   0.075688372697483   0.075671863349026   0.075656203680712   0.075641355130411   0.075627280663181
#    0.075613944683082   0.075601313041424   0.075589352951103   0.075578032972476   0.075567322886643   0.075557193809495
#    0.075547617971512   0.075538568748389   0.075530020678689   0.075521949328142   0.075514331329455   0.075507144204705
#    0.075500366586104   0.075493977911053   0.075487958565826   0.075482289799842   0.075476953583582   0.075471932792116
#    0.075467211049068   0.075462772706798   0.075458597845113   0.075454680601919   0.075451004011641   0.075447563919988
#    0.075444331344609   0.075441302133883   0.075438464954376   0.075435809155661   0.075433324580042   0.075431001465519
#    0.07542883058444    0.075426803212711   0.075424910985105   0.075423145928266   0.07542150055884    0.075419967737548
#    0.075418540737159   0.075417213053389   0.075415978633262   0.075414831785148   0.075413766874929   0.075412778857942
#    0.075411862833681   0.075411014182769   0.075410228581015   0.075409501843674   0.075408830108132   0.075408209715359
#    0.075407637236192   0.075407109317487   0.075406623007423   0.07540617531001    0.075405763577787   0.075405385346764
#    0.075405038097379   0.07540471963063    0.075404427843755   0.075404160788313   0.075403916566273   0.07540369351946
#    0.075403489948862   0.075403304392633   0.075403135495963   0.075402981859161   0.075402842278459   0.075402715652927
#    0.075402600911083   0.075402497052538   0.075402403200529   0.075402318456203   0.075402242106875   0.075402173310142
#    0.075402077490022   0.075402055956806   0.075402006188812   0.075401961715402   0.075401921944725   0.075401886440758
#    0.075401854849219   0.075401826756887   0.075401801817313   0.075401779725906   0.07540176019147    0.075401742998516
#    0.075401727792738   0.075401714437474   0.075401702739095   0.075401692500632   0.075401683553432   0.075401675733756
#    0.075401668931236   0.075401663064684   0.075401657968704   0.075401653580308   0.075401649801423   0.075401646561472
#    0.075401643788117   0.075401641503476   0.07540163947335    0.075401637779363   0.075401636324669   0.075401635086386
#    0.075401634089571   0.075401633272885   0.075401632869968   0.075401632300521   0.07540163182984    0.075401631443174
#    0.075401631133086   0.075401630919398   0.075401630732509   0.075401630612349   0.075401630455613   0.075401630331652
#    0.075401630238072   0.075401630164361   0.075401630093644   0.075401630076966   0.07540163005507    0.075401630046483
#    0.07540163002076    0.075401629999788   0.075401629981107   0.075401630021577   0.075401630020543   0.075401630018305
#    0.075401630063453   0.075401630033285   0.075401630020323   0.075401630009793   0.075401629987988   0.075401629971966
#    0.075401629951375   0.075401629980281   0.075401629988031   0.075401629982539   0.075401629975656   0.075401629978872
#    0.075401629972815   0.075401630007653   0.07540162990269    0.075401629747123   0.075401629626969   0.075401629461423
#    0.075401629292522   0.075401629119211]
# sender_1_signal_0_h_array = 
# [ 0.98130283817541   0.984426264423407  0.987522750774396  0.990457353854291  0.992985744898457  0.994923092124734
#   0.996280467490327  0.997194125052372  0.997807921808223  0.998227668234224  0.998521967813277  0.998733685780776
#   0.998889626943717  0.999006830440909  0.999096368018728  0.999165599635507  0.999219523018271  0.999261595074762
#   0.999294242277627  0.999319184578288  0.99933764515487   0.999350488704962  0.99935831401217   0.999361516597289
#   0.999360331357956  0.99935486151236   0.999345097939693  0.999330931605534  0.999312160857308  0.999288494783236
#   0.999259553434297  0.999224865443331  0.999183863392746  0.999135877158902  0.999080125376175  0.999015705108939
#   0.998941579789255  0.99885656546611   0.998759315421179  0.998648303234294  0.99852180443214   0.998377876928477
#   0.99821434056775   0.998028756218833  0.997818405035516  0.99758026870748   0.9973110117682    0.997006967300448
#   0.996664127673043  0.996278142236967  0.995844324171018  0.995357668857143  0.994812886225736  0.994204449376899
#   0.993526661385397  0.992773741470781  0.991939930621151  0.991019615299282  0.990007466098679  0.988898586293723
#   0.987688663362008  0.986374115026348  0.984952220470455  0.98342122738369   0.981780426541038  0.980030187714919
#   0.978171953645806  0.976208192206823  0.974142310315699  0.97197853610433   0.969721777970879  0.967377470195687
#   0.964951414774306  0.962449628157917  0.959878199958311  0.957243168685715  0.954550417541487  0.951805591419292
#   0.949014034730179  0.946180748528993  0.943310364682427  0.940407134437457  0.937474928653222  0.93451724706957
#   0.931537234231444  0.928537700006286  0.925521142975608  0.922489775319966  0.919445548126294  0.916390176318326
#   0.9133251626396    0.910251820305309  0.90717129408716   0.904084579708729  0.90099254151402   0.897895928432857
#   0.894795388308977  0.891691480684314  0.888584688149451  0.885475426377559  0.882364052962072  0.879250875174991
#   0.876136156758686  0.873020123856073  0.869902970177379  0.86678486149214   0.863665939527951  0.86054632534826
#   0.857426122274739  0.854305418411907  0.851184288825248  0.848062797418323  0.844940998548406  0.841818938415589
#   0.838696656255972  0.835574185365458  0.832451553977342  0.829328786013942  0.826205901729679  0.823082918260837
#   0.819959850095116  0.816836709472295  0.813713506725908  0.810590250574237  0.807466948368033  0.804343606301173
#   0.801220229589682  0.798096822623633  0.794973389096126  0.791849932112471  0.788726454282715  0.785602957799936
#   0.782479444506392  0.779355915949396  0.776232373428474  0.773108818035158  0.769985250686453  0.766861672153041
#   0.763738083082993  0.760614484021707  0.757490875428604  0.754367257691211  0.75124363113684   0.74811999604252
#   0.744996352643192  0.741872701138619  0.738749041699111  0.735625374470382  0.732501699577517  0.729378017128341
#   0.726254327216181  0.723130629922223  0.720006925317402  0.716883213464039  0.713759494417188  0.710635768225704
#   0.707512034933226  0.704388294578886  0.701264547198027  0.698140792822662  0.695017031481979  0.691893263202675
#   0.688769488009293  0.685645705924476  0.682521916969185  0.679398121162869  0.676274318523668  0.673150509068494
#   0.670026692813176  0.666902869772549  0.663779039960539  0.660655203390243  0.657531360073974  0.654407510023337
#   0.651283653249265  0.648159789762085  0.645035919571522  0.64191204268676   0.638788159116486  0.63566426886888
#   0.632540371951683  0.629416468372195  0.626292558137313  0.623168641253551  0.620044717727049  0.616920787563609
#   0.613796850768709  0.610672907347505  0.607548957304868  0.604425000645383  0.601301037373375  0.598177067492923
#   0.595053091007856  0.591929107921785  0.588805118238104  0.585681121960015  0.582557119090517  0.579433109632429
#   0.576309093588406  0.57318507096093   0.570061041752335  0.566937005964818  0.563812963600426  0.560688914661079
#   0.557564859148581  0.554440797064615  0.551316728410757  0.548192653188474  0.54506857139915   0.541944483044068
#   0.538820388124428  0.535696286641347  0.532572178595863  0.529448063988954  0.526323942821523  0.523199815094409
#   0.520075680808394  0.516951539964206  0.513827392562513  0.51070323860395   0.507579078089091  0.504454911018474
#   0.501330737392602  0.498206557211924  0.495082370476877  0.491958177187846  0.488833977345199  0.485709770949263
#   0.482585558000355  0.479461338498757  0.476337112444722  0.473212879838495  0.470088640680298  0.466964394970332
#   0.463840142708777  0.460715883895808  0.45759161853158   0.454467346616231  0.451343068149891  0.448218783132676
#   0.4450944915647    0.441970193446052  0.438845888776826  0.435721577557101  0.432597259786952  0.429472935466439
#   0.426348604595629  0.423224267174569  0.420099923203313  0.416975572681904  0.413851215610378  0.410726851988773
#   0.407602481817122  0.404478105095451  0.401353721823784  0.398229332002148  0.395104935630559  0.391980532709039
#   0.388856123237599  0.385731707216257  0.382607284645025  0.379482855523912  0.37635841985293   0.373233977632087
#   0.370109528861389  0.366985073540843  0.363860611670458  0.360736143250235  0.357611668280179  0.354487186760296
#   0.351362698690586  0.348238204071054  0.345113702901702  0.341989195182532  0.338864680913545  0.335740160094745
#   0.33261563272613   0.329491098807704  0.326366558339465  0.323242011321416  0.320117457753558  0.316992897635888
#   0.313868330968411  0.310743757751125  0.307619177984031  0.304494591667127  0.301369998800417  0.298245399383898
#   0.295120793417572  0.291996180901438  0.288871561835496  0.285746936219747  0.282622304054191  0.279497665338827
#   0.276373020073656  0.273248368258678  0.270123709893892  0.266999044979299  0.263874373514899  0.260749695500692
#   0.257625010936677  0.254500319822855  0.251375622159227  0.24825091794579   0.245126207182547  0.242001489869496
#   0.238876766006638  0.235752035593973  0.2326272986315    0.229502555119221  0.226377805057134  0.223253048445239
#   0.220128285283538  0.217003515572029  0.213878739310713  0.21075395649959   0.20762916713866   0.204504371227923
#   0.201379568767378  0.198254759757025]
# sender_1_signal_1_h_array = 
# [ 0.98130283817541   0.984426264423407  0.987522750774396  0.990457353854291  0.992985744898457  0.994923092124734
#   0.996280467490327  0.997194125052372  0.997807921808223  0.998227668234224  0.998521967813277  0.998733685780776
#   0.998889626943717  0.999006830440909  0.999096368018728  0.999165599635507  0.999219523018271  0.999261595074762
#   0.999294242277627  0.999319184578288  0.99933764515487   0.999350488704962  0.99935831401217   0.999361516597289
#   0.999360331357956  0.99935486151236   0.999345097939693  0.999330931605534  0.999312160857308  0.999288494783236
#   0.999259553434297  0.999224865443331  0.999183863392746  0.999135877158902  0.999080125376175  0.999015705108939
#   0.998941579789255  0.99885656546611   0.998759315421179  0.998648303234294  0.99852180443214   0.998377876928477
#   0.99821434056775   0.998028756218833  0.997818405035516  0.99758026870748   0.9973110117682    0.997006967300448
#   0.996664127673043  0.996278142236967  0.995844324171018  0.995357668857143  0.994812886225736  0.994204449376899
#   0.993526661385397  0.992773741470781  0.991939930621151  0.991019615299282  0.990007466098679  0.988898586293723
#   0.987688663362008  0.986374115026348  0.984952220470455  0.98342122738369   0.981780426541038  0.980030187714919
#   0.978171953645806  0.976208192206823  0.974142310315699  0.97197853610433   0.969721777970879  0.967377470195687
#   0.964951414774306  0.962449628157917  0.959878199958311  0.957243168685715  0.954550417541487  0.951805591419292
#   0.949014034730179  0.946180748528993  0.943310364682427  0.940407134437457  0.937474928653222  0.93451724706957
#   0.931537234231444  0.928537700006286  0.925521142975608  0.922489775319966  0.919445548126294  0.916390176318326
#   0.9133251626396    0.910251820305309  0.90717129408716   0.904084579708729  0.90099254151402   0.897895928432857
#   0.894795388308977  0.891691480684314  0.888584688149451  0.885475426377559  0.882364052962072  0.879250875174991
#   0.876136156758686  0.873020123856073  0.869902970177379  0.86678486149214   0.863665939527951  0.86054632534826
#   0.857426122274739  0.854305418411907  0.851184288825248  0.848062797418323  0.844940998548406  0.841818938415589
#   0.838696656255972  0.835574185365458  0.832451553977342  0.829328786013942  0.826205901729679  0.823082918260837
#   0.819959850095116  0.816836709472295  0.813713506725908  0.810590250574237  0.807466948368033  0.804343606301173
#   0.801220229589682  0.798096822623633  0.794973389096126  0.791849932112471  0.788726454282715  0.785602957799936
#   0.782479444506392  0.779355915949396  0.776232373428474  0.773108818035158  0.769985250686453  0.766861672153041
#   0.763738083082993  0.760614484021707  0.757490875428604  0.754367257691211  0.75124363113684   0.74811999604252
#   0.744996352643192  0.741872701138619  0.738749041699111  0.735625374470382  0.732501699577517  0.729378017128341
#   0.726254327216181  0.723130629922223  0.720006925317402  0.716883213464039  0.713759494417188  0.710635768225704
#   0.707512034933226  0.704388294578886  0.701264547198027  0.698140792822662  0.695017031481979  0.691893263202675
#   0.688769488009293  0.685645705924476  0.682521916969185  0.679398121162869  0.676274318523668  0.673150509068494
#   0.670026692813176  0.666902869772549  0.663779039960539  0.660655203390243  0.657531360073974  0.654407510023337
#   0.651283653249265  0.648159789762085  0.645035919571522  0.64191204268676   0.638788159116486  0.63566426886888
#   0.632540371951683  0.629416468372195  0.626292558137313  0.623168641253551  0.620044717727049  0.616920787563609
#   0.613796850768709  0.610672907347505  0.607548957304868  0.604425000645383  0.601301037373375  0.598177067492923
#   0.595053091007856  0.591929107921785  0.588805118238104  0.585681121960015  0.582557119090517  0.579433109632429
#   0.576309093588406  0.57318507096093   0.570061041752335  0.566937005964818  0.563812963600426  0.560688914661079
#   0.557564859148581  0.554440797064615  0.551316728410757  0.548192653188474  0.54506857139915   0.541944483044068
#   0.538820388124428  0.535696286641347  0.532572178595863  0.529448063988954  0.526323942821523  0.523199815094409
#   0.520075680808394  0.516951539964206  0.513827392562513  0.51070323860395   0.507579078089091  0.504454911018474
#   0.501330737392602  0.498206557211924  0.495082370476877  0.491958177187846  0.488833977345199  0.485709770949263
#   0.482585558000355  0.479461338498757  0.476337112444722  0.473212879838495  0.470088640680298  0.466964394970332
#   0.463840142708777  0.460715883895808  0.45759161853158   0.454467346616231  0.451343068149891  0.448218783132676
#   0.4450944915647    0.441970193446052  0.438845888776826  0.435721577557101  0.432597259786952  0.429472935466439
#   0.426348604595629  0.423224267174569  0.420099923203313  0.416975572681904  0.413851215610378  0.410726851988773
#   0.407602481817122  0.404478105095451  0.401353721823784  0.398229332002148  0.395104935630559  0.391980532709039
#   0.388856123237599  0.385731707216257  0.382607284645025  0.379482855523912  0.37635841985293   0.373233977632087
#   0.370109528861389  0.366985073540843  0.363860611670458  0.360736143250235  0.357611668280179  0.354487186760296
#   0.351362698690586  0.348238204071054  0.345113702901702  0.341989195182532  0.338864680913545  0.335740160094745
#   0.33261563272613   0.329491098807704  0.326366558339465  0.323242011321416  0.320117457753558  0.316992897635888
#   0.313868330968411  0.310743757751125  0.307619177984031  0.304494591667127  0.301369998800417  0.298245399383898
#   0.295120793417572  0.291996180901438  0.288871561835496  0.285746936219747  0.282622304054191  0.279497665338827
#   0.276373020073656  0.273248368258678  0.270123709893892  0.266999044979299  0.263874373514899  0.260749695500692
#   0.257625010936677  0.254500319822855  0.251375622159227  0.24825091794579   0.245126207182547  0.242001489869496
#   0.238876766006638  0.235752035593973  0.2326272986315    0.229502555119221  0.226377805057134  0.223253048445239
#   0.220128285283538  0.217003515572029  0.213878739310713  0.21075395649959   0.20762916713866   0.204504371227923
#   0.201379568767378  0.198254759757025]
# sender_1_signal_2_h_array = 
# [ 0.98130283817541   0.984426264423407  0.987522750774396  0.990457353854291  0.992985744898457  0.994923092124734
#   0.996280467490327  0.997194125052372  0.997807921808223  0.998227668234224  0.998521967813277  0.998733685780776
#   0.998889626943717  0.999006830440909  0.999096368018728  0.999165599635507  0.999219523018271  0.999261595074762
#   0.999294242277627  0.999319184578288  0.99933764515487   0.999350488704962  0.99935831401217   0.999361516597289
#   0.999360331357956  0.99935486151236   0.999345097939693  0.999330931605534  0.999312160857308  0.999288494783236
#   0.999259553434297  0.999224865443331  0.999183863392746  0.999135877158902  0.999080125376175  0.999015705108939
#   0.998941579789255  0.99885656546611   0.998759315421179  0.998648303234294  0.99852180443214   0.998377876928477
#   0.99821434056775   0.998028756218833  0.997818405035516  0.99758026870748   0.9973110117682    0.997006967300448
#   0.996664127673043  0.996278142236967  0.995844324171018  0.995357668857143  0.994812886225736  0.994204449376899
#   0.993526661385397  0.992773741470781  0.991939930621151  0.991019615299282  0.990007466098679  0.988898586293723
#   0.987688663362008  0.986374115026348  0.984952220470455  0.98342122738369   0.981780426541038  0.980030187714919
#   0.978171953645806  0.976208192206823  0.974142310315699  0.97197853610433   0.969721777970879  0.967377470195687
#   0.964951414774306  0.962449628157917  0.959878199958311  0.957243168685715  0.954550417541487  0.951805591419292
#   0.949014034730179  0.946180748528993  0.943310364682427  0.940407134437457  0.937474928653222  0.93451724706957
#   0.931537234231444  0.928537700006286  0.925521142975608  0.922489775319966  0.919445548126294  0.916390176318326
#   0.9133251626396    0.910251820305309  0.90717129408716   0.904084579708729  0.90099254151402   0.897895928432857
#   0.894795388308977  0.891691480684314  0.888584688149451  0.885475426377559  0.882364052962072  0.879250875174991
#   0.876136156758686  0.873020123856073  0.869902970177379  0.86678486149214   0.863665939527951  0.86054632534826
#   0.857426122274739  0.854305418411907  0.851184288825248  0.848062797418323  0.844940998548406  0.841818938415589
#   0.838696656255972  0.835574185365458  0.832451553977342  0.829328786013942  0.826205901729679  0.823082918260837
#   0.819959850095116  0.816836709472295  0.813713506725908  0.810590250574237  0.807466948368033  0.804343606301173
#   0.801220229589682  0.798096822623633  0.794973389096126  0.791849932112471  0.788726454282715  0.785602957799936
#   0.782479444506392  0.779355915949396  0.776232373428474  0.773108818035158  0.769985250686453  0.766861672153041
#   0.763738083082993  0.760614484021707  0.757490875428604  0.754367257691211  0.75124363113684   0.74811999604252
#   0.744996352643192  0.741872701138619  0.738749041699111  0.735625374470382  0.732501699577517  0.729378017128341
#   0.726254327216181  0.723130629922223  0.720006925317402  0.716883213464039  0.713759494417188  0.710635768225704
#   0.707512034933226  0.704388294578886  0.701264547198027  0.698140792822662  0.695017031481979  0.691893263202675
#   0.688769488009293  0.685645705924476  0.682521916969185  0.679398121162869  0.676274318523668  0.673150509068494
#   0.670026692813176  0.666902869772549  0.663779039960539  0.660655203390243  0.657531360073974  0.654407510023337
#   0.651283653249265  0.648159789762085  0.645035919571522  0.64191204268676   0.638788159116486  0.63566426886888
#   0.632540371951683  0.629416468372195  0.626292558137313  0.623168641253551  0.620044717727049  0.616920787563609
#   0.613796850768709  0.610672907347505  0.607548957304868  0.604425000645383  0.601301037373375  0.598177067492923
#   0.595053091007856  0.591929107921785  0.588805118238104  0.585681121960015  0.582557119090517  0.579433109632429
#   0.576309093588406  0.57318507096093   0.570061041752335  0.566937005964818  0.563812963600426  0.560688914661079
#   0.557564859148581  0.554440797064615  0.551316728410757  0.548192653188474  0.54506857139915   0.541944483044068
#   0.538820388124428  0.535696286641347  0.532572178595863  0.529448063988954  0.526323942821523  0.523199815094409
#   0.520075680808394  0.516951539964206  0.513827392562513  0.51070323860395   0.507579078089091  0.504454911018474
#   0.501330737392602  0.498206557211924  0.495082370476877  0.491958177187846  0.488833977345199  0.485709770949263
#   0.482585558000355  0.479461338498757  0.476337112444722  0.473212879838495  0.470088640680298  0.466964394970332
#   0.463840142708777  0.460715883895808  0.45759161853158   0.454467346616231  0.451343068149891  0.448218783132676
#   0.4450944915647    0.441970193446052  0.438845888776826  0.435721577557101  0.432597259786952  0.429472935466439
#   0.426348604595629  0.423224267174569  0.420099923203313  0.416975572681904  0.413851215610378  0.410726851988773
#   0.407602481817122  0.404478105095451  0.401353721823784  0.398229332002148  0.395104935630559  0.391980532709039
#   0.388856123237599  0.385731707216257  0.382607284645025  0.379482855523912  0.37635841985293   0.373233977632087
#   0.370109528861389  0.366985073540843  0.363860611670458  0.360736143250235  0.357611668280179  0.354487186760296
#   0.351362698690586  0.348238204071054  0.345113702901702  0.341989195182532  0.338864680913545  0.335740160094745
#   0.33261563272613   0.329491098807704  0.326366558339465  0.323242011321416  0.320117457753558  0.316992897635888
#   0.313868330968411  0.310743757751125  0.307619177984031  0.304494591667127  0.301369998800417  0.298245399383898
#   0.295120793417572  0.291996180901438  0.288871561835496  0.285746936219747  0.282622304054191  0.279497665338827
#   0.276373020073656  0.273248368258678  0.270123709893892  0.266999044979299  0.263874373514899  0.260749695500692
#   0.257625010936677  0.254500319822855  0.251375622159227  0.24825091794579   0.245126207182547  0.242001489869496
#   0.238876766006638  0.235752035593973  0.2326272986315    0.229502555119221  0.226377805057134  0.223253048445239
#   0.220128285283538  0.217003515572029  0.213878739310713  0.21075395649959   0.20762916713866   0.204504371227923
#   0.201379568767378  0.198254759757025]
# fixed_theta_1 = 0.6703125
# fixed_theta_2 = 0.3359375
# sender_1_signal_0_fixed_theta_1_fixed_theta_2_h_array = 
# [  9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01
#    9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01
#    9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01
#    9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01
#    9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01
#    9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01
#    9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01
#    9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01
#    9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01
#    9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01
#    9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01
#    9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01
#    9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01
#    9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01
#    9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01
#    9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01
#    9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01
#    9.996287521788654e-01   9.996287521788655e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01
#    9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01
#    9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01
#    9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01
#    9.996287521788654e-01   9.996287521788654e-01   9.996287521788654e-01   1.000000000000000e+00   1.000000000000000e+00
#    1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00
#    1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00
#    1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00
#    1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00
#    1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00
#    1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00
#    1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00
#    1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00
#    1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00
#    1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00
#    1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00
#    1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00
#    1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00
#    1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00
#    1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00
#    1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00
#    1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00
#    1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00
#    1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00
#    1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00
#    1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.000000000000000e+00   1.336734578636394e-14
#    1.336734578636394e-14   1.336734578636394e-14   1.336734578636394e-14   1.336734578636394e-14   1.336734578636394e-14
#    1.336734578636394e-14   1.336734578636394e-14   1.336734578636394e-14   1.336734578636394e-14   1.336734578636394e-14
#    1.336734578636394e-14   1.336734578636394e-14   1.336734578636394e-14   1.336734578636394e-14   1.336734578636394e-14
#    1.336734578636393e-14   1.336734578636393e-14   1.336734578636393e-14   1.336734578636395e-14   1.336734578636393e-14
#    1.336734578636393e-14   1.336734578636394e-14   1.336734578636394e-14   1.336734578636394e-14   1.336734578636394e-14
#    1.336734578636394e-14   1.336734578636394e-14   1.336734578636394e-14   1.336734578636394e-14   1.336734578636394e-14
#    1.336734578636394e-14   1.336734578636394e-14   1.336734578636394e-14   1.336734578636394e-14   1.336734578636393e-14
#    1.336734578636394e-14   1.336734578636394e-14   1.336734578636394e-14   1.336734578636394e-14   1.336734578636394e-14
#    1.336734578636394e-14   1.336734578636394e-14   1.336734578636394e-14   1.336734578636394e-14   1.336734578636394e-14
#    1.336734578636394e-14   1.336734578636394e-14   1.336734578636394e-14   1.336734578636392e-14   1.336734578636394e-14
#    1.336734578636392e-14   1.336734578636392e-14   1.336734578636392e-14   1.336734578636392e-14   1.336734578636392e-14
#    1.336734578636392e-14   1.336734578636392e-14   1.336734578636392e-14   1.336734578636392e-14   1.336734578636392e-14
#    1.336734578636392e-14   1.336734578636392e-14   1.336734578636392e-14   1.336734578636392e-14   1.336734578636392e-14
#    1.336734578636397e-14   1.336734578636392e-14   1.336734578636392e-14   1.336734578636392e-14   1.336734578636392e-14
#    1.336734578636397e-14   1.336734578636392e-14   1.336734578636392e-14   1.336734578636392e-14   1.336734578636392e-14
#    1.336734578636392e-14   1.336734578636392e-14   1.336734578636392e-14   1.336734578636392e-14   1.336734578636392e-14
#    1.336734578636392e-14   1.336734578636392e-14   1.336734578636392e-14   1.336734578636392e-14   1.336734578636392e-14
#    1.336734578636392e-14   1.336734578636392e-14   1.336734578636392e-14   1.336734578636392e-14   1.336734578636392e-14
#    1.336734578636392e-14   1.336734578636392e-14   1.336734578636392e-14   1.336734578636392e-14   1.336734578636392e-14
#    1.336734578636392e-14   1.336734578636392e-14   1.336734578636392e-14   1.336734578636392e-14   1.336734578636392e-14
#    1.336734578636392e-14   1.336734578636392e-14   1.336734578636392e-14   1.336734578636373e-14   1.336734578636411e-14]
# sender_1_signal_1_fixed_theta_1_h_array = 
# [ 0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.999999999999982  0.999999999999982
#   0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982
#   0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982
#   0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982
#   0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982
#   0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982
#   0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982
#   0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982
#   0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982
#   0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982
#   0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982
#   0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982
#   0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982
#   0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982
#   0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982
#   0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982
#   0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982
#   0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982  0.999999999999982
#   0.999999999999982  0.999999999999982]
# sender_1_signal_2_fixed_theta_2_h_array = 
# [ 0.000371247409709  0.000371246949152  0.000371246433414  0.000371245855683  0.000371245208277  0.000371244482538
#   0.000371243668696  0.000371242755733  0.000371241731208  0.000371240581077  0.000371239289474  0.000371237838478
#   0.000371236207827  0.000371234374618  0.000371232312945  0.000371229993499  0.000371227383113  0.00037122444424
#   0.000371221134363  0.000371217405326  0.00037121320257   0.00037120846427   0.000371203120349  0.000371197091365
#   0.000371190287245  0.000371182605847  0.000371173931341  0.000371164132359  0.000371153059925  0.000371140545099
#   0.000371126396334  0.000371110396499  0.00037109229954   0.000371071826742  0.000371048662565  0.000371022450016
#   0.000370992785545  0.00037095921342   0.000370921219612  0.000370878225164  0.000370829579106  0.000370774550962
#   0.000370712322965  0.00037064198214   0.000370562512469  0.000370472787451  0.000370371563432  0.000370257474226
#   0.000370129027601  0.000369984604368  0.000369822460873  0.000369640735776  0.000369437462019  0.000369210584854
#   0.000368957986612  0.000368677518677  0.000368367040683  0.000368024466426  0.000367647815332  0.000367235267606
#   0.00036678522048   0.000366296342432  0.000365767621908  0.000365198407062  0.000364588433449  0.000363937837362
#   0.000363247153591  0.000362517297666  0.00036174953389   0.000360945431582  0.00036010681274   0.000359235694714
#   0.000358334231461  0.000357404656639  0.000356449231125  0.00035547019687   0.000354469738197  0.000353449950971
#   0.000352412819504  0.000351360200634  0.000350293814125  0.000349215238416  0.000348125910708  0.000347027130399
#   0.000345920065001  0.000344805757749  0.000343685136294  0.000342559021938  0.000341428139036  0.000340293124248
#   0.000339154535446  0.00033801286013   0.00033686852325   0.000335721894416  0.000334573294453  0.000333423001333
#   0.000332271255496  0.000331118264592  0.000329964207706  0.000328809239072  0.000327653491366  0.000326497078577
#   0.000325340098536  0.000324182635118  0.000323024760153  0.000321866535104  0.000320708012507  0.000319549237231  0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.                 0.                 0.                 0.                 0.                 0.                 0.
#   0.               ]
# here now
# 1493873844.26
# receiver_2_signal_0_normalization_factor_h_gte_theta_1_gt_theta_2 = 0.014157417045967964
# 1493873926.67
# receiver_2_signal_0_normalization_factor_h_lt_theta_1_lte_theta_2 = 0.8160334535892759
# 1493874204.2
# receiver_2_signal_0_normalization_factor_h_lt_theta_1_gt_theta_2 = 0.08181818181818182
# 1493874208.22
# receiver_2_signal_0_normalization_factor_h_gte_theta_1_lte_theta_2 = 0.07346331723644361
# 1493874422.2
# receiver_2_signal_0_normalization_factor = 0.9854723696898693
# 1493874422.2
# receiver_2_signal_1_normalization_factor_h_lte_theta_2 = 0.00830689131565488
# 1493874614.14
# receiver_2_signal_1_normalization_factor_h_gt_theta_2 = 0.004024401135850221
# 1493874707.57
# receiver_2_signal_1_normalization_factor = 0.012331292451505101
# 1493874707.57
# receiver_2_signal_2_normalization_factor_h_lt_theta_1 = 0.002148364592542277
# 1493875040.4
# receiver_2_signal_2_normalization_factor_h_gte_theta_1 = 4.797327620031842e-05
# 1493875334.88
# receiver_2_signal_2_normalization_factor = 0.0021963378687425953
# 1493875334.88
# receiver_2_signal_0_h_array = 
# [  8.864604827693391e+00   8.672463385568722e+00   8.483023307110633e+00   8.294643937638780e+00   8.105560124724983e+00
#    7.915328723500205e+00   7.724913211135548e+00   7.535664677473618e+00   7.348653620890868e+00   7.164563706967775e+00
#    6.983789642619922e+00   6.806543801772649e+00   6.632929322392497e+00   6.462984390949492e+00   6.296708196030609e+00
#    6.134076156184914e+00   5.975048991940841e+00   5.819578243888937e+00   5.667609704761568e+00   5.519085601351058e+00
#    5.373946010412925e+00   5.232129794744314e+00   5.093575232242189e+00   4.958220444499524e+00   4.826003691991676e+00
#    4.696863578864151e+00   4.570739195415574e+00   4.447570216939972e+00   4.327296971527326e+00   4.209860485456934e+00
#    4.095202512187736e+00   3.983265549182831e+00   3.873992845604669e+00   3.767328403096655e+00   3.663216971304583e+00
#    3.561604039409071e+00   3.462435824687050e+00   3.365659258962205e+00   3.271221973719206e+00   3.179072284630176e+00
#    3.089159176264269e+00   3.001432287816315e+00   2.915841900792752e+00   2.832338929727457e+00   2.750874917160361e+00
#    2.671402034287703e+00   2.593873088871337e+00   2.518241542154255e+00   2.444461536642475e+00   2.372487936642333e+00
#    2.302276383338405e+00   2.233783365907237e+00   2.166966309625076e+00   2.101783681089686e+00   2.038195109498115e+00
#    1.976161521397989e+00   1.915645284506352e+00   1.856610354183942e+00   1.799022414163580e+00   1.742849001437841e+00
#    1.688059604147120e+00   1.634625721214327e+00   1.582520873618829e+00   1.531720559713984e+00   1.482202150775650e+00
#    1.433944727679960e+00   1.386928864682972e+00   1.341136371016446e+00   1.296550004733100e+00   1.253153175394358e+00
#    1.210929652524654e+00   1.169863295306597e+00   1.129937816093932e+00   1.091136586503228e+00   1.053442490705145e+00
#    1.016837826617018e+00   9.813042524079799e-01   9.468227732982234e-01   9.133737621291264e-01   8.809370065274724e-01
#    8.494917755333008e-01   8.190168991094049e-01   7.894908548144915e-01   7.608918569340314e-01   7.331979443966959e-01
#    7.063870647724573e-01   6.804371524990237e-01   6.553262001927757e-01   6.310323224652641e-01   6.075338120962676e-01
#    5.848091887261178e-01   5.628372404431926e-01   5.415970587768725e-01   5.210680676800622e-01   5.012300471142747e-01
#    4.820631518472923e-01   4.635479260488673e-01   4.456653142318333e-01   4.283966690402824e-01   4.117237563374779e-01
#    3.956287579968523e-01   3.800942727518789e-01   3.651033154159462e-01   3.506393147423888e-01   3.366861101578054e-01
#    3.232279475687720e-01   3.102494744129164e-01   2.977357340998176e-01   2.856721599650271e-01   2.740445688413663e-01
#    2.628391543351912e-01   2.520424798812347e-01   2.416414716376292e-01   2.316234112725016e-01   2.219759286848953e-01
#    2.126869946954590e-01   2.037449137361864e-01   1.951383165633057e-01   1.868561530130660e-01   1.788876848165257e-01
#    1.712224784863902e-01   1.638503982864055e-01   1.567615992916858e-01   1.499465205465865e-01   1.433958783252662e-01
#    1.371006594988612e-01   1.310521150121857e-01   1.252417534720398e-01   1.196613348485133e-01   1.143028642901139e-01
#    1.091585860530753e-01   1.042209775448249e-01   9.948274348127634e-02   9.493681015735791e-02   9.057631982998487e-02
#    8.639462521251580e-02   8.238528407959819e-02   7.854205398120694e-02   7.485888706459018e-02   7.132992500277532e-02
#    6.794949402823566e-02   6.471210007028073e-02   6.161242399470498e-02   5.864531694420934e-02   5.580579577810391e-02
#    5.308903860977644e-02   5.049038044042994e-02   4.800530888757782e-02   4.562946000679508e-02   4.335861420523033e-02
#    4.118869224540474e-02   3.911575133773343e-02   3.713598132052307e-02   3.524570092569696e-02   3.344135412904396e-02
#    3.171950658338443e-02   3.007684213344220e-02   2.851015941082163e-02   2.701636850782345e-02   2.559248772869351e-02
#    2.423564041695423e-02   2.294305185747930e-02   2.171204625198661e-02   2.054004376664141e-02   1.942455765047512e-02
#    1.836319142334131e-02   1.735363613214588e-02   1.639366767410315e-02   1.548114418578475e-02   1.461400349674352e-02
#    1.379026064650935e-02   1.300800546376912e-02   1.226540020673824e-02   1.156067726230045e-02   1.089213690656729e-02
#    1.025814511940269e-02   9.657131458123816e-03   9.087586985482560e-03   8.548062252107084e-03   8.037165332150969e-03
#    7.553559911092742e-03   7.095963424642358e-03   6.663145247725352e-03   6.253924932529104e-03   5.867170494609475e-03
#    5.501796746070006e-03   5.156763674839016e-03   4.831074869083904e-03   4.523775985815106e-03   4.233953262745466e-03
#    3.960732072484202e-03   3.703275518157528e-03   3.460783069561324e-03   3.232489238963853e-03   3.017662296294394e-03
#    2.815603018628455e-03   2.625643485825163e-03   2.447145900320653e-03   2.279501451424165e-03   2.122129210611449e-03
#    1.974475061254915e-03   1.836010661403919e-03   1.706232438845808e-03   1.584660617690072e-03   1.470838275730042e-03
#    1.364330431848203e-03   1.264723162743166e-03   1.171622748267739e-03   1.084654844679193e-03   1.003463685114239e-03
#    9.277113066125446e-04   8.570768030239503e-04   7.912556031456053e-04   7.299587734463997e-04   6.729123447469609e-04
#    6.198566622344188e-04   5.705457582019023e-04   5.247467469134262e-04   4.822392410054061e-04   4.428147888465550e-04
#    4.062763322883323e-04   3.724376842483876e-04   3.411230254657516e-04   3.121664206883480e-04   2.854113513723162e-04
#    2.607102683648626e-04   2.379241600743244e-04   2.169221380231790e-04   1.975810384982364e-04   1.797850399335617e-04
#    1.634252955571895e-04   1.483995808420403e-04   1.346119553107490e-04   1.219724382532853e-04   1.103966979253342e-04
#    9.980575380441302e-05   9.012569148959007e-05   8.128738983948567e-05   7.322625995194472e-05   6.588199559738809e-05
#    5.919833472638275e-05   5.312283168039117e-05   4.760663974300705e-05   4.260430367722011e-05   3.807356190240325e-05
#    3.397515797277208e-05   3.027266102702465e-05   2.693229488673705e-05   2.392277548886168e-05   2.121515634535257e-05
#    1.878268173052876e-05   1.660064730427323e-05   1.464626788656302e-05   1.289855210612706e-05   1.133818365323410e-05
#    9.947408873730144e-06   8.709930448459912e-06   7.610807367247341e-06   6.636357694462524e-06   5.774073889722757e-06
#    5.012533285378775e-06   4.341321425365608e-06   3.750956536501306e-06   3.232819265358204e-06   2.779086640928348e-06
#    2.382670120238783e-06   2.037157526068863e-06   1.736758691905912e-06   1.476254635165050e-06   1.250950085494819e-06
#    1.056629200691673e-06   8.895143083537383e-07   7.462275169175079e-07   6.237550451402072e-07   5.194141244154558e-07
#    4.308223335407846e-07   3.558692306921521e-07   2.926901524031815e-07   2.396420542952084e-07   1.952812731584857e-07
#    1.583430947450424e-07   1.277230162996050e-07   1.024595974268604e-07   8.171885801024576e-08   6.477976655686843e-08
#    5.102160687364648e-08   3.991198878955021e-08   3.099624779057193e-08   2.388783561686714e-08   1.825967449986708e-08
#    1.383640181994304e-08   1.038743568971251e-08   7.720795598609542e-09   5.677615790824587e-09   4.128277235350483e-09
#    2.963832324207560e-09   2.099365273766608e-09   1.465435649230607e-09   1.006736499836877e-09   6.796404712863700e-10
#    4.500934893224011e-10   2.918186277446429e-10   1.847954510705634e-10   1.139827061584602e-10   6.825472856375869e-11
#    3.952432830382496e-11   2.203300692642607e-11   1.174903978571773e-11   5.950429414361604e-12   2.835606341993074e-12
#    1.256049903089642e-12   5.088318853013944e-13   1.843460995670430e-13   5.784759768811967e-14   1.498393341298401e-14
#    2.965370050911895e-15   3.912762884559532e-16   2.611696618108300e-17   4.320125350668185e-19   6.483163453454905e-23]
# receiver_2_signal_1_h_array = 
# [  3.818465385478143e-04   1.186929409732369e-03   2.051765116900197e-03   2.981714644425506e-03   3.981802756581742e-03
#    5.057340353183988e-03   6.214822892649542e-03   7.462115154821362e-03   8.808257857758045e-03   1.026332300965376e-02
#    1.183840706248590e-02   1.354570234513112e-02   1.539860483704395e-02   1.741184149294486e-02   1.960161229718265e-02
#    2.198574666002685e-02   2.458387526841411e-02   2.741761903704969e-02   3.051079700114624e-02   3.388965512753920e-02
#    3.758311812318491e-02   4.162306642226231e-02   4.604464062799749e-02   5.088657576465201e-02   5.619156774634847e-02
#    6.200667447550248e-02   6.838375393584289e-02   7.537994147172522e-02   8.305816822775061e-02   9.148772224496790e-02
#    1.007448531060286e-01   1.109134201023094e-01   1.220855826433365e-01   1.343625299363368e-01   1.478552447238383e-01
#    1.626852929513455e-01   1.789856274970791e-01   1.969013893656459e-01   2.165906838483335e-01   2.382253019019534e-01
#    2.619913482261184e-01   2.880897270942960e-01   3.167364248629068e-01   3.481625143225609e-01   3.826137909490064e-01
#    4.203499352412385e-01   4.616430796973489e-01   5.067756451134208e-01   5.560373010012361e-01   6.097209019693929e-01
#    6.681172596381119e-01   7.315086324515088e-01   8.001608583181596e-01   8.743141217370789e-01   9.541724410332864e-01
#    1.039892082986487e+00   1.131569257766710e+00   1.229227607327814e+00   1.332806159318741e+00   1.442148553862979e+00
#    1.556994435528592e+00   1.676973910335745e+00   1.801605875944141e+00   1.930300832327174e+00   2.062368477794625e+00
#    2.197030018604487e+00   2.333434714755584e+00   2.470679805499199e+00   2.607832660804676e+00   2.743953832376518e+00
#    2.878119651362816e+00   3.009443135789410e+00   3.137092202372655e+00   3.260304482404310e+00   3.378398372356554e+00
#    3.490780263129754e+00   3.596948154897807e+00   3.696492058715767e+00   3.789091706334625e+00   3.874512141903049e+00
#    3.952597765517938e+00   4.023265354735986e+00   4.086496521084158e+00   4.142329977702617e+00   4.190853911604208e+00
#    4.232198676639325e+00   4.266529955255712e+00   4.294042480428411e+00   4.314954363977135e+00   4.329502043120732e+00
#    4.337935832200456e+00   4.340516049463077e+00   4.337509678050972e+00   4.329187514447676e+00   4.315821755317494e+00
#    4.297683973917406e+00   4.275043439221552e+00   4.248165733944715e+00   4.217311631305121e+00   4.182736194284421e+00
#    4.144688065083323e+00   4.103408916275897e+00   4.059133038733849e+00   4.012087044668599e+00   3.962489667098814e+00
#    3.910551639690671e+00   3.856475643247798e+00   3.800456307166870e+00   3.742680255946699e+00   3.683326192369416e+00
#    3.622565010288079e+00   3.560559931081110e+00   3.497466658794096e+00   3.433433549805352e+00   3.368601793543080e+00
#    3.303105601366201e+00   3.237072401213622e+00   3.170623036041204e+00   3.103871964413679e+00   3.036927461910612e+00
#    2.969891822249413e+00   2.902861557232396e+00   2.835927594794828e+00   2.769175474572470e+00   2.702685540524811e+00
#    2.636533130247814e+00   2.570788760690966e+00   2.505518310060343e+00   2.440783195744613e+00   2.376640548146608e+00
#    2.313143380340382e+00   2.250340753504528e+00   2.188277938107668e+00   2.126996570842495e+00   2.066534807321524e+00
#    2.006927470561224e+00   1.948206195292013e+00   1.890399568140499e+00   1.833533263737177e+00   1.777630176808347e+00
#    1.722710550315316e+00   1.668792099707212e+00   1.615890133356256e+00   1.564017669246222e+00   1.513185547985570e+00
#    1.463402542218917e+00   1.414675462509190e+00   1.367009259764379e+00   1.320407124282104e+00   1.274870581485156e+00
#    1.230399584419605e+00   1.186992603094570e+00   1.144646710717650e+00   1.103357666920347e+00   1.063119998024620e+00
#    1.023927074434042e+00   9.857711852018466e-01   9.486436098594004e-01   9.125346875622911e-01   8.774338836225506e-01
#    8.433298534914175e-01   8.102105042564096e-01   7.780630537155371e-01   7.468740870905491e-01   7.166296114401203e-01
#    6.873151078329449e-01   6.589155813397638e-01   6.314156089023875e-01   6.047993851368310e-01   5.790507661267573e-01
#    5.541533112624694e-01   5.300903231797761e-01   5.068448858376954e-01   4.843999008886011e-01   4.627381220890096e-01
#    4.418421883073256e-01   4.216946546726920e-01   4.022780222172259e-01   3.835747659585274e-01   3.655673614840537e-01
#    3.482383100836389e-01   3.315701624755835e-01   3.155455411708851e-01   3.001471615193387e-01   2.853578514804002e-01
#    2.711605701608846e-01   2.575384251607365e-01   2.444746887673143e-01   2.319528130378129e-01   2.199564438086516e-01
#    2.084694336698735e-01   1.974758539418066e-01   1.869600056904762e-01   1.769064298174737e-01   1.672999162109103e-01
#    1.581255123299836e-01   1.493685302416179e-01   1.410145538335593e-01   1.330494445442526e-01   1.254593466557562e-01
#    1.182306918419237e-01   1.113502030500172e-01   1.048048977449054e-01   9.858209054428889e-02   9.266939527272475e-02
#    8.705472646153077e-02   8.172630032099644e-02   7.667263521064958e-02   7.188255163268306e-02   6.734517177299429e-02
#    6.304991861365004e-02   5.898651463996123e-02   5.514498016472157e-02   5.151563129155174e-02   4.808907753867771e-02
#    4.485621914387145e-02   4.180824407068755e-02   3.893662473554192e-02   3.623311447459839e-02   3.368974376885787e-02
#    3.129881624528112e-02   2.905290447121590e-02   2.694484556796358e-02   2.496773659590270e-02   2.311492991810338e-02
#    2.138002823874301e-02   1.975687964973404e-02   1.823957250836691e-02   1.682243022331542e-02   1.550000595298266e-02
#    1.426707722878781e-02   1.311864051550950e-02   1.204990572032963e-02   1.105629066175384e-02   1.013341550912500e-02
#    9.277097202996144e-03   8.483343866183771e-03   7.748349214886778e-03   7.068486978826752e-03   6.440305338943849e-03
#    5.860521390769110e-03   5.326015641186737e-03   4.833826545901925e-03   4.381145094537587e-03   3.965309449899938e-03
#    3.583799647576449e-03   3.234232361660979e-03   2.914355742039448e-03   2.622044328315850e-03   2.355294045111928e-03
#    2.112217283135530e-03   1.891038070081113e-03   1.690087335102507e-03   1.507798270281821e-03   1.342701792209697e-03
#    1.193422106491362e-03   1.058672377699034e-03   9.372504703925119e-04   8.280350237778248e-04   7.299810625592502e-04
#    6.421165176308997e-04   5.635382250383109e-04   4.934083242803850e-04   4.309507095700632e-04   3.754476013659826e-04
#    3.262362340706550e-04   2.827056598946446e-04   2.442936686755489e-04   2.104838232373433e-04   1.808026096800190e-04
#    1.548167018012881e-04   1.321303386715187e-04   1.123828142099821e-04   9.524607744493381e-05   8.042244198201193e-05
#    6.764240305494973e-05   5.666256038961280e-05   4.726364497692843e-05   3.924864772235077e-05   3.244104781911430e-05
#    2.668313857966718e-05   2.183444835433085e-05   1.777025406842933e-05   1.438017997258880e-05   1.156690766227426e-05
#    9.244924228003779e-06   7.339397277639017e-06   5.785107666941880e-06   4.525462375550685e-06   3.511580612684149e-06
#    2.701450172558479e-06   2.059151029436054e-06   1.554143139674280e-06   1.160615403109196e-06   8.568140292880765e-07
#    6.248483687230100e-07   4.496369007939028e-07   3.188926836879309e-07   2.226126144175627e-07   1.527295285385793e-07
#    1.028043352811183e-07   6.775528803798427e-08   4.362154383748219e-08   2.735822653126330e-08   1.666027693685713e-08
#    9.812449082304942e-09   5.562433013695989e-09   3.017574366225158e-09   1.555022022291981e-09   7.541138667502532e-10
#    3.399953025261638e-10   1.402135238781263e-10   5.172207854626100e-11   1.652848600046959e-11   4.360749557537062e-12
#    8.791971090684543e-13   1.182088968632648e-13   8.041531305475739e-15   1.355985044421788e-16   2.074838579199494e-20]
# receiver_2_signal_1_theta_1_array = 
# [  2.831121840083249e-02   3.082251010422280e-02   3.356275351047803e-02   3.655348071155381e-02   3.981858185353016e-02
#    4.338457679180252e-02   4.728073488936847e-02   5.153920344344870e-02   5.619524014087913e-02   6.128753238076472e-02
#    6.685856760388748e-02   7.295503857353897e-02   7.962828511023576e-02   8.693477270823409e-02   9.493661869895181e-02
#    1.037021665495069e-01   1.133066153128277e-01   1.238327088123103e-01   1.353714894843926e-01   1.480231217272371e-01
#    1.618977894307772e-01   1.771166720305717e-01   1.938130028638564e-01   2.121332127268064e-01   2.322381602338747e-01
#    2.543044487349820e-01   2.785258269976162e-01   3.051146674113743e-01   3.343035108864404e-01   3.663466616044727e-01
#    4.015218069955928e-01   4.401316283507954e-01   4.825053548634406e-01   5.290001980939167e-01   5.800025842869572e-01
#    6.359290780375284e-01   6.972268619206247e-01   7.643736023950154e-01   8.378764923076298e-01   9.182702148060723e-01
#    1.006113523201458e+00   1.101984078084131e+00   1.206471129969370e+00   1.320165588077227e+00   1.443646981222172e+00
#    1.577466806062501e+00   1.722127785652353e+00   1.878058645680326e+00   2.045584178607980e+00   2.224890630489927e+00
#    2.415986833419516e+00   2.618662033351908e+00   2.832442029104695e+00   3.056546020068707e+00   3.289847399423652e+00
#    3.530842525173598e+00   3.777632106287159e+00   4.027920072211806e+00   4.279034455987168e+00   4.527973750435906e+00
#    4.771480317896724e+00   5.006139817180385e+00   5.228502512317625e+00   5.435219183451564e+00   5.623181728794546e+00
#    5.789656989378346e+00   5.932402263438540e+00   6.049752548343568e+00   6.140672558495939e+00   6.204770511253169e+00
#    6.242274855745515e+00   6.253978834589277e+00   6.241160463898513e+00   6.205486904378893e+00   6.148912275059815e+00
#    6.073576965163030e+00   5.981714797559957e+00   5.875572386424664e+00   5.757343053702876e+00   5.629115963375388e+00
#    5.492839826222471e+00   5.350299651207420e+00   5.203104534901088e+00   5.052684312052428e+00   4.900292950460467e+00
#    4.747016778465374e+00   4.593785914609162e+00   4.441387574832180e+00   4.290480228391099e+00   4.141607839098298e+00
#    3.995213653785834e+00   3.851653182754796e+00   3.711206159628748e+00   3.574087375202972e+00   3.440456357289414e+00
#    3.310425921986087e+00   3.184069656570710e+00   3.061428414928376e+00   2.942515916870401e+00   2.827323545914777e+00
#    2.715824438419904e+00   2.607976952156404e+00   2.503727595759074e+00   2.403013492943418e+00   2.305764447535231e+00
#    2.211904667668665e+00   2.121354200223553e+00   2.034030119848501e+00   1.949847510825961e+00   1.868720274599569e+00
#    1.790561790987510e+00   1.715285456912734e+00   1.642805122843402e+00   1.573035444001091e+00   1.505892160705711e+00
#    1.441292319931020e+00   1.379154448193417e+00   1.319398684243365e+00   1.261946878631980e+00   1.206722666048548e+00
#    1.153651515335393e+00   1.102660761256805e+00   1.053679621404059e+00   1.006639201038040e+00   9.614724881867092e-01
#    9.181143409112423e-01   8.765014683190597e-01   8.365724066233585e-01   7.982674913174358e-01   7.615288263406915e-01
#    7.263002509546506e-01   6.925273049164351e-01   6.601571924290368e-01   6.291387452586126e-01   5.994223853357136e-01
#    5.709600870969983e-01   5.437053397744455e-01   5.176131097984378e-01   4.926398034477634e-01   4.687432298523643e-01
#    4.458825644323765e-01   4.240183128388349e-01   4.031122754466421e-01   3.831275124383068e-01   3.640283095072363e-01
#    3.457801442013935e-01   3.283496529217411e-01   3.117045985847111e-01   2.958138389537512e-01   2.806472956416814e-01
#    2.661759237828845e-01   2.523716823722312e-01   2.392075052659467e-01   2.266572728382624e-01   2.146957842866762e-01
#    2.032987305778028e-01   1.924426680251753e-01   1.821049924898885e-01   1.722639141946004e-01   1.628984331411746e-01
#    1.539883151220484e-01   1.455140683153104e-01   1.374569204533999e-01   1.297987965553150e-01   1.225222972122264e-01
#    1.156106774164300e-01   1.090478259236119e-01   1.028182451384862e-01   9.690703151392761e-02   9.129985645382473e-02
#    8.598294770997315e-02   8.094307126342794e-02   7.616751368085163e-02   7.164406493649378e-02   6.736100169056096e-02
#    6.330707101484651e-02   5.947147455660633e-02   5.584385313178746e-02   5.241427173882711e-02   4.917320498436364e-02
#    4.611152291231577e-02   4.322047722790234e-02   4.049168790829596e-02   3.791713019171448e-02   3.548912193687481e-02
#    3.320031134484687e-02   3.104366503545890e-02   2.901245647052223e-02   2.710025471625285e-02   2.530091353738178e-02
#    2.360856081555791e-02   2.201758828475534e-02   2.052264157651080e-02   1.911861056792167e-02   1.780062002544646e-02
#    1.656402053765615e-02   1.540437973019025e-02   1.431747375627961e-02   1.329927905630033e-02   1.234596437992892e-02
#    1.145388306457190e-02   1.061956556384475e-02   9.839712219977956e-03   9.111186274127237e-03   8.431007108666077e-03
#    7.796343715637538e-03   7.204508385640111e-03   6.652950611520751e-03   6.139251201343652e-03   5.661116595200208e-03
#    5.216373380519752e-03   4.802963000635079e-03   4.418936651450235e-03   4.062450361149812e-03   3.731760247981269e-03
#    3.425217951232177e-03   3.141266230614173e-03   2.878434729354866e-03   2.635335896386529e-03   2.410661063108451e-03
#    2.203176670286006e-03   2.011720640735337e-03   1.835198893527530e-03   1.672581995529999e-03   1.522901946186220e-03
#    1.385249091517287e-03   1.258769163410192e-03   1.142660440338679e-03   1.036171025742119e-03   9.385962403670963e-04
#    8.492761249545268e-04   7.675930497324526e-04   6.929694272513144e-04   6.248655251740157e-04   5.627773757081002e-04
#    5.062347784413162e-04   4.547993934150257e-04   4.080629213423027e-04   3.656453679490114e-04   3.271933894868839e-04
#    2.923787165374366e-04   2.608966532945558e-04   2.324646495817765e-04   2.068209429275655e-04   1.837232680884453e-04
#    1.629476314754471e-04   1.442871480041823e-04   1.275509379528319e-04   1.125630814754724e-04   9.916162848048678e-05
#    8.719766164527900e-05   7.653441039913999e-05   6.704641376592924e-05   5.861873001718308e-05   5.114619114440458e-05
#    4.453270021657465e-05   3.869056974537944e-05   3.353989923627457e-05   2.900799015828071e-05   2.502879661935874e-05
#    2.154241008732101e-05   1.849457654850688e-05   1.583624454789315e-05   1.352314260490446e-05   1.151538454895928e-05
#    9.777101367712619e-06   8.276098209048500e-06   6.983535225132865e-06   5.873630993253623e-06   4.923387293760157e-06
#    4.112334070159649e-06   3.422293440337799e-06   2.837161670945818e-06   2.342708069231664e-06   1.926389787995587e-06
#    1.577181579913958e-06   1.285419577203505e-06   1.042658211490206e-06   8.415394268019116e-07   6.756733758232449e-07
#    5.395298259337535e-07   4.283395370964312e-07   3.380049083738074e-07   2.650192237219936e-07   2.063938607503871e-07
#    1.595928583353249e-07   1.224742703403174e-07   9.323776322346839e-08   7.037794500417998e-08   5.264294191648179e-08
#    3.899776709507961e-08   2.859205282264410e-08   2.073174421288940e-08   1.485437771363279e-08   1.050759248730191e-08
#    7.330546563236416e-09   5.037932656632002e-09   3.406310713099078e-09   2.262495565542559e-09   1.473758581429682e-09
#    9.396218334860979e-10   5.850421454534069e-10   3.548003467526729e-10   2.089282157272574e-10   1.190219005858673e-10
#    6.530608023185043e-11   3.432776384815607e-11   1.717242840897257e-11   8.108157208521173e-12   3.575570035267913e-12
#    1.452626775524264e-12   5.338410042829922e-13   1.730585130873890e-13   4.773773477041455e-14   1.061465303699888e-14
#    1.744031085106762e-15   1.816577775373703e-16   8.792283345394026e-18   8.860595870073530e-20   4.501649072839294e-24]
# receiver_2_signal_2_h_array = 
# [  6.933609036799389e+01   5.531836596253101e+01   4.226694152199563e+01   3.087720844693450e+01   2.190105705319126e+01
#    1.549783420841304e+01   1.119636490943297e+01   8.352161484747851e+00   6.447232740188783e+00   5.137746384163909e+00
#    4.210106425717502e+00   3.533103485794053e+00   3.025108906192594e+00   2.634196699928955e+00   2.326490338397929e+00
#    2.079322061985968e+00   1.877154225954011e+00   1.709095231368154e+00   1.567351385668347e+00   1.446239648620098e+00
#    1.341544020231282e+00   1.250087147681397e+00   1.169439602511686e+00   1.097719002902773e+00   1.033448881539851e+00
#    9.754579863652902e-01   9.228073952023488e-01   8.747370569862549e-01   8.306260952142256e-01   7.899629904662728e-01
#    7.523229425235352e-01   7.173505106854670e-01   6.847461765375517e-01   6.542558512411172e-01   6.256626143444024e-01
#    5.987801589133180e-01   5.734475524219893e-01   5.495250203313784e-01   5.268905305622988e-01   5.054370096602729e-01
#    4.850700605900771e-01   4.657060814681791e-01   4.472707067532914e-01   4.296975093409622e-01   4.129269149825688e-01
#    3.969052905197041e-01   3.815841751563864e-01   3.669196301592593e-01   3.528716871217608e-01   3.394038787097020e-01
#    3.264828388189646e-01   3.140779613711248e-01   3.021611088838574e-01   2.907063633536709e-01   2.796898131082244e-01
#    2.690893701168011e-01   2.588846128645063e-01   2.490566503504840e-01   2.395880031189667e-01   2.304624975327939e-01
#    2.216651698106416e-01   2.131821767244955e-01   2.050007103316784e-01   1.971089147119295e-01   1.894958033807894e-01
#    1.821511768106793e-01   1.750655402400117e-01   1.682300226054984e-01   1.616362979161884e-01   1.552765106456618e-01
#    1.491432067328923e-01   1.432292715735202e-01   1.375278760055743e-01   1.320324308205727e-01   1.267365498394153e-01
#    1.216340211485171e-01   1.167187857410507e-01   1.119849225733442e-01   1.074266389287037e-01   1.030382649658304e-01
#    9.881425139328896e-02   9.474916932906170e-02   9.083771155065863e-02   8.707469449649712e-02   8.345506052866580e-02
#    7.997388010136702e-02   7.662635359339759e-02   7.340781265550765e-02   7.031372099522670e-02   6.733967458162488e-02
#    6.448140124368079e-02   6.173475979732956e-02   5.909573863516185e-02   5.656045392216127e-02   5.412514743116479e-02
#    5.178618408476673e-02   4.954004928138282e-02   4.738334592488923e-02   4.531279160456253e-02   4.332521530809873e-02
#    4.141755435270250e-02   3.958685118460253e-02   3.783025019249277e-02   3.614499454123401e-02   3.452842304194038e-02
#    3.297796707182277e-02   3.149114755421399e-02   3.006557200669959e-02   2.869893166317445e-02   2.738899867389710e-02
#    2.613362338616979e-02   2.493073165249762e-02   2.377832254889965e-02   2.267446535658219e-02   2.161729771698599e-02
#    2.060502304800163e-02   1.963590836619373e-02   1.870828213091219e-02   1.782053216275218e-02   1.697110363410322e-02
#    1.615849712944977e-02   1.538126677304942e-02   1.463801842160812e-02   1.392740791959012e-02   1.324813941483826e-02
#    1.259896373223019e-02   1.197867680315891e-02   1.138611814869477e-02   1.082016941435962e-02   1.027975295452192e-02
#    9.763830464499145e-03   9.271401658532945e-03   8.801502991880868e-03   8.353206425344338e-03   7.925618230628019e-03
#    7.517877834997577e-03   7.129156703772575e-03   6.758657259258403e-03   6.405611834785018e-03   6.069281662581267e-03
#    5.748955894272129e-03   5.443950652841191e-03   5.153608114953241e-03   4.877295622581745e-03   4.614404822932900e-03
#    4.364350835702809e-03   4.126571446746217e-03   3.900526327275350e-03   3.685696277744988e-03   3.481582495615572e-03
#    3.287705866219838e-03   3.103606275990291e-03   2.928842488127589e-03   2.762988794476835e-03   2.605639799601338e-03
#    2.456404408677471e-03   2.314907946347687e-03   2.180791049300141e-03   2.053709117560744e-03   1.933331783162378e-03
#    1.819342395668207e-03   1.711437524044466e-03   1.609326474395680e-03   1.512730823092067e-03   1.421383964834596e-03
#    1.335030675218428e-03   1.253426687369802e-03   1.176338282245200e-03   1.103541892194646e-03   1.034823717403559e-03
#    9.699793548395075e-04   9.088134393415756e-04   8.511392965010916e-04   7.967786069928040e-04   7.455610820257252e-04
#    6.973241495924426e-04   6.519126512049615e-04   6.091785488140327e-04   5.689806416174440e-04   5.311842924709850e-04
#    4.956611636237222e-04   4.622889615068169e-04   4.309511903125078e-04   4.015369141069272e-04   3.739411571459337e-04
#    3.480614282645017e-04   3.238043285876188e-04   3.010780016010246e-04   2.797961284749991e-04   2.598765841661756e-04
#    2.412413570941468e-04   2.238163710738461e-04   2.075313137578591e-04   1.923194713918273e-04   1.781175696911594e-04
#    1.648656206521640e-04   1.525067751154843e-04   1.409871809043706e-04   1.302558463648443e-04   1.202645091392038e-04
#    1.109675100086439e-04   1.023216716449123e-04   9.428618211503480e-05   8.682248298710153e-05   7.989416188900762e-05
#    7.346684937582220e-05   6.750811996518299e-05   6.198739720372324e-05   5.687586263109704e-05   5.214636851162747e-05
#    4.777335420700493e-05   4.373276606678663e-05   4.000198071670359e-05   3.655973162796951e-05   3.338603885391087e-05
#    3.046214182329384e-05   2.777043508272032e-05   2.529440688339566e-05   2.301858051044355e-05   2.092845825575074e-05
#    1.901046793807920e-05   1.725191187687155e-05   1.564091822881548e-05   1.416639459880901e-05   1.281798383949480e-05
#    1.158602195600516e-05   1.046149803497641e-05   9.436316199907899e-06   8.501974903020732e-06   7.651592934464460e-06
#    6.878409329169931e-06   6.176143280275887e-06   5.538964673398121e-06   4.961466143972181e-06   4.438636593224881e-06
#    3.965836100479638e-06   3.538772171612682e-06   3.153477265548060e-06   2.806476905332571e-06   2.494015617258270e-06
#    2.213159442196879e-06   1.961040253533888e-06   1.735019757185028e-06   1.532673563857459e-06   1.351776189177263e-06
#    1.190286938071693e-06   1.046336631346804e-06   9.182151339156175e-07   8.043596456092238e-07   7.033437169439213e-07
#    6.138669536221627e-07   5.347453749140149e-07   4.649023923999378e-07   4.033603768552115e-07   3.492327823218505e-07
#    3.017167976460738e-07   2.600864969583700e-07   2.236864617399370e-07   1.919258482537969e-07   1.642728752219199e-07
#    1.402497077015999e-07   1.194277141555079e-07   1.014230747208256e-07   8.589271966399786e-08   7.253057795940799e-08
#    6.106411685310123e-08   5.125115416696026e-08   4.287692596495170e-08   3.575139304160304e-08   2.970677050421076e-08
#    2.459526550483836e-08   2.028700893638595e-08   1.666816763935233e-08   1.363922437280091e-08   1.111341348493914e-08
#    9.015300876076183e-09   7.279497479974834e-09   5.849496099225235e-09   4.676622016675666e-09   3.719088368618466e-09
#    2.941147806789910e-09   2.312332495749203e-09   1.806774990302266e-09   1.402603014774064e-09   1.081401622553013e-09
#    8.277366508789331e-10   6.287338027606981e-10   4.737080856372405e-10   3.538387153753877e-10   2.618849548729423e-10
#    1.919386993464729e-10   1.392099457717181e-10   9.984159235236938e-11   7.075030576134305e-11   4.950229483418927e-11
#    3.414567668353453e-11   2.319912449768071e-11   1.550474802126552e-11   1.017816384827542e-11   6.551609159336015e-12
#    4.127166198723892e-12   2.538628120334589e-12   1.520699513658250e-12   8.843755521578310e-13   4.974827252373677e-13
#    2.694919576032393e-13   1.398686579913845e-13   6.905975187491525e-14   3.217796223997722e-14   1.400062414289729e-14
#    5.611027971068172e-15   2.033784623121467e-15   6.501397509412755e-16   1.768112864494731e-16   3.875249113119484e-17
#    6.274835174823798e-18   6.439650596695834e-19   3.070246824392767e-20   3.047179216271007e-22   1.524288627304665e-26]
# receiver_2_signal_2_theta_2_array = 
# [  6.362108722847368  18.688136688082917  29.15583028936614   34.739557268382072  33.813425433591433  28.617074440760344
#   22.509635876886087  17.266049383851865  13.256386403358785  10.306749681613187   8.148675289691072   6.555736508793488
#    5.362760010775442   4.45467811389095    3.752146792670551   3.200189988243565   2.760278167101018   2.405024034529496
#    2.114667106350688   1.874736546853981   1.674482675327787   1.505811309913171   1.36254992182369    1.239935286406213
#    1.134250897804747   1.042566990398788   0.962551786342201   0.892332820768136   0.830393910367987   0.775497790582158
#    0.726627445117291   0.682941192051932   0.643737995814533   0.608430452747778   0.576523586766096   0.547598081566411
#    0.521296927835777   0.497314719164838   0.475389017172484   0.455293344216408   0.436831464656136   0.419832692572386
#    0.404148021974223   0.389646919753867   0.376214655529702   0.363750068639957   0.35216369281403    0.341376174861829
#    0.331316936131391   0.321923035276606   0.313138198642618   0.304911990766039   0.29719910244373    0.289958737811148
#    0.283154085092444   0.276751858297365   0.270721899267764   0.265036831215458   0.25967175631952    0.254603991125142
#    0.249812834456095   0.245279363357291   0.240986253254066   0.236917619075873   0.233058874564387   0.229396607385857
#    0.225918468007867   0.222613070591587   0.219469904399831   0.216479254434939   0.213632130203181   0.210920201657992
#    0.208335741506115   0.205871573171698   0.203521023806371   0.201277881811145   0.199136358401041   0.19709105279816
#    0.195136920685296   0.193269245592013   0.191483612919665   0.189775886342093   0.188142186345682   0.18657887069646
#    0.185082516643521   0.183649904687556   0.182278003760817   0.180963957680706   0.179705072753445   0.178498806417162
#    0.177342756825133   0.176234653280305   0.175172347441915   0.174153805231716   0.173177099377393   0.172240402534203
#    0.171341980933605   0.170480188513238   0.169653461484735   0.168860313304135   0.168099330008336   0.167369165884942
#    0.166668539458035   0.165996229747985   0.165351072796392   0.164731958432038   0.164137827251777   0.163567667808361
#    0.16302051398627    0.162495442546711   0.161991570839889   0.161508054656702   0.161044086224147   0.160598892322873
#    0.160171732529222   0.159761897553538   0.159368707699624   0.158991511395746   0.158629683829495   0.158282625657227
#    0.157949761794753   0.157630540265568   0.157324431114211   0.157030925425901   0.156749534367975   0.156479788252785
#    0.156221235705433   0.155973442848395   0.155735992542499   0.155508483663952   0.155290530524209   0.155081761717559
#    0.154881820366039   0.154690362915983   0.15450705872868    0.154331589600128   0.154163649214623   0.15400294322381
#    0.153849186686744   0.153702106700076   0.153561440033241   0.153426933191754   0.15329834207026    0.15317543156486
#    0.153057975264037   0.15294575507435    0.152838560993077   0.152736190732401   0.152638449461741   0.15254514955956
#    0.152456110317259   0.152371157707981   0.152290124148554   0.152212848260008   0.152139174646      0.152068953708969
#    0.152002041396251   0.1519382990614     0.151877592209548   0.151819793755061   0.151764779697153   0.151712433918411
#    0.151662636856423   0.151615280658173   0.151570259351998   0.151527471135732   0.151486817966516   0.151448205734315
#    0.15141154388142    0.151376745393865   0.151343726657037   0.151312407356171   0.151282710264026   0.151254561325499
#    0.151227889369575   0.151202626081824   0.151178705901131   0.15115606594429    0.151134645784794   0.151114387618912
#    0.15109523593687    0.151077137500291   0.151060041356315   0.151043898652335   0.151028662601566   0.151014288409176
#    0.151000733145316   0.150987955828429   0.150975917130517   0.150964579536694   0.150953907138302   0.15094386559269
#    0.150934422112605   0.150925545418073   0.150917195613448   0.150909361218836   0.150902023351261   0.150895127870482
#    0.150888662684307   0.150882604183999   0.150876929908097   0.150871618315822   0.150866649133373   0.150862002929896
#    0.150857661192132   0.150853606420607   0.150849821917321   0.150846291855605   0.150843001128793   0.150839935518041
#    0.150837081455787   0.150834426106042   0.150831957270573   0.150829663485762   0.150827533699748   0.150825557715199
#    0.150823725685571   0.150822028413942   0.150820457161213   0.150819003685852   0.150817660215088   0.150816419416301
#    0.150815274378848   0.150814218584756   0.150813245933572   0.150812350618485   0.150811527207033   0.15081077065984
#    0.150810076165585   0.150809439244387   0.150808855673603   0.15080832153222    0.1508078331088     0.150807386950923
#    0.150806979879669   0.150806608780975   0.15080627092031    0.150805963685035   0.150805684531847   0.150805431282469
#    0.150805201811215   0.15080499410351    0.150804806353181   0.150804636866247   0.150804484074552   0.15080434649725
#    0.150804154979196   0.150804111842116   0.150804012339786   0.150803923274135   0.150803843714184   0.150803772711407
#    0.150803709506936   0.150803653304653   0.150803603430394   0.150803559271787   0.150803520203565   0.150803485729675
#    0.150803455426524   0.150803428866062   0.150803405431536   0.150803384927575   0.150803367055139   0.150803351425895
#    0.150803337854585   0.150803326113568   0.150803315940318   0.150803307166868   0.150803299607245   0.150803293122097
#    0.150803287575387   0.150803282846957   0.150803278885278   0.150803275450687   0.150803273104948   0.150803270703264
#    0.15080326874228    0.150803267102413   0.150803265725058   0.150803264596532   0.150803263654955   0.150803262889144
#    0.150803262264875   0.150803261759157   0.150803261356209   0.150803261053938   0.150803260774685   0.150803260617978
#    0.150803260463531   0.150803260348589   0.150803260242614   0.150803260164525   0.150803260104029   0.15080326005755
#    0.150803260038218   0.150803259997255   0.15080325997524    0.150803259958122   0.150803259948674   0.15080325994093
#    0.150803259952193   0.150803259934549   0.150803260005696   0.150803260008905   0.150803259972237   0.150803259972449
#    0.150803259970399   0.15080325996024    0.150803259952614   0.15080325994732    0.150803259951575   0.150803259939388
#    0.150803259944344   0.150803259933418   0.150803259928568   0.150803259937848   0.150803259111825   0.150803258879289
#    0.150803258540961   0.150803258237627]

