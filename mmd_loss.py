import tensorflow as tf
from functools import partial


def compute_pairwise_distances(x, y):
  norm = lambda x: tf.reduce_sum(tf.square(x), 1)
  return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def gaussian_kernel_matrix(x, y, sigmas):
  beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
  dist = compute_pairwise_distances(x, y)
  s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
  return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))


def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
  cost = tf.reduce_mean(kernel(x, x))
  cost += tf.reduce_mean(kernel(y, y))
  cost -= 2 * tf.reduce_mean(kernel(x, y))
  # We do not allow the loss to become negative.
  cost = tf.where(cost > 0, cost, 0, name='value')
  return cost


def mmd_loss(source_samples, target_samples):
  sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
  gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=tf.constant(sigmas))
  loss_value = maximum_mean_discrepancy(source_samples, target_samples, kernel=gaussian_kernel)
  loss_value = tf.maximum(1e-4, loss_value)
  return loss_value