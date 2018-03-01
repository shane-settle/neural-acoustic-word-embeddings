import tensorflow as tf


def cos_sim(x, y):
    x_dot_y = tf.reduce_sum(tf.multiply(x, y), reduction_indices=1)
    x_dot_x = tf.reduce_sum(tf.square(x), reduction_indices=1)
    y_dot_y = tf.reduce_sum(tf.square(y), reduction_indices=1)
    return tf.divide(x_dot_y, tf.multiply(tf.sqrt(x_dot_x), tf.sqrt(y_dot_y)))


def triplet_hinge(anchor, same, diff, margin):
    return tf.maximum(0., margin + cos_sim(anchor, diff) - cos_sim(anchor, same))


def triplet_loss(logits, same_partition, diff_partition, margin=0.3):
    logits = tf.segment_mean(logits, same_partition)
    batch_size = tf.reduce_max(diff_partition) + 1
    anchor, same, diff = tf.split(logits, [batch_size, batch_size, -1])
    anchor = tf.gather(anchor, diff_partition)
    same = tf.gather(same, diff_partition)
    losses = triplet_hinge(anchor, same, diff, margin)
    losses = tf.segment_max(losses, diff_partition)
    return tf.reduce_mean(losses)
