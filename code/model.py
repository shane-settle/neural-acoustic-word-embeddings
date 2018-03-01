from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from triplet_loss import triplet_loss


class LSTM(object):
    """LSTM class to set up model."""

    def __init__(self, is_train, n, h, p, scope):
        """Initialize lstm hparams."""

        self.n = n  # -- number of layers
        self.h = h  # -- number of hidden units
        self.p = p if is_train else 0.0  # -- keep
        self.scope = scope  # -- variable scope

    def run(self, x, ts, bidirectional=False, reuse=None):
        """Run model."""

        with tf.variable_scope(self.scope, reuse=reuse):
            for l in range(self.n):
                with tf.variable_scope("l{}".format(l + 1), reuse=reuse):
                    fw = tf.nn.rnn_cell.LSTMCell(self.h)
                    if l < self.n - 1 and self.p > 0.0:
                        fw = tf.nn.rnn_cell.DropoutWrapper(fw, output_keep_prob=self.p)

                    if bidirectional:
                        bw = tf.nn.rnn_cell.LSTMCell(self.h)
                        if l < self.n - 1 and self.p > 0.0:
                            bw = tf.nn.rnn_cell.DropoutWrapper(bw, output_keep_prob=self.p)

                        x, state = tf.nn.bidirectional_dynamic_rnn(fw, bw, x, sequence_length=ts, dtype=tf.float32)
                        x = tf.concat(x, 2)
                    else:
                        x, state = tf.nn.dynamic_rnn(fw, x, sequence_length=ts, dtype=tf.float32)

            return tf.concat([direction.h for direction in state], 1) if bidirectional else state.h


class Model(object):
    """Neural embedding model."""

    def __init__(self, is_train, config, reuse):
        """Initialize model."""

        self.x = tf.placeholder(tf.float32, [None, None, config.feature_dim])
        self.ts = tf.placeholder(tf.int32, [None])
        self.same_partition = tf.placeholder(tf.int32, [None])
        self.diff_partition = tf.placeholder(tf.int32, [None])

        self.lstm = LSTM(is_train=is_train, n=config.num_layers,
                         h=config.hidden_size, p=config.keep_prob, scope="lstm")

        self.embeddings = self.lstm.run(self.x, self.ts, bidirectional=config.bidirectional, reuse=reuse)

        if is_train:
            self.loss = triplet_loss(self.embeddings, self.same_partition, self.diff_partition, margin=config.margin)
            optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)
            self.optim = optimizer.minimize(self.loss)

    def get_loss(self, sess, x, ts, same_partition, diff_partition):
        """Calculate loss (for training)."""

        return sess.run([self.optim, self.loss], feed_dict={
            self.x: x,  # input
            self.ts: ts,  # input lengths
            self.same_partition: same_partition,  # -- track same indices
            self.diff_partition: diff_partition  # -- track diff indices
        })

    def get_embeddings(self, sess, x, ts):
        """Calculate average precision (for evaluation)."""

        return sess.run(self.embeddings, feed_dict={
            self.x: x,  # input
            self.ts: ts  # input lengths
        })
