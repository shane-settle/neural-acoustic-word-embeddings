from os import path
import tensorflow as tf
from lstm import LSTMRunner
from triplet_loss import loss


class Model(object):
    """Neural embedding model."""

    def __init__(self, is_train, config, reuse):
        """Initialize model."""
        self.x = tf.placeholder(tf.float32, [None, config.max_length, config.num_features])
        self.ts = tf.placeholder(tf.int32, [None])
        self.same_partition = tf.placeholder(tf.int32, [None])
        self.diff_partition = tf.placeholder(tf.int32, [None])

        self.lstm = LSTMRunner(is_train=is_train, n=config.num_layers,
                               h=config.hidden_size, p=config.keep_prob, scope="lstm")

        self.embeddings = self.lstm.run(self.x, self.ts, bidirectional=config.bidirectional, reuse=reuse)

        self.loss = loss(self.embeddings, self.same_partition, self.diff_partition, margin=config.margin)

        if is_train:
            # optimizer = tf.train.MomentumOptimizer(learning_rate=config.lr, momentum=config.mom, use_nesterov=True)
            optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)
            self.optim = optimizer.minimize(self.loss)
            logdir = path.join(config.logdir, "train" if is_train else "dev")
            self.writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
            tf.summary.scalar("loss", self.loss)
            self.summarize = tf.summary.merge_all()

    def calculate_loss(self, sess, x, ts, same_partition, diff_partition):
        """Calculate loss (for training)."""
        feed_dict = {
            self.x: x,  # -- input data
            self.ts: ts,  # -- input sequence lengths
            self.same_partition: same_partition,  # -- track same indices
            self.diff_partition: diff_partition  # -- track diff indices
        }
        return sess.run([self.optim, self.loss, self.summarize], feed_dict=feed_dict)
