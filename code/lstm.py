import tensorflow as tf


class LSTMRunner(object):
    """LSTMRunner class to set up model."""

    def __init__(self, is_train, n, h, p, scope):
        """Initialize lstm hparams."""
        self.n = n  # -- number of layers
        self.h = h  # -- number of hidden units
        self.p = p if is_train else 1.0  # -- dropout
        self.scope = scope  # -- variable scope

    def run(self, x, ts, bidirectional=False, reuse=None):
        """Run model."""
        with tf.variable_scope(self.scope, reuse=reuse):
            for l in range(self.n):
                with tf.variable_scope("l{}".format(l+1), reuse=reuse):
                    fw = tf.nn.rnn_cell.LSTMCell(self.h)
                    if bidirectional:
                        bw = tf.nn.rnn_cell.LSTMCell(self.h)
                        x, state = tf.nn.bidirectional_dynamic_rnn(fw, bw, x, sequence_length=ts, dtype=tf.float32)
                        x = tf.concat(x, 2)
                    else:
                        x, state = tf.nn.dynamic_rnn(fw, x, sequence_length=ts, dtype=tf.float32)

            if bidirectional:
                final_state = tf.concat([direction.h for direction in state], 1)
            else:
                final_state = state.h

            return final_state
