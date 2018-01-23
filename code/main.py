import numpy as np
import tensorflow as tf
from model import Model
from data import Data


class DebugConfig(object):
    """Set up model for debugging."""

    trainfile = "/Users/shane/GitHub/dawe/data/mc3/swbd.train.npz"
    devfile = "/Users/shane/GitHub/dawe/data/mc3/swbd.dev.npz"
    batch_size = 10
    current_epoch = 0
    num_epochs = 10
    max_length = 200
    num_features = 39
    num_layers = 3
    hidden_size = 128
    bidirectional = True
    keep_prob = 0.7
    margin = 0.5
    max_same = 1
    max_diff = 5
    lr = 0.001
    mom = 0.9
    logdir = "/Users/shane/GitHub/dawe/logs/debug"
    ckpt = None


def main():
    config = DebugConfig()
    train_model = Model(is_train=True, config=config, reuse=None)
    train_data = Data(is_train=True, datapath=config.trainfile, batch_size=config.batch_size)

    dev_model = Model(is_train=False, config=config, reuse=True)
    dev_data = Data(is_train=False, datapath=config.devfile, batch_size=config.batch_size)

    batch_size = config.batch_size
    current_epoch = config.current_epoch
    num_epochs = config.num_epochs

    saver = tf.train.Saver()

    proto = tf.ConfigProto(intra_op_parallelism_threads=2)
    with tf.Session(config=proto) as sess:
        if config.ckpt is None:  # -- initialize
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
        else:  # -- or restore
            saver.restore(sess, config.ckpt)

        batch = 0
        for epoch in range(current_epoch, num_epochs):
            print("epoch: ", epoch)
            for index in range(0, train_data.num_examples, batch_size):
                indices = range(index, min(index + batch_size, train_data.num_examples))
                x, ts, same, diff = train_data.get_next(indices, max_same=config.max_same, max_diff=config.max_diff)
                _, loss, train_summary = train_model.calculate_loss(sess, x, ts, same, diff)
                # train_model.writer.add_summary(train_summary, batch)
                print("loss:", loss)
                batch += 1

            # for index in range(0, dev_data.num_examples, batch_size):
            #    indices = range(index, min(index + batch_size, dev_data.num_examples))
            #    x, ts, same_partition, diff_partition = dev_data.get_next(indices)
            #    _, _, eval_summary = dev_model.calculate_loss(sess, x, ts, same_partition, diff_partition)
            #    dev_model.writer.add_summary(eval_summary, batch)

            # saver.save(sess, 'model', global_step=epoch)


if __name__ == "__main__":
    main()
