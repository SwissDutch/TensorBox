#!/usr/bin/env python
import json
import datetime
import random
import time
import string
import argparse
import os
from scipy import misc
import tensorflow as tf
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

random.seed(0)
np.random.seed(0)

import data_input
import encoder
import optimizer
import objective

from utils import train_utils, googlenet_load


@ops.RegisterGradient("Hungarian")
def _hungarian_grad(op, *args):
    return map(array_ops.zeros_like, op.inputs)


def build(hypes, q):
    '''
    Build full model for training, including forward / backward passes,
    optimizers, and summary statistics.
    '''
    gpu_options = tf.GPUOptions()
    config = tf.ConfigProto(gpu_options=gpu_options)
    encoder_net = googlenet_load.init(hypes, config=config)

    learning_rate = tf.placeholder(tf.float32)

    images, labels, decoded_logits, losses = {}, {}, {}, {}
    for phase in ['train', 'test']:
        # Load images and Labels
        images[phase], labels[phase] = data_input.inputs(hypes, q[phase])

        # Run inference on the encoder network
        logits = encoder.inference(hypes, images[phase], encoder_net)

        # Build decoder on top of the logits
        decoded_logits[phase] = objective.decoder(hypes, logits, phase)

        # Compute losses
        losses[phase] = objective.loss(hypes, decoded_logits[phase],
                                       labels[phase], phase)

    total_loss = losses['train'][0]
    global_step = tf.Variable(0, trainable=False)

    # Build training operation
    train_op = optimizer.training(hypes, total_loss,
                                  global_step, learning_rate)

    # Write Values to summary
    accuracy, smooth_op = objective.evaluation(
        hypes, images, labels, decoded_logits, losses, global_step)

    summary_op = tf.merge_all_summaries()

    return (config, total_loss, accuracy, summary_op, train_op,
            smooth_op, global_step, learning_rate, encoder_net)


def train(H, test_images):
    '''
    Setup computation graph, run 2 prefetch data threads, and then run the main loop
    '''

    if not os.path.exists(H['save_dir']): os.makedirs(H['save_dir'])

    ckpt_file = H['save_dir'] + '/save.ckpt'
    with open(H['save_dir'] + '/hypes.json', 'w') as f:
        json.dump(H, f, indent=4)

    q = {}
    enqueue_op = {}
    for phase in ['train', 'test']:
        q[phase] = data_input.create_queues(H, phase)

    (config, loss, accuracy, summary_op, train_op,
     smooth_op, global_step, learning_rate, encoder_net) = build(H, q)

    saver = tf.train.Saver(max_to_keep=None)
    writer = tf.train.SummaryWriter(
        logdir=H['save_dir'],
        flush_secs=10
    )

    with tf.Session(config=config) as sess:
        tf.train.start_queue_runners(sess=sess)
        for phase in ['train', 'test']:
            # enqueue once manually to avoid thread start delay
            data_input.start_enqueuing_threads(H, q[phase], phase, sess)

        tf.set_random_seed(H['solver']['rnd_seed'])
        sess.run(tf.initialize_all_variables())
        writer.add_graph(sess.graph)
        weights_str = H['solver']['weights']
        if len(weights_str) > 0:
            print('Restoring from: %s' % weights_str)
            saver.restore(sess, weights_str)

        # train model for N iterations
        start = time.time()
        max_iter = H['solver'].get('max_iter', 10000000)
        for i in xrange(max_iter):
            display_iter = H['logging']['display_iter']
            adjusted_lr = (H['solver']['learning_rate'] *
                           0.5 ** max(0, (i / H['solver']['learning_rate_step']) - 2))
            lr_feed = {learning_rate: adjusted_lr}

            if i % display_iter != 0:
                # train network
                batch_loss_train, _ = sess.run([loss, train_op], feed_dict=lr_feed)
            else:
                # test network every N iterations; log additional info
                if i > 0:
                    dt = (time.time() - start) / (H['batch_size'] * display_iter)
                start = time.time()
                (train_loss, test_accuracy, summary_str,
                    _, _) = sess.run([loss, accuracy['test'],
                                      summary_op, train_op, smooth_op,
                                     ], feed_dict=lr_feed)
                writer.add_summary(summary_str, global_step=global_step.eval())
                print_str = string.join([
                    'Step: %d',
                    'lr: %f',
                    'Train Loss: %.2f',
                    'Test Accuracy: %.1f%%',
                    'Time/image (ms): %.1f'
                ], ', ')
                print(print_str %
                      (i, adjusted_lr, train_loss,
                       test_accuracy * 100, dt * 1000 if i > 0 else 0))

            if global_step.eval() % H['logging']['save_iter'] == 0 or global_step.eval() == max_iter - 1:
                saver.save(sess, ckpt_file, global_step=global_step)


def main():
    '''
    Parse command line arguments and return the hyperparameter dictionary H.
    H first loads the --hypes hypes.json file and is further updated with
    additional arguments as needed.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=None, type=str)
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--hypes', required=True, type=str)
    parser.add_argument('--logdir', default='output', type=str)
    args = parser.parse_args()
    with open(args.hypes, 'r') as f:
        H = json.load(f)
    if args.gpu is not None:
        H['solver']['gpu'] = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(H['solver']['gpu'])
    if len(H.get('exp_name', '')) == 0:
        H['exp_name'] = args.hypes.split('/')[-1].replace('.json', '')
    H['save_dir'] = args.logdir + '/%s_%s' % (H['exp_name'],
        datetime.datetime.now().strftime('%Y_%m_%d_%H.%M'))
    if args.weights is not None:
        H['solver']['weights'] = args.weights
    train(H, test_images=[])

if __name__ == '__main__':
    main()
