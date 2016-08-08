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
try:
    from tensorflow.models.rnn import rnn_cell
except ImportError:
    rnn_cell = tf.nn.rnn_cell
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


def build_lstm_inner(H, lstm_input):
    '''
    build lstm decoder
    '''
    lstm_cell = rnn_cell.BasicLSTMCell(H['lstm_size'], forget_bias=0.0)
    if H['num_lstm_layers'] > 1:
        lstm = rnn_cell.MultiRNNCell([lstm_cell] * H['num_lstm_layers'])
    else:
        lstm = lstm_cell

    batch_size = H['batch_size'] * H['grid_height'] * H['grid_width']
    state = tf.zeros([batch_size, lstm.state_size])

    outputs = []
    with tf.variable_scope('RNN', initializer=tf.random_uniform_initializer(-0.1, 0.1)):
        for time_step in range(H['rnn_len']):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            output, state = lstm(lstm_input, state)
            outputs.append(output)
    return outputs

def build_overfeat_inner(H, lstm_input):
    '''
    build simple overfeat decoder
    '''
    if H['rnn_len'] > 1:
        raise ValueError('rnn_len > 1 only supported with use_lstm == True')
    outputs = []
    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    with tf.variable_scope('Overfeat', initializer=initializer):
        w = tf.get_variable('ip', shape=[1024, H['lstm_size']])
        outputs.append(tf.matmul(lstm_input, w))
    return outputs

def deconv(x, output_shape, channels):
    k_h = 2
    k_w = 2
    w = tf.get_variable('w_deconv', initializer=tf.random_normal_initializer(stddev=0.01),
                        shape=[k_h, k_w, channels[1], channels[0]])
    y = tf.nn.conv2d_transpose(x, w, output_shape, strides=[1, k_h, k_w, 1], padding='VALID')
    return y

def rezoom(H, pred_boxes, early_feat, early_feat_channels, w_offsets, h_offsets):
    '''
    Rezoom into a feature map at multiple interpolation points in a grid. 

    If the predicted object center is at X, len(w_offsets) == 3, and len(h_offsets) == 5,
    the rezoom grid will look as follows:

    [o o o]
    [o o o]
    [o X o]
    [o o o]
    [o o o]

    Where each letter indexes into the feature map with bilinear interpolation
    '''


    grid_size = H['grid_width'] * H['grid_height']
    outer_size = grid_size * H['batch_size']
    indices = []
    for w_offset in w_offsets:
        for h_offset in h_offsets:
            indices.append(train_utils.bilinear_select(H,
                                                       pred_boxes,
                                                       early_feat,
                                                       early_feat_channels,
                                                       w_offset, h_offset))

    interp_indices = tf.concat(0, indices)
    rezoom_features = train_utils.interp(early_feat,
                                         interp_indices,
                                         early_feat_channels)
    rezoom_features_r = tf.reshape(rezoom_features,
                                   [len(w_offsets) * len(h_offsets),
                                    outer_size,
                                    H['rnn_len'],
                                    early_feat_channels])
    rezoom_features_t = tf.transpose(rezoom_features_r, [1, 2, 0, 3])
    return tf.reshape(rezoom_features_t,
                      [outer_size,
                       H['rnn_len'],
                       len(w_offsets) * len(h_offsets) * early_feat_channels])

def build_forward(H, x, googlenet, phase, reuse):
    '''
    Construct the forward model
    '''

    grid_size = H['grid_width'] * H['grid_height']
    outer_size = grid_size * H['batch_size']

    logits = encoder.inference(H, x, googlenet)
    
    return objective.decoder(H, logits, phase)

def build_forward_backward(H, x, googlenet, phase, boxes, flags):
    '''
    Call build_forward() and then setup the loss functions
    '''

    grid_size = H['grid_width'] * H['grid_height']
    outer_size = grid_size * H['batch_size']
    reuse = {'train': None, 'test': True}[phase]

    decoded_logits = build_forward(H, x, googlenet, phase, reuse)
    labels = flags, None, boxes

    pred_boxes, pred_confidences, loss, confidences_loss, boxes_loss \
     = objective.loss(H, decoded_logits, labels, phase)


    return pred_boxes, pred_confidences, loss, confidences_loss, boxes_loss

def build(H, q):
    '''
    Build full model for training, including forward / backward passes,
    optimizers, and summary statistics.
    '''
    arch = H
    solver = H["solver"]

    os.environ['CUDA_VISIBLE_DEVICES'] = str(solver['gpu'])

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    gpu_options = tf.GPUOptions()
    config = tf.ConfigProto(gpu_options=gpu_options)

    encoder_net = googlenet_load.init(H, config)

    learning_rate = tf.placeholder(tf.float32)

    loss, accuracy, confidences_loss, boxes_loss = {}, {}, {}, {}
    for phase in ['train', 'test']:
        # generate predictions and losses from forward pass
        x, confidences, boxes = q[phase].dequeue_many(arch['batch_size'])
        flags = tf.argmax(confidences, 3)


        grid_size = H['grid_width'] * H['grid_height']

        (pred_boxes, pred_confidences,
         loss[phase], confidences_loss[phase],
         boxes_loss[phase]) = build_forward_backward(H, x, encoder_net, phase, boxes, flags)
        pred_confidences_r = tf.reshape(pred_confidences, [H['batch_size'], grid_size, H['rnn_len'], arch['num_classes']])
        pred_boxes_r = tf.reshape(pred_boxes, [H['batch_size'], grid_size, H['rnn_len'], 4])


        # Set up summary operations for tensorboard
        a = tf.equal(tf.argmax(confidences[:, :, 0, :], 2), tf.argmax(pred_confidences_r[:, :, 0, :], 2))
        accuracy[phase] = tf.reduce_mean(tf.cast(a, 'float32'), name=phase+'/accuracy')

        if phase == 'train':
            global_step = tf.Variable(0, trainable=False)

            train_op = optimizer.training(H, loss['train'],
                                          global_step, learning_rate)

        elif phase == 'test':
            moving_avg = tf.train.ExponentialMovingAverage(0.95)
            smooth_op = moving_avg.apply([accuracy['train'], accuracy['test'],
                                          confidences_loss['train'], boxes_loss['train'],
                                          confidences_loss['test'], boxes_loss['test'],
                                          ])

            for p in ['train', 'test']:
                tf.scalar_summary('%s/accuracy' % p, accuracy[p])
                tf.scalar_summary('%s/accuracy/smooth' % p, moving_avg.average(accuracy[p]))
                tf.scalar_summary("%s/confidences_loss" % p, confidences_loss[p])
                tf.scalar_summary("%s/confidences_loss/smooth" % p,
                    moving_avg.average(confidences_loss[p]))
                tf.scalar_summary("%s/regression_loss" % p, boxes_loss[p])
                tf.scalar_summary("%s/regression_loss/smooth" % p,
                    moving_avg.average(boxes_loss[p]))

        if phase == 'test':
            test_image = x
            # show ground truth to verify labels are correct
            test_true_confidences = confidences[0, :, :, :]
            test_true_boxes = boxes[0, :, :, :]

            # show predictions to visualize training progress
            test_pred_confidences = pred_confidences_r[0, :, :, :]
            test_pred_boxes = pred_boxes_r[0, :, :, :]

            def log_image(np_img, np_confidences, np_boxes, np_global_step, pred_or_true):
                
                merged = train_utils.add_rectangles(H, np_img, np_confidences, np_boxes,
                                                    use_stitching=True,
                                                    rnn_len=H['rnn_len'])[0]
                
                num_images = 10
                img_path = os.path.join(H['save_dir'], '%s_%s.jpg' % ((np_global_step / H['logging']['display_iter']) % num_images, pred_or_true))
                misc.imsave(img_path, merged)
                return merged

            pred_log_img = tf.py_func(log_image,
                                      [test_image, test_pred_confidences, test_pred_boxes, global_step, 'pred'],
                                      [tf.float32])
            true_log_img = tf.py_func(log_image,
                                      [test_image, test_true_confidences, test_true_boxes, global_step, 'true'],
                                      [tf.float32])
            tf.image_summary(phase + '/pred_boxes', pred_log_img)
            tf.image_summary(phase + '/true_boxes', true_log_img)

    summary_op = tf.merge_all_summaries()

    return (config, loss, accuracy, summary_op, train_op,
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
                batch_loss_train, _ = sess.run([loss['train'], train_op], feed_dict=lr_feed)
            else:
                # test network every N iterations; log additional info
                if i > 0:
                    dt = (time.time() - start) / (H['batch_size'] * display_iter)
                start = time.time()
                (train_loss, test_accuracy, summary_str,
                    _, _) = sess.run([loss['train'], accuracy['test'],
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
