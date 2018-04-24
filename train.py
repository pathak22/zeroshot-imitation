import sys
import os
PD = os.getcwd() + '/caffe/python/'
if PD not in sys.path:
    sys.path.append(PD)
from data import rope_data
import numpy as np
import subprocess
import collections
import copy
import tensorflow as tf
import time
import matplotlib.pyplot as plt

slim = tf.contrib.slim
from nets import alexnet_geurzhoy

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import embedding_ops

CONFIG = tf.ConfigProto()
CONFIG.gpu_options.allow_growth = True

GRAD_CLIP_NORM = 40

# from original poke paper
FEAT_SIZE = 400
BATCH_SIZE = 64
ENCODING_SIZE = 200 # latent feature space representation of image
FEATURE_SIZE = 2 * ENCODING_SIZE
LOCATION_BINS = 400 # number of possible grasp locations
LOCATION_EMBEDDING_SIZE = 50 # discrete to continuous representation
THETA_BINS = 36 # discretization of angle bins
THETA_EMBEDDING_SIZE = 36 # discrete to cintinusous representation
LENGTH_BINS = 10 # 1-10 cm movement

def init_weights(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.random_normal_initializer(0, 0.01))

def make_network(x, network_size):
    """Makes fully connected network with input x and given layer sizes.
    Assume len(network_size) >= 2
    """
    input_size = network_size[0]
    output_size = network_size.pop()
    a = input_size
    cur = x
    i = 0
    for a, b in zip(network_size, network_size[1:]):
        W = init_weights("W" + str(i), [a, b])
        B = init_weights("B" + str(i), [1, b])
        cur = tf.nn.elu(tf.matmul(cur, W) + B)
        i += 1
    W = init_weights("W" + str(i), [b, output_size])
    B = init_weights("B" + str(i), [1, output_size])
    prediction = tf.matmul(cur, W) + B
    return prediction

def leaky_relu(x, alpha):
    return tf.maximum(x, alpha * x)

class RopeImitator():

    def __init__(self, name, unfreeze_time=30000, autoencode=False,
        action_lr=1e-4, deconv_lr=1e-3, fwd_consist=False, baseline_reg=False, softmaxBackprop=True,
        gtAction=False):
        self.unfreeze_time = unfreeze_time
        self.autoencode = autoencode
        self.gtAction = gtAction
        self.name = '{0}_{1}_{2}_{3}_{4}_{5}K_{6}_{7}'.format(name, 'fwdconsist' + str(fwd_consist), 'baselinereg' + str(baseline_reg), 
            'deconv_lr' + str(deconv_lr), 'autoencode' + str(autoencode),
            'unfreeze' + str(int(unfreeze_time/1000.)), 'softmax' + str(softmaxBackprop),
            'gtAction' + str(gtAction))
        self.fwd_consist = fwd_consist
        self.start = 0

        self.batch_loader = rope_data

        self.image_ph = tf.placeholder(tf.float32, [None, 200, 200, 3], name='image_ph')
        self.goal_image_ph = tf.placeholder(tf.float32, [None, 200, 200, 3], name='goal_image_ph')
        self.location_ph = tf.placeholder(tf.float32, [None, LOCATION_BINS], name='location_ph')
        self.theta_ph = tf.placeholder(tf.float32, [None, THETA_BINS], name='theta_ph')
        self.length_ph = tf.placeholder(tf.float32, [None, LENGTH_BINS], name='length_ph')
        self.ignore_flag_ph = tf.placeholder(tf.float32, [None], name='ignore_flag_ph')
        self.is_training_ph = tf.placeholder(tf.bool, name='is_training_ph')
        self.autoencode_ph = tf.placeholder(tf.bool)
        self.gtAction_ph = tf.placeholder(tf.bool)

        # get latent representations for both the images
        latent_image, latent_conv5_image = alexnet_geurzhoy.network(self.image_ph, trainable=True, num_outputs=ENCODING_SIZE)
        latent_goal_image, latent_conv5_goal_image = alexnet_geurzhoy.network(self.goal_image_ph, trainable=True, num_outputs=ENCODING_SIZE, reuse=True)

        # concatenate the latent representations and share information
        features = tf.concat(1, [latent_image, latent_goal_image])

        with tf.variable_scope("concat_fc"):
            x = tf.nn.relu(features)
            x = slim.fully_connected(x, FEAT_SIZE, scope="concat_fc")

        #################################
        # ACTION PREDICTION
        #################################
        location_embedding = init_weights('location_embedding', [LOCATION_BINS, LOCATION_EMBEDDING_SIZE])
        theta_embedding = init_weights('theta_embedding', [THETA_BINS, THETA_EMBEDDING_SIZE])

        # layer for predicting X, Y
        with tf.variable_scope('location_pred'):
            loc_network_layers = [FEATURE_SIZE, 200, 200, LOCATION_BINS]
            location_pred = make_network(x, loc_network_layers)
            location_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(location_pred, self.location_ph))

            location_sample = math_ops.argmax(tf.cond(self.is_training_ph, lambda: self.location_ph, lambda: location_pred), 1)
            location_embed = embedding_ops.embedding_lookup(location_embedding, location_sample)

        # layer for predicting theta
        with tf.variable_scope('theta_pred'):
            x_with_loc = tf.concat(1, [x, location_embed])
            theta_network_layers = [FEATURE_SIZE + LOCATION_EMBEDDING_SIZE, 200, 200, THETA_BINS]
            theta_pred = make_network(x_with_loc, theta_network_layers)
            theta_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(theta_pred, self.theta_ph))

            theta_sample = math_ops.argmax(tf.cond(self.is_training_ph, lambda: self.theta_ph, lambda: theta_pred), 1)
            theta_embed = embedding_ops.embedding_lookup(theta_embedding, theta_sample)

        # layer for predicting length of movement
        with tf.variable_scope('length_pred'):
            x_with_loc_theta = tf.concat(1, [x_with_loc, theta_embed])
            length_network_layers = [FEATURE_SIZE + LOCATION_EMBEDDING_SIZE + THETA_EMBEDDING_SIZE, 200, 200, LENGTH_BINS]
            length_pred = make_network(x_with_loc_theta, length_network_layers)
            length_softmax = tf.nn.softmax_cross_entropy_with_logits(length_pred, self.length_ph)
            length_loss = tf.reduce_mean(length_softmax * self.ignore_flag_ph)

        # add to collections for retrieval
        tf.add_to_collection('location_logit', location_pred)
        tf.add_to_collection('theta_logit', theta_pred)
        tf.add_to_collection('len_logit', length_pred)

        # variables of only inverse model without features
        inv_vars_no_alex = [v for v in tf.trainable_variables() if 'alexnet' not in v.name]
        print('Action prediction tensors consist {0} out of {1}'.format(len(inv_vars_no_alex), len(tf.trainable_variables())))


        total_loss = location_loss + theta_loss + length_loss

        action_optimizer = tf.train.AdamOptimizer(action_lr)

        action_grads, _ = zip(*action_optimizer.compute_gradients(total_loss, inv_vars_no_alex))
        action_grads, _ = tf.clip_by_global_norm(action_grads, GRAD_CLIP_NORM)
        action_grads = zip(action_grads, inv_vars_no_alex)

        action_grads_full, _ = zip(*action_optimizer.compute_gradients(total_loss, tf.trainable_variables()))
        action_grads_full, _ = tf.clip_by_global_norm(action_grads_full, GRAD_CLIP_NORM)
        action_grads_full = zip(action_grads_full, tf.trainable_variables())

        #################################
        # FORWARD CONSISTENCY
        #################################
        if self.fwd_consist:
            with tf.variable_scope('fwd_consist'):
                if softmaxBackprop:
                    location_pred = tf.nn.softmax(location_pred)
                    theta_pred = tf.nn.softmax(theta_pred)
                    length_pred = tf.nn.softmax(length_pred)

                # baseline regularization => gradients flow only to alexnet, not action pred
                if baseline_reg:
                    print('baseline')
                    action_embed = tf.concat(1, [self.location_ph, self.theta_ph, self.length_ph])
                else:
                    # fwd_consist => gradients flow through action prediction
                    latent_conv5_image = tf.stop_gradient(latent_conv5_image)
                    action_embed = tf.cond(self.gtAction_ph,
                        lambda: tf.concat(1, [self.location_ph, self.theta_ph, self.length_ph]),
                        lambda: tf.concat(1, [location_pred, theta_pred, length_pred]))

                action_embed = slim.fully_connected(action_embed, 363)
                action_embed = tf.reshape(action_embed, [-1, 11, 11, 3])
                # concat along depth
                fwd_features = tf.concat(3, [latent_conv5_image, action_embed])
                # deconvolution
                batch_size = tf.shape(fwd_features)[0]

                wt1 = tf.Variable(tf.truncated_normal([5, 5, 64, 259], stddev=0.1))
                deconv1 = tf.nn.conv2d_transpose(fwd_features, wt1, [batch_size, 22, 22, 64], [1, 2, 2, 1])
                deconv1 = leaky_relu(deconv1, 0.2)
                wt2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
                deconv2 = tf.nn.conv2d_transpose(deconv1, wt2, [batch_size, 44, 44, 32], [1, 2, 2, 1])
                deconv2 = leaky_relu(deconv2, 0.2)
                wt3 = tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=0.1))
                deconv3 = tf.nn.conv2d_transpose(deconv2, wt3, [batch_size, 88, 88, 3], [1, 2, 2, 1])
                deconv3 = tf.nn.tanh(deconv3)
                # loss from upsampled deconvolution and goal image
                upsampled_deconv_img = tf.image.resize_images(deconv3, [200, 200])
                tf.add_to_collection('upsampled_deconv_img', upsampled_deconv_img)

                # image inputs are -255 to 255 ??? for some reason
                # whether to autoencode or not

                normalized_goal_img = tf.cond(self.autoencode_ph, lambda: self.image_ph / 255.0, lambda: self.goal_image_ph / 255.0)

                # just to visualize
                deconv_log_img = (upsampled_deconv_img + 1.0) * 127.5

                # variables of only forward model
                fwd_vars = [v for v in tf.trainable_variables() if 'fwd_consist' in v.name]
                print('Forward consistency tensors consist {0} out of {1}'.format(len(fwd_vars), len(tf.trainable_variables())))

                fwd_consist_loss = tf.reduce_mean(tf.abs(upsampled_deconv_img - normalized_goal_img))
                deconv_optimizer = tf.train.AdamOptimizer(deconv_lr)

                fwd_consist_grads, _ = zip(*deconv_optimizer.compute_gradients(fwd_consist_loss, fwd_vars))
                fwd_consist_grads, _ = tf.clip_by_global_norm(fwd_consist_grads, GRAD_CLIP_NORM)
                fwd_consist_grads = zip(fwd_consist_grads, fwd_vars)

                fwd_consist_grads_full, _ = zip(*deconv_optimizer.compute_gradients(fwd_consist_loss, tf.trainable_variables()))
                fwd_consist_grads_full, _ = tf.clip_by_global_norm(fwd_consist_grads_full, GRAD_CLIP_NORM)
                fwd_consist_grads_full = zip(fwd_consist_grads_full, tf.trainable_variables())

                self.optimize_fwd_freeze = deconv_optimizer.apply_gradients(fwd_consist_grads)

                with tf.control_dependencies([fwd_consist_grads_full[0][0][0], action_grads_full[0][0][0]]):
                    self.optimize_fwd_full = deconv_optimizer.apply_gradients(fwd_consist_grads_full)
                    self.optimize_action_full = action_optimizer.apply_gradients(action_grads_full)

        self.optimize_action_no_alex = action_optimizer.apply_gradients(action_grads)
        self.optimize_action_alex = action_optimizer.apply_gradients(action_grads_full)

        #################################
        # LOGGING AND SAVING OPERATIONS
        #################################
        loc_correct_pred = tf.equal(tf.argmax(location_pred, 1), tf.argmax(self.location_ph, 1))
        self.loc_accuracy = tf.reduce_mean(tf.cast(loc_correct_pred, tf.float32))

        theta_correct_pred = tf.equal(tf.argmax(theta_pred, 1), tf.argmax(self.theta_ph, 1))
        self.theta_accuracy = tf.reduce_mean(tf.cast(theta_correct_pred, tf.float32))

        length_correct_pred = tf.equal(tf.argmax(length_pred, 1), tf.argmax(self.length_ph, 1))
        self.length_accuracy = tf.reduce_mean(tf.cast(length_correct_pred, tf.float32))

        # logging
        tf.summary.scalar('model/location_loss', location_loss, collections=['train'])
        tf.summary.scalar('model/theta_loss', theta_loss, collections=['train'])
        tf.summary.scalar('model/length_loss', length_loss, collections=['train'])
        if self.fwd_consist:
            tf.summary.scalar('model/fwd_consist_loss', fwd_consist_loss, collections=['train'])
            tf.summary.image('upsampled_deconv_image', deconv_log_img, max_outputs=5, collections=['train'])

        tf.summary.image('before', (self.image_ph + 255.0) / 2.0, max_outputs=5, collections=['train'])
        tf.summary.image('after', (self.goal_image_ph + 255.0) / 2.0, max_outputs=5, collections=['train'])

        self.train_summaries = tf.summary.merge_all('train')

        self.writer = tf.summary.FileWriter('./results/{0}/logs/{1}'.format(self.name, time.time()))

        self.saver = tf.train.Saver(max_to_keep=None)

        self.sess = tf.Session(config=CONFIG)
        self.sess.run(tf.global_variables_initializer())

        self.model_directory = './results/{0}/models/'.format(self.name)
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)

    def get_batch(self, batch_size, is_training):
        dataset = 'train' if is_training else 'val'

        image, goal_image, location, theta, length, d, c, ignore_flag = self.batch_loader.get_batch(dataset, batch_size)
        print(dataset, location.shape, theta.shape, length.shape)

        feed_dict = {
            self.image_ph: image,
            self.goal_image_ph: goal_image,
            self.location_ph: location,
            self.theta_ph: theta,
            self.length_ph: length,
            self.ignore_flag_ph: ignore_flag,
            self.is_training_ph: is_training,
            self.autoencode_ph: False,
            self.gtAction_ph: False
        }

        return feed_dict

    def train(self, iterations):
        for i in range(self.start, iterations):
            print(i)
            feed_dict = self.get_batch(BATCH_SIZE, True)

            ops_to_run = []
            if i < self.unfreeze_time:
                ops_to_run.append(self.optimize_action_no_alex)
                if self.fwd_consist:
                    ops_to_run.append(self.optimize_fwd_freeze)
                    if self.autoencode and i < self.unfreeze_time * (2/3):
                        feed_dict[self.autoencode_ph] = True
                if self.gtAction:
                    feed_dict[self.gtAction_ph] = True
            else:
                if self.fwd_consist:
                    ops_to_run.append(self.optimize_fwd_full)
                    ops_to_run.append(self.optimize_action_full)
                else:
                    ops_to_run.append(self.optimize_action_alex)


            ops_to_run.append(self.train_summaries)
            op_results = self.sess.run(ops_to_run, feed_dict=feed_dict)
            train_summaries = op_results[-1]

            if i % 100 == 0:
                self.writer.add_summary(train_summaries, i)

            # validate on 1000 images
            # split into batches of 100 because of memory issues
            if i % 1000 == 0:
                self.saver.save(self.sess, self.model_directory + 'inverse', global_step=i)
                print('Saved at timestep {0}'.format(i))

                cum_loc_acc, cum_theta_acc, cum_len_acc = 0, 0, 0

                for _ in range(10):
                    val_dict = self.get_batch(100, False)
                    loc_acc, theta_acc, len_acc = self.sess.run([self.loc_accuracy, self.theta_accuracy, self.length_accuracy], feed_dict=val_dict)

                    cum_loc_acc += loc_acc
                    cum_theta_acc += theta_acc
                    cum_len_acc += len_acc

                cum_loc_acc, cum_theta_acc, cum_len_acc = cum_loc_acc / 10.0, cum_theta_acc / 10.0, cum_len_acc / 10.0

                summaries = tf.Summary(value=[tf.Summary.Value(tag='val/loc_acc', simple_value=cum_loc_acc), tf.Summary.Value(tag='val/theta_acc', simple_value=cum_theta_acc), tf.Summary.Value(tag='val/len_acc', simple_value=cum_len_acc)])
                self.writer.add_summary(summaries, i)

            self.writer.flush()

    def restore(self, iteration, model_name=None):
        if model_name == None:
            model_name = self.name
        self.start = iteration

        saved_model_directory = './results/{0}/models/'.format(model_name)
        self.saver.restore(self.sess, saved_model_directory + 'inverse-{0}'.format(iteration))
        print('Loaded model {0} at iteration {1}'.format(model_name, iteration))

    # print statistics of data
    # use to check to see if you've downloaded the correct dataset
    def stats(self):
        # validation data
        v_loc, v_theta, v_len, t_loc, t_theta, t_len = [], [], [], [], [], []
        for i in range(3):
            val_dict = self.get_batch(1000, False)
            v_loc.append(np.argmax(val_dict[self.location_ph], axis=1))
            v_theta.append(np.argmax(val_dict[self.theta_ph], axis=1))
            v_len.append(np.argmax(val_dict[self.length_ph], axis=1))

        for i in range(10):
            train_dict = self.get_batch(1000, True)
            t_loc.append(np.argmax(train_dict[self.location_ph], axis=1))
            t_theta.append(np.argmax(train_dict[self.theta_ph], axis=1))
            t_len.append(np.argmax(train_dict[self.length_ph], axis=1))

        fig, axes = plt.subplots(2, 3)
        axes[0, 0].set_title('val_locs')
        axes[0, 0].hist(np.concatenate(v_loc))
        axes[0, 1].set_title('val_theta')
        axes[0, 1].hist(np.concatenate(v_theta))
        axes[0, 2].set_title('val_lens')
        axes[0, 2].hist(np.concatenate(v_len))
        axes[1, 0].set_title('train_locs')
        axes[1, 0].hist(np.concatenate(t_loc))
        axes[1, 1].set_title('train_theta')
        axes[1, 1].hist(np.concatenate(t_theta))
        axes[1, 2].set_title('train_lens')
        axes[1, 2].hist(np.concatenate(t_len))
        plt.show()
