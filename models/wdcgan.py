import code
import os
import time
import numpy as np
import keras.backend as K
import tensorflow as tf

import matplotlib.pyplot as plt

from keras.layers.convolutional import Convolution2D, Deconvolution2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Reshape, Flatten, Activation
from keras.layers import Input
from keras.models import Model
from keras import initializers

# ----------------------------------------------------------------------------

default_opt = { 'lr' : 5e-5, 'c' : 1e-2, 'n_critic' : 5 }

class WDCGAN(object):
  """Wasserstein Deep Convolutional Generative Adversarial Network"""

  def __init__(self, n_dim, n_chan=1, opt_alg='rmsprop', opt_params=default_opt):
    self.n_critic = opt_params['n_critic']
    self.c        = opt_params['c']
    self.dataset  = opt_params['dataset']
    n_lat = 100

    # create session
    self.sess = tf.Session()
    K.set_session(self.sess) # pass keras the session

    # create generator
    with tf.name_scope('generator'):
      Xk_g = Input(shape=(n_lat,))
      g = make_dcgan_generator(Xk_g, n_lat, self.dataset.width, self.dataset.height, n_chan)

    # create discriminator
    with tf.name_scope('discriminator'):
      Xk_d = Input(shape=(n_chan, n_dim, n_dim))
      d = make_dcgan_discriminator(Xk_d)

    # instantiate networks
    g_net = Model(input=Xk_g, output=g)
    d_net = Model(input=Xk_d, output=d)

    # save inputs
    X_g = tf.placeholder(tf.float32, shape=(None, n_lat), name='X_g')
    X_d = tf.placeholder(tf.float32, shape=(None, n_chan, n_dim, n_dim), name='X_d')
    self.inputs = X_g, X_d

    # get the weights
    self.w_g = [w for w in tf.global_variables() if 'generator' in w.name]
    self.w_d = [w for w in tf.global_variables() if 'discriminator' in w.name]

    # create predictions
    d_real = d_net(X_d)
    d_fake = d_net(g_net(X_g))
    self.P = g_net(X_g)

    # create losses
    self.loss_g = tf.reduce_mean(d_fake)
    self.loss_d = tf.reduce_mean(d_real) - tf.reduce_mean(d_fake)

    # compute and store discriminator probabilities
    self.d_real = tf.reduce_mean(d_real)
    self.d_fake = tf.reduce_mean(d_fake)
    self.p_real = tf.reduce_mean(tf.sigmoid(d_real))
    self.p_fake = tf.reduce_mean(tf.sigmoid(d_fake))

    # Craete the tensorboard summaries
    summary_loss_g = tf.summary.scalar('loss_g', self.loss_g)
    summary_loss_d = tf.summary.scalar('loss_d', self.loss_d)    
    summary_d_real = tf.summary.scalar('d_real', self.d_real)
    summary_d_fake = tf.summary.scalar('d_fake', self.d_fake)
    summary_p_real = tf.summary.scalar('p_real', self.p_real)
    summary_p_fake = tf.summary.scalar('p_fake', self.p_fake)
    self.loss_summary = tf.summary.merge([
      summary_loss_g, summary_loss_d, summary_d_real, summary_d_fake, summary_p_real, summary_p_fake]);
    self.im_summary_image = tf.placeholder(tf.uint8);
    self.im_summary = tf.summary.image('samples',self.im_summary_image)
    
    # create an optimizer
    lr = opt_params['lr']
    optimizer_g = tf.train.RMSPropOptimizer(lr)
    optimizer_d = tf.train.RMSPropOptimizer(lr)

    # get gradients
    gv_g = optimizer_g.compute_gradients(self.loss_g, self.w_g)
    gv_d = optimizer_d.compute_gradients(self.loss_d, self.w_d)

    # create training operation
    self.train_op_g = optimizer_g.apply_gradients(gv_g)
    self.train_op_d = optimizer_d.apply_gradients(gv_d)

    # clip the weights, so that they fall in [-c, c]
    self.clip_updates = [w.assign(tf.clip_by_value(w, -self.c, self.c)) for w in self.w_d]

  def fit(self, X_train, X_val, n_epoch=10, n_batch=128, logdir='dcgan-run'):
    # initialize log directory                  
    if tf.gfile.Exists(logdir): tf.gfile.DeleteRecursively(logdir)
    tf.gfile.MakeDirs(logdir)

    # init model
    init = tf.global_variables_initializer()
    self.sess.run(init)

    # summarization
    summary_writer = tf.summary.FileWriter(logdir, self.sess.graph)
    
    # train the model
    step, g_step, epoch = 0, 0, 0
    while self.dataset.train.epochs_completed < n_epoch:
    
      n_critic = 100 if g_step < 25 or (g_step+1) % 500 == 0 else self.n_critic

      start_time = time.time()
      for i in range(n_critic):
        losses_d = []

        # load the batch
        X_batch = self.dataset.train.next_batch(n_batch)[0]
        self.data_shape = X_batch.shape
        X_batch = X_batch.reshape((self.data_shape[0], self.data_shape[1],
                                   self.data_shape[2], self.data_shape[3]))
        noise = np.random.rand(n_batch,100).astype('float32')
        feed_dict = self.load_batch(X_batch, noise)

        # train the critic/discriminator
        loss_d = self.train_d(feed_dict)
        losses_d.append(loss_d)

      loss_d = np.array(losses_d).mean()

      # train the generator
      noise = np.random.rand(n_batch,100).astype('float32')
      # noise = np.random.uniform(-1.0, 1.0, [n_batch, 100]).astype('float32')
      feed_dict = self.load_batch(X_batch, noise)
      loss_g = self.train_g(feed_dict)
      g_step += 1

      if g_step < 100 or g_step % 100 == 0:
        tot_time = time.time() - start_time
        print 'Epoch: %3d, Gen step: %4d (%3.1f s), Disc loss: %.6f, Gen loss %.6f' % \
          (self.dataset.train.epochs_completed, g_step, tot_time, loss_d, loss_g)
        # store the losses
        print("Tensorboard: Storing loss")
        loss_summary = self.sess.run(self.loss_summary, feed_dict=feed_dict)
        summary_writer.add_summary(loss_summary, g_step)
        
      # take samples
      if g_step % 25 == 0:
        noise = np.random.rand(n_batch,100).astype('float32')
        samples = self.gen(noise)
        samples = samples[:42] # instead, try getting 42 examples from the dataset
        fname = logdir + '.data_samples-%d.png' % g_step
        viz_samples = self.dataset.train.next_batch(42)[0]
        #code.interact(local=locals())
        #image_of_samples = viz_samples
        #image_of_samples = image_of_samples.transpose(1, 0, 2,3)
        #image_of_samples = image_of_samples.reshape(self.data_shape[1], 6, 7, self.data_shape[2], self.data_shape[3])
        #image_of_samples = image_of_samples.reshape(
        #  self.data_shape[1],
        #  6*self.data_shape[2],
        #  7*self.data_shape[3])

        image_of_samples = samples
        #image_of_samples = viz_samples;
        image_of_samples = image_of_samples.reshape(6,7,self.data_shape[1],self.data_shape[2], self.data_shape[3])
        image_of_samples = image_of_samples.transpose(0,3,1,4,2)
        #image_of_samples = viz_samples.reshape(6, 7, self.data_shape[2], self.data_shape[3], self.data_shape[1])
        #image_of_samples = image_of_samples.transpose(2, 0, 1, 3, 4)
        image_of_samples = image_of_samples.reshape(
          6*self.data_shape[2],
          7*self.data_shape[3],
          self.data_shape[1])
        #image_of_samples = image_of_samples.transpose(1,2,0)
        image_of_samples = np.uint8(image_of_samples)
        plt.imsave(fname, image_of_samples)#,cmap='gray')

        # send the visualization to tensorboard
        print("Tensorboard: Saving Image")
        if self.data_shape[1] == 1:
          image_of_samples3d = np.uint8(
            255*np.rollaxis(np.tile(image_of_samples,(3,1,1,1)),0,4))
        else:
          image_of_samples3d = image_of_samples.reshape((1,) + image_of_samples.shape)
        feed_dict[self.im_summary_image] = image_of_samples3d
        summary = self.sess.run(self.im_summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary, g_step)
        #saver.save(self.sess, checkpoint_root, global_step=step)        
        
      # saver.save(self.sess, checkpoint_root, global_step=step)

  def gen(self, noise):
    X_g_in, X_d_in = self.inputs
    feed_dict = { X_g_in : noise, K.learning_phase() : True }
    return self.sess.run(self.P, feed_dict=feed_dict)

  def train_g(self, feed_dict):
    _, loss_g = self.sess.run([self.train_op_g, self.loss_g], feed_dict=feed_dict)
    return loss_g

  def train_d(self, feed_dict):
    # clip the weights, so that they fall in [-c, c]
    self.sess.run(self.clip_updates, feed_dict=feed_dict)

    # take a step of RMSProp
    self.sess.run(self.train_op_d, feed_dict=feed_dict)

    # return discriminator loss
    return self.sess.run(self.loss_d, feed_dict=feed_dict)

  def train(self, feed_dict):
    self.sess.run(self.train_op, feed_dict=feed_dict)

  def load_batch(self, X_train, noise, train=True):
    X_g_in, X_d_in = self.inputs
    return {X_g_in : noise, X_d_in : X_train, K.learning_phase() : train}

  def eval_err(self, X, n_batch=128):
    batch_iterator = iterate_minibatches(X, n_batch, shuffle=True)
    loss_g, loss_d, p_real, p_fake = 0, 0, 0, 0
    tot_loss_g, tot_loss_d, tot_p_real, tot_p_fake = 0, 0, 0, 0
    for bn, batch in enumerate(batch_iterator):
      noise = np.random.rand(n_batch,100)
      feed_dict = self.load_batch(batch, noise)
      loss_g, loss_d, p_real, p_fake \
        = self.sess.run([self.d_real, self.d_fake, self.p_real, self.p_fake], 
                        feed_dict=feed_dict)
      tot_loss_g += loss_g
      tot_loss_d += loss_d
      tot_p_real += p_real
      tot_p_fake += p_fake
    return tot_loss_g / (bn+1), tot_loss_d / (bn+1), \
           tot_p_real / (bn+1), tot_p_fake / (bn+1)

    
# ----------------------------------------------------------------------------

def conv2D_init():
   return initializers.RandomNormal(stddev=0.02);

def make_dcgan_discriminator(Xk_d):
  x = Convolution2D(nb_filter=64, nb_row=4, nb_col=4, subsample=(2,2),
        activation=None, border_mode='same', init=conv2D_init(),
        dim_ordering='th')(Xk_d)
  # x = BatchNormalization(axis=1)(x) # <- makes things much worse!
  x = LeakyReLU(0.2)(x)

  x = Convolution2D(nb_filter=128, nb_row=4, nb_col=4, subsample=(2,2),
        activation=None, border_mode='same', init=conv2D_init(),
        dim_ordering='th')(x)
  x = BatchNormalization(axis=1)(x)
  x = LeakyReLU(0.2)(x)

  x = Flatten()(x)
  x = Dense(1024, init=conv2D_init())(x)
  x = BatchNormalization()(x)
  x = LeakyReLU(0.2)(x)

  d = Dense(1, activation=None)(x)  
  
  return d

def make_dcgan_generator(Xk_g, n_lat, im_width, im_height, n_chan=1):
  n_g_hid1 = 1024 # size of hidden layer in generator layer 1
  n_g_hid2 = 128  # size of hidden layer in generator layer 2

  x = Dense(n_g_hid1, init=conv2D_init())(Xk_g)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  hid_init_sz = 8
  x = Dense(n_g_hid2*hid_init_sz*hid_init_sz, init=conv2D_init())(x)
  x = Reshape((n_g_hid2, hid_init_sz, hid_init_sz))(x)
  x = BatchNormalization(axis=1)(x)
  x = Activation('relu')(x)
  xg = x
  s = xg.shape
  
  l = Deconvolution2D(64, 5, 5,
        border_mode='same', activation=None, subsample=(2,2), 
        init=conv2D_init(), dim_ordering='th')
  s = l.compute_output_shape((s))
  print("dcgan shape = " + str(s))
  x = l(x)
  x = BatchNormalization(axis=1)(x)
  x = Activation('relu')(x)

  l = Deconvolution2D(n_chan, 5, 5, 
                      border_mode='same', activation='sigmoid', subsample=(2,2), 
                      init=conv2D_init(), dim_ordering='th')
  s = l.compute_output_shape((s))
  print("dcgan shape = " + str(s))
  x = l(x)
  
  return x
