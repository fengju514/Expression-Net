import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages



class Pose_Estimation(object):
  

  def __init__(self, images, labels, mode, ifdropout, keep_rate_fc6, keep_rate_fc7, lr_rate_fac, net_data, batch_size, mean_labels, std_labels):
    
    self.batch_size = batch_size
    self._images = images
    self.labels = labels
    
    self.mode = mode
    self.ifdropout = ifdropout
    self.keep_rate_fc6 = keep_rate_fc6
    self.keep_rate_fc7 = keep_rate_fc7
    self.ifadd_weight_decay = 0 #ifadd_weight_decay
    self.net_data = net_data
    self.lr_rate_fac = lr_rate_fac
    self._extra_train_ops = []
    self.optimizer = 'Adam'
    self.mean_labels = mean_labels
    self.std_labels = std_labels
    #self.train_mean_vec = train_mean_vec

  def _build_graph(self):
    """Build a whole graph for the model."""
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self._build_model()
    
    if self.mode == 'train':
      self._build_train_op()
    
    #self.summaries = tf.merge_all_summaries()

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self):
    """Build the core model within the graph."""
   
    with tf.variable_scope('Spatial_Transformer'):
      x = self._images
      x = tf.image.resize_bilinear(x, tf.constant([227,227], dtype=tf.int32)) # the image should be 227 x 227 x 3
      
      self.resized_img = x
      theta = self._ST('ST2', x, 3, (16,16), 3, 16, self._stride_arr(1))
      #print "*** ", x.get_shape()
   

  
    with tf.variable_scope('costs'):
      self.predictions = theta
      self.preds_unNormalized = theta * (self.std_labels + 0.000000000000000001) + self.mean_labels
      pred_dim1 = theta.get_shape()[0]
      pred_dim2 = theta.get_shape()[1]

      del theta

      pow_res = tf.pow(self.predictions-self.labels, 2)
      xent = tf.reduce_sum(pow_res,1)
      self.cost = tf.reduce_mean(xent, name='xent')
    
      if self.ifadd_weight_decay == 1:
        self.cost += self._decay()
      

  



  def conv(self, input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])





  def _ST(self, name, x, channel_x, out_size, filter_size, out_filters, strides):
    """ Spatial Transformer. """

    with tf.variable_scope(name):

      # zero-mean input [B,G,R]: [93.5940, 104.7624, 129.1863] --> provided by vgg-face
      """
      with tf.name_scope('preprocess') as scope:
        mean = tf.constant(tf.reshape(self.train_mean_vec*255.0, [3]), dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        x = x - mean
      """

      # conv1
      with tf.name_scope('conv1') as scope:
        #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
        k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
        conv1W = tf.Variable(self.net_data["conv1"]["weights"], trainable=False, name='W')
        conv1b = tf.Variable(self.net_data["conv1"]["biases"], trainable=False, name='baises')
        conv1_in = self.conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        conv1 = tf.nn.relu(conv1_in, name='conv1')
        #print x.get_shape(), conv1.get_shape()
        
        
        self.conv1 = conv1



        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool1 = tf.nn.max_pool(conv1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name='pool1')
                                                                                       
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn1 = tf.nn.local_response_normalization(maxpool1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias, name='norm1')



      # conv2
      with tf.name_scope('conv2') as scope:
        #conv(5, 5, 256, 1, 1, group=2, name='conv2')
        k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv2W = tf.Variable(self.net_data["conv2"]["weights"], trainable=False, name='W')
        conv2b = tf.Variable(self.net_data["conv2"]["biases"], trainable=False, name='baises')
        conv2_in = self.conv(lrn1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv2 = tf.nn.relu(conv2_in, name='conv2')
        #print conv2.get_shape(), self.net_data["conv2"]["weights"].shape, self.net_data["conv2"]["biases"].shape

        self.conv2 = conv2


        #maxpool2                                                                                                              
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                                    
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool2 = tf.nn.max_pool(conv2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name='pool2')
        #print maxpool2.get_shape()



        #lrn2
        #lrn(2, 2e-05, 0.75, name='norm2')
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn2 = tf.nn.local_response_normalization(maxpool2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias, name='norm2')

        

      # conv3                                                                                                                                   
      with tf.name_scope('conv3') as scope:
        #conv(3, 3, 384, 1, 1, name='conv3')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
        conv3W = tf.Variable(self.net_data["conv3"]["weights"], trainable=False, name='W')
        conv3b = tf.Variable(self.net_data["conv3"]["biases"], trainable=False, name='baises')
        conv3_in = self.conv(lrn2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv3 = tf.nn.relu(conv3_in, name='conv3')
        #print conv3.get_shape(), self.net_data["conv3"]["weights"].shape, self.net_data["conv3"]["biases"].shape
    
      # conv4                                                                                                                                                            
      with tf.name_scope('conv4') as scope:
        #conv(3, 3, 384, 1, 1, group=2, name='conv4')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
        conv4W = tf.Variable(self.net_data["conv4"]["weights"], trainable=False, name='W')
        conv4b = tf.Variable(self.net_data["conv4"]["biases"], trainable=False, name='baises')
        conv4_in = self.conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv4 = tf.nn.relu(conv4_in, name='conv4')
        #print conv4.get_shape()

        self.conv4 = conv4

      # conv5                                                                                                                                             
      with tf.name_scope('conv5') as scope:
        #conv(3, 3, 256, 1, 1, group=2, name='conv5')
        k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv5W = tf.Variable(self.net_data["conv5"]["weights"], trainable=False, name='W')
        conv5b = tf.Variable(self.net_data["conv5"]["biases"], trainable=False, name='baises')
        self.conv5b = conv5b
        conv5_in = self.conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv5 = tf.nn.relu(conv5_in, name='conv5')
        #print conv5.get_shape()

        self.conv5 = conv5

        #maxpool5
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name='pool5')
        #print maxpool5.get_shape(), maxpool5.get_shape()[1:], int(np.prod(maxpool5.get_shape()[1:]))
        
      
      # fc6
      with tf.variable_scope('fc6') as scope:
        #fc(4096, name='fc6')
        fc6W = tf.Variable(self.net_data["fc6"]["weights"], trainable=False, name='W')
        fc6b = tf.Variable(self.net_data["fc6"]["biases"], trainable=False, name='baises')
        self.fc6W = fc6W
        self.fc6b = fc6b
        fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b, name='fc6')
        #print fc6.get_shape()
        if self.ifdropout == 1:
          fc6 = tf.nn.dropout(fc6, self.keep_rate_fc6, name='fc6_dropout')
            
      # fc7 
      with tf.variable_scope('fc7') as scope:
        #fc(4096, name='fc7')
        fc7W = tf.Variable(self.net_data["fc7"]["weights"], trainable=False, name='W')
        fc7b = tf.Variable(self.net_data["fc7"]["biases"], trainable=False, name='baises')
        self.fc7b = fc7b
        fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b, name='fc7')
        #print fc7.get_shape()
        if self.ifdropout == 1:
          fc7 = tf.nn.dropout(fc7, self.keep_rate_fc7, name='fc7_dropout')
                                                                                                   
      # fc8  
      with tf.variable_scope('fc8') as scope:
       
        # Move everything into depth so we can perform a single matrix multiplication.                            
        fc7 = tf.reshape(fc7, [self.batch_size, -1])
        dim = fc7.get_shape()[1].value
        #print "fc7 dim:\n"
        #print fc7.get_shape(), dim
        fc8W = tf.Variable(tf.random_normal(tf.stack([dim, self.labels.shape[1]]), mean=0.0, stddev=0.01), trainable=False, name='W')                                                                    
        fc8b = tf.Variable(tf.zeros([self.labels.shape[1]]), trainable=False, name='baises')                                                                                                      
        self.fc8b = fc8b
        theta = tf.nn.xw_plus_b(fc7, fc8W, fc8b)  

      

        self.theta = theta
        self.fc8W = fc8W
        self.fc8b = fc8b
        # %% We'll create a spatial transformer module to identify discriminative
        # %% patches
        #h_trans = self._transform(theta, x, out_size, channel_x)
        #print h_trans.get_shape()
      return theta


 

  def _build_train_op(self):
    """Build training specific ops for the graph."""
    #self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
    #tf.scalar_summary('learning rate', self.lrn_rate)
    """
    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(self.cost, trainable_variables)
    """
    if self.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
    elif self.optimizer == 'Adam':
      optimizer = tf.train.AdamOptimizer(0.001 * self.lr_rate_fac)
    elif self.optimizer == 'mom':
      optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
    
   

    self.train_op = optimizer.minimize(self.cost)

