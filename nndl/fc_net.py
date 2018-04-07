import numpy as np
import pdb

from .layers import *
from .layer_utils import *

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize all parameters of the network in the self.params dictionary.
    #   The weights and biases of layer 1 are W1 and b1; and in general the 
    #   weights and biases of layer i are Wi and bi. The
    #   biases are initialized to zero and the weights are initialized
    #   so that each parameter has mean 0 and standard deviation weight_scale.
    #
    #   BATCHNORM: Initialize the gammas of each layer to 1 and the beta
    #   parameters to zero.  The gamma and beta parameters for layer 1 should
    #   be self.params['gamma1'] and self.params['beta1'].  For layer 2, they
    #   should be gamma2 and beta2, etc. Only use batchnorm if self.use_batchnorm 
    #   is true and DO NOT batch normalize the output scores.
    # ================================================================ #
    layer_list = [input_dim]+hidden_dims+[num_classes]
    

    for i in range(0,self.num_layers-1):
      if self.use_batchnorm:
        self.params['gamma'+str(i+1)] = np.ones(layer_list[i+1])
        self.params['beta'+str(i+1)] = np.zeros(layer_list[i+1])
      self.params['W'+str(i+1)] = np.random.normal(scale = weight_scale,size = (layer_list[i],layer_list[i+1]))
      self.params['b'+str(i+1)] = np.zeros(layer_list[i+1])
    
    self.params['W'+str(self.num_layers)] = np.random.normal(scale = weight_scale,size = (hidden_dims[-1],num_classes))
    self.params['b'+str(self.num_layers)] = np.zeros(num_classes)
    
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in np.arange(self.num_layers - 1)]

    # Cast all parameters to the correct datatype
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the FC net and store the output
    #   scores as the variable "scores".
    #
    #   BATCHNORM: If self.use_batchnorm is true, insert a bathnorm layer
    #   between the affine_forward and relu_forward layers.  You may
    #   also write an affine_batchnorm_relu() function in layer_utils.py.
    #
    #   DROPOUT: If dropout is non-zero, insert a dropout layer after
    #   every ReLU layer.
    # ================================================================ #

    out_fcrl = X
        
    self.cache = {}
    cache_drop = None
    for i in range(1,self.num_layers+1):
      W_i = 'W'+ str(i)
      b_i = 'b' +str(i)
      gamma_i = 'gamma'+str(i)
      beta_i = 'beta'+str(i)
      cache_i = 'c' +str(i)
      if i == self.num_layers:
        out_fcrl,cache = affine_forward(out_fcrl,self.params[W_i],self.params[b_i])
      else:
        if self.use_batchnorm:
          out_fcrl,cache = affine_bn_relu_forward(out_fcrl,self.params[W_i],self.params[b_i],self.params[gamma_i],self.params[beta_i],self.bn_params[i-1])
        else:
          out_fcrl,cache = affine_relu_forward(out_fcrl,self.params[W_i],self.params[b_i])
        if self.use_dropout:
          out_fcrl,cache_drop = dropout_forward(out_fcrl,self.dropout_param)

      self.cache[cache_i] = {'cache':cache,'drop':cache_drop}
          
    scores = out_fcrl

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backwards pass of the FC net and store the gradients
    #   in the grads dict, so that grads[k] is the gradient of self.params[k]
    #   Be sure your L2 regularization includes a 0.5 factor.
    #
    #   BATCHNORM: Incorporate the backward pass of the batchnorm.
    #
    #   DROPOUT: Incorporate the backward pass of dropout.
    # ================================================================ #

    loss,grads_out = softmax_loss(scores,y)
    for i in range(self.num_layers,0,-1):
      W_i = 'W'+str(i)
      b_i = 'b'+str(i)
      gamma_i = 'gamma'+str(i)
      beta_i = 'beta'+str(i)
      cache_i = 'c'+str(i)
     
      loss+=0.5*self.reg*(np.sum(self.params[W_i]**2))
      if i ==self.num_layers:
        dx,grads[W_i],grads[b_i] = affine_backward(grads_out,self.cache[cache_i]['cache'])
      else:
        if self.use_dropout:
          dx = dropout_backward(dx,self.cache[cache_i]['drop'])
        if self.use_batchnorm:
          dx,grads[W_i],grads[b_i],grads[gamma_i],grads[beta_i] = affine_bn_relu_backward(dx,self.cache[cache_i]['cache'])
        else:
          dx,grads[W_i],grads[b_i] = affine_relu_backward(dx,self.cache[cache_i]['cache'])

      grads[W_i]+=self.reg*self.params[W_i]
            



    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    return loss, grads
