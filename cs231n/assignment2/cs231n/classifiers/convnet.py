import time
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *


class ConvNet(object):
    """Convolutional network
    """

    def __init__(self, input_dim=(3, 32, 32), layers=(), weight_scale=1e-3, reg=0.):
        """Initialize a network

        Args:
            input_dim: Tuple (C, H, W) giving size of input data
            layers: Tuple (
                {'type': 'affine':
                 'param': {
                    'num_output': ,
                 },
                },
                {'type': 'relu'},
                {'type': 'dropout'
                'param': {
                    'p': 0.5,
                 },
                },
                {'type': 'conv',
                 'param': {
                    'num_output': ,
                    'filter_size': ,
                    'pad': ,
                    'stride': ,
                 },
                },
                {'type': 'pool',
                 'param': {
                    'pool': MAX,
                    'filter_size': ,
                    'stride': ,
                 },
                },
                {'type': 'batchnorm',
                'param': {
                    'eps': 1e-5',
                    'momentum': 0.9
                 },
                },
                {'type': 'spatial_batchnorm',
                'param': {
                    'eps': 1e-5',
                    'momentum': 0.9
                 },
                },
                {'type': 'svm'},
                {'type': 'softmax'}
                )
            weight_scale: Scalar giving standard deviation for random initialization
                of weights.
            reg: Scalar giving L2 regularization strength
        """
        self.params = {}
        self.reg = reg
        self.layers = layers

        af_cv_idx = 0
        batchnorm_idx = 0
        last_dim = input_dim
        for i, layer in enumerate(self.layers[:-1]):
            if layer['type'] == 'affine':
                af_cv_idx += 1
                self.layers[i]['param']['names'] = ('W' + str(af_cv_idx), 'b' + str(af_cv_idx))
                self.params[layer['param']['names'][0]] = np.random.randn(np.prod(last_dim), layer['param']['num_output']) * weight_scale
                self.params[layer['param']['names'][1]] = np.zeros(layer['param']['num_output'])

                last_dim = layer['param']['num_output']
            elif layer['type'] == 'relu':
                pass
            elif layer['type'] == 'dropout':
                pass
            elif layer['type'] == 'conv':
                af_cv_idx += 1
                C, H, W = last_dim
                num_filters, filter_size, pad, stride = layer['param']['num_output'], layer['param']['filter_size'], layer['param']['pad'], layer['param']['stride']
                self.layers[i]['param']['names'] = ('W' + str(af_cv_idx), 'b' + str(af_cv_idx))
                self.params[layer['param']['names'][0]] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale
                self.params[layer['param']['names'][1]] = np.zeros(num_filters)

                out_height = (H - filter_size + 2 * pad) / stride + 1
                out_width = (W - filter_size + 2 * pad) / stride + 1
                last_dim = (num_filters, out_height, out_width)
            elif layer['type'] == 'pool':
                C, H, W = last_dim
                out_height = (H - layer['param']['filter_size']) / layer['param']['stride'] + 1
                out_width = (W - layer['param']['filter_size']) / layer['param']['stride'] + 1

                last_dim = (C, out_height, out_width)
            elif layer['type'] == 'spatial_batchnorm':
                batchnorm_idx += 1

                self.layers[i]['param']['names'] = ('gamma' + str(batchnorm_idx), 'beta' + str(batchnorm_idx))
                self.params[layer['param']['names'][0]] = np.random.randn(last_dim[-3]) * weight_scale
                self.params[layer['param']['names'][1]] = np.random.randn(last_dim[-3]) * weight_scale
            elif layer['type'] == 'batchnorm':
                batchnorm_idx += 1

                self.layers[i]['param']['names'] = ('gamma' + str(batchnorm_idx), 'beta' + str(batchnorm_idx))
                self.params[layer['param']['names'][0]] = np.random.randn(np.prod(last_dim)) * weight_scale
                self.params[layer['param']['names'][1]] = np.random.randn(np.prod(last_dim)) * weight_scale
            else:
                raise ValueError('Invalid layer type "%s"' % layer['type'])

    def loss(self, X, y=None):
        mode = 'test' if y is None else 'train'
        in_vec = []
        out_vec = []
        cache_vec = []

        in_vec.append(X)
        for i, layer in enumerate(self.layers[:-1]):
            out = None
            cache = None
            if layer['type'] == 'affine':
                out, cache = affine_forward(in_vec[i], self.params[layer['param']['names'][0]], self.params[layer['param']['names'][1]])
            elif layer['type'] == 'relu':
                out, cache = relu_forward(in_vec[i])
            elif layer['type'] == 'dropout':
                dropout_param = {'mode': mode, 'p': layer['param']['p']}
                out, cache = dropout_forward(in_vec[i], dropout_param)
            elif layer['type'] == 'conv':
                conv_param = {'stride': layer['param']['stride'], 'pad': layer['param']['pad']}
                out, cache = conv_forward_fast(in_vec[i], self.params[layer['param']['names'][0]], self.params[layer['param']['names'][1]], conv_param)
            elif layer['type'] == 'pool':
                pool_param = {'pool_height': layer['param']['filter_size'], 'pool_width': layer['param']['filter_size'], 'stride': layer['param']['stride']}
                out, cache = max_pool_forward_fast(in_vec[i], pool_param)
            elif layer['type'] == 'spatial_batchnorm':
                bn_param = {'mode': mode, 'eps': layer['param']['eps'], 'momentum': layer['param']['momentum']}
                out, cache = spatial_batchnorm_forward(in_vec[i], self.params[layer['param']['names'][0]], self.params[layer['param']['names'][1]], bn_param)
            elif layer['type'] == 'batchnorm':
                bn_param = {'mode': mode, 'eps': layer['param']['eps'], 'momentum': layer['param']['momentum']}
                out, cache = batchnorm_forward(in_vec[i], self.params[layer['param']['names'][0]], self.params[layer['param']['names'][1]], bn_param)
            else:
                raise ValueError('Invalid layer type "%s"' % layer['type'])

            out_vec.append(out)
            cache_vec.append(cache)
            in_vec.append(out)

        score = out_vec[-1]

        if y is None:
            return score

        loss = 0
        grads = {}

        # last layer
        if self.layers[-1]['type'] == 'softmax':
            loss, dscore = softmax_loss(score, y)
        elif self.layer[-1]['type'] == 'svm':
            loss, dscore = svm_loss(score, y)

        dout_vec = [None] * (len(self.layers) - 1)
        dout_vec[-1] = dscore
        i = len(self.layers) - 2
        for layer in self.layers[-2::-1]:
            dx = None
            if layer['type'] == 'affine':
                dx, dw, db = affine_backward(dout_vec[i], cache_vec[i])
                grads[layer['param']['names'][0]] = dw
                grads[layer['param']['names'][1]] = db

                loss += 0.5 * self.reg * np.sum(self.params[layer['param']['names'][0]] * self.params[layer['param']['names'][0]])
                grads[layer['param']['names'][0]] += self.reg * self.params[layer['param']['names'][0]]
            elif layer['type'] == 'relu':
                dx = relu_backward(dout_vec[i], cache_vec[i])
            elif layer['type'] == 'dropout':
                dx = dropout_backward(dout_vec[i], cache_vec[i])
            elif layer['type'] == 'conv':
                dx, dw, db = conv_backward_fast(dout_vec[i], cache_vec[i])
                grads[layer['param']['names'][0]] = dw
                grads[layer['param']['names'][1]] = db

                loss += 0.5 * self.reg * np.sum(self.params[layer['param']['names'][0]] * self.params[layer['param']['names'][0]])
                grads[layer['param']['names'][0]] += self.reg * self.params[layer['param']['names'][0]]
            elif layer['type'] == 'pool':
                dx = max_pool_backward_fast(dout_vec[i], cache_vec[i])
            elif layer['type'] == 'spatial_batchnorm':
                dx, dgamma, dbeta = spatial_batchnorm_backward(dout_vec[i], cache_vec[i])
                grads[layer['param']['names'][0]] = dgamma
                grads[layer['param']['names'][1]] = dbeta
            elif layer['type'] == 'batchnorm':
                dx, dgamma, dbeta = batchnorm_backward(dout_vec[i], cache_vec[i])
                grads[layer['param']['names'][0]] = dgamma
                grads[layer['param']['names'][1]] = dbeta
            else:
                raise ValueError('Invalid layer type "%s"' % layer['type'])

            if i > 0:
                dout_vec[i - 1] = dx
            i -= 1

        return loss, grads
