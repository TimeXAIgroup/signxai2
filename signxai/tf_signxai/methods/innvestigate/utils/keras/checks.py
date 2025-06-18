# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


###############################################################################
###############################################################################
###############################################################################


import tensorflow as tf
from version_parser.version import Version
tf_v = Version(tf.__version__)

assert tf_v.get_major_version() == 2
assert tf_v.get_minor_version() >= 2


# Prevents circular imports.
def get_kgraph():
    from . import graph as kgraph
    return kgraph


__all__ = [
    # "get_current_layers",
    "get_known_layers",
    "get_activation_search_safe_layers",

    "contains_activation",
    "contains_kernel",
    "only_relu_activation",
    "is_network",
    "is_convnet_layer",
    "is_relu_convnet_layer",
    "is_average_pooling",
    "is_max_pooling",
    "is_input_layer",
    "is_batch_normalization_layer",
]


###############################################################################
###############################################################################
###############################################################################


# def get_current_layers():
#     """
#     Returns a list of currently available layers in Keras.
#     """
#     class_set = set([(getattr(keras_layers, name), name)
#                      for name in dir(keras_layers)
#                      if (inspect.isclass(getattr(keras_layers, name)) and
#                          issubclass(getattr(keras_layers, name),
#                                     'Layer'))])
#     return [x[1] for x in sorted((str(x[0]), x[1]) for x in class_set)]


def get_known_layers():
    """
    Returns a list of keras layer we are aware of.
    """

    # Inside function to not break import if Keras changes.
    KNOWN_LAYERS = (
        'ELU',
        'LeakyReLU',
        'PReLU',
        'Softmax',
        'ThresholdedReLU',
        'Conv1D',
        'Conv2D',
        'Conv2DTranspose',
        'Conv3D',
        'Conv3DTranspose',
        'Cropping1D',
        'Cropping2D',
        'Cropping3D',
        'SeparableConv1D',
        'SeparableConv2D',
        'UpSampling1D',
        'UpSampling2D',
        'UpSampling3D',
        'ZeroPadding1D',
        'ZeroPadding2D',
        'ZeroPadding3D',
        'ConvLSTM2D',
        'ConvRecurrent2D',
        'Activation',
        'ActivityRegularization',
        'Dense',
        'Dropout',
        'Flatten',
        'InputLayer',
        'Lambda',
        'Masking',
        'Permute',
        'RepeatVector',
        'Reshape',
        'SpatialDropout1D',
        'SpatialDropout2D',
        'SpatialDropout3D',
        'CuDNNGRU',
        'CuDNNLSTM',
        'Embedding',
        'LocallyConnected1D',
        'LocallyConnected2D',
        'Add',
        'Average',
        'Concatenate',
        'Dot',
        'Maximum',
        'Minimum',
        'Multiply',
        'Subtract',
        'AlphaDropout',
        'GaussianDropout',
        'GaussianNoise',
        'BatchNormalization',
        'BatchNorm',
        'AveragePooling1D',
        'AveragePooling2D',
        'AveragePooling3D',
        'GlobalAveragePooling1D',
        'GlobalAveragePooling2D',
        'GlobalAveragePooling3D',
        'GlobalMaxPooling1D',
        'GlobalMaxPooling2D',
        'GlobalMaxPooling3D',
        'MaxPooling1D',
        'MaxPooling2D',
        'MaxPooling3D',
        'GRU',
        'GRUCell',
        'LSTM',
        'LSTMCell',
        'RNN',
        'SimpleRNN',
        'SimpleRNNCell',
        'StackedRNNCells',
        'Bidirectional',
        'TimeDistributed',
        'Wrapper',
    )
    return KNOWN_LAYERS


def get_known_activations(lowercase=False):
    ACTIVATIONS = (
            'ReLU',
            'ELU',
            'LeakyReLU',
            'PReLU',
            'Softmax',
            'ThresholdedReLU')

    if lowercase:
        return [str.lower(x) for x in ACTIVATIONS]
    else:
        return ACTIVATIONS


def get_activation_search_safe_layers():
    """
    Returns a list of keras layer that we can walk along
    in an activation search.
    """

    # Inside function to not break import if Keras changes.
    ACTIVATION_SEARCH_SAFE_LAYERS = (
        'ELU',
        'LeakyReLU',
        'PReLU',
        'Softmax',
        'ThresholdedReLU',
        'Activation',
        'ActivityRegularization',
        'Dropout',
        'Flatten',
        'Reshape',
        'Add',
        'GaussianNoise',
        'BatchNormalization',
        'BatchNorm',
    )
    return ACTIVATION_SEARCH_SAFE_LAYERS


###############################################################################
###############################################################################
###############################################################################


def contains_activation(layer, activation=None):
    """
    Check whether the layer contains an activation function.
    activation is None then we only check if layer can contain an activation.
    """

    if hasattr(layer, "activation"):
        # print(layer.activation, activation)
        if activation is not None:
            return activation.lower() in str(layer.activation).lower()
        else:
            return True
    elif isInstanceOf(layer, get_known_activations()):
        if activation is not None:
            return activation.lower() in str(type(layer)).lower()
        else:
            return True
    else:
        return False


def contains_kernel(layer):
    """
    Check whether the layer contains a kernel.
    """

    # TODO: add test and check this more throughroughly.
    # rely on Keras convention.
    if hasattr(layer, "kernel") or hasattr(layer, "depthwise_kernel") or hasattr(layer, "pointwise_kernel"):
        return True
    else:
        return False


def contains_bias(layer):
    """
    Check whether the layer contains a bias.
    """

    # todo: add test and check this more throughroughly.
    # rely on Keras convention.
    if hasattr(layer, "bias"):
        return True
    else:
        return False


def only_relu_activation(layer):
    """Checks if layer contains no or only a ReLU activation."""
    return (not contains_activation(layer) or
            contains_activation(layer, None) or
            contains_activation(layer, "linear") or
            contains_activation(layer, "relu"))


def is_network(layer):
    """
    Is network in network?
    """
    return isInstanceOf(layer, 'Model')


def is_conv_layer(layer, *args, **kwargs):
    """Checks if layer is a convolutional layer."""
    CONV_LAYERS = (
        'Conv1D',
        'Conv2D',
        'Conv2DTranspose',
        'Conv3D',
        'Conv3DTranspose',
        'SeparableConv1D',
        'SeparableConv2D',
        'DepthwiseConv2D'
    )
    return isInstanceOf(layer, CONV_LAYERS)


def is_batch_normalization_layer(layer, *args, **kwargs):
    """Checks if layer is a batchnorm layer."""
    BN_LAYERS = (
        'BatchNormalization',
        'BatchNorm',
    )
    return isInstanceOf(layer, BN_LAYERS)


def is_add_layer(layer, *args, **kwargs):
    """Checks if layer is an addition-merge layer."""
    return isInstanceOf(layer, 'Add')


def is_dense_layer(layer, *args, **kwargs):
    """Checks if layer is a dense layer."""
    return isInstanceOf(layer, 'Dense')


def is_convnet_layer(layer):
    """Checks if layer is from a convolutional network."""
    # Inside function to not break import if Keras changes.
    CONVNET_LAYERS = (
        'ELU',
        'LeakyReLU',
        'PReLU',
        'Softmax',
        'ThresholdedReLU',
        'Conv1D',
        'Conv2D',
        'Conv2DTranspose',
        'Conv3D',
        'Conv3DTranspose',
        'Cropping1D',
        'Cropping2D',
        'Cropping3D',
        'SeparableConv1D',
        'SeparableConv2D',
        'UpSampling1D',
        'UpSampling2D',
        'UpSampling3D',
        'ZeroPadding1D',
        'ZeroPadding2D',
        'ZeroPadding3D',
        'Activation',
        'ActivityRegularization',
        'Dense',
        'Dropout',
        'Flatten',
        'InputLayer',
        'Lambda',
        'Masking',
        'Permute',
        'RepeatVector',
        'Reshape',
        'SpatialDropout1D',
        'SpatialDropout2D',
        'SpatialDropout3D',
        'Embedding',
        'LocallyConnected1D',
        'LocallyConnected2D',
        'Add',
        'Average',
        'Concatenate',
        'Dot',
        'Maximum',
        'Minimum',
        'Multiply',
        'Subtract',
        'AlphaDropout',
        'GaussianDropout',
        'GaussianNoise',
        'BatchNormalization',
        'BatchNorm',
        'AveragePooling1D',
        'AveragePooling2D',
        'AveragePooling3D',
        'GlobalAveragePooling1D',
        'GlobalAveragePooling2D',
        'GlobalAveragePooling3D',
        'GlobalMaxPooling1D',
        'GlobalMaxPooling2D',
        'GlobalMaxPooling3D',
        'MaxPooling1D',
        'MaxPooling2D',
        'MaxPooling3D',
    )
    return isInstanceOf(layer, CONVNET_LAYERS)


def is_relu_convnet_layer(layer):
    """Checks if layer is from a convolutional network with ReLUs."""
    return (is_convnet_layer(layer) and only_relu_activation(layer))


def is_average_pooling(layer):
    """Checks if layer is an average-pooling layer."""
    AVERAGEPOOLING_LAYERS = (
        'AveragePooling1D',
        'AveragePooling2D',
        'AveragePooling3D',
        'GlobalAveragePooling1D',
        'GlobalAveragePooling2D',
        'GlobalAveragePooling3D',
    )
    return isInstanceOf(layer, AVERAGEPOOLING_LAYERS)


def is_max_pooling(layer):
    """Checks if layer is a max-pooling layer."""
    MAXPOOLING_LAYERS = (
        'MaxPooling1D',
        'MaxPooling2D',
        'MaxPooling3D',
        'GlobalMaxPooling1D',
        'GlobalMaxPooling2D',
        'GlobalMaxPooling3D',
    )
    return isInstanceOf(layer, MAXPOOLING_LAYERS)


def is_input_layer(layer, ignore_reshape_layers=True):
    """Checks if layer is an input layer."""
    # Triggers if ALL inputs of layer are connected
    # to a Keras input layer object.
    # Note: In the sequential api the Sequential object
    # adds the Input layer if the user does not.
    kgraph = get_kgraph()

    layer_inputs = kgraph.get_input_layers(layer)
    # We ignore certain layers, that do not modify
    # the data content.
    # todo: update this list!
    IGNORED_LAYERS = (
        'Flatten',
        'Permute',
        'Reshape',
        'ZeroPadding1D',
        'ZeroPadding2D',
        'ZeroPadding3D'
    )
    while any([isInstanceOf(x, IGNORED_LAYERS) for x in layer_inputs]):
        tmp = set()
        for l in layer_inputs:
            if(ignore_reshape_layers and
               isInstanceOf(l, IGNORED_LAYERS)):
                tmp.update(kgraph.get_input_layers(l))
            else:
                tmp.add(l)
        layer_inputs = tmp

    # print([(type(x), 'InputLayer') for x in layer_inputs])
    if all(['InputLayer' in str(type(x)) for x in layer_inputs]):
        # print('IsInputLayer = True')
        return True
    else:
        # print('IsInputLayer = False')
        return False


def isInstanceOf(obj, valid):
    obj_type = str(type(obj))
    if type(valid) == str:
        if obj_type.endswith(".{}'>".format(valid)):
            return True
    elif type(valid) == list or type(valid) == tuple:
        for x in valid:
            if obj_type.endswith(".{}'>".format(x)):
                return True
    else:
        raise Exception('Unknown type: {}'.format(type(valid)))

    return False

def is_layer_at_idx(layer, index, ignore_reshape_layers=True):
    """Checks if layer is a layer at index index."""
    # Triggers if ALL inputs of layer are connected
    # to a Keras input layer object.
    # Note: In the sequential api the Sequential object
    # adds the Input layer if the user does not.
    kgraph = get_kgraph()

    layer_inputs = [layer]
    # We ignore certain layers, that do not modify
    # the data content.
    # todo: update this list!
    IGNORED_LAYERS = (
        'Flatten',
        'Permute',
        'Reshape',
    )

    for i in range(index):

        while any([isInstanceOf(x, IGNORED_LAYERS) for x in layer_inputs]):
            tmp = set()
            for l in layer_inputs:
                if (ignore_reshape_layers and
                        isInstanceOf(l, IGNORED_LAYERS)):
                    tmp.update(kgraph.get_input_layers(l))
                else:
                    tmp.add(l)
            layer_inputs = tmp

        tmp = set()
        for l in layer_inputs:
            tmp.update(kgraph.get_input_layers(l))
        layer_inputs = tmp

        if any([isInstanceOf(x, 'InputLayer')
                for x in layer_inputs]):
            return False

    ret = all([is_input_layer(x, ignore_reshape_layers) for x in layer_inputs])
    return ret
