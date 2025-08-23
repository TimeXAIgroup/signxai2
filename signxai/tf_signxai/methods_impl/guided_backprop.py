""" Source: https://github.com/Crispy13/crispy """

import tensorflow as tf
from tensorflow import keras


def build_guided_model(model):
    """
    Builds guided model

    """

    model_copied = keras.models.clone_model(model)
    model_copied.set_weights(model.get_weights())

    @tf.custom_gradient
    def guidedRelu(x):
        def grad(dy):
            return tf.cast(dy > 0, tf.float32) * tf.cast(x > 0, tf.float32) * dy

        return tf.nn.relu(x), grad

    layer_dict = [layer for layer in model_copied.layers[1:] if hasattr(layer, 'activation')]
    for layer in layer_dict:
        if layer.activation == tf.keras.activations.relu:
            layer.activation = guidedRelu

    return model_copied


def guided_backprop_on_guided_model(model_no_softmax, img, layer_name):
    """
    Returns guided backpropagation image.

    Parameters
    ----------
    model_no_softmax : a keras model object

    img : an img to inspect with guided backprop.

    layer_name : a string
        a layer name for calculating gradients.


    """
    guided_model = build_guided_model(model_no_softmax)

    part_model = keras.Model(guided_model.inputs, guided_model.get_layer(layer_name).output)

    with tf.GradientTape() as tape:
        f32_img = tf.cast(img, tf.float32)
        tape.watch(f32_img)
        part_output = part_model(f32_img)

    grads = tape.gradient(part_output, f32_img)[0].numpy()

    # delete copied model
    del part_model
    del guided_model

    return grads


