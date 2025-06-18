import json
import logging

from tensorflow.python.keras.saving.model_config import model_from_json

def remove_softmax_from_model(loaded_model_json):
    # Convert JSONized model config to dictionary
    dct = json.loads(loaded_model_json)

    # Extract layer list
    layers = dct['config']['layers']

    # Remove Softmax layer if present
    for i, layer in enumerate(layers):
        if layer['class_name'] == 'Softmax':
            layers.pop(i)
            break

    # Convert dict back to json string
    loaded_model_json = json.dumps(dct)

    # Replace Softmax by ReLU activations if present
    loaded_model_json = loaded_model_json.replace('"activation": "softmax"', '"activation": "relu"')

    return loaded_model_json


def load_model_and_weights_from_paths(modelpath, weightspath, remove_softmax=False):
    with open(modelpath, 'r') as json_file:
        loaded_model_json = json_file.read()

    if remove_softmax:
        loaded_model_json = remove_softmax_from_model(loaded_model_json)

    model = model_from_json(loaded_model_json)
    logging.info('Loaded and initialized model from disk.')

    model.load_weights(weightspath)
    logging.info('Loaded weights from disk into model.')

    return model


def load_models_from_paths(modelpath, weightspath):
    model = load_model_and_weights_from_paths(modelpath, weightspath, remove_softmax=False)
    model_wo_softmax = load_model_and_weights_from_paths(modelpath, weightspath, remove_softmax=True)

    return model, model_wo_softmax