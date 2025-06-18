import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

from signxai.methods.wrappers import calculate_relevancemap
from signxai.utils.utils import normalize_heatmap, download_model

# Load train and test data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Scale images to the [-1, 0] range
x_train = x_train.astype("float32") / -255.0
x_test = x_test.astype("float32") / -255.0
x_train = -(np.ones_like(x_train) + x_train)
x_test = -(np.ones_like(x_test) + x_test)

# Load model
path = 'model.h5'
download_model(path)
model = load_model(path)

# Remove softmax
model.layers[-1].activation = None

# Calculate relevancemaps
x = x_test[231]
R1 = calculate_relevancemap('gradient_x_input', np.array(x), model, neuron_selection=3)
R2 = calculate_relevancemap('gradient_x_sign_mu_neg_0_5', np.array(x), model, neuron_selection=3)
R3 = calculate_relevancemap('gradient_x_input', np.array(x), model, neuron_selection=8)
R4 = calculate_relevancemap('gradient_x_sign_mu_neg_0_5', np.array(x), model, neuron_selection=8)

# Visualize heatmaps
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(18, 12))
axs[0][0].imshow(x, cmap='seismic', clim=(-1, 1))
axs[1][0].imshow(x, cmap='seismic', clim=(-1, 1))
axs[0][1].matshow(normalize_heatmap(R1), cmap='seismic', clim=(-1, 1))
axs[0][2].matshow(normalize_heatmap(R2), cmap='seismic', clim=(-1, 1))
axs[1][1].matshow(normalize_heatmap(R3), cmap='seismic', clim=(-1, 1))
axs[1][2].matshow(normalize_heatmap(R4), cmap='seismic', clim=(-1, 1))

plt.show()
