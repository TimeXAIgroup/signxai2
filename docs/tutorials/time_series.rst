=================
ECG Time Series
=================

This tutorial demonstrates how to use SignXAI2 for explaining time series models, specifically focusing on ECG (electrocardiogram) data.

.. contents:: Contents
   :local:
   :depth: 2

Introduction
------------

Time series data presents unique challenges for explainability. In this tutorial, we'll use SignXAI2 to explain predictions from ECG classification models built with both TensorFlow and PyTorch.

ECG signals are particularly interesting because they have specific patterns (P-wave, QRS complex, T-wave) that domain experts recognize, allowing us to validate if our explainability methods highlight medically relevant features.

Setup
-----

First, let's install the required packages:

.. code-block:: bash

    # For TensorFlow
    pip install signxai2[tensorflow]
    
    # For PyTorch
    pip install signxai2[pytorch]
    
    # Additional packages for this tutorial
    pip install wfdb matplotlib scikit-learn

Let's download a sample ECG record from PhysioNet:

.. code-block:: python

    import wfdb
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Download a sample ECG record
    record = wfdb.rdrecord('100', pn_dir='mitdb', sampto=3600)
    
    # Extract the first lead
    ecg_signal = record.p_signal[:, 0]
    
    # Plot the signal
    plt.figure(figsize=(15, 5))
    plt.plot(ecg_signal)
    plt.title('ECG Signal')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude (mV)')
    plt.grid(True)
    plt.show()
    
    # Save a segment for our analysis
    segment = ecg_signal[1000:2000]
    np.save('ecg_segment.npy', segment)

TensorFlow ECG Model
--------------------

Let's build a simple CNN model for ECG classification with TensorFlow:

.. code-block:: python

    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
    
    # Generate synthetic data (in practice, you would use real ECG datasets)
    def generate_synthetic_ecg_data(n_samples=1000, seq_length=1000, n_classes=2):
        X = np.random.randn(n_samples, seq_length, 1) * 0.1
        # Add synthetic patterns for different classes
        for i in range(n_samples):
            if i % n_classes == 0:  # Class 0: Normal
                # Add normal QRS complex
                X[i, 400:420, 0] += np.sin(np.linspace(0, np.pi, 20)) * 1.0
                X[i, 350:370, 0] += np.sin(np.linspace(0, np.pi, 20)) * 0.2  # P wave
                X[i, 450:480, 0] += np.sin(np.linspace(0, np.pi, 30)) * 0.3  # T wave
            else:  # Class 1: Abnormal
                # Add abnormal QRS complex
                X[i, 380:410, 0] += np.sin(np.linspace(0, np.pi, 30)) * 0.8
                X[i, 420:460, 0] -= np.sin(np.linspace(0, np.pi, 40)) * 0.4
            
        # Create labels
        y = np.array([i % n_classes for i in range(n_samples)])
        return X, y
    
    # Generate data
    X_train, y_train = generate_synthetic_ecg_data(800, 1000, 2)
    X_test, y_test = generate_synthetic_ecg_data(200, 1000, 2)
    
    # Create a CNN model for ECG classification
    def create_ecg_model(seq_length=1000):
        model = Sequential([
            Conv1D(16, kernel_size=5, activation='relu', input_shape=(seq_length, 1)),
            MaxPooling1D(pool_size=2),
            Conv1D(32, kernel_size=5, activation='relu'),
            MaxPooling1D(pool_size=2),
            Conv1D(64, kernel_size=5, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(2)  # No activation (logits)
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    # Create and train the model
    model = create_ecg_model()
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_acc:.4f}')
    
    # Save the model
    model.save('ecg_model_tf.h5')
    
    # Save a sample for explanation
    np.save('ecg_sample.npy', X_test[0, :, 0])

Now let's use SignXAI to explain the ECG model's predictions:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from signxai import explain, list_methods
    
    # Load the model and sample
    model = tf.keras.models.load_model('ecg_model_tf.h5')
    ecg_sample = np.load('ecg_sample.npy')
    
    # Prepare input
    x = ecg_sample.reshape(1, 1000, 1)
    
    # Get prediction
    preds = model.predict(x)
    predicted_class = np.argmax(preds[0])
    print(f"Predicted class: {predicted_class} (confidence: {tf.nn.softmax(preds)[0, predicted_class]:.4f})")
    
    # Calculate explanations with different methods
    methods = [
        'gradient',
        'gradient_x_input',
        'integrated_gradients',
        'grad_cam',  # Works for time series too
        'lrp_z',
        'lrp_epsilon_0_1',
        'lrpsign_z'  # The SIGN method
    ]
    
    explanations = {}
    for method in methods:
        if method == 'grad_cam':
            explanations[method] = explain(
                model=model,
                x=x,
                method_name=method,
                target_class=predicted_class,
                last_conv_layer_name='conv1d_2'
            )
        else:
            explanations[method] = explain(
                model=model,
                x=x,
                method_name=method,
                target_class=predicted_class
            )
    
    # Visualize explanations
    fig, axs = plt.subplots(len(methods) + 1, 1, figsize=(15, 3*(len(methods) + 1)))
    
    # Original signal
    axs[0].plot(ecg_sample)
    axs[0].set_title('Original ECG Signal')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid(True)
    
    # Explanations
    for i, method in enumerate(methods):
        # Reshape explanation to 1D
        expl = explanations[method][0, :, 0]
        
        # Plot explanation
        axs[i+1].plot(expl)
        axs[i+1].set_title(f'Method: {method}')
        axs[i+1].set_ylabel('Attribution')
        axs[i+1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Alternative visualization: Overlay explanation on signal
    plt.figure(figsize=(15, 10))
    
    for i, method in enumerate(methods):
        plt.subplot(len(methods), 1, i+1)
        
        # Original signal
        plt.plot(ecg_sample, 'gray', alpha=0.5, label='ECG Signal')
        
        # Explanation
        expl = explanations[method][0, :, 0]
        expl_norm = (expl - expl.min()) / (expl.max() - expl.min()) if expl.max() > expl.min() else expl
        plt.plot(expl_norm, 'r', label='Attribution')
        
        plt.title(f'Method: {method}')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

PyTorch ECG Model
-----------------

Now let's implement a similar model in PyTorch:

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.utils.data import TensorDataset, DataLoader
    
    # Create a PyTorch CNN model for ECG classification
    class ECG_CNN(nn.Module):
        def __init__(self, seq_length=1000):
            super(ECG_CNN, self).__init__()
            self.conv1 = nn.Conv1d(1, 16, kernel_size=5)
            self.pool1 = nn.MaxPool1d(2)
            self.conv2 = nn.Conv1d(16, 32, kernel_size=5)
            self.pool2 = nn.MaxPool1d(2)
            self.conv3 = nn.Conv1d(32, 64, kernel_size=5)
            self.pool3 = nn.MaxPool1d(2)
            
            # Calculate size after convolutions and pooling
            self.flat_size = 64 * (((seq_length - 4) // 2 - 4) // 2 - 4) // 2
            
            self.fc1 = nn.Linear(self.flat_size, 64)
            self.dropout = nn.Dropout(0.2)
            self.fc2 = nn.Linear(64, 2)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            # Conv blocks
            x = self.pool1(self.relu(self.conv1(x)))
            x = self.pool2(self.relu(self.conv2(x)))
            x = self.pool3(self.relu(self.conv3(x)))
            
            # Flatten
            x = x.view(-1, self.flat_size)
            
            # Fully connected
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            
            return x
    
    # Generate the same synthetic data as before
    X_train, y_train = generate_synthetic_ecg_data(800, 1000, 2)
    X_test, y_test = generate_synthetic_ecg_data(200, 1000, 2)
    
    # Convert to PyTorch tensors and prepare data loaders
    # PyTorch expects [batch, channels, time] format
    X_train_pt = torch.tensor(X_train.transpose(0, 2, 1), dtype=torch.float32)
    y_train_pt = torch.tensor(y_train, dtype=torch.long)
    X_test_pt = torch.tensor(X_test.transpose(0, 2, 1), dtype=torch.float32)
    y_test_pt = torch.tensor(y_test, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_pt, y_train_pt)
    test_dataset = TensorDataset(X_test_pt, y_test_pt)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize model, loss, and optimizer
    model = ECG_CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {correct/total:.4f}')
    
    # Save the model
    torch.save(model.state_dict(), 'ecg_model_pt.pth')
    
    # Save the same sample for explanation
    sample = X_test[0]
    torch.save(torch.tensor(sample.transpose(1, 0), dtype=torch.float32), 'ecg_sample_pt.pt')

Now let's use SignXAI to explain the PyTorch ECG model:

.. code-block:: python

    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from signxai import explain, list_methods
    
    # Load the model
    model = ECG_CNN()
    model.load_state_dict(torch.load('ecg_model_pt.pth'))
    model.eval()
    
    # Remove softmax (modify the last layer)
    model.fc2 = torch.nn.Linear(64, 2, bias=True)
    model.load_state_dict(torch.load('ecg_model_pt.pth'), strict=False)
    
    # Load the sample
    ecg_sample_pt = torch.load('ecg_sample_pt.pt')
    ecg_sample_np = ecg_sample_pt.numpy()[0]  # Convert to numpy for visualization
    
    # Add batch dimension
    input_tensor = ecg_sample_pt.unsqueeze(0)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
    
    _, predicted_idx = torch.max(output, 1)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    print(f"Predicted class: {predicted_idx.item()} (confidence: {probabilities[0, predicted_idx.item()]:.4f})")
    
    # Calculate explanations with different methods
    methods = [
        "gradient",
        "gradient_x_input",
        "integrated_gradients",
        "smoothgrad",
        "lrp_epsilon_0_1",
        "lrp_alpha_1_beta_0"
    ]
    
    explanations = {}
    for method in methods:
        explanations[method] = explain(
            model=model,
            x=input_tensor,
            method_name=method,
            target_class=predicted_idx.item()
        )
    
    # Visualize explanations
    fig, axs = plt.subplots(len(methods) + 1, 1, figsize=(15, 3*(len(methods) + 1)))
    
    # Original signal
    axs[0].plot(ecg_sample_np)
    axs[0].set_title('Original ECG Signal')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid(True)
    
    # Explanations
    for i, method in enumerate(methods):
        # Reshape explanation to 1D (PyTorch format is [batch, channel, time])
        expl = explanations[method][0, 0, :]
        
        # Plot explanation
        axs[i+1].plot(expl)
        axs[i+1].set_title(f'Method: {method}')
        axs[i+1].set_ylabel('Attribution')
        axs[i+1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Alternative visualization: Colorful time series
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    
    plt.figure(figsize=(15, 15))
    
    for i, method in enumerate(methods):
        plt.subplot(len(methods), 1, i+1)
        
        # Get explanation
        expl = explanations[method][0, 0, :].numpy()
        
        # Normalize between -1 and 1
        norm = Normalize(vmin=-1, vmax=1)
        normalized_expl = 2 * (expl - expl.min()) / (expl.max() - expl.min()) - 1 if expl.max() > expl.min() else expl
        
        # Create colormap
        cmap = plt.cm.seismic
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        
        # Plot time series with color based on explanation
        for j in range(len(ecg_sample_np) - 1):
            plt.plot(
                [j, j+1], 
                [ecg_sample_np[j], ecg_sample_np[j+1]], 
                color=cmap(norm(normalized_expl[j])), 
                linewidth=2
            )
        
        plt.colorbar(sm, label='Attribution Value')
        plt.title(f'Method: {method}')
        plt.ylabel('Amplitude')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

Advanced Analysis
-----------------

Let's perform a more detailed analysis focusing on characteristic ECG features:

.. code-block:: python

    # Define characteristic ECG components (these would be expert-identified in real applications)
    p_wave_region = slice(350, 370)
    qrs_complex_region = slice(400, 420)
    t_wave_region = slice(450, 480)
    
    # Calculate the mean attribution for each region using TensorFlow LRP-SIGN method
    lrpsign_expl = explanations['lrpsign_z'][0, :, 0]
    
    p_wave_attr = np.mean(np.abs(lrpsign_expl[p_wave_region]))
    qrs_complex_attr = np.mean(np.abs(lrpsign_expl[qrs_complex_region]))
    t_wave_attr = np.mean(np.abs(lrpsign_expl[t_wave_region]))
    
    # Visualize with region highlighting
    plt.figure(figsize=(15, 6))
    
    # Plot original ECG
    plt.subplot(2, 1, 1)
    plt.plot(ecg_sample)
    
    # Highlight ECG components
    plt.axvspan(350, 370, color='blue', alpha=0.2, label='P-wave')
    plt.axvspan(400, 420, color='red', alpha=0.2, label='QRS Complex')
    plt.axvspan(450, 480, color='green', alpha=0.2, label='T-wave')
    
    plt.title('ECG Signal with Components')
    plt.legend()
    plt.grid(True)
    
    # Plot explanation with component attribution
    plt.subplot(2, 1, 2)
    plt.plot(lrpsign_expl)
    
    # Highlight attribution in ECG components
    plt.axvspan(350, 370, color='blue', alpha=0.2)
    plt.axvspan(400, 420, color='red', alpha=0.2)
    plt.axvspan(450, 480, color='green', alpha=0.2)
    
    # Add component attribution values
    plt.text(360, max(lrpsign_expl), f'P-wave: {p_wave_attr:.4f}', 
             horizontalalignment='center', backgroundcolor='white')
    plt.text(410, max(lrpsign_expl), f'QRS: {qrs_complex_attr:.4f}', 
             horizontalalignment='center', backgroundcolor='white')
    plt.text(465, max(lrpsign_expl), f'T-wave: {t_wave_attr:.4f}', 
             horizontalalignment='center', backgroundcolor='white')
    
    plt.title('LRP-SIGN Attribution')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Compare attribution across methods
    methods_to_compare = ['gradient', 'gradient_x_input', 'lrp_z', 'lrpsign_z']
    components = ['P-wave', 'QRS Complex', 'T-wave']
    regions = [p_wave_region, qrs_complex_region, t_wave_region]
    
    # Calculate attribution for each method and component
    component_attribution = {}
    for method in methods_to_compare:
        expl = explanations[method][0, :, 0]
        component_attribution[method] = [np.mean(np.abs(expl[region])) for region in regions]
    
    # Visualize component attribution comparison
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(components))
    width = 0.2
    offsets = np.linspace(-0.3, 0.3, len(methods_to_compare))
    
    for i, method in enumerate(methods_to_compare):
        plt.bar(x + offsets[i], component_attribution[method], width, label=method)
    
    plt.xlabel('ECG Component')
    plt.ylabel('Mean Absolute Attribution')
    plt.title('Attribution Comparison Across Methods')
    plt.xticks(x, components)
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.show()

Conclusion
----------

In this tutorial, we've demonstrated how SignXAI can be used to explain time series models, specifically:

1. Building and training ECG classification models in both TensorFlow and PyTorch
2. Using various explainability methods to generate attributions
3. Visualizing attributions for time series data
4. Performing component-specific analysis to identify which ECG features are most important for the model's predictions

Time series explainability offers unique insights that can be particularly valuable in domains like healthcare, where understanding why a model made a specific prediction can be critical.

The methods we've seen can be applied to other time series data types such as financial data, sensor readings, or any sequential data where understanding the model's focus is important.

Interactive Notebooks
---------------------

For hands-on experience with time series explanations using ECG data, check out these interactive Jupyter notebooks:

**TensorFlow:**
- `examples/tutorials/tensorflow/tensorflow_time_series.ipynb` - ECG classification with TensorFlow and iNNvestigate

**PyTorch:**
- `examples/tutorials/pytorch/pytorch_time_series.ipynb` - ECG classification with PyTorch and Zennit

These notebooks provide complete implementations including data preprocessing, model training, and explanation generation with real ECG datasets.