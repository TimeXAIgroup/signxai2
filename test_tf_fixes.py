#!/usr/bin/env python3
"""
Test TensorFlow fixes with a more realistic model
"""
import tensorflow as tf
import numpy as np
import sys
import os

# Add project root to sys.path
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(current_script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the calculate_explanation_innvestigate function
from signxai.utils.utils import calculate_explanation_innvestigate

def create_cnn_model():
    """Create a CNN model similar to what's used in the comparison"""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='linear')  # No softmax for XAI
    ])
    return model

def test_cnn_with_innvestigate():
    """Test iNNvestigate with a CNN model"""
    print("Creating CNN model...")
    model = create_cnn_model()
    
    print("Model summary:")
    model.summary()
    
    # Create test input
    x = np.random.random((1, 28, 28, 1)).astype(np.float32)
    print(f"Input shape: {x.shape}")
    
    # Test prediction
    pred = model.predict(x, verbose=0)
    print(f"Prediction shape: {pred.shape}")
    target_class = np.argmax(pred[0])
    print(f"Target class: {target_class}")
    
    # Test gradient-based methods (should work now)
    gradient_methods = ['gradient', 'input_t_gradient', 'guided_backprop', 'deconvnet']
    
    working_methods = []
    failing_methods = []
    
    for method in gradient_methods:
        print(f"\nTesting method: {method}")
        try:
            explanation = calculate_explanation_innvestigate(
                model=model,
                x=x,
                method=method,
                neuron_selection=target_class
            )
            print(f"  SUCCESS: Explanation shape: {explanation.shape}")
            working_methods.append(method)
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            failing_methods.append((method, str(e)))
    
    print(f"\n--- SUMMARY ---")
    print(f"Working methods: {working_methods}")
    print(f"Failing methods: {[m[0] for m in failing_methods]}")
    
    if working_methods:
        print(f"\nTensorFlow gradient-based methods are now WORKING!")
        return True
    else:
        print(f"\nTensorFlow methods still failing")
        return False

if __name__ == "__main__":
    # Disable GPU to avoid compatibility issues
    tf.config.set_visible_devices([], 'GPU')
    success = test_cnn_with_innvestigate()
    if success:
        print("\n✅ TensorFlow compatibility fixes are successful!")
    else:
        print("\n❌ TensorFlow compatibility issues remain")