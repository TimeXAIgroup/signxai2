#!/usr/bin/env python3
"""
Simple test to reproduce the InputLayer error in TensorFlow iNNvestigate
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

def create_simple_model():
    """Create a simple TensorFlow model for testing"""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(10,)),
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dense(3, activation='linear')  # No softmax for XAI
    ])
    return model

def test_innvestigate():
    """Test iNNvestigate with a simple model"""
    print("Creating simple model...")
    model = create_simple_model()
    
    print("Model summary:")
    model.summary()
    
    # Create test input
    x = np.random.random((1, 10)).astype(np.float32)
    print(f"Input shape: {x.shape}")
    
    # Test prediction
    pred = model.predict(x, verbose=0)
    print(f"Prediction: {pred}")
    target_class = np.argmax(pred[0])
    print(f"Target class: {target_class}")
    
    # Test various iNNvestigate methods
    methods = ['gradient', 'lrp.epsilon', 'lrp.alpha_1_beta_0']
    
    for method in methods:
        print(f"\nTesting method: {method}")
        try:
            explanation = calculate_explanation_innvestigate(
                model=model,
                x=x,
                method=method,
                neuron_selection=target_class
            )
            print(f"  SUCCESS: Explanation shape: {explanation.shape}")
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Disable GPU to avoid compatibility issues
    tf.config.set_visible_devices([], 'GPU')
    test_innvestigate()