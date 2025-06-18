#!/usr/bin/env python3
"""
Test what weight attributes are available on TensorFlow layers
"""
import tensorflow as tf
import numpy as np

def test_weights_attributes():
    """Test what weight attributes are available on layers"""
    # Create a simple model
    inputs = tf.keras.layers.Input(shape=(10,))
    dense1 = tf.keras.layers.Dense(5, activation='relu')(inputs)
    outputs = tf.keras.layers.Dense(3)(dense1)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Test with some data to ensure layers are built
    test_input = np.random.random((1, 10))
    _ = model(test_input)
    
    print("Model summary:")
    model.summary()
    
    print("\nTesting weight attributes:")
    for i, layer in enumerate(model.layers):
        print(f"\n--- Layer {i}: {layer.name} ({type(layer).__name__}) ---")
        
        # Check various weight-related attributes
        weight_attrs_to_check = [
            '_trainable_weights', '_non_trainable_weights',
            'trainable_weights', 'non_trainable_weights',
            'weights', 'trainable_variables', 'non_trainable_variables'
        ]
        
        for attr in weight_attrs_to_check:
            if hasattr(layer, attr):
                try:
                    value = getattr(layer, attr)
                    if isinstance(value, list):
                        print(f"  {attr}: list with {len(value)} items")
                        for j, item in enumerate(value[:3]):  # Show first 3
                            print(f"    [{j}]: {item.shape if hasattr(item, 'shape') else type(item).__name__}")
                    else:
                        print(f"  {attr}: {type(value).__name__}")
                except Exception as e:
                    print(f"  {attr}: ERROR - {e}")
            else:
                print(f"  {attr}: NOT FOUND")

if __name__ == "__main__":
    tf.config.set_visible_devices([], 'GPU')
    test_weights_attributes()