#!/usr/bin/env python3
"""
Test what attributes are available on TensorFlow layers
"""
import tensorflow as tf
import numpy as np

def test_layer_attributes():
    """Test what attributes are available on layers"""
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
    
    print("\nTesting layer attributes:")
    for i, layer in enumerate(model.layers):
        print(f"\n--- Layer {i}: {layer.name} ({type(layer).__name__}) ---")
        
        # Check various shape-related attributes
        attrs_to_check = [
            'input_shape', 'output_shape', 
            'get_input_shape_at', 'get_output_shape_at',
            'input_spec', 'built', 'batch_input_shape',
            '_build_input_shape'
        ]
        
        for attr in attrs_to_check:
            if hasattr(layer, attr):
                try:
                    value = getattr(layer, attr)
                    if callable(value):
                        try:
                            result = value(0)  # Try with index 0
                            print(f"  {attr}(0): {result}")
                        except:
                            print(f"  {attr}: <callable>")
                    else:
                        print(f"  {attr}: {value}")
                except Exception as e:
                    print(f"  {attr}: ERROR - {e}")
            else:
                print(f"  {attr}: NOT FOUND")
        
        # Check if layer has input/output tensors
        if hasattr(layer, '_inbound_nodes') and layer._inbound_nodes:
            print(f"  _inbound_nodes: {len(layer._inbound_nodes)} nodes")
            try:
                first_node = layer._inbound_nodes[0]
                print(f"  first_node input_tensors: {[t.shape for t in first_node.input_tensors]}")
                print(f"  first_node output_tensors: {[t.shape for t in first_node.output_tensors]}")
            except Exception as e:
                print(f"  node access error: {e}")

if __name__ == "__main__":
    tf.config.set_visible_devices([], 'GPU')
    test_layer_attributes()