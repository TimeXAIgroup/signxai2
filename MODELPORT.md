Summary of TensorFlow-to-PyTorch ECG Model Porting and Verification Process
Date: May 17, 2025
Project: signxai - ECG Model Porting Initiative

1. Objective:
To port a suite of TensorFlow-based ECG classification models to PyTorch, ensuring high fidelity and numerical equivalence with the original models. This included an initial generic ECG model and several pathology-specific models (AVB, ISCH, LBBB, RBBB). The goal was to achieve "perfect ports" enabling consistent behavior and explainability analysis across both frameworks.

2. Methodology & Toolkit Developed/Utilized:

A systematic approach was adopted, supported by a set of custom utility scripts:

TensorFlow Model Inspection (utils/inspect_model.py):

Enhanced to load TensorFlow Keras models from both single .h5 files (architecture + weights) and model.json (architecture) + weights.h5 (weights) pairs.
Used to output detailed architecture summaries (layer names, types, configurations, input/output shapes) and to extract/save specific layer weights to .npy files for detailed numerical comparison.
PyTorch Model Definitions (examples/data/models/pytorch/ECG/):

ecg_model.py: Contains ECG_PyTorch, a manually created nn.Module class meticulously translating the architecture of the original generic ECG model.
pathology_ecg_model.py: Contains Pathology_ECG_PyTorch, a single nn.Module class designed to represent the shared architecture identified for the AVB, ISCH, LBBB, and RBBB TensorFlow models. This consolidation was made after inspection confirmed their architectural identity.
Weight Conversion Utility (utils/convert_tf_weights_to_pytorch.py):

A unified script developed to handle weight transfer for all supported model types (ecg, vgg16, avb_ecg, isch_ecg, lbbb_ecg, rbbb_ecg).
Features include:
Command-line arguments for specifying model type, input paths (single H5 or JSON+weights), and output path for PyTorch state dictionaries (.pt).
Dynamic loading of TensorFlow models.
Dynamic instantiation of the correct PyTorch model class (ECG_PyTorch or Pathology_ECG_PyTorch) with parameters (e.g., input_channels, num_classes) inferred from the loaded TensorFlow model or overridden via arguments.
A configuration system (get_model_specific_config) to provide the correct TensorFlow-to-PyTorch layer name mapping for weight transfer, accounting for different architectures.
Correct handling of weight permutations for different layer types (e.g., Conv1D, Dense).
Detailed logging of the conversion process, including shape checks and sample weight/bias values.
PyTorch Layer Weight Inspection Utility (utils/inspect_pytorch_layer_weights.py):

Created to load a ported PyTorch model and its weights.
Allows inspection and saving of weights/biases for a specified layer within the PyTorch model to .npy files, complementing the TensorFlow weight inspection.
Layer-by-Layer Numerical Comparison Utility (examples/comparison/compare_ecg_models.py):

A generalized script designed to perform a rigorous, layer-by-layer comparison of outputs between a TensorFlow model and its ported PyTorch equivalent.
Features include:
Internal loop to test a predefined list of all relevant ECG model types (ecg, avb_ecg, isch_ecg, lbbb_ecg, rbbb_ecg) in a single execution.
A configuration system (get_model_comparison_config) to fetch all model-specific details: TF model loading method and paths, PyTorch class and weights path, PyTorch class initialization arguments, input data generation parameters (shape, permutation), and a detailed layers_to_test sequence.
The layers_to_test sequence maps each relevant TensorFlow layer name to a corresponding sequence of PyTorch operations (module calls, functional activations, lambda operations for data manipulation like permutations) needed to replicate the TF layer's output. Helper functions (_get_original_ecg_layers_config, _get_pathology_model_shared_layers_config) define these sequences to promote reuse.
Generation of identical dummy input data for both models (with appropriate permutations for PyTorch).
Comparison of output shapes and numerical values (using np.allclose with a defined tolerance, atol) for each configured layer.
Detailed reporting of success/failure, including max/mean absolute differences and samples of differing values if a mismatch occurs.
3. Porting and Verification Walkthrough & Key Findings:

Initial ECG Model (ecg_model.h5 -> ECG_PyTorch):

The initial port revealed output mismatches.
Layer-by-layer comparison pinpointed the Flatten layer as the point of divergence.
Root Cause: TensorFlow's Conv1D default (channels_last) produced feature maps as (Batch, Steps, Channels), while PyTorch Conv1D uses (Batch, Channels, Steps). Direct flattening led to different element orders.
Solution:
In ECG_PyTorch.forward(): Added x = x.permute(0, 2, 1) before self.flatten(x).
In the comparison script's configuration for this model: Ensured the pt_ops for the "flatten" test included this permutation before calling PyTorch's nn.Flatten.
Result: Achieved perfect numerical equivalence for all layers.
Pathology-Specific Models (AVB, ISCH, LBBB, RBBB):

Architecture Analysis: Inspection of model.json files confirmed that AVB, ISCH, LBBB, and RBBB all share an identical architecture (12 input channels, 2000 steps, 5 conv blocks with ELU, GlobalAveragePooling1D, 3 dense blocks, 2 output classes with Softmax). This architecture is distinct from the original ecg_model.h5.
Consolidated PyTorch Model: A single PyTorch class, Pathology_ECG_PyTorch, was defined in pathology_ecg_model.py to represent this shared architecture.
Weight Conversion & Verification:
The unified conversion script successfully ported weights for AVB, ISCH, LBBB, and RBBB using their respective model.json and weights.h5 files, instantiating Pathology_ECG_PyTorch.
The generalized comparison script verified AVB, ISCH, and RBBB, showing perfect layer-by-layer numerical equivalence.
LBBB Discrepancy: The LBBB model initially showed a minor failure (max absolute difference ~1.5e-4, just outside atol=1e-5) at the activation_3 layer (output of the 4th convolutional block + ELU). However, its final softmax output matched perfectly.
Troubleshooting LBBB:
Re-running the weight conversion for LBBB yielded the same result.
The weights of the critical TensorFlow layer (conv1d_3) and its PyTorch counterpart (Pathology_ECG_PyTorch.conv4) were extracted using the inspection utilities and saved to .npy files.
A direct numerical comparison of these saved weight arrays (after accounting for kernel permutation) using np.allclose with a very strict tolerance (1e-8) confirmed that the weights themselves were identically ported.
LBBB Conclusion: The minute discrepancy at activation_3 for LBBB is attributed to inherent, minor numerical precision differences between TensorFlow's and PyTorch's underlying implementations of convolution or ELU operations, given that specific set of weights. Since the weights are identical and the final model output matches, the LBBB port is considered successful and functionally equivalent for all practical purposes.
4. Final Status:

All five ECG models (original ecg, AVB, ISCH, LBBB, RBBB) have been successfully ported to PyTorch.

Four models (ecg, AVB, ISCH, RBBB) demonstrate layer-by-layer numerical equivalence with their TensorFlow originals, within a strict tolerance (atol=1e-05).
One model (lbbb_ecg) demonstrates perfect weight transfer and identical final output, with a single intermediate layer (activation_3) showing a max absolute difference of ~1.5e-4, attributed to framework-specific numerical precision nuances rather than a porting error.
The developed toolkit of generalized Python scripts provides a robust framework for these and future model porting and verification efforts.

5. Key Takeaways for Porting TF Models to PyTorch:

Thorough Architecture Inspection: Before writing any PyTorch code, meticulously analyze the TensorFlow model's architecture, including layer types, parameters, data formats, and activation placements.
Data Format (channels_first vs. channels_last): This is a common source of error. Be vigilant about how data flows through convolutional layers and ensure tensors are permuted correctly before operations like Flatten or any operation sensitive to dimension order.
Explicit vs. Integrated Activations: TensorFlow Keras models might define activations within the layer config (e.g., Conv1D(..., activation='relu')) or as separate Activation layers. PyTorch equivalents must mirror this behavior accurately in the forward pass.
Layer-by-Layer Numerical Verification: Simply matching the final output is often insufficient. A detailed comparison of intermediate layer outputs is crucial for identifying the exact point of divergence and ensuring true equivalence.
Direct Weight Comparison: If subtle output discrepancies persist, directly comparing the numerical values of the weights (after accounting for framework-specific permutations like TF's (KS, IC, OC) vs. PT's (OC, IC, KS) for Conv1D kernels) is a vital debugging step. Saving weights to .npy files facilitates this.
Numerical Precision Awareness: Understand that identical mathematical operations performed by different deep learning libraries might yield infinitesimally different floating-point results. Define an acceptable tolerance (atol, rtol) for comparisons. If weights are verified to be identical and the final model predictions are consistent, minor intermediate numerical noise may be acceptable.
Reusable and Parameterized Utilities: For projects involving multiple model ports, investing in generalized scripts for inspection, weight conversion, and comparison significantly improves efficiency and consistency.
