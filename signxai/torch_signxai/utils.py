# signxai/torch_signxai/utils.py

import torch
import torch.nn as nn
import json
import os
from typing import Union, Type, List, Optional, Dict, Any, Tuple


# 1. Corrected top-level remove_softmax
def remove_softmax(model: nn.Module) -> nn.Module:
    """
    Removes the softmax layer from a PyTorch model if it's the last one
    in model.classifier or a common sequential structure.
    This version modifies the model in-place if a Softmax layer is found.
    For models like the custom VGG16_PyTorch that don't have an explicit Softmax layer
    but output logits, this function will effectively be a no-op regarding layer removal,
    which is correct.
    """
    classifier_attr_names = ['classifier', 'fc', 'output_layer']
    modified = False

    for attr_name in classifier_attr_names:
        if hasattr(model, attr_name):
            classifier_block = getattr(model, attr_name)
            if isinstance(classifier_block, nn.Sequential):
                if len(classifier_block) > 0 and isinstance(classifier_block[-1], (nn.Softmax, nn.LogSoftmax)):
                    new_classifier_block = nn.Sequential(*list(classifier_block.children())[:-1])
                    setattr(model, attr_name, new_classifier_block)
                    modified = True
                    break

    if not modified:
        if isinstance(model, nn.Sequential) and len(model) > 0 and isinstance(model[-1], (nn.Softmax, nn.LogSoftmax)):
            pass  # In-place modification of top-level Sequential is more complex and usually handled by reassignment

    return model


# 2. NoSoftmaxWrapper class
class NoSoftmaxWrapper(nn.Module):
    """Wrapper class that removes softmax from a PyTorch model."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        if isinstance(output, tuple):
            return output[0]
        return output


# 3. Corrected top-level decode_predictions
def decode_predictions(preds: torch.Tensor,
                       top: int = 5,
                       class_list_path: Optional[str] = None
                       ) -> List[List[Tuple[str, str, float]]]:
    """Decodes the prediction of an ImageNet model."""
    labels_loaded = False
    labels: Dict[str, Tuple[str, str]] = {}

    if class_list_path is None:
        try:
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            default_path = os.path.join(current_script_dir, "..", "..", "data", "imagenet_class_index.json")

            if os.path.exists(default_path):
                class_list_path = default_path
            else:
                try:
                    import signxai
                    package_root = os.path.dirname(signxai.__file__)
                    class_list_path = os.path.join(package_root, "data", "imagenet_class_index.json")
                except ImportError:
                    pass
            if not class_list_path or not os.path.exists(class_list_path):
                raise FileNotFoundError(f"Default class index not found. Checked: {default_path}" +
                                        (f" and package path." if 'package_root' in locals() else "."))
        except Exception as e:
            print(f"Warning: Could not determine default ImageNet class index path. {e}")

    if class_list_path and os.path.exists(class_list_path):
        try:
            with open(class_list_path) as f:
                idx2label_data = json.load(f)
                labels = {idx_str: (data_tuple[0], data_tuple[1]) for idx_str, data_tuple in idx2label_data.items()}
                labels_loaded = True
        except Exception as e:
            print(f"Error loading ImageNet class index from {class_list_path}: {e}.")

    if not labels_loaded:
        print(f"Warning: ImageNet class index not loaded. Predictions will be generic.")
        num_classes = preds.shape[-1]
        labels = {str(i): (f"idx_{i}", f"Class_{i}") for i in range(num_classes)}

    if preds.ndim == 1:
        preds = preds.unsqueeze(0)

    sum_preds = preds.sum(dim=-1)
    is_logits = not (
            (preds.min() >= 0.0) and
            (preds.max() <= 1.0) and
            torch.all(torch.isclose(sum_preds, torch.ones_like(sum_preds)))
    )

    if is_logits:
        preds = torch.softmax(preds, dim=-1)

    top_probs, top_idxs = torch.topk(preds, top, dim=-1)

    final_output: List[List[Tuple[str, str, float]]] = []
    for i in range(preds.shape[0]):
        sample_predictions: List[Tuple[str, str, float]] = []
        for j in range(top):
            class_idx_tensor = top_idxs[i, j]
            probability_tensor = top_probs[i, j]
            class_idx_str = str(class_idx_tensor.item())
            probability_float = probability_tensor.item()
            imagenet_id_str, class_name_str = labels.get(class_idx_str, (class_idx_str, "UnknownClassName"))
            sample_predictions.append((imagenet_id_str, class_name_str, probability_float))
        final_output.append(sample_predictions)

    return final_output