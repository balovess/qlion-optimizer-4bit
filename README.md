# QLion: Quantized Lion Optimizer for PyTorch

[![PyPI Version](https://badge.fury.io/py/qlion.svg)](https://badge.fury.io/py/qlion)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**QLion** is a PyTorch optimizer based on the [Lion optimizer](https://arxiv.org/abs/2302.06675).  Its **key feature is 4-bit quantization of the optimizer state**, designed to enhance training efficiency and reduce memory footprint while maintaining the excellent performance of the Lion optimizer.

## Core Features

*   **4-bit Quantized State:** Compresses the optimizer state (momentum) using 4-bit quantization, significantly reducing memory footprint, especially beneficial for large-scale model training.
*   **Parameter Grouping Configuration:** Supports grouping model parameters and setting independent quantization strategies and hyperparameters for different groups, enabling fine-grained optimization control.
*   **Adaptive Quantization Threshold:** Quantizes only large parameter tensors exceeding a specified element count threshold, balancing quantization benefits and precision loss.
*   **Built-in Gradient Clipping:** Integrates gradient clipping to prevent gradient explosion, enhance training stability, and mitigate the risk of quantization error amplification.
*   **Cautious Update Mode:** An optional cautious update strategy that adjusts the update direction to the gradient sign direction when the gradient and update directions conflict, improving training robustness.
*   **Strict Numerical Stability Protection:** Implemented with a focus on numerical stability, automatically handling all-zero tensors and incorporating minimum scale protection to ensure numerical safety during quantization.
*   **Easy to Use and Extend:** API design conforms to PyTorch optimizer interface specifications, making it easy to integrate into existing PyTorch projects and convenient for further development and feature extension.
*   **Distributed Training Friendly:** Enhanced memory and computational efficiency makes QLion highly suitable for distributed training environments, improving efficiency and scalability.

## Code Structure

*   **`SymmetricQuantizer` Class:** Implements the symmetric quantizer responsible for tensor quantization and dequantization operations.
    *   **Features:**
        *   Symmetric Quantization: Uses a symmetric quantization scheme with a quantization range of `[-2^(bits-1) + 1, 2^(bits-1) - 1]`.
        *   Automatic Handling of All-Zero Tensors: Returns zero tensors and scale=0.0 directly when the input tensor is all zeros, avoiding quantization errors.
        *   Error Compensation Mechanism (Optional): [Planned for future versions to further enhance quantization accuracy]
        *   Minimum Scale Protection: Ensures the scale value is no less than `min_scale`, preventing division-by-zero errors and numerical instability.
    *   **Methods:**
        *   `quantize(tensor: torch.Tensor) -> Tuple[torch.Tensor, float]`: Performs quantization, returning the quantized `int8` tensor and the scale value.
        *   `dequantize(quantized: torch.Tensor, scale: float) -> torch.Tensor`: Performs dequantization, restoring the quantized tensor to a floating-point tensor.

*   **`QLion` Class:** Implements the QLion optimizer, inheriting from `torch.optim.Optimizer`.
    *   **`__init__` Method (Initialization):**
        *   Parameter Validation: Validates the effectiveness of parameters such as learning rate `lr`, momentum coefficients `betas`, weight decay coefficient `weight_decay`, etc.
        *   Default Parameter Settings: Sets default hyperparameters for the optimizer, including learning rate, momentum coefficients, weight decay, quantization enable, cautious update mode, quantization threshold, gradient clipping threshold, etc.
        *   Quantizer Initialization: Creates a `SymmetricQuantizer` instance. Each `QLion` optimizer instance has an independent quantizer.
        *   State Consistency Check: Performs a state data consistency check at initialization to prevent quantized state errors.
    *   **`_check_consistency()` Method:**
        *   State Data Consistency Check: Iterates through parameter groups, checking if `quant_exp_avg` and `scale` in the optimizer state appear in pairs to prevent quantized state data corruption.
    *   **`_apply_quantization(p: torch.Tensor, state: dict, group: dict)` Method:**
        *   Applies Quantization to Optimizer State: Based on parameter group configurations (`use_4bit`, `quant_threshold`), quantizes the momentum `exp_avg` of parameter `p` that meets the quantization conditions. Saves the quantized tensor and scale value to the optimizer state `state`, and releases memory of the original `exp_avg` tensor.
    *   **`_prepare_update(p: torch.Tensor, state: dict, group: dict) -> torch.Tensor` Method:**
        *   Prepares Update Amount (Handles Quantized State): Before calculating the parameter update amount, dequantizes the quantized state, restores `exp_avg` to a floating-point tensor, then performs the momentum update formula of the Lion optimizer, and calls `_apply_quantization` again to quantize the updated `exp_avg` (delayed until the next step).
        *   Lion Core Update Formula: `update = exp_avg * beta1 + grad * (1 - beta1)`
        *   Sign Update: Returns the sign of the update amount `update.sign()`, which is a core feature of the Lion optimizer.
    *   **`_apply_cautious_update(update: torch.Tensor, grad: torch.Tensor)` Method:**
        *   Cautious Update Strategy: When `cautious` mode is enabled, if the update direction conflicts with the gradient direction (`(update * grad) < 0`), the update direction is adjusted to the gradient sign direction to improve training robustness.
    *   **`step(closure: Optional[Callable] = None) -> Optional[float]` Method:**
        *   Performs a Single Optimization Step: Iterates through parameter groups and performs the following operations for each parameter:
            *   Gradient Clipping: If `grad_clip > 0`, gradient clipping is performed to prevent gradient explosion.
            *   State Initialization: If `exp_avg` is not in the optimizer state, it is initialized as a zero tensor.
            *   Prepare Update Amount: Calls the `_prepare_update` method to calculate the parameter update amount.
            *   Apply Cautious Update: If `cautious=True`, calls the `_apply_cautious_update` method to apply the cautious update strategy.
            *   Weight Decay: If `weight_decay > 0`, applies AdamW-style weight decay.
            *   Parameter Update: Updates model parameter `p` based on the calculated update amount and learning rate.
        *   Returns the loss value (if a closure is provided).
    *   **`__repr__() -> str` Method:**
        *   Optimizer Information Printing: Prints the configuration information of the optimizer, including the learning rate and whether quantization is enabled.

## How to Use

1.  **Installation:**

    ```bash
    # Not yet released to PyPI, please download the qlion.py file directly for use.
    # Future versions may be released to PyPI.
    ```

2.  **Import `QLion` Optimizer:**

    ```python
    from qlion import QLion
    import torch.nn as nn
    import torch.optim as optim
    ```

3.  **Configure Model and Data (Example):**

    ```python
    model = nn.Linear(10, 2)
    x = torch.randn(32, 10)
    y = torch.randn(32, 2)
    loss_fn = nn.MSELoss()
    ```

4.  **Initialize `QLion` Optimizer:**

    *   **Basic Usage:**

        ```python
        optimizer = QLion(model.parameters(), lr=1e-4)
        ```

    *   **Enable 4-bit Quantization:**

        ```python
        optimizer = QLion(model.parameters(), lr=1e-4, use_4bit=True)
        ```

    *   **Grouped Parameter Configuration (Example):** Apply different quantization strategies or hyperparameters to different parts of the model's parameters.

        ```python
        optimizer = QLion([
            {'params': model.weight, 'use_4bit': True, 'lr': 1e-3},  # Enable quantization for weights, using a different learning rate
            {'params': model.bias,   'use_4bit': False}             # Do not quantize biases, using the default learning rate
        ], lr=1e-4, weight_decay=0.1) # Default learning rate and weight decay will apply to all parameter groups
        ```

5.  **Training Loop:**

    ```python
    for epoch in range(100):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    ```

## Experiments and Chart Interpretation

The code includes a `detailed_test()` function for more detailed testing and performance monitoring. Running `detailed_test()` will generate three charts to help analyze the training behavior and engineering characteristics of the QLion optimizer:

1.  **Training Loss:** Shows the change in loss value during training. Decreasing loss indicates model learning. A rapid decrease and stabilization usually signify convergence.


2.  **Quantization Coverage:** Shows the proportion of parameters being quantized during training. A high and stable quantization coverage (close to 1.0) indicates that the quantization technique is effectively applied.

    

3.  **Gradient vs Update Norms:** Compares the changing trends of Gradient Norm and Update Norm. An extremely low Update Norm is a significant characteristic of the Lion optimizer (and QLion), potentially offering memory efficiency and training stability advantages.

  
**Key Points for Chart Interpretation:**

*   **Training Loss:** Focus on the decreasing trend and convergence speed of the loss value.
*   **Quantization Coverage:** Focus on the quantization ratio and stability to verify the effective application of the quantization technique.
*   **Gradient vs Update Norms:** Focus on the magnitude of the Update Norm and the trend of both norms to analyze the optimizer's behavior and engineering characteristics. An extremely low Update Norm is one of QLion's engineering advantages.

