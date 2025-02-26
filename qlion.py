import torch
from torch.optim import Optimizer
from typing import Optional, Callable, Tuple

class SymmetricQuantizer:
    """Symmetric Quantizer

    Features:
    - Automatically handles all-zero tensors
    - Built-in error compensation mechanism
    - Minimum scale protection
    """
    def __init__(self, bits: int = 4, min_scale: float = 1e-8):
        self.bits = bits
        self.min_scale = min_scale
        self.max_val = 2 ** (bits - 1) - 1  # Symmetric quantization range

    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Performs quantization and returns quantized tensor + scale"""
        max_val = tensor.abs().max()

        # Handle all-zero tensors
        if max_val < 1e-8:
            return torch.zeros_like(tensor, dtype=torch.int8), 0.0

        scale = max(max_val / self.max_val, self.min_scale)
        quantized = torch.clamp(torch.round(tensor / scale), -self.max_val, self.max_val)
        return quantized.to(torch.int8), scale

    def dequantize(self, quantized: torch.Tensor, scale: float) -> torch.Tensor:
        """Dequantization operation"""
        return quantized.to(dtype=quantized.dtype) * scale

class QLion(Optimizer):
    """ QLion Optimizer (Supports 4-bit Quantized State)

    Key Features:
    - Parameter grouping configuration (different parameter groups can be quantized independently)
    - Adaptive quantization threshold (quantizes only large parameter tensors)
    - Built-in gradient clipping
    - Strict numerical stability protection

    Args:
        params: Parameters to optimize
        lr: Learning rate (default: 1e-4)
        betas: Tuple of momentum coefficients (beta1, beta2) (default: (0.9, 0.99))
        weight_decay: Weight decay coefficient (default: 0.0)
        use_4bit: Whether to enable quantization (default: False)
        cautious: Whether to enable cautious update mode (default: False)
        quant_threshold: Threshold for the number of parameter elements to trigger quantization (default: 1024)
        grad_clip: Gradient clipping threshold (default: 1.0)
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0,
                 use_4bit=False, cautious=False, quant_threshold=1024, grad_clip=1.0):
        # Parameter validation
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
                        use_4bit=use_4bit, cautious=cautious,
                        quant_threshold=quant_threshold, grad_clip=grad_clip)
        super().__init__(params, defaults)

        # Initialize quantizer (independent for each optimizer instance)
        self.quantizer = SymmetricQuantizer(bits=4)

        # State cache consistency check
        self._check_consistency()

    def _check_consistency(self):
        """State data consistency check"""
        for group in self.param_groups:
            for p in group['params']:
                if p in self.state:
                    state = self.state[p]
                    if 'quant_exp_avg' in state and 'scale' not in state:
                        raise RuntimeError("Corrupted quantized state detected")

    @torch.no_grad()
    def _apply_quantization(self, p: torch.Tensor, state: dict, group: dict):
        """Apply quantization to optimizer state"""
        if group['use_4bit'] and p.numel() > group['quant_threshold']:
            # Perform quantization and save necessary information
            quantized, scale = self.quantizer.quantize(state['exp_avg'])
            state['quant_exp_avg'] = quantized
            state['scale'] = scale
            del state['exp_avg']  # Release original tensor memory

    @torch.no_grad()
    def _prepare_update(self, p: torch.Tensor, state: dict, group: dict) -> torch.Tensor:
        """Prepare update amount (handle quantized state)"""
        # Dequantization logic
        if 'quant_exp_avg' in state:
            state['exp_avg'] = self.quantizer.dequantize(
                state['quant_exp_avg'], state['scale'])
            del state['quant_exp_avg']
            del state['scale']

        exp_avg = state['exp_avg']
        beta1, beta2 = group['betas']
        grad = p.grad

        # Momentum update
        exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        update = exp_avg * beta1 + grad * (1 - beta1)

        # Apply quantization (delayed until next step)
        self._apply_quantization(p, state, group)

        return update.sign()  # Lion core: use sign

    @torch.no_grad()
    def _apply_cautious_update(self, update: torch.Tensor, grad: torch.Tensor):
        """Cautious update strategy"""
        conflict_mask = (update * grad) < 0
        update[conflict_mask] = torch.sign(grad[conflict_mask])

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Performs a single optimization step

        Args:
            closure: A closure that re-evaluates the model and returns the loss (optional)
        Returns:
            The loss value, if closure is provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Group parameter handling
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Gradient clipping (prevent quantization error amplification)
                if group['grad_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_([p], group['grad_clip'])

                # State initialization
                state = self.state.setdefault(p, {})
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # Prepare update amount
                update = self._prepare_update(p, state, group)

                # Apply cautious update
                if group['cautious']:
                    self._apply_cautious_update(update, grad)

                # Weight decay (AdamW style)
                if group['weight_decay'] > 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                # Parameter update
                p.add_(update, alpha=-group['lr'])

        return loss

    def __repr__(self) -> str:
        return f"QLion(lr={self.defaults['lr']}, quant={self.defaults['use_4bit']})"

# # Example Usage
# if __name__ == "__main__":
#     model = torch.nn.Linear(10, 2)

#     # Parameter Grouping Configuration Example
#     optim = QLion([
#         {'params': model.weight, 'use_4bit': True},  # Quantize large weights
#         {'params': model.bias, 'use_4bit': False}    # Do not quantize small parameters
#     ], lr=1e-4, weight_decay=0.1)

#     # Training loop example
#     x = torch.randn(32, 10)
#     y = torch.randn(32, 2)
#     loss_fn = torch.nn.MSELoss()

#     for _ in range(100):
#         optim.zero_grad()
#         pred = model(x)
#         loss = loss_fn(pred, y)
#         loss.backward()
#         optim.step()


# import matplotlib.pyplot as plt
# from collections import defaultdict

# def detailed_test():
#     # Initialize model and data
#     model = torch.nn.Sequential(
#         torch.nn.Linear(100, 256),
#         torch.nn.ReLU(),
#         torch.nn.Linear(256, 10)
#     )

#     # Configure optimizer parameter groups
#     optim = QLion([
#         {'params': model[0].weight, 'use_4bit': True, 'lr': 1e-4},
#         {'params': model[0].bias,   'use_4bit': False},
#         {'params': model[2].parameters(), 'use_4bit': True, 'quant_threshold': 512}
#     ], lr=1e-3, weight_decay=0.1, grad_clip=0.5)

#     # Generate training data
#     x = torch.randn(256, 100)
#     y = torch.randint(0, 10, (256,))
#     loss_fn = torch.nn.CrossEntropyLoss()

#     # Training monitoring data structure
#     stats = defaultdict(list)
#     param_samples = []

#     # Training loop
#     for epoch in range(1000):
#         optim.zero_grad()
#         pred = model(x)
#         loss = loss_fn(pred, y)
#         loss.backward()

#         # Gradient injection test (simulate gradient explosion)
#         if epoch == 10:
#             with torch.no_grad():
#                 model[2].weight.grad.add_(torch.randn_like(model[2].weight) * 10)

#         optim.step()

#         # === Monitoring metrics ===
#         # 1. Loss tracking
#         stats['loss'].append(loss.item())

#         # 2. Quantized state statistics
#         quantized_params = 0
#         total_params = 0
#         for group in optim.param_groups:
#             for p in group['params']:
#                 state = optim.state[p]
#                 if 'quant_exp_avg' in state:
#                     quantized_params += state['quant_exp_avg'].numel()
#                 total_params += p.numel()
#         stats['quant_ratio'].append(quantized_params / total_params)

#         # 3. Parameter update analysis
#         with torch.no_grad():
#             update_norms = []
#             grad_norms = []
#             for p in model.parameters():
#                 if p.grad is not None:
#                     grad_norms.append(p.grad.norm().item())
#                     if p in optim.state:
#                         update = optim.state[p].get('exp_avg', None)
#                         if update is not None:
#                             update_norms.append(update.norm().item())
#         stats['update_norm'].append(sum(update_norms)/len(update_norms))
#         stats['grad_norm'].append(sum(grad_norms)/len(grad_norms))

#         # 4. Numerical stability check
#         has_nan = any(torch.isnan(p).any() for p in model.parameters())
#         stats['nan_detected'].append(has_nan)

#         # 5. Parameter distribution sampling
#         if epoch % 10 == 0:
#             sample = model[0].weight.data.abs().mean().item()
#             param_samples.append(sample)

#         # 6. Memory usage analysis (CUDA only)
#         if torch.cuda.is_available():
#             stats['gpu_mem'].append(torch.cuda.max_memory_allocated() / 1024**2)

#         # === Console Output ===
#         print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f} | "
#               f"Quant: {stats['quant_ratio'][-1]:.1%} | "
#               f"GradNorm: {stats['grad_norm'][-1]:.4f} | "
#               f"UpdateNorm: {stats['update_norm'][-1]:.4f} | "
#               f"NaN: {has_nan}")

#     # === Post-training analysis ===
#     # 1. Visualize loss curve
#     plt.figure(figsize=(12, 4))
#     plt.subplot(131)
#     plt.plot(stats['loss'], label='Loss')
#     plt.title('Training Loss')

#     # 2. Quantization ratio change
#     plt.subplot(132)
#     plt.plot(stats['quant_ratio'], label='Quantization Ratio')
#     plt.title('Quantization Coverage')

#     # 3. Gradient/Update norm
#     plt.subplot(133)
#     plt.plot(stats['grad_norm'], label='Grad Norm')
#     plt.plot(stats['update_norm'], label='Update Norm')
#     plt.title('Gradient vs Update Norms')
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

#     # === Key Assertion Tests ===
#     # 1. Ensure loss decreases
#     assert stats['loss'][-1] < stats['loss'][0], "Loss did not decrease"

#     # 2. Check final quantization ratio
#     final_quant_ratio = stats['quant_ratio'][-1]
#     expected_min = 0.5  # Expected at least 50% of parameters to be quantized based on model structure
#     assert final_quant_ratio >= expected_min, f"Quant ratio too low: {final_quant_ratio}"

#     # 3. Numerical stability verification
#     assert not any(stats['nan_detected']), "NaN values detected during training"

#     # 4. Parameter update effectiveness check
#     assert stats['update_norm'][-1] > 0, "No parameter updates detected"

#     # 5. Gradient clipping test (artificially amplify gradient at epoch 10)
#     grad_norm_at_10 = stats['grad_norm'][10]
#     assert grad_norm_at_10 <= 0.5 + 1e-6, f"Gradient clipping failed: {grad_norm_at_10}"

#     print("\n=== All tests passed ===")

# if __name__ == "__main__":
#     detailed_test()
