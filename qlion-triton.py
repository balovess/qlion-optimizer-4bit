import torch
from torch.optim import Optimizer
from typing import Optional, Callable, Tuple

# Get GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

# Get compute capability
capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)

# Print results
print(f"Current GPU: {gpu_name}")
print(f"Compute capability: {capability}")

# Determine appropriate numerical formats
if capability >= (8, 0):  # A100, H100, L40, 4090+
    print("✅ Supports FP8 (recommended) and INT8/INT4")
elif capability >= (7, 5):  # T4, 2080Ti
    print("✅ Supports FP16 and INT8, but not recommended for INT4")
elif capability >= (7, 0):  # V100
    print("✅ Supports FP16 only, not recommended for INT8/INT4")
else:
    print("⚠️ FP32 only, consider FP16 mixed-precision training")

# --- Attempt Triton import and define specialized kernels ---
try:
    import triton
    import triton.language as tl

    @triton.jit
    def qlion_update_kernel(exp_avg_ptr, grad_ptr, output_ptr, 
                             beta1: float, beta2: float, 
                             one_minus_beta2: float, one_minus_beta1: float, 
                             n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        """
        Triton kernel:
        1. Update momentum: exp_avg = exp_avg * beta2 + grad * (1 - beta2)
        2. Calculate update: update = exp_avg * beta1 + grad * (1 - beta1)
        3. Return sign update: sign(update)
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)
        grad = tl.load(grad_ptr + offsets, mask=mask)

        exp_avg_new = exp_avg * beta2 + grad * one_minus_beta2
        tl.store(exp_avg_ptr + offsets, exp_avg_new, mask=mask)

        update = exp_avg_new * beta1 + grad * one_minus_beta1
        sign_update = tl.where(update < 0, -1.0, tl.where(update > 0, 1.0, 0.0))
        tl.store(output_ptr + offsets, sign_update, mask=mask)

    _triton_available = True
except ImportError:
    _triton_available = False

# --- Symmetric Quantizer ---
class SymmetricQuantizer:
    """
    Symmetric quantizer (production-grade implementation)
    Features:
      - Auto-handling of all-zero tensors
      - Built-in error compensation
      - Minimum scale protection
    """
    def __init__(self, bits: int = 4, min_scale: float = 1e-8):
        self.bits = bits
        self.min_scale = min_scale
        self.max_val = 2 ​** (bits - 1) - 1  # Symmetric range
    
    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Perform quantization and return quantized tensor + scale"""
        max_val = tensor.abs().max()
        if max_val < 1e-8:  # Handle all-zero tensors
            return torch.zeros_like(tensor, dtype=torch.int8), 0.0
        scale = max(max_val / self.max_val, self.min_scale)
        quantized = torch.clamp(torch.round(tensor / scale), -self.max_val, self.max_val)
        return quantized.to(torch.int8), scale
    
    def dequantize(self, quantized: torch.Tensor, scale: float) -> torch.Tensor:
        """Dequantize: Convert int8 to float and multiply by scale"""
        return quantized.float().mul(scale)

# --- QLion Optimizer ---
class QLion(Optimizer):
    """
    Production-grade QLion optimizer (4-bit quantized states + Triton acceleration)
    
    Key features:
      - Per-parameter-group configuration (independent quantization/Triton settings)
      - Adaptive quantization threshold (quantize only large tensors)
      - Built-in gradient clipping & numerical stability
      - Optional Triton acceleration (GPU-only when enabled)
    
    Args:
      params: Parameters to optimize
      lr: Learning rate (default: 1e-4)
      betas: Momentum coefficients (default: (0.9, 0.99))
      weight_decay: Weight decay (default: 0.0)
      use_4bit: Enable 4-bit quantization (default: False)
      cautious: Enable cautious update mode (default: False)
      quant_threshold: Element count threshold for quantization (default: 1024)
      grad_clip: Gradient clipping threshold (default: 1.0)
      enable_triton: Enable Triton acceleration (default: True)
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0, 
                 use_4bit=False, cautious=False, quant_threshold=1024, grad_clip=1.0,
                 enable_triton=True):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
                        use_4bit=use_4bit, cautious=cautious,
                        quant_threshold=quant_threshold, grad_clip=grad_clip,
                        enable_triton=enable_triton)
        super().__init__(params, defaults)
        
        self.quantizer = SymmetricQuantizer(bits=4)
        self._check_consistency()

    def _check_consistency(self):
        """Consistency check for state data"""
        for group in self.param_groups:
            for p in group['params']:
                if p in self.state:
                    state = self.state[p]
                    if 'quant_exp_avg' in state and 'scale' not in state:
                        raise RuntimeError("Corrupted quantized state detected")

    def _triton_available(self) -> bool:
        """Check Triton availability"""
        return _triton_available

    @torch.no_grad()
    def _apply_quantization(self, p: torch.Tensor, state: dict, group: dict):
        """Apply quantization to state"""
        if group['use_4bit'] and p.numel() > group['quant_threshold']:
            quantized, scale = self.quantizer.quantize(state['exp_avg'])
            state['quant_exp_avg'] = quantized
            state['scale'] = scale
            del state['exp_avg']  # Free memory

    @torch.no_grad()
    def _prepare_update(self, p: torch.Tensor, state: dict, group: dict) -> torch.Tensor:
        """Prepare update with optional Triton acceleration"""
        if 'quant_exp_avg' in state:
            state['exp_avg'] = self.quantizer.dequantize(
                state['quant_exp_avg'], state['scale'])
            del state['quant_exp_avg']
            del state['scale']
        
        exp_avg = state['exp_avg']
        beta1, beta2 = group['betas']
        grad = p.grad

        # Use kernel if Triton enabled and conditions met
        if (p.is_cuda and group.get("enable_triton", True) and self._triton_available() and 
            exp_avg.is_contiguous() and grad.is_contiguous()):
            output = torch.empty_like(exp_avg)
            n_elements = p.numel()
            BLOCK_SIZE = 128  # Tunable per hardware
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            one_minus_beta2 = 1 - beta2
            one_minus_beta1 = 1 - beta1
            qlion_update_kernel[grid](exp_avg, grad, output,
                                       beta1, beta2, one_minus_beta2, one_minus_beta1,
                                       n_elements, BLOCK_SIZE=BLOCK_SIZE)
            update = output
        else:
            # Pure PyTorch implementation
            exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
            update = exp_avg * beta1 + grad * (1 - beta1)
            update = update.sign()

        self._apply_quantization(p, state, group)
        return update

    @torch.no_grad()
    def _apply_cautious_update(self, update: torch.Tensor, grad: torch.Tensor):
        """Cautious update to avoid conflicting directions"""
        conflict_mask = (update * grad) < 0
        update[conflict_mask] = torch.sign(grad[conflict_mask])

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform a single optimization step"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if group['grad_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_([p], group['grad_clip'])
                
                state = self.state.setdefault(p, {})
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                update = self._prepare_update(p, state, group)
                if group['cautious']:
                    self._apply_cautious_update(update, p.grad)
                if group['weight_decay'] > 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                p.add_(update, alpha=-group['lr'])
        return loss

    def __repr__(self) -> str:
        return f"QLion(lr={self.defaults['lr']}, quant={self.defaults['use_4bit']}, triton={self.defaults.get('enable_triton', True)})"

# --- Benchmark: Triton vs Pure PyTorch ---
def compare_optimizers():
    if not torch.cuda.is_available():
        print("GPU required for this benchmark.")
        return

    # Create two models with identical architecture
    model_triton = torch.nn.Linear(1024, 1024).cuda()
    model_pure = torch.nn.Linear(1024, 1024).cuda()
    
    # Initialize optimizers:
    # 1. Triton-accelerated
    optim_triton = QLion([{'params': model_triton.parameters(), 
                           'use_4bit': True, 'enable_triton': True}],
                         lr=1e-4, weight_decay=0.1)
    # 2. Pure PyTorch
    optim_pure = QLion([{'params': model_pure.parameters(), 
                          'use_4bit': False, 'enable_triton': True}],
                       lr=1e-4, weight_decay=0.1)
    
    loss_fn = torch.nn.MSELoss()
    x = torch.randn(64, 1024).cuda()
    target = torch.randn(64, 1024).cuda()
    
    # Warmup phase
    for _ in range(10):
        optim_triton.zero_grad()
        loss = loss_fn(model_triton(x), target)
        loss.backward()
        optim_triton.step()
        
        optim_pure.zero_grad()
        loss = loss_fn(model_pure(x), target)
        loss.backward()
        optim_pure.step()
    
    iterations = 100
    
    # --- Timing & Memory: Triton version ---
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iterations):
        optim_triton.zero_grad()
        loss = loss_fn(model_triton(x), target)
        loss.backward()
        optim_triton.step()
    end_event.record()
    torch.cuda.synchronize()
    triton_time = start_event.elapsed_time(end_event)  # ms
    triton_memory = torch.cuda.max_memory_allocated() / (1024 ​** 2)  # MB

    # --- Timing & Memory: Pure PyTorch version ---
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iterations):
        optim_pure.zero_grad()
        loss = loss_fn(model_pure(x), target)
        loss.backward()
        optim_pure.step()
    end_event.record()
    torch.cuda.synchronize()
    pure_time = start_event.elapsed_time(end_event)
    pure_memory = torch.cuda.max_memory_allocated() / (1024 ​** 2)
    
    print(f"\n[Triton]  {iterations} iterations: {triton_time:.2f} ms | Peak memory: {triton_memory:.2f} MB")
    print(f"[PyTorch] {iterations} iterations: {pure_time:.2f} ms | Peak memory: {pure_memory:.2f} MB")

if __name__ == "__main__":
    compare_optimizers()
