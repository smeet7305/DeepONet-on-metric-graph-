# Trained & Frozen Weights and Biases: Physics-Informed DeepONet

This directory contains the completely trained and frozen weights and biases for the **Physics-Informed DeepONet on Metric Graphs** (Blechschmidt et al., ICML 2025).

The models are pre-trained on 20,000 PDE simulations with 20,000 optimization epochs (Width 200, 7 hidden layers) and verified to achieve $\sim 1.4\%$ relative space-time error against the high-fidelity Finite Volume Method (FVM) solver.

---

## Directory Contents

| Filename | Format | Description |
| :--- | :---: | :--- |
| `model_metadata.json` | JSON | Architecture specs, dimensions, activations, PDE parameters. |
| `load_weights.py` | Python | Helper module with 1-line functions to load weights into NumPy, PyTorch, or JAX. |
| `inflow_weights_biases.npz` | NumPy | Named arrays for the Inflow Edge DeepONet surrogate. |
| `inner_weights_biases.npz` | NumPy | Named arrays for the Inner Edge DeepONet surrogate. |
| `outflow_weights_biases.npz` | NumPy | Named arrays for the Outflow Edge DeepONet surrogate. |
| `inflow_edge_frozen_params.pkl` | Pickle / JAX | Exact JAX tuple `(branch_params, trunk_params)` for Inflow edges. |
| `inner_edge_frozen_params.pkl` | Pickle / JAX | Exact JAX tuple `(branch_params, trunk_params)` for Inner edges. |
| `outflow_edge_frozen_params.pkl` | Pickle / JAX | Exact JAX tuple `(branch_params, trunk_params)` for Outflow edges. |
| `all_edge_models_frozen.pt` | PyTorch | Combined PyTorch state dictionary for all three edge models. |

---

## How to Load and Use in Your Code

### 1. In Pure Python / NumPy (Zero Framework Dependencies)
```python
from training_weights_biases.load_weights import load_numpy_weights

# Load inner edge weights
weights = load_numpy_weights('inner')  # or 'inflow', 'outflow'

# Access individual layers
w0 = weights['branch_layer_0_weight']  # Shape: (304, 200)
b0 = weights['branch_layer_0_bias']    # Shape: (200,)
u_w = weights['branch_U_weight']       # Shape: (304, 200)
v_w = weights['branch_V_weight']       # Shape: (304, 200)
t_w0 = weights['trunk_layer_0_weight'] # Shape: (200, 200)
```

### 2. In PyTorch
```python
from training_weights_biases.load_weights import load_torch_weights

weights = load_torch_weights('inner')
w0 = weights['branch_layer_0_weight']  # PyTorch Tensor: torch.Size([304, 200])
```

### 3. In JAX
```python
from training_weights_biases.load_weights import load_jax_params

branch_params, trunk_params = load_jax_params('inner')
```
