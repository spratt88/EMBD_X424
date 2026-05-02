import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def generate_checkerboard(n=800, grid=4, seed=42):
    np.random.seed(seed)
    X = np.random.uniform(0, grid, (n, 2))
    y = ((np.floor(X[:, 0]) + np.floor(X[:, 1])) % 2).astype(int).reshape(-1, 1)
    return X, y

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def init_params(layer_dims, seed=0):
    np.random.seed(seed)
    params = {}
    for l in range(1, len(layer_dims)):
        params[f"W{l}"] = np.random.randn(layer_dims[l-1], layer_dims[l]) * np.sqrt(2.0 / layer_dims[l-1])
        params[f"b{l}"] = np.zeros((1, layer_dims[l]))
    return params

def forward(X, params):
    L = len([k for k in params if k.startswith("W")])
    cache = {"A0": X}
    A = X
    for l in range(1, L + 1):
        Z = A @ params[f"W{l}"] + params[f"b{l}"]
        cache[f"Z{l}"] = Z
        A = relu(Z) if l < L else sigmoid(Z)
        cache[f"A{l}"] = A
    return A, cache

def compute_loss(y_hat, y):
    eps = 1e-9
    return -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

def backward(y, cache, params):
    m = y.shape[0]
    L = len([k for k in params if k.startswith("W")])
    grads = {}
    dZ = (cache[f"A{L}"] - y) / m
    for l in reversed(range(1, L + 1)):
        grads[f"dW{l}"] = cache[f"A{l-1}"].T @ dZ
        grads[f"db{l}"] = np.sum(dZ, axis=0, keepdims=True)
        if l > 1:
            dZ = (dZ @ params[f"W{l}"].T) * relu_derivative(cache[f"Z{l-1}"])
    return grads

def init_adam(params):
    L = len([k for k in params if k.startswith("W")])
    m_a = {}; v_a = {}
    for l in range(1, L+1):
        for k in [f"dW{l}", f"db{l}"]:
            m_a[k] = np.zeros_like(params[k[1:]])
            v_a[k] = np.zeros_like(params[k[1:]])
    return m_a, v_a

def update_adam(params, grads, m_a, v_a, t, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    L = len([k for k in params if k.startswith("W")])
    for l in range(1, L + 1):
        for dk, pk in [(f"dW{l}", f"W{l}"), (f"db{l}", f"b{l}")]:
            m_a[dk] = b1*m_a[dk] + (1-b1)*grads[dk]
            v_a[dk] = b2*v_a[dk] + (1-b2)*grads[dk]**2
            params[pk] -= lr * (m_a[dk]/(1-b1**t)) / (np.sqrt(v_a[dk]/(1-b2**t)) + eps)
    return params, m_a, v_a

def train(X, y, layer_dims, lr=0.003, epochs=5000, print_every=500):
    params = init_params(layer_dims)
    m_a, v_a = init_adam(params)
    history = []
    for epoch in range(1, epochs + 1):
        y_hat, cache = forward(X, params)
        loss = compute_loss(y_hat, y)
        grads = backward(y, cache, params)
        params, m_a, v_a = update_adam(params, grads, m_a, v_a, t=epoch, lr=lr)
        history.append(loss)
        if epoch % print_every == 0 or epoch == 1:
            acc = np.mean((y_hat >= 0.5).astype(int) == y) * 100
            print(f"Epoch {epoch:>5} | Loss: {loss:.4f} | Acc: {acc:.1f}%")
    return params, history

def plot_results(X, y, params, history):
    fig = plt.figure(figsize=(16, 5), facecolor="#0a0a0f")
    gs = gridspec.GridSpec(1, 3, wspace=0.38)
    ax1, ax2, ax3 = [fig.add_subplot(gs[i]) for i in range(3)]

    xx, yy = np.meshgrid(np.linspace(0, 1, 300), np.linspace(0, 1, 300))
    Z, _ = forward(np.c_[xx.ravel(), yy.ravel()], params)
    Z = Z.reshape(xx.shape)

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor("#0a0a0f")
        ax.tick_params(colors="#555")
        for sp in ax.spines.values(): sp.set_color("#222")

    ax1.contourf(xx, yy, Z, levels=100, cmap="RdYlGn", alpha=0.9)
    ax1.contour(xx, yy, Z, levels=[0.5], colors="white", linewidths=2, linestyles="--")
    c = np.where(y.ravel()==1, "#00ffcc", "#ff3366")
    ax1.scatter(X[:,0], X[:,1], c=c, s=10, alpha=0.5, linewidths=0)
    ax1.set_title("Decision Boundary", color="white", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("x₁", color="#888", fontsize=9); ax1.set_ylabel("x₂", color="#888", fontsize=9)

    ax2.scatter(X[y.ravel()==0,0], X[y.ravel()==0,1], c="#ff3366", s=10, alpha=0.5, label="Class 0", linewidths=0)
    ax2.scatter(X[y.ravel()==1,0], X[y.ravel()==1,1], c="#00ffcc", s=10, alpha=0.5, label="Class 1", linewidths=0)
    ax2.set_title("Checkerboard Dataset", color="white", fontsize=13, fontweight="bold", pad=10)
    ax2.legend(facecolor="#111", labelcolor="white", edgecolor="#333")
    ax2.set_xlabel("x₁", color="#888", fontsize=9); ax2.set_ylabel("x₂", color="#888", fontsize=9)

    ax3.plot(history, color="#00ffcc", linewidth=1.5)
    ax3.fill_between(range(len(history)), history, alpha=0.1, color="#00ffcc")
    ax3.set_title("Training Loss (BCE)", color="white", fontsize=13, fontweight="bold", pad=10)
    ax3.set_xlabel("Epoch", color="#888", fontsize=9); ax3.set_ylabel("Loss", color="#888", fontsize=9)

    fig.suptitle("Checkerboard NN  ·  2→16→8→1  ·  ReLU hidden, Sigmoid output  ·  Adam",
                 color="#ccc", fontsize=12, y=1.02)
    plt.savefig("/mnt/user-data/outputs/checkerboard_results.png", dpi=150,
                bbox_inches="tight", facecolor="#0a0a0f")
    plt.close()
    print("Plot saved.")

if __name__ == "__main__":
    LAYER_DIMS = [2, 16, 8, 1]
    X_raw, y = generate_checkerboard(n=800, grid=4)
    X = X_raw / 4.0

    print(f"Architecture: {' → '.join(map(str, LAYER_DIMS))}")
    print(f"Hidden: ReLU  |  Output: Sigmoid  |  Optimizer: Adam\n")

    params, history = train(X, y, LAYER_DIMS, lr=0.003, epochs=5000)

    y_hat, _ = forward(X, params)
    acc = np.mean((y_hat >= 0.5).astype(int) == y) * 100
    print(f"\nFinal Accuracy: {acc:.2f}%")
    plot_results(X, y, params, history)
