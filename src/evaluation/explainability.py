from __future__ import annotations

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def compute_gradcam_1d(
    model: tf.keras.Model,
    x: np.ndarray,
    layer_name: str = "conv1d_1"
) -> np.ndarray:
    """Compute 1D Grad-CAM heatmap for a single input sequence.

    Parameters
    ----------
    model : tf.keras.Model
        The trained CNN + BiLSTM model.
    x : np.ndarray
        Input signal of shape (1, 625, 1), (625, 1), or (625,)
    layer_name : str
        Name of the target convolutional layer (e.g. 'conv1d_1')

    Returns
    -------
    heatmap : np.ndarray
        1D array of shape (625,) normalized to [0, 1]
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = np.expand_dims(np.expand_dims(x, axis=0), axis=-1)
    elif x.ndim == 2:
        x = np.expand_dims(x, axis=0)

    # 1. Create a sub-model that outputs both the target layer activations and model output
    try:
        target_layer = model.get_layer(layer_name)
    except ValueError:
        # Fallback to the first conv1d layer if name not found
        conv_layers = [l.name for l in model.layers if "conv1d" in l.name]
        if conv_layers:
            layer_name = conv_layers[-1] # use the last conv layer
            target_layer = model.get_layer(layer_name)
        else:
            raise ValueError(f"No Conv1D layer found in model.")

    grad_model = tf.keras.Model(
        inputs=[model.inputs],
        outputs=[target_layer.output, model.output]
    )

    # 2. Compute gradients of predicted class w.r.t target layer activations
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        class_channel = predictions[:, 0]

    # Gradients of prediction w.r.t target layer activations
    grads = tape.gradient(class_channel, conv_outputs)

    # 3. Global average pooling of gradients (channel-wise weights)
    # grads shape: (1, conv_len, channels)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    # 4. Weighted combination of activation maps
    conv_outputs = conv_outputs[0] # remove batch dim -> (conv_len, channels)
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # 5. Apply ReLU to keep only positive contributions
    heatmap = tf.maximum(heatmap, 0)

    # Normalize to [0, 1]
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    heatmap = heatmap.numpy()

    # 6. Interpolate heatmap to original signal length (625)
    original_len = x.shape[1]
    x_orig = np.linspace(0, 1, original_len)
    x_heat = np.linspace(0, 1, len(heatmap))
    interpolated_heatmap = np.interp(x_orig, x_heat, heatmap)

    return interpolated_heatmap


def plot_gradcam_1d(
    signal: np.ndarray,
    heatmap: np.ndarray,
    title: str = "Grad-CAM 1D Signal Attribution"
):
    """Plot 1D signal colored by Grad-CAM heatmap values."""
    signal = np.asarray(signal).reshape(-1)
    heatmap = np.asarray(heatmap).reshape(-1)
    
    fig, ax = plt.subplots(figsize=(10, 3.5))
    
    # Plot base signal in light gray for contrast
    ax.plot(signal, color='#cbd5e1', alpha=0.7, label='PPG Signal', linewidth=1.5)
    
    # Overlay colored segments using scatter plot
    sc = ax.scatter(
        np.arange(len(signal)),
        signal,
        c=heatmap,
        cmap='coolwarm',
        s=12,
        alpha=0.9,
        label='Attribution (Red = High AF correlation)'
    )
    
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Importance / Weight of Contribution')
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Samples')
    ax.set_ylabel('Amplitude (normalized)')
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right')
    plt.tight_layout()
    
    return fig, ax


def plot_rf_feature_importances(
    feature_importances: np.ndarray,
    feature_names: list[str],
    top_n: int = 15,
    title: str = "Feature Importances"
):
    """Plot the top N feature importances."""
    feature_importances = np.asarray(feature_importances)
    indices = np.argsort(feature_importances)[::-1][:top_n]
    
    names = [feature_names[i] for i in indices]
    importances = feature_importances[indices]
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(names, importances, color='#4f46e5') # indigo color
    ax.invert_yaxis()  # top-down view
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Importance score')
    ax.grid(True, axis='x', alpha=0.2)
    
    # Add values on top of bars
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.002,
            bar.get_y() + bar.get_height()/2,
            f'{width:.4f}',
            ha='left',
            va='center',
            fontsize=9,
            color='#475569'
        )
        
    plt.tight_layout()
    return fig, ax
