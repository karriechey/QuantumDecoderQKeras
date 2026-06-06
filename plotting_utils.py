import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_pareto_frontier(results_df, output_path=None):
    """
    Plot Pareto frontier: model_size_kb vs logical_error_rate.
    Highlights non-dominated configurations.

    Args:
        results_df: DataFrame with columns [w_bits, a_bits, p_L, model_size_kb]
        output_path: if provided, save figure to this path
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Extract coordinates
    x = results_df['model_size_kb'].values
    y = results_df['p_L'].values

    # Find Pareto frontier (minimize both size and error)
    pareto_mask = np.ones(len(results_df), dtype=bool)
    for i in range(len(results_df)):
        for j in range(len(results_df)):
            if i != j and x[j] <= x[i] and y[j] <= y[i]:
                if x[j] < x[i] or y[j] < y[i]:
                    pareto_mask[i] = False
                    break

    # Plot all points
    ax.scatter(x[~pareto_mask], y[~pareto_mask], s=50, alpha=0.5, label='Dominated')
    ax.scatter(x[pareto_mask], y[pareto_mask], s=100, alpha=0.9, color='red',
               edgecolors='darkred', linewidth=2, label='Pareto optimal')

    # Annotate with (w, a) for Pareto points
    for i in np.where(pareto_mask)[0]:
        w, a = results_df.iloc[i][['w_bits', 'a_bits']].values
        ax.annotate(f'({int(w)},{int(a)})',
                   (x[i], y[i]),
                   xytext=(5, 5),
                   textcoords='offset points',
                   fontsize=8)

    ax.set_xlabel('Model Size (KB)', fontsize=12)
    ax.set_ylabel('Logical Error Rate', fontsize=12)
    ax.set_title('Quantization Pareto Frontier', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig, ax


def plot_learning_curves(history, output_path=None):
    """
    Plot training vs validation loss and accuracy.

    Args:
        history: Keras history object from model.fit()
        output_path: if provided, save figure to this path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(history.history['loss'], label='Train loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Progress: Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(history.history['accuracy'], label='Train accuracy', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Val accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Progress: Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig, (ax1, ax2)


def plot_distance_scaling(results_df, output_path=None):
    """
    Plot logical error rate vs code distance for different quantization levels.

    Args:
        results_df: DataFrame with columns [code_distance, quantization_level, logical_error_rate]
        output_path: if provided, save figure to this path
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    quantization_levels = results_df['quantization_level'].unique()
    colors = cm.viridis(np.linspace(0, 1, len(quantization_levels)))

    for quant_level, color in zip(quantization_levels, colors):
        subset = results_df[results_df['quantization_level'] == quant_level]
        subset = subset.sort_values('code_distance')
        ax.loglog(subset['code_distance'], subset['logical_error_rate'],
                 'o-', linewidth=2, markersize=8, label=quant_level, color=color)

    ax.set_xlabel('Code Distance', fontsize=12)
    ax.set_ylabel('Logical Error Rate', fontsize=12)
    ax.set_title('Distance Scaling', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig, ax


def plot_threshold_analysis(results_df, output_path=None):
    """
    Plot logical error rate vs physical error rate for threshold estimation.

    Args:
        results_df: DataFrame with columns [physical_error_rate, logical_error_rate, quantization_config]
        output_path: if provided, save figure to this path
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    configs = results_df['quantization_config'].unique()
    colors = cm.viridis(np.linspace(0, 1, len(configs)))

    for config, color in zip(configs, colors):
        subset = results_df[results_df['quantization_config'] == config]
        subset = subset.sort_values('physical_error_rate')
        ax.loglog(subset['physical_error_rate'], subset['logical_error_rate'],
                 'o-', linewidth=2, markersize=8, label=config, color=color)

    ax.set_xlabel('Physical Error Rate', fontsize=12)
    ax.set_ylabel('Logical Error Rate', fontsize=12)
    ax.set_title('Threshold Analysis (p* estimation)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig, ax
