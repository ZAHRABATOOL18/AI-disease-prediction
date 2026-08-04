"""
pca_viz.py
----------
PCA visualization utilities: 2D scatter of the data projected onto its
first two principal components (colored by disease/no-disease), plus a
cumulative explained-variance plot to show how many components are really
needed to capture the signal.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def plot_pca_scatter(X_scaled, y, title, out_path):
    """2D PCA projection, points colored by class (disease vs no disease)."""
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_

    plt.figure(figsize=(6.5, 5.5))
    for label, color, name in [(0, "#4C72B0", "No Disease"), (1, "#C44E52", "Disease")]:
        mask = np.asarray(y) == label
        plt.scatter(components[mask, 0], components[mask, 1],
                    c=color, label=name, alpha=0.6, s=25, edgecolor="none")

    plt.xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
    plt.title(title, fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return explained


def plot_pca_explained_variance(X_scaled, title, out_path, max_components=None):
    """Cumulative explained variance vs number of PCA components."""
    n_features = X_scaled.shape[1]
    n_components = min(max_components or n_features, n_features)

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(6.5, 5))
    plt.plot(range(1, n_components + 1), cumulative, marker="o", markersize=4, color="#55A868")
    plt.axhline(0.95, color="grey", linestyle="--", linewidth=1, label="95% variance")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title(title, fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    n_for_95 = int(np.argmax(cumulative >= 0.95) + 1) if (cumulative >= 0.95).any() else n_components
    return n_for_95
