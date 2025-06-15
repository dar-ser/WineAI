import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_data(X_train):
    """
    Wizualizacja danych:
    - Heatmapa korelacji na zbiorze treningowym
    - PCA 
    """
    # --- Correlation Heatmap
    corr = X_train.corr()
    plt.figure()
    plt.imshow(corr, aspect='auto', cmap='coolwarm')
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Feature Correlation Heatmap (Train Set)")
    plt.tight_layout()
    plt.show()

    # --- PCA 
    pca = PCA()
    pca.fit(X_train)
    cum_var = pca.explained_variance_ratio_.cumsum()
    plt.figure()
    plt.plot(range(1, len(cum_var) + 1), cum_var, marker='o')
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance Curve (Train Set)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()