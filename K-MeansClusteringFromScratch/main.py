import numpy as np
from sklearn.datasets import load_wine
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from CustomMeans import CustomMeans
from sklearn.metrics import silhouette_score

# scroll down to the bottom to implement your solution


def plot_comparison(data: np.ndarray, predicted_clusters: np.ndarray, true_clusters: np.ndarray = None,
                    centers: np.ndarray = None, show: bool = True):

    # Use this function to visualize the results on Stage 6.

    if true_clusters is not None:
        plt.figure(figsize=(20, 10))

        plt.subplot(1, 2, 1)
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=predicted_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.title('Predicted clusters')
        plt.xlabel('alcohol')
        plt.ylabel('malic_acid')
        plt.grid()

        plt.subplot(1, 2, 2)
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=true_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.title('Ground truth')
        plt.xlabel('alcohol')
        plt.ylabel('malic_acid')
        plt.grid()
    else:
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=predicted_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.title('Predicted clusters')
        plt.xlabel('alcohol')
        plt.ylabel('malic_acid')
        plt.grid()

    plt.savefig('Visualization.png', bbox_inches='tight')
    if show:
        plt.show()


if __name__ == '__main__':

    # Load data
    data = load_wine(as_frame=True, return_X_y=True)
    X_full, y_full = data

    # Permutate it to make things more interesting
    rnd = np.random.RandomState(42)
    permutations = rnd.permutation(len(X_full))
    X_full = X_full.iloc[permutations]
    y_full = y_full.iloc[permutations]

    # From dataframe to ndarray
    X_full = X_full.values
    y_full = y_full.values

    # Scale data
    scaler = StandardScaler()
    X_full = scaler.fit_transform(X_full)

    # Finding right value of K
    silhouette_list =[]
    inertia = []
    for i in range(2, 11):
        K_means = CustomMeans(i)
        train = K_means.fit(X_full, eps=1e-6)
        predicted_clusters = K_means.predict(X_full)
        score = K_means.calculate_inertia(X_full, predicted_clusters)
        silhouette = silhouette_score(X_full, predicted_clusters)
        inertia.append(int(score))
        silhouette_list.append(silhouette)

    plt.figure(figsize=(10, 10))
    plt.plot(range(2,11), inertia, c='r', label='Inertia', linewidth=2)
    plt.legend()
    plt.xlabel('Number of clusters', fontsize=20)
    plt.ylabel('Inertia', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(alpha=0.7)
    #plt.show()

    plt.figure(figsize=(10, 10))
    plt.plot(range(2, 11), silhouette_list, c='b', label='Silhouette', linewidth=2)
    plt.legend()
    plt.xlabel('Number of clusters', fontsize=20)
    plt.ylabel('Silhouette score', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #plt.grid(alpha=0.7)


    # Creating model with 3 cluster (Right value of K)
    K_means_3_clusters = CustomMeans(3)
    training = K_means_3_clusters.fit(X_full, eps=1e-6)
    predicted = K_means_3_clusters.predict(X_full)
    print(predicted[:20])

    plot_comparison(X_full, predicted, y_full, K_means_3_clusters.centroids)


