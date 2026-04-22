import numpy as np

class CustomMeans:
    def __init__(self, k):
        self.k = k
        self.centroids = None

    def find_nearest_center(self, X):
        result = {i: [] for i in range(self.k)}

        for sample in X:
            distance = [np.linalg.norm(self.centroids[i] - sample) for i in range(len(self.centroids))]
            min_dist = np.argmin(distance)
            result[int(min_dist)].append(sample)

        return result

    def calculate_new_centers(self, assigned_samples):
        new_centers = []

        for i in range(self.k):
            new_centers.append(np.mean(assigned_samples[i], axis=0))

        return np.array(new_centers)

    def fit(self, X, eps=1e-6):
        self.centroids = X[:self.k]
        distance = np.inf

        while distance >= eps:
            first_assigned_samples = self.find_nearest_center(X)
            new_centers = self.calculate_new_centers(first_assigned_samples)
            distance = np.linalg.norm(self.centroids - new_centers)
            self.centroids = new_centers

    def predict(self, X):
        result = []

        for sample in X:
            distance = [np.linalg.norm(self.centroids[i] - sample) for i in range(len(self.centroids))]
            min_dist = np.argmin(distance)
            result.append(int(min_dist))

        return result

    def calculate_inertia(self, X, predicted_labels):
        inertia = 0
        for i in range(len(X)):
            sample = X[i]
            label = predicted_labels[i]
            cluster = self.centroids[label]

            inertia += np.linalg.norm(sample - cluster) ** 2

        return inertia