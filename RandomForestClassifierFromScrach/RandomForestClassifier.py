import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class RandomForestClassifier:
    def __init__(self, n_trees=10, max_depth=np.iinfo(np.int64).max, min_error=1e-6):

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_error = min_error
        self.forest = []
        self.is_fit = False

    @staticmethod
    def create_bootstrap(X, y):
        random_choice = np.random.choice(a=len(X), size=len(X), replace=True)
        X_sample = X[random_choice]
        y_sample = y[random_choice]
        return X_sample, y_sample

    def fit(self, X_train, y_train):
        for _ in tqdm(range(self.n_trees)):
            X_sample, y_sample = self.create_bootstrap(X_train, y_train)
            tree_clf = DecisionTreeClassifier(max_features='sqrt', max_depth=self.max_depth, min_impurity_decrease=self.min_error)
            tree_clf.fit(X_sample, y_sample)
            self.forest.append(tree_clf)

        self.is_fit = True

    def predict(self, X_test):

        if not self.is_fit:
            raise AttributeError('The forest is not fit yet! Consider calling .fit() method.')

        tree_preds = np.array([tree.predict(X_test) for tree in self.forest])
        final_predictions = [np.bincount(sample_votes).argmax() for sample_votes in tree_preds.T]

        return np.array(final_predictions)