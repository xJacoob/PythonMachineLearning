import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from RandomForestClassifier import RandomForestClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt


np.random.seed(52)


def convert_embarked(x):
    if x == 'S':
        return 0
    elif x == 'C':
        return 1
    else:
        return 2


if __name__ == '__main__':
    data = pd.read_csv('https://www.dropbox.com/s/4vu5j6ahk2j3ypk/titanic_train.csv?dl=1')

    data.drop(
        ['PassengerId', 'Name', 'Ticket', 'Cabin'],
        axis=1,
        inplace=True
    )
    data.dropna(inplace=True)

    # Separate these back
    y = data['Survived'].astype(int)
    X = data.drop('Survived', axis=1)

    X['Sex'] = X['Sex'].apply(lambda x: 0 if x == 'male' else 1)
    X['Embarked'] = X['Embarked'].apply(lambda x: convert_embarked(x))

    X_train, X_val, y_train, y_val = train_test_split(X.values, y.values, stratify=y, train_size=0.8)

    scores = []
    for i in range(1, 20):
        Random_Forest = RandomForestClassifier(n_trees=i)
        Random_Forest.fit(X_train, y_train)
        predictions = Random_Forest.predict(X_val)

        acc = accuracy_score(y_val, predictions)
        scores.append(acc)

    formated = [f"{s:.3f}" for s in scores[:20]]
    print(f"[{', '.join(formated)}]")

    plt.figure(figsize=[15, 5])
    plt.plot(range(1, 20), scores, color='blue', label='Random Forest')
    plt.legend(loc='upper left')
    plt.show()


