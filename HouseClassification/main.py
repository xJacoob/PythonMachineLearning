import os
import requests
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from category_encoders import TargetEncoder
from sklearn.metrics import classification_report

def target_encoder(X_train, X_test, y_train, y_test):
    enc = TargetEncoder(cols=['Zip_area', 'Zip_loc', 'Room'])
    X_train_final = enc.fit_transform(X_train, y_train)
    X_test_final = enc.transform(X_test)

    X_train_final.columns = X_train_final.columns.astype(str)
    X_test_final.columns = X_test_final.columns.astype(str)

    tree_clf = DecisionTreeClassifier(criterion='entropy', max_features=3, splitter='best', max_depth=6, min_samples_split=4, random_state=3)
    tree_clf.fit(X_train_final, y_train)

    test_pred = tree_clf.predict(X_test_final)

    return classification_report(y_test, test_pred, output_dict=True)


def one_hot_encoder(X_train, X_test, y_train, y_test):
    enc = OneHotEncoder(drop="first")
    X_train_transformed = pd.DataFrame(enc.fit_transform(X_train[['Zip_area', 'Zip_loc', 'Room']]).toarray(), index=X_train.index)
    X_test_transformed = pd.DataFrame(enc.transform(X_test[['Zip_area', 'Zip_loc', 'Room']]).toarray(), index=X_test.index)
    X_train_final = X_train[['Area', 'Lon', 'Lat']].join(X_train_transformed)
    X_test_final = X_test[['Area', 'Lon', 'Lat']].join(X_test_transformed)

    X_train_final.columns = X_train_final.columns.astype(str)
    X_test_final.columns = X_test_final.columns.astype(str)

    tree_clf = DecisionTreeClassifier(criterion='entropy', max_features=3, splitter='best', max_depth=6,
                                      min_samples_split=4, random_state=3)

    tree_clf.fit(X_train_final, y_train)

    test_pred = tree_clf.predict(X_test_final)

    return classification_report(y_test, test_pred, output_dict=True)


def ordinal_encoder(X_train, X_test, y_train, y_test):
    enc = OrdinalEncoder()
    X_train_transformed = pd.DataFrame(enc.fit_transform(X_train[['Zip_area', 'Zip_loc', 'Room']]),
                                           index=X_train.index)
    X_test_transformed = pd.DataFrame(enc.transform(X_test[['Zip_area', 'Zip_loc', 'Room']]),
                                          index=X_test.index)
    X_train_final = X_train[['Area', 'Lon', 'Lat']].join(X_train_transformed)
    X_test_final = X_test[['Area', 'Lon', 'Lat']].join(X_test_transformed)

    X_train_final.columns = X_train_final.columns.astype(str)
    X_test_final.columns = X_test_final.columns.astype(str)

    tree_clf = DecisionTreeClassifier(criterion='entropy', max_features=3, splitter='best', max_depth=6,
                                          min_samples_split=4, random_state=3)

    tree_clf.fit(X_train_final, y_train)

    test_pred = tree_clf.predict(X_test_final)

    return classification_report(y_test, test_pred, output_dict=True)



if __name__ == '__main__':
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    if 'house_class.csv' not in os.listdir('../Data'):
        sys.stderr.write("[INFO] Dataset is loading.\n")
        url = "https://www.dropbox.com/s/7vjkrlggmvr5bc1/house_class.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/house_class.csv', 'wb').write(r.content)
        sys.stderr.write("[INFO] Loaded.\n")


    df = pd.read_csv('../Data/house_class.csv')
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    zip_loc_values = X['Zip_loc'].values
    zip_loc_dict = {}


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=zip_loc_values, random_state=1)

    result1 = target_encoder(X_train, X_test, y_train, y_test)

    result2 = one_hot_encoder(X_train, X_test, y_train, y_test)

    result3 = ordinal_encoder(X_train, X_test, y_train, y_test)

    print(f"OneHotEncoder:{round(result2['macro avg']['f1-score'], 2)}")
    print(f"OrdinalEncoder:{round(result3['macro avg']['f1-score'], 2)}")
    print(f"TargetEncoder:{round(result1['macro avg']['f1-score'], 2)}")


