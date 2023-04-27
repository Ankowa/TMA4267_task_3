import numpy as np
import itertools
import pandas as pd
from tqdm import tqdm


from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning

simplefilter("ignore", category=ConvergenceWarning)
simplefilter("ignore", category=DataConversionWarning)


from copy import deepcopy
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

CONCURRENT_PROCESSES_CNT = 4


def get_data():
    try:
        X = pd.read_csv("data/X.csv", index_col=0)
        y = pd.read_csv("data/y.csv", index_col=0)
    except:
        X, y = fetch_openml("mnist_784", data_home="data", version=1, return_X_y=True)
    return X, y


def train_test_data(X, y, test_size=10_000, seed=123):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return X_train.values, X_test.values, y_train.values, y_test.values


def standard_scaling(X_train, X_test, apply):
    if apply:
        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_test = ss.transform(X_test)
    return X_train, X_test


def pca(X_train, X_test, apply, n_components=10):
    if apply:
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    return X_train, X_test


def augment_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    apply: bool,
    seed=123,
):
    if apply:
        np.random.seed(seed)
        mask = np.random.permutation(X_train.shape[0])[: X_train.shape[0] // 2]
        X_train_augmented = (
            X_train[mask].reshape(-1, 28, 28)[:, :, ::-1].reshape(-1, 28 * 28)
        )
        y_train_augmented = y_train[mask]
        X_train = np.concatenate([X_train, X_train_augmented], axis=0)
        y_train = np.concatenate([y_train, y_train_augmented], axis=0)

    return X_train, y_train


def train_model(X_train, y_train, model):
    if model:
        clf = LogisticRegression(random_state=123, max_iter=5)
    else:
        clf = DecisionTreeClassifier(random_state=123)
    clf.fit(X_train, y_train)
    return clf


def evaluate(X_test, y_test, clf):
    return clf.score(X_test, y_test) * 100


def run_single(X_train, X_test, y_train, y_test, A, B, C, D):
    X_train, X_test = standard_scaling(X_train, X_test, apply=A)
    X_train, y_train = augment_data(X_train, y_train, apply=C)
    X_train, X_test = pca(X_train, X_test, apply=B)
    clf = train_model(X_train, y_train, model=D)
    return evaluate(X_test, y_test, clf)


def run(X, y):
    X_train, X_test, y_train, y_test = train_test_data(X, y)
    results = []
    factors = ["A", "B", "C", "D"]
    response = ["Y"]
    for A, B, C, D in tqdm(itertools.product([True, False], repeat=4)):
        results.append(
            [A, B, C, D, run_single(X_train, X_test, y_train, y_test, A, B, C, D)]
        )
    df = pd.DataFrame(results, columns=factors + response)
    df[factors] = df[factors].applymap(lambda x: {True: 1, False: -1}[x])
    for comb_size in range(1, len(factors) + 1):
        for comb in itertools.combinations(factors, comb_size):
            df["".join(comb)] = df[list(comb)].prod(axis=1)

    df["intercept"] = 1
    df = df[
        ["intercept"] + [c for c in df.columns if c != "Y" and c != "intercept"] + ["Y"]
    ]
    df.Y = df.Y.apply(lambda x: round(x, 2))
    return df


def save_results(df):
    df.to_csv("data/results.csv")


def main():
    X, y = get_data()
    print("Running experiments...")
    df = run(X, y)
    print("Saving results...")
    save_results(df)


if __name__ == "__main__":
    main()
