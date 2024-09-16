import argparse
import os
import pathlib

import pandas as pd
from scipy.cluster import hierarchy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from joblib import dump


def preprocess_dataset(df):
    splits_dir = 'splits'
    pathlib.Path(splits_dir).mkdir(parents=True, exist_ok=True)

    # cleaning
    df.drop(columns=df.filter(regex='2.00 µm|0.50 µm', axis=1).columns, inplace=True)
    df.drop(df[df['Classification'].isin(('Other', 'ROI'))].index, inplace=True)
    df.dropna(axis=1, how='all', inplace=True)

    df.rename(columns={'Classification': 'Class'}, inplace=True)
    df.rename(columns=lambda x: x.replace('ROI: 1.00 µm per pixel: ', ''), inplace=True)

    # OD Sum Mean custom scaling
    mean_values = df.groupby('Image')['OD Sum: Mean'].mean()
    min_mean_value = mean_values.min()
    normalized_mean_values = mean_values * min_mean_value
    df['OD Sum: Mean'] = df.apply(lambda row: row['OD Sum: Mean'] / normalized_mean_values[row['Image']], axis=1)

    df.drop(columns=['Object ID', 'Object type', 'Parent', 'ROI', 'Centroid X µm', 'Centroid Y µm'], inplace=True)

    features_df = df[['Image',
                      'Class',
                      'Area µm^2',
                      'Length µm',
                      'Circularity',
                      'Solidity',
                      'Max diameter µm',
                      'Min diameter µm',
                      'Perimeter µm',
                      'OD Sum: Mean',
                      'OD Sum: Haralick Angular second moment (F0)',
                      'OD Sum: Haralick Contrast (F1)',
                      'OD Sum: Haralick Correlation (F2)',
                      'OD Sum: Haralick Sum of squares (F3)',
                      'OD Sum: Haralick Inverse difference moment (F4)',
                      'OD Sum: Haralick Sum average (F5)',
                      'OD Sum: Haralick Sum variance (F6)',
                      'OD Sum: Haralick Sum entropy (F7)',
                      'OD Sum: Haralick Entropy (F8)',
                      'OD Sum: Haralick Difference variance (F9)',
                      'OD Sum: Haralick Difference entropy (F10)',
                      'OD Sum: Haralick Information measure of correlation 1 (F11)',
                      'OD Sum: Haralick Information measure of correlation 2 (F12)',
                      ]].copy()

    features_df.dropna(axis=0, how='any', inplace=True)

    image_sr = features_df[['Image']].copy()
    features_df.drop(['Image'], axis=1, inplace=True)
    class_sr = features_df[['Class']].copy()
    features_df.drop(['Class'], axis=1, inplace=True)

    # correlation
    corr = features_df.corr(method='pearson')

    # hierarchical clustering
    Z = hierarchy.linkage(y=corr, method='ward', metric='euclidean', optimal_ordering=True)  # linkage matrix
    # form flat clusters from the hierarchical clustering defined by the linkage matrix
    threshold = 0.7  # the maximum inter-cluster distance allowed when forming flat clusters
    clusters = hierarchy.fcluster(Z, t=threshold, criterion='distance')

    # feature selection
    selected_features = set()
    for cluster_id in range(1, max(clusters) + 1):
        features_in_cluster = corr.columns[clusters == cluster_id]
        representative_feature = features_in_cluster[0]  # select the first feature as the representative
        selected_features.add(representative_feature)
    final_selected_features = list(selected_features)
    columns_to_keep = final_selected_features
    selected_features_df = features_df[columns_to_keep].copy()

    X = selected_features_df
    y = class_sr
    g = image_sr
    X.to_csv(f'{splits_dir}/X.csv', index=False)
    y.to_csv(f'{splits_dir}/y.csv', index=False)
    g.to_csv(f'{splits_dir}/g.csv', index=False)

    return X, y, g


def forest_classifier(X_train, X_test, y_train, y_test, scale=True, mol=False):
    splits_dir = 'splits'
    pathlib.Path(splits_dir).mkdir(parents=True, exist_ok=True)
    results_dir = 'results'
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)

    index = X_train.columns
    if scale:  # linearly scale each feature in the range (0, 1)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

    # train
    forest = RandomForestClassifier(random_state=42, n_jobs=-1)
    model = forest.fit(X_train, y_train['Class'].values)
    dump(model, f'{results_dir}/forest{"_mol" if mol else ""}.joblib')

    # feature importance based on mean decrease in impurity
    forest_importances_sr = pd.Series(forest.feature_importances_, index=index).sort_values(ascending=True)
    forest_importances_sr.to_csv(f'{results_dir}/forest_feature_importances_scale_{scale}{"_mol" if mol else ""}.csv')

    # test
    y_pred = forest.predict(X_test)
    class_report = classification_report(y_test, y_pred)
    with open(f'{results_dir}/forest_classification_report_{"mol" if mol else ""}.txt', 'w') as stream:
        stream.write(class_report)


def tree_classifier(X_train, X_test, y_train, y_test, scale=True):
    results_dir = 'results'
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)

    if scale:  # linearly scale each feature in the range (0, 1)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

    # train
    tree = DecisionTreeClassifier(random_state=42)
    model = tree.fit(X_train, y_train['Class'].values)
    dump(model, f'{results_dir}/tree.joblib')

    # test
    y_pred = tree.predict(X_test)
    class_report = classification_report(y_test, y_pred)
    with open(f'{results_dir}/tree_classification_report.txt', 'w') as stream:
        stream.write(class_report)


def pca_analysis(X, y, scale=True, mol=False):
    results_dir = 'results'
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)
    if scale:  # standardize features by removing the mean and scaling to unit variance
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    pca = PCA(n_components=2)  # linear, deterministic
    model = pca.fit(X)
    dump(model, f'{results_dir}/pca{"_mol" if mol else ""}.joblib')
    # X_embedded = model.transform(X)


def main(args):
    file_paths = args.qupath_exported_measurements  # one or more QuPath exported project measurements
    splits_dir = 'splits'
    pathlib.Path(splits_dir).mkdir(parents=True, exist_ok=True)
    splits = ('X', 'y', 'g')
    if not all(map(lambda x: os.path.isfile(f'{splits_dir}/{x}.csv'), splits)):
        dfs = []
        for file_path in file_paths:
            dfs.append(pd.read_csv(file_path))
        df = pd.concat(dfs, ignore_index=True)
        X, y, g = preprocess_dataset(df)
    else:
        X, y, g = (pd.read_csv(f'{splits_dir}/{x}.csv') for x in splits)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    forest_classifier(X_train, X_test, y_train, y_test)

    pca_analysis(X, y)

    tree_classifier(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    def tuple_of_strings(arg):
        return tuple(arg.split(','))


    parser.add_argument(
        '--qupath-exported-measurements',
        type=tuple_of_strings,
        required=True,
        help='QuPath exported measurements files',
    )
    arguments = parser.parse_args()
    main(arguments)
