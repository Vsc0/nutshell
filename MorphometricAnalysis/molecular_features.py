import pandas as pd
from sklearn.model_selection import train_test_split

from morphometric_features import forest_classifier, pca_analysis


def ras_mutant_vs_wild_type_niftp():
    splits_dir = 'splits'

    X = pd.read_csv(f'{splits_dir}/X.csv')
    g = pd.read_csv(f'{splits_dir}/g.csv')
    y = pd.read_csv(f'{splits_dir}/y.csv')

    df = pd.concat([X, g, y], axis=1)

    df = df[df.Class.eq('NIFTP')]

    with open('data/ras_mutated_wsi_names.txt', 'r') as stream:
        ras_mutated_wsi_names = stream.read().splitlines()

    with open('data/wild_type_wsi_names.txt', 'r') as stream:
        wild_type_wsi_names = stream.read().splitlines()

    molecular_wsi_names = ras_mutated_wsi_names + wild_type_wsi_names

    df = df[df['Image'].isin(molecular_wsi_names)]

    def molecular_label(row):
        if row['Image'] in ras_mutated_wsi_names:
            return 'RAS'
        elif row['Image'] in wild_type_wsi_names:
            return 'WT'

    df['Class'] = df.apply(molecular_label, axis=1)

    df.drop(['Image'], axis=1, inplace=True)

    y_mol = df[['Class']].copy()
    y_mol.to_csv(f'{splits_dir}/y_mol.csv', index=False)

    df.drop(['Class'], axis=1, inplace=True)
    df.to_csv(f'{splits_dir}/X_mol.csv', index=False)

    X_train, X_test, y_train, y_test = train_test_split(df, y_mol, stratify=y_mol, test_size=0.2, random_state=42)

    forest_classifier(X_train, X_test, y_train, y_test, mol=True)

    pca_analysis(df, y_mol, mol=True)


def main():
    ras_mutant_vs_wild_type_niftp()


if __name__ == '__main__':
    main()
