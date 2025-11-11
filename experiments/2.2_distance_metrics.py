""" Compute a bunch of metrics on the split data

Derek van Tilborg
Eindhoven University of Technology
Augustus 2024
"""

import argparse
import os
from os.path import join as ospj
from tqdm.auto import tqdm
import pandas as pd
from jcm.training_logistics import get_all_dataset_names, load_dataset_df
from cheminformatics.molecular_similarity import tani_sim_to_train, mean_cosine_cats_to_train, \
    mcsf_to_train, applicability_domain_kNN, applicability_domain_SDC
from cheminformatics.complexity import molecular_complexity
from cheminformatics.descriptors import num_rings, n_smiles_branches, n_smiles_tokens_no_specials, mol_weight
from constants import ROOTDIR


def normalize_dataset_name(name: str) -> str:
    """Return base dataset name that matches files under data/split/."""
    dataset = name
    if dataset.endswith('.csv'):
        dataset = dataset[:-4]
    if dataset.endswith('_split'):
        dataset = dataset[:-6]
    return dataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Compute similarity and complexity metrics for selected datasets.")
    parser.add_argument('--datasets', nargs='+',
                        help="Dataset names (e.g. CHEMBL4203_Ki). Use 'ChEMBL_36' for the pretraining set.")
    parser.add_argument('--include-chembl', action='store_true',
                        help="Include ChEMBL_36 when automatically discovering datasets.")
    args = parser.parse_args()

    # move to root dir
    os.chdir(ROOTDIR)

    # settings for the KNN OOD metric
    KNN_AD_NMIN = 10
    KNN_AD_SCUTOFF = 0.25

    if args.datasets:
        all_dataset_names = [normalize_dataset_name(name) for name in args.datasets]
    else:
        all_dataset_names = get_all_dataset_names()
        chembl_split = ospj('data', 'split', 'ChEMBL_36_split.csv')
        if args.include_chembl and os.path.exists(chembl_split) and 'ChEMBL_36' not in all_dataset_names:
            all_dataset_names.append('ChEMBL_36')

    metrics_dir = ospj('data', 'datasets_with_metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    all_data = []
    for i, dataset_name in enumerate(all_dataset_names):
        print(f"\n{i}\t{dataset_name}")

        # load the data and keep the columns that are needed downstream
        df = load_dataset_df(dataset_name)
        required_columns = {'smiles', 'split'}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"{dataset_name} is missing required columns: {', '.join(sorted(missing))}")

        optional_columns = [col for col in ['y', 'cluster'] if col in df.columns]
        missing_optional = [col for col in ['y', 'cluster'] if col not in df.columns]
        if missing_optional:
            print(f"\t\tSkipping unavailable columns for {dataset_name}: {', '.join(missing_optional)}")

        df = df.loc[:, ['smiles'] + optional_columns + ['split']]

        # get the smiles strings
        train_smiles = df[df['split'] == 'train'].smiles.tolist()
        all_smiles = df.smiles.tolist()

        # ECFP Tanimoto similarity
        print('\t\tComputing Tanimoto similarities between ECFPs')
        tani = tani_sim_to_train(all_smiles, train_smiles, scaffold=False)
        df['Tanimoto_to_train'] = tani

        tani_scaf = tani_sim_to_train(all_smiles, train_smiles, scaffold=True)
        df['Tanimoto_scaffold_to_train'] = tani_scaf

        # Cats Cosine similarity
        print('\t\tComputing cosine similarities between Cats descriptors')
        cats_cos = mean_cosine_cats_to_train(all_smiles, train_smiles)
        df['Cats_cos'] = cats_cos

        # Maximum Common Substructure Fraction (MCSF)
        print('\t\tComputing substructure similarities')
        mcsf = mcsf_to_train(all_smiles, train_smiles, scaffold=False)
        df['MCSF'] = mcsf

        # Compute "complexity" of molecules
        print('\t\tComputing molecular complexity measures')
        complex = [molecular_complexity(smi, bottcher=False, motifs=False) for smi in tqdm(all_smiles)]
        df = pd.concat((df, pd.DataFrame(complex)), axis=1)

        # KNN OOD metric; 10.1021/acs.chemrestox.9b00498
        print('\t\tComputing the applicability domain (KNN)')
        knn_ad = applicability_domain_kNN(all_smiles, train_smiles, k=KNN_AD_NMIN, sim_cutoff=KNN_AD_SCUTOFF)
        df['knn_ad'] = knn_ad

        print('\t\tComputing the applicability domain (SDC)')
        # Sum of distance-weighted contributions; 10.1021/acs.jcim.8b00597
        sdc_ad = applicability_domain_SDC(all_smiles, train_smiles)
        df['sdc_ad'] = sdc_ad

        # write to file
        df.to_csv(ospj(metrics_dir, f'{dataset_name}.csv'), index=False)

        df['dataset'] = dataset_name
        all_data.append(df)

    pd.concat(all_data).to_csv(ospj(metrics_dir, 'all_datasets.csv'), index=False)

    # second round
    all_data2 = []
    for i, dataset_name in enumerate(all_dataset_names):
        print(f"\n{i}\t{dataset_name}")

        df_ = pd.read_csv(ospj(metrics_dir, f'{dataset_name}.csv'))

        # Number of SMILES tokens
        df_['n_smiles_tokens'] = [n_smiles_tokens_no_specials(smi) for smi in df_.smiles]

        # Number of rings
        df_['n_rings'] = [num_rings(smi) for smi in df_.smiles]

        # Number of branches
        df_['n_smiles_branches'] = [n_smiles_branches(smi) for smi in df_.smiles]

        # Mol weight
        df_['mol_weight'] = [mol_weight(smi) for smi in df_.smiles]

        # write to file
        df_.to_csv(ospj(metrics_dir, f'{dataset_name}.csv'), index=False)

        df_['dataset'] = dataset_name
        all_data2.append(df_)

    pd.concat(all_data2).to_csv(ospj(metrics_dir, 'all_datasets.csv'), index=False)
