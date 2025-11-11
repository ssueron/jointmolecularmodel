"""
Plot the finetuning dataset splits using a TSNE projection

Derek van Tilborg
Eindhoven University of Technology
June 2024
"""

import argparse
import os
from os.path import join as ospj
import pandas as pd
from rdkit import Chem
from cheminformatics.descriptors import mols_to_ecfp
from cheminformatics.utils import get_scaffold
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from constants import ROOTDIR
from sklearn.decomposition import TruncatedSVD

SPLIT_COLOR_MAP = {
    'ood': '#e9c46a',
    'train': '#0a9396',
    'val': '#f4a259',
    'test': '#94d2bd'
}


def to_split_filename(name: str) -> str:
    """Normalize user-provided dataset names to '<dataset>_split.csv'."""
    filename = name
    if filename.endswith('.csv'):
        filename = filename[:-4]
    if filename.endswith('_split'):
        filename = filename[:-6]
    return f"{filename}_split.csv"


def tsne_mols(mols: list, split: list[str], split_categories: list[str] | None = None,
              random_state: int | None = None, **kwargs) -> pd.DataFrame:
    """ Perform a TSNE on a set of molcules using their ECFPs """
    all_ecfps = mols_to_ecfp(mols, radius=2, nbits=2048, to_array=True)

    # perform SVD to reduce the dimension of the binary vectors
    svd = TruncatedSVD(n_components=100, random_state=random_state)
    X_reduced = svd.fit_transform(all_ecfps)

    # TSNE.
    reducer = TSNE(n_components=2, random_state=random_state, **kwargs)
    projection = reducer.fit_transform(X_reduced)
    print(f"KL: {reducer.kl_divergence_:.4f}")

    # Create a DataFrame for the UMAP results
    df = pd.DataFrame(projection, columns=['x', 'y'])

    split_list = list(split)
    if split_categories is None:
        split_categories = list(pd.unique(split_list))
    df['Split'] = pd.Categorical(split_list, categories=split_categories, ordered=True)

    return df


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Visualize dataset splits with TSNE.")
    parser.add_argument('--datasets', nargs='+',
                        help="Dataset names (e.g. CHEMBL4203_Ki). Use 'ChEMBL_36' for the pretraining set.")
    parser.add_argument('--include-chembl', action='store_true',
                        help="Include ChEMBL_36 when automatically discovering datasets.")
    parser.add_argument('--max-per-split', type=int, default=5000,
                        help="Cap the number of molecules sampled per split before running TSNE. "
                             "Set to 0 to disable downsampling.")
    parser.add_argument('--random-state', type=int, default=0,
                        help="Random seed used for sampling and TSNE.")
    args = parser.parse_args()

    IN_DIR_PATH = 'data/split'
    OUT_DIR_PATH = 'results/dataset_clustering'
    os.chdir(ROOTDIR)
    os.makedirs(OUT_DIR_PATH, exist_ok=True)

    if args.datasets:
        datasets = [to_split_filename(name) for name in args.datasets]
    else:
        datasets = [i for i in os.listdir(IN_DIR_PATH) if i.endswith('split.csv')]
        if not args.include_chembl:
            datasets = [i for i in datasets if i != 'ChEMBL_36_split.csv']
    datasets = [f for f in datasets if os.path.exists(ospj(IN_DIR_PATH, f))]

    tsne_coordinates = []

    for dataset in tqdm(datasets):
        df = pd.read_csv(ospj(IN_DIR_PATH, dataset))

        max_per_split = args.max_per_split if args.max_per_split and args.max_per_split > 0 else None
        original_size = len(df)
        if max_per_split is not None:
            sampled = []
            for split_name, split_df in df.groupby('split'):
                if len(split_df) > max_per_split:
                    sampled.append(split_df.sample(n=max_per_split, random_state=args.random_state))
                else:
                    sampled.append(split_df)
            df = pd.concat(sampled).sample(frac=1, random_state=args.random_state).reset_index(drop=True)
            if len(df) < original_size:
                print(f"Downsampled {dataset} from {original_size} to {len(df)} molecules "
                      f"({max_per_split} per split).")

        smiles = df['smiles'].tolist()

        # Determine split ordering and colors present in this dataset
        split_categories = [s for s in ['ood', 'train', 'val', 'test'] if s in df['split'].unique()]
        remainder = sorted(set(df['split']) - set(split_categories))
        split_categories += remainder
        palette = [SPLIT_COLOR_MAP.get(cat, '#bbbbbb') for cat in split_categories]

        # Get molecules and their scaffolds
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        scaffold_mols = [get_scaffold(m, scaffold_type='cyclic_skeleton') for m in mols]

        # perform TSNE. We choose a perplexity of 20 because it gave a low KL divergence across the board
        projection_mols = tsne_mols(mols, split=df['split'], split_categories=split_categories,
                                    perplexity=20, random_state=args.random_state)
        projection_scaffolds = tsne_mols(scaffold_mols, split=df['split'], split_categories=split_categories,
                                         perplexity=20, random_state=args.random_state)

        # save TSNE coordinates
        projection_mols['smiles'] = smiles
        projection_mols['fingerprint'] = 'full'
        projection_mols['dataset'] = dataset.replace('_split.csv', '')
        tsne_coordinates.append(projection_mols)

        projection_scaffolds['smiles'] = smiles
        projection_scaffolds['fingerprint'] = 'scaffold'
        projection_scaffolds['dataset'] = dataset.replace('_split.csv', '')
        tsne_coordinates.append(projection_scaffolds)

        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        # Create the first scatter plot
        sns.scatterplot(ax=axes[0], x='x', y='y', hue='Split', data=projection_mols,
                        palette=palette, hue_order=split_categories)
        axes[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        axes[0].set_xlabel('Whole molecule ECFPs')
        axes[0].set_ylabel('')

        # Create the second scatter plot
        sns.scatterplot(ax=axes[1], x='x', y='y', hue='Split', data=projection_scaffolds,
                        palette=palette, hue_order=split_categories)
        axes[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        axes[1].set_xlabel('Cyclic skeleton ECFPs')
        axes[1].set_ylabel('')

        fig.suptitle(f'TSNE of {dataset.replace("_split.csv", "")} (n={len(smiles)})')
        plt.savefig(ospj(OUT_DIR_PATH, dataset.replace('.csv', '.pdf')))
        plt.close(fig)

    # save the file with all results
    tsne_coordinates = pd.concat(tsne_coordinates)
    tsne_coordinates.to_csv(ospj(OUT_DIR_PATH, "TSNE_coordinates.csv"), index=False)
