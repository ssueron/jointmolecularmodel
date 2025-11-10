"""
Script to clean the data of all finetuning sets from Lit-PCBA, MoleculeACE, and Ames mutagenicity and the pre-training
data from ChEBML33

Derek van Tilborg
June 2024
Eindhoven University of Technology
"""

import os
import argparse
from collections import Counter
import pandas as pd
from cheminformatics.cleaning import clean_mols
from constants import ROOTDIR


def process_litpcba(dataset_name: str) -> pd.DataFrame:

    # read data
    with open(f'data/LitPCBA/{dataset_name}/actives.smi', 'r') as file:
        actives = [line.strip().split(' ')[0] for line in file]
    with open(f'data/LitPCBA/{dataset_name}/inactives.smi', 'r') as file:
        inactives = [line.strip().split(' ')[0] for line in file]

    # Clean molecules
    actives, actives_failed = clean_mols(actives)
    inactives, inactives_failed = clean_mols(inactives)

    print('Parsing errors:')
    [print(f"{k}: {v}") for k, v in Counter(actives_failed['reason'] + inactives_failed['reason']).items()]

    # Get the sets
    actives_clean = set(actives['clean'])
    inactives_clean = set(inactives['clean'])

    # Check if there are overlapping molecules in both the set of inactives and actives
    intersecting_smiles = actives_clean & inactives_clean
    actives_clean = [smi for smi in actives_clean if smi not in intersecting_smiles]
    inactives_clean = [smi for smi in inactives_clean if smi not in intersecting_smiles]

    # Put it all together
    y = [1] * len(actives_clean) + [0] * len(inactives_clean)
    smiles = list(actives_clean) + list(inactives_clean)

    # put together in a dataframe and shuffle the rows
    df = pd.DataFrame({'smiles': smiles, 'y': y})
    df = df.sample(frac=1).reset_index(drop=True)

    df.to_csv(f'data/clean/{dataset_name}.csv', index=False)


def process_moleculeace(dataset_name: str) -> pd.DataFrame:
    activity_threshold = 100  # in nM

    df_original = pd.read_csv(f'data/moleculeace_original/{dataset_name}.csv')

    actives = df_original.loc[df_original['exp_mean [nM]'] <= activity_threshold, 'smiles'].tolist()
    inactives = df_original.loc[df_original['exp_mean [nM]'] > activity_threshold, 'smiles'].tolist()

    # Clean molecules
    actives, actives_failed = clean_mols(actives)
    inactives, inactives_failed = clean_mols(inactives)

    print('Parsing errors:')
    [print(f"{k}: {v}") for k, v in Counter(actives_failed['reason'] + inactives_failed['reason']).items()]

    # Get the sets
    actives_clean = set(actives['clean'])
    inactives_clean = set(inactives['clean'])

    # Check if there are overlapping molecules in both the set of inactives and actives
    intersecting_smiles = actives_clean & inactives_clean
    actives_clean = [smi for smi in actives_clean if smi not in intersecting_smiles]
    inactives_clean = [smi for smi in inactives_clean if smi not in intersecting_smiles]

    # Put it all together
    y = [1] * len(actives_clean) + [0] * len(inactives_clean)
    smiles = list(actives_clean) + list(inactives_clean)

    # put together in a dataframe and shuffle the rows
    df = pd.DataFrame({'smiles': smiles, 'y': y})
    df = df.sample(frac=1).reset_index(drop=True)

    df.to_csv(f'data/clean/{dataset_name}.csv', index=False)


def process_ames():

    actives, inactives = [], []
    with open('data/Ames_mutagenicity/smiles_cas_N6512.smi', 'r') as file:
        for line in file:
            label = int(line.strip().split('\t')[-1])
            smi = line.strip().split(' ')[0]
            if label == 1:
                actives.append(smi)
            else:
                inactives.append(smi)

    # Clean molecules
    actives, actives_failed = clean_mols(actives)
    inactives, inactives_failed = clean_mols(inactives)

    print('Parsing errors:')
    [print(f"{k}: {v}") for k, v in Counter(actives_failed['reason'] + inactives_failed['reason']).items()]

    # Get the sets
    actives_clean = set(actives['clean'])
    inactives_clean = set(inactives['clean'])

    # Check if there are overlapping molecules in both the set of inactives and actives
    intersecting_smiles = actives_clean & inactives_clean
    actives_clean = [smi for smi in actives_clean if smi not in intersecting_smiles]
    inactives_clean = [smi for smi in inactives_clean if smi not in intersecting_smiles]

    # Put it all together
    y = [1] * len(actives_clean) + [0] * len(inactives_clean)
    smiles = list(actives_clean) + list(inactives_clean)

    # put together in a dataframe and shuffle the rows
    df = pd.DataFrame({'smiles': smiles, 'y': y})
    df = df.sample(frac=1).reset_index(drop=True)

    df.to_csv(f'data/clean/Ames_mutagenicity.csv', index=False)


def process_chembl(version: int = 36):

    # Read ChEMBL
    chembl_path = f"data/ChEMBL/chembl_{version}_chemreps.txt"
    if not os.path.exists(chembl_path):
        raise FileNotFoundError(f"Could not locate {chembl_path}. Place the chemreps file there or pass the correct version.")
    chembl_smiles = pd.read_table(chembl_path).canonical_smiles.tolist()

    print('started with', len(chembl_smiles))  # 2,372,674
    # Clean smiles and get rid of duplicates
    chembl_smiles_clean, chembl_smiles_failed = clean_mols(chembl_smiles)

    print('Parsing errors:')
    [print(f"{k}: {v}") for k, v in Counter(chembl_smiles_failed['reason']).items()]

    print('clean smiles', len(chembl_smiles_clean['clean']))

    chembl_smiles_clean = list(set(chembl_smiles_clean['clean']))
    chembl_smiles_clean = [smi for smi in chembl_smiles_clean if type(smi) is str and smi != '']
    '''1,974,867 were unique '''
    print('uniques', len(chembl_smiles_clean))

    # Save cleaned SMILES strings to a csv file for later use
    output_path = f"data/clean/ChEMBL_{version}.csv"
    pd.DataFrame({'smiles': chembl_smiles_clean}).to_csv(output_path, index=False)


def required_files_exist(paths: list[str]) -> bool:
    """Return True if all paths exist, else print warning and return False."""
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        print(f"Skipping step, missing required files: {missing}")
        return False
    return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Clean raw datasets into standardized CSV files.")
    parser.add_argument('--datasets', nargs='+', choices=['litpcba', 'moleculeace', 'ames', 'chembl'],
                        help="Subset of dataset groups to process. Defaults to all available.")
    parser.add_argument('--chembl-version', type=int, default=36,
                        help="ChEMBL release number to use for chemreps (default: 36).")
    args = parser.parse_args()

    selected_groups = set(args.datasets) if args.datasets else {'litpcba', 'moleculeace', 'ames', 'chembl'}

    os.chdir(ROOTDIR)

    if 'litpcba' in selected_groups:
        litpcba_sets = ['ESR1_ant', 'TP53', 'PPARG']
        litpcba_processed = False
        for dataset in litpcba_sets:
            required = [f'data/LitPCBA/{dataset}/actives.smi', f'data/LitPCBA/{dataset}/inactives.smi']
            if required_files_exist(required):
                process_litpcba(dataset)
                litpcba_processed = True
        if not litpcba_processed:
            print("No Lit-PCBA datasets were processed (missing files?).")

    if 'moleculeace' in selected_groups:
        moleculeace_sets = ['CHEMBL4203_Ki', 'CHEMBL2034_Ki', 'CHEMBL233_Ki', 'CHEMBL4616_EC50', 'CHEMBL287_Ki',
                            'CHEMBL218_EC50', 'CHEMBL264_Ki', 'CHEMBL219_Ki', 'CHEMBL2835_Ki', 'CHEMBL2147_Ki',
                            'CHEMBL231_Ki', 'CHEMBL3979_EC50', 'CHEMBL237_EC50', 'CHEMBL244_Ki', 'CHEMBL4792_Ki',
                            'CHEMBL1871_Ki', 'CHEMBL237_Ki', 'CHEMBL262_Ki', 'CHEMBL2047_EC50', 'CHEMBL2971_Ki',
                            'CHEMBL237_Ki', 'CHEMBL204_Ki', 'CHEMBL214_Ki', 'CHEMBL1862_Ki', 'CHEMBL234_Ki',
                            'CHEMBL238_Ki', 'CHEMBL235_EC50', 'CHEMBL4005_Ki', 'CHEMBL236_Ki', 'CHEMBL228_Ki']
        for dataset in moleculeace_sets:
            path = f'data/moleculeace_original/{dataset}.csv'
            if required_files_exist([path]):
                process_moleculeace(dataset)

    if 'ames' in selected_groups:
        ames_source = 'data/Ames_mutagenicity/smiles_cas_N6512.smi'
        if required_files_exist([ames_source]):
            process_ames()

    if 'chembl' in selected_groups:
        process_chembl(version=args.chembl_version)

    # Parsing errors:
    # Does not fit vocab: 231632
    # P with a valency of 5: 27595
    # Isotope: 1501
    # None: 53
    # Other: 1
    # clean smiles:  2,111,892
    # uniques: 1,952,050
