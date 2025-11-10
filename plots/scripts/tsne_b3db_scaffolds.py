import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.manifold import TSNE

# Load data
df = pd.read_csv('data/cns_data/B3DB_classification.csv')

# Generate ECFP fingerprints
fps = []
valid_indices = []
for idx, smiles in enumerate(df['SMILES']):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fps.append(np.array(fp))
        valid_indices.append(idx)

fps = np.array(fps)
df_valid = df.iloc[valid_indices].reset_index(drop=True)

# Run t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(fps)

# Plot 1: Color by reliability
fig, ax = plt.subplots(figsize=(10, 8))
colors = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c', 'D': '#d62728'}
for group in ['A', 'B', 'C', 'D']:
    mask = df_valid['group'] == group
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               c=colors[group], label=f'Group {group}', alpha=0.6, s=10)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_title('t-SNE of B3DB Dataset by Reliability Group')
ax.legend()
plt.tight_layout()
plt.savefig('plots/tsne_b3db_reliability.png', dpi=300)
plt.close()

# Plot 2: Color by BBB label
fig, ax = plt.subplots(figsize=(10, 8))
colors_bbb = {'BBB+': '#2ca02c', 'BBB-': '#d62728'}
for label in ['BBB+', 'BBB-']:
    mask = df_valid['BBB+/BBB-'] == label
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               c=colors_bbb[label], label=label, alpha=0.6, s=10)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_title('t-SNE of B3DB Dataset by BBB Permeability')
ax.legend()
plt.tight_layout()
plt.savefig('plots/tsne_b3db_bbb_label.png', dpi=300)
plt.close()

print(f"Generated t-SNE plots for {len(df_valid)} molecules")
print(f"Saved: plots/tsne_b3db_reliability.png")
print(f"Saved: plots/tsne_b3db_bbb_label.png")
