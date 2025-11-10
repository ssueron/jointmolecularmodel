import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('data/cns_data/B3DB_classification.csv')

# Remove missing values
df_clean = df.dropna(subset=['logBB'])

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Linear scale
ax1.hist(df_clean['logBB'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax1.set_xlabel('logBB')
ax1.set_ylabel('Frequency')
ax1.set_title('logBB Distribution (Linear Scale)')
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Log scale
ax2.hist(df_clean['logBB'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax2.set_xlabel('logBB')
ax2.set_ylabel('Frequency (log scale)')
ax2.set_yscale('log')
ax2.set_title('logBB Distribution (Log Scale)')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/logbb_distribution.png', dpi=300)
plt.close()

print(f"Analyzed {len(df_clean)} molecules with logBB values")
print(f"logBB range: {df_clean['logBB'].min():.2f} to {df_clean['logBB'].max():.2f}")
print(f"Mean logBB: {df_clean['logBB'].mean():.2f}")
print(f"Median logBB: {df_clean['logBB'].median():.2f}")
print(f"Saved: plots/logbb_distribution.png")
