import pandas as pd
import matplotlib.pyplot as plt
import sys

file_path = sys.argv[1] if len(sys.argv) > 1 else 'data/results/pytorch_t1.0s_n1.parquet'

print(f"Loading {file_path}...")
df = pd.read_parquet(file_path)
print(f"Total spikes: {len(df)}")
print(f"Unique neurons: {df['neuron_index'].nunique()}")
print(f"Time range: {df['time_ms'].min():.2f}ms - {df['time_ms'].max():.2f}ms")

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 1. Raster plot
ax = axes[0]
ax.scatter(df['time_ms'], df['neuron_index'], s=1, alpha=0.3, c='black')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Neuron index')
ax.set_title(f'Spike Raster Plot — {len(df)} spikes, {df["neuron_index"].nunique()} active neurons')
ax.grid(True, alpha=0.3)

# 2. PSTH (spike rate over time)
ax = axes[1]
ax.hist(df['time_ms'], bins=200, color='steelblue', edgecolor='black', alpha=0.7)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Spike count')
ax.set_title('Spike Count Over Time (PSTH)')
ax.grid(True, alpha=0.3)

# 3. Top 20 active neurons
ax = axes[2]
top20 = df['neuron_index'].value_counts().head(20)
ax.barh(range(len(top20)), top20.values, color='coral', edgecolor='black')
ax.set_yticks(range(len(top20)))
ax.set_yticklabels([f"N{idx}" for idx in top20.index], fontsize=8)
ax.set_xlabel('Spike count')
ax.set_title('Top 20 Most Active Neurons')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
output = file_path.replace('.parquet', '_plot.png')
plt.savefig(output, dpi=150)
print(f"Saved plot to: {output}")
plt.show()