import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the results
try:
    df = pd.read_csv('results.csv')
    print("Data loaded successfully:")
    print(df.head())
except FileNotFoundError:
    print("Error: results.csv not found. Please run the C program first.")
    exit(1)

# Set up the plotting style
plt.style.use('seaborn-v0_8')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Colors for consistency
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# Plot 1: Runtime vs Thread Count
ax1.plot(df['Threads'], df['Runtime(s)'], 'o-', color=colors[0], linewidth=2, markersize=8)
ax1.set_xlabel('Number of Threads', fontsize=12)
ax1.set_ylabel('Runtime (seconds)', fontsize=12)
ax1.set_title('Runtime vs Number of Threads\n(Min-Max-Mean Computation)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(df['Threads'])

# Add runtime values as text annotations
for i, (x, y) in enumerate(zip(df['Threads'], df['Runtime(s)'])):
    ax1.annotate(f'{y:.2f}s', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

# Plot 2: Speedup vs Thread Count
ax2.plot(df['Threads'], df['Speedup'], 'o-', color=colors[1], linewidth=2, markersize=8, label='Actual Speedup')

# Add ideal speedup line for comparison
ideal_speedup = df['Threads'].values
ax2.plot(df['Threads'], ideal_speedup, '--', color=colors[3], alpha=0.7, label='Ideal Speedup')

ax2.set_xlabel('Number of Threads', fontsize=12)
ax2.set_ylabel('Speedup', fontsize=12)
ax2.set_title('Speedup vs Number of Threads\n(Min-Max-Mean Computation)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(df['Threads'])
ax2.legend()

# Add speedup values as text annotations
for i, (x, y) in enumerate(zip(df['Threads'], df['Speedup'])):
    ax2.annotate(f'{y:.1f}x', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()

# Save the combined plot
plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
print("Combined performance analysis saved as 'performance_analysis.png'")

# Create individual plots as well
# Runtime plot
plt.figure(figsize=(10, 6))
plt.plot(df['Threads'], df['Runtime(s)'], 'o-', color=colors[0], linewidth=3, markersize=10)
plt.xlabel('Number of Threads', fontsize=14)
plt.ylabel('Runtime (seconds)', fontsize=14)
plt.title('Runtime Performance: Min-Max-Mean Computation', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(df['Threads'])

# Add annotations
for i, (x, y) in enumerate(zip(df['Threads'], df['Runtime(s)'])):
    plt.annotate(f'{y:.2f}s', (x, y), textcoords="offset points", xytext=(0,15), ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('runtime.png', dpi=300, bbox_inches='tight')
print("Runtime plot saved as 'runtime.png'")

# Speedup plot
plt.figure(figsize=(10, 6))
plt.plot(df['Threads'], df['Speedup'], 'o-', color=colors[1], linewidth=3, markersize=10, label='Actual Speedup')
plt.plot(df['Threads'], ideal_speedup, '--', color=colors[3], linewidth=2, alpha=0.7, label='Ideal Speedup')
plt.xlabel('Number of Threads', fontsize=14)
plt.ylabel('Speedup', fontsize=14)
plt.title('Speedup Analysis: Min-Max-Mean Computation', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(df['Threads'])
plt.legend(fontsize=12)

# Add annotations
for i, (x, y) in enumerate(zip(df['Threads'], df['Speedup'])):
    plt.annotate(f'{y:.1f}x', (x, y), textcoords="offset points", xytext=(0,15), ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('speedup.png', dpi=300, bbox_inches='tight')
print("Speedup plot saved as 'speedup.png'")

# Print some statistics
print(f"\nPerformance Summary:")
print(f"Best speedup: {df['Speedup'].max():.2f}x with {df.loc[df['Speedup'].idxmax(), 'Threads']} threads")
print(f"Serial runtime: {df[df['Threads'] == 1]['Runtime(s)'].values[0]:.2f} seconds")
print(f"Best parallel runtime: {df['Runtime(s)'].min():.2f} seconds with {df.loc[df['Runtime(s)'].idxmin(), 'Threads']} threads")

# Calculate efficiency
df['Efficiency'] = df['Speedup'] / df['Threads'] * 100
print(f"Best efficiency: {df['Efficiency'].max():.1f}% with {df.loc[df['Efficiency'].idxmax(), 'Threads']} threads")

plt.show()