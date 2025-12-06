#!/usr/bin/env python3
"""
Script to create a labeled GO enrichment dot plot with specific terms highlighted
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Read the GO enrichment results
go_results = pd.read_csv('results/task4_go_enrichment_bp.csv')

# Select top 20 terms
results = go_results.head(20).copy()
results['-log10(FDR)'] = -np.log10(results['Adjusted P-value'])
results['Gene_Count'] = results['Overlap'].apply(lambda x: int(x.split('/')[0]))

# Create figure
fig, ax = plt.subplots(figsize=(16, 12))

# Create scatter plot
scatter = ax.scatter(results['Gene_Count'], results['-log10(FDR)'],
                     s=results['Gene_Count']*30,  # Larger bubbles
                     c=results['Adjusted P-value'],
                     cmap='RdYlBu_r', alpha=0.7, edgecolors='black', linewidth=2)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, label='Adjusted P-value', shrink=0.8)
cbar.ax.tick_params(labelsize=14)
cbar.set_label('Adjusted P-value', size=16, weight='bold')

# Label specific bubbles mentioned in the table
labels_to_add = {
    'Cell Population Proliferation': 'upper right (large)',
    'Cell Migration': 'middle right',
    'Inflammatory Response': 'lower middle',
    'Angiogenesis': 'left side'
}

# Find these terms in the data and label them
for idx, row in results.iterrows():
    term_name = row['Term']
    
    # Check if this term should be labeled (partial match)
    for label_key in labels_to_add.keys():
        if label_key.lower() in term_name.lower():
            # Add annotation with arrow
            ax.annotate(term_name,
                       xy=(row['Gene_Count'], row['-log10(FDR)']),
                       xytext=(20, 20), textcoords='offset points',
                       fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7, edgecolor='black', linewidth=2),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', 
                                     color='black', lw=2))
            break

# Formatting
ax.set_xlabel('Gene Count', fontsize=18, fontweight='bold')
ax.set_ylabel('-log10(Adjusted P-value)', fontsize=18, fontweight='bold')
ax.set_title('GO Biological Process Enrichment\n(Key Terms Labeled)', 
             fontsize=20, fontweight='bold', pad=20)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.grid(True, alpha=0.3, linestyle='--')

# Add legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Bubble size = Gene count',
                         markerfacecolor='steelblue', markersize=15, markeredgecolor='black', markeredgewidth=2)]
ax.legend(handles=legend_elements, loc='upper left', fontsize=14, framealpha=0.9)

plt.tight_layout()
plt.savefig('results/task4_go_enrichment_labeled.png', dpi=300, bbox_inches='tight')
print("✓ Created labeled GO enrichment plot: results/task4_go_enrichment_labeled.png")

# Also create a simple table showing bubble positions
print("\n=== GO Term Bubble Locations ===")
print(f"{'GO Term':<50} | Gene Count | -log10(FDR) | Position")
print("="*100)

# Determine positions based on coordinates
for idx, row in results.iterrows():
    term_name = row['Term']
    gene_count = row['Gene_Count']
    log_fdr = row['-log10(FDR)']
    
    # Determine position category
    if gene_count > 100 and log_fdr > 6:
        position = "Upper right (large)"
    elif gene_count > 80 and 4 < log_fdr < 6:
        position = "Middle right"
    elif 40 < gene_count < 60 and log_fdr < 4:
        position = "Lower middle"
    elif gene_count < 40 and log_fdr > 5:
        position = "Left side"
    else:
        position = "Other"
    
    # Only print key terms
    for label_key in labels_to_add.keys():
        if label_key.lower() in term_name.lower():
            print(f"{term_name[:48]:<50} | {gene_count:>10} | {log_fdr:>11.2f} | {position}")
            break

plt.show()
