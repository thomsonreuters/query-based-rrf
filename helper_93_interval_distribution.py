import matplotlib.pyplot as plt
import numpy as np

# Data Setup
datasets = ['MS MARCO', 'NQ', 'ACORD', 'NFCorpus']
combinations = ['BM25 vs Biencoder', 'BM25 vs Qwen3', 'RM3 vs Biencoder', 'RM3 vs Qwen3']
categories = ['0', 'Single', '1 Interval', 'Multi Interval']

# Raw data inputs
data_raw = {
    'MS MARCO': [
        [32, 1, 65, 2],
        [30, 1, 67, 2],
        [33, 1, 65, 2],
        [31, 1, 67, 2]
    ],
    'NQ': [
        [33, 1, 65, 1],
        [23, 1, 75, 1],
        [34, 1, 64, 1],
        [22, 1, 76, 1]
       
    ],
    'ACORD': [
        [0, 22, 78, 0],
        [0, 29, 71, 0],
        [0, 22, 76, 2],
        [0, 14, 86, 0]
        
    ],
    'NFCorpus': [
        [0, 27, 72, 1],
        [0, 23, 75, 2],
        [0, 27, 71, 2],
        [0, 24, 74, 2]   
    ]
}

# Normalize data to sum to 100%
data_normalized = {}
for ds in datasets:
    norm_comb = []
    for comb in data_raw[ds]:
        total = sum(comb)
        norm_comb.append([x / total * 100 for x in comb])
    data_normalized[ds] = norm_comb

# Plotting Setup
fig, ax = plt.subplots(figsize=(14, 8))

bar_width = 0.2
index = np.arange(len(datasets)) 
offsets = [-1.5 * bar_width, -0.5 * bar_width, 0.5 * bar_width, 1.5 * bar_width]
colors = ["#49B549", "#5f5ff2", "#8ac7f9", "#ef8e2c"] # Red, Blue, Green, Orange

# Create Bars
for i, ds in enumerate(datasets):
    comb_data = data_normalized[ds]
    
    for j, (comb_vals, offset) in enumerate(zip(comb_data, offsets)):
        x = index[i] + offset
        bottom = 0
        
        for k, val in enumerate(comb_vals):
            # Only add label for the first bar for the legend
            label = categories[k] if i == 0 and j == 0 else ""
            
            ax.bar(x, val, bar_width, bottom=bottom, color=colors[k], label=label, edgecolor='white')
            bottom += val
            
            # Add percentage text if segment is large enough
            if val > 5:
                ax.text(x, bottom - val/2, f"{val:.0f}%", ha='center', va='center', fontsize=8, color='black')

# Formatting
ax.set_ylabel('Percentage')
ax.set_title('Optimal Sparse Retriever Weight Pattern Analysis')
ax.set_xticks(index)
ax.set_xticklabels(datasets, fontsize=12, fontweight='bold')
plt.ylim(0, 101)

# Add labels for combinations below the bars
for i, ds in enumerate(datasets):
    for j, offset in enumerate(offsets):
        x = index[i] + offset
        c_name = combinations[j].replace(' vs ', '\nvs\n').replace('Biencoder', 'BiEnc')
        ax.text(x, -6, c_name, ha='center', va='top', fontsize=8)

# Legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), title='Optimal Weights', loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)

plt.subplots_adjust(bottom=0.2)
plt.tight_layout()
plt.savefig('dataset_stats_plot.jpg')