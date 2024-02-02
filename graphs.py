import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import rc

sns.set_palette("tab10")
plt.rcParams["font.family"] = "serif"


file_paths = [
    './run1000/dice/dice_A.csv',
    './run1000/dice/dice_B.csv',
    './run1000/dice/dice_C.csv',
    './run1000/dice/dice_D.csv',
    './run1000/dice/dice_E.csv',
    './run1000/dice/dice_F.csv',
    './run1000/dice/dice_G.csv',
]

model_dataframes = {
    'Clean Model': pd.DataFrame(),
    'Gaussian': pd.DataFrame(), 
    'Salt and Pepper': pd.DataFrame(), 
    'Grad Cam Gaussian': pd.DataFrame(), 
    'Grad Cam Salt and Pepper': pd.DataFrame()
    }


labels = {
    'Clean Model': {'A': 'Standard U-Net', 'B': 'Standard U-Net', 'C': 'Standard U-Net', 'D': 'Standard U-Net', 'E': 'Standard U-Net', 'F': 'Standard U-Net', 'G': 'Standard U-Net'},
    'Gaussian': {'A': '$\sigma=0.01$', 'B': '$\sigma=0.02$', 'C': '$\sigma=0.03$', 'D': '$\sigma=0.04$', 'E': '$\sigma=0.05$', 'F': '$\sigma=0.1$', 'G': '$\sigma=0.2$'},
    'Salt and Pepper': {'A': '$p=0.001$', 'B': '$p=0.005$', 'C': '$p=0.01$', 'D': '$p=0.02$', 'E': '$p=0.03$', 'F': '$p=0.04$', 'G': '$p=0.05$'},
    'Grad Cam Gaussian': {'A': '$\sigma=0.01$', 'B': '$\sigma=0.02$', 'C': '$\sigma=0.03$', 'D': '$\sigma=0.04$', 'E': '$\sigma=0.05$', 'F': '$\sigma=0.1$', 'G': '$\sigma=0.2$'},
    'Grad Cam Salt and Pepper': {'A': '$p=0.001$', 'B': '$p=0.005$', 'C': '$p=0.01$', 'D': '$p=0.02$', 'E': '$p=0.03$', 'F': '$p=0.04$', 'G': '$p=0.05$'},
}
used = False

for file_path in file_paths:
    df = pd.read_csv(file_path)
    config = file_path.split('_')[1][0]

    for column in df.columns[1:]:
        model_name = column.split(' - ')[-1].strip()

        if model_name == 'Clean Model' and not used:
            model_dataframes[model_name] = pd.concat([model_dataframes[model_name], df[['Epoch', column]].set_index('Epoch')], axis=1)
            used = True
        else:
            model_dataframes[model_name] = pd.concat([model_dataframes[model_name], df[['Epoch', column]].set_index('Epoch').rename(columns={column: config})], axis=1)


# Plot a line chart for each model type
for model_type, df in model_dataframes.items():
    grouped_df = df.groupby(np.arange(len(df)) // 10).mean()
    model_dataframes[model_type] = grouped_df

titles = {
    'Clean Model': 'Standard',
    'Gaussian': 'Gaussian Model',
    'Salt and Pepper': 'Salt and Pepper Model',
    'Grad Cam Gaussian': 'Grad-CAM Gaussian Model',
    'Grad Cam Salt and Pepper': 'Grad-CAM Salt and Pepper Model'
}

fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex='col', sharey='row')
plt.subplots_adjust(hspace=2, wspace=0)

# Subplot 1
df = model_dataframes['Gaussian']
xvals = np.arange(len(df) + 1) * 10
axs[0, 0].plot(xvals, np.insert(model_dataframes['Clean Model']['B'].values, 0, 0), label='Standard Model', linestyle='--', color='black', linewidth=1.5)

for column in df.columns:
    config_label = labels['Gaussian'][column]
    axs[0, 0].plot(xvals, np.insert(df[column].values, 0, 0), label=f'{config_label}', linewidth=1.5)
    
axs[0, 0].set_xlim(left=0, right=1000)
axs[0, 0].set_ylim(0.65, 0.95)
axs[0, 0].set_xticks([])
axs[0, 0].set_yticks([])
axs[0, 0].set_xlabel('')
axs[0, 0].set_ylabel('Dice Score', fontsize=14)
axs[0, 0].legend(fontsize=12, )
axs[0, 0].set_title(f'{titles["Gaussian"]}', fontsize=16, fontweight='bold')
axs[0, 0].grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)

# Subplot 2
df = model_dataframes['Grad Cam Gaussian']
xvals = np.arange(len(df) + 1) * 10
axs[0, 1].plot(xvals, np.insert(model_dataframes['Clean Model']['B'].values, 0, 0), label='Standard Model', linestyle='--', color='black', linewidth=1.5)
for column in df.columns:
    config_label = labels['Grad Cam Gaussian'][column]
    axs[0, 1].plot(xvals, np.insert(df[column].values, 0, 0), label=f'{config_label}', linewidth=1.5)
axs[0, 1].set_xlim(left=0, right=1000)
axs[0, 1].set_ylim(0.65, 0.95)
axs[0, 1].set_xticks([])
axs[0, 1].set_yticks(np.arange(0.65, 0.96, 0.05))
axs[0, 1].set_xlabel('')
axs[0, 1].set_ylabel('')
axs[0, 1].legend(fontsize=12)
axs[0, 1].set_title(f'{titles["Grad Cam Gaussian"]}', fontsize=16, fontweight='bold')
axs[0, 1].grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)

# Subplot 3
df = model_dataframes['Salt and Pepper']
xvals = np.arange(len(df) + 1) * 10
axs[1, 0].plot(xvals, np.insert(model_dataframes['Clean Model']['B'].values, 0, 0), label='Standard Model', linestyle='--', color='black', linewidth=1.5)
for column in df.columns:
    config_label = labels['Salt and Pepper'][column]
    axs[1, 0].plot(xvals, np.insert(df[column].values, 0, 0), label=f'{config_label}', linewidth=1.5)
axs[1, 0].set_xlim(left=-10, right=1000)
axs[1, 0].set_ylim(0.65, 0.95)
axs[1, 0].set_xticks(np.arange(0, 901, 100))
axs[1, 0].set_yticks([])
axs[1, 0].set_xlabel('Epoch', fontsize=14)
axs[1, 0].set_ylabel('Dice Score', fontsize=14)
axs[1, 0].legend(fontsize=12)
axs[1, 0].set_title(f'{titles["Salt and Pepper"]}', fontsize=16, fontweight='bold')
axs[1, 0].grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)

# Subplot 4
df = model_dataframes['Grad Cam Salt and Pepper']
xvals = np.arange(len(df) + 1) * 10
axs[1, 1].plot(xvals, np.insert(model_dataframes['Clean Model']['B'].values, 0, 0), label='Standard Model', linestyle='--', color='black', linewidth=1.5)
for column in df.columns:
    config_label = labels['Grad Cam Salt and Pepper'][column]
    axs[1, 1].plot(xvals, np.insert(df[column].values, 0, 0), label=f'{config_label}', linewidth=1.5)
axs[1, 1].set_xlim(left=-10, right=1000)
axs[1, 1].set_ylim(0.65, 0.95)
axs[1, 1].set_xticks(np.arange(0, 901, 100))
axs[1, 1].set_yticks(np.arange(0.65, 0.96, 0.05))
axs[1, 1].set_xlabel('Epoch', fontsize=14)
axs[1, 1].set_ylabel('')
axs[1, 1].legend(fontsize=12)
axs[1, 1].set_title(f'{titles["Grad Cam Salt and Pepper"]}', fontsize=16, fontweight='bold')
axs[1, 1].grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)

plt.tight_layout(h_pad=.5, w_pad=-.5)
plt.savefig('./run1000/dice/subplots.pdf')
plt.show()