# %%
import numpy as np
import pandas as pd
import torch
import os 

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

from tqdm import tqdm
import argparse

try:
    import seaborn as sns
except ImportError:
    sns = None

def main(args):
    dataname = args.dataname        
    sample_file_name = args.sample_file_name

    syn_path = f'synthetic/{dataname}/{sample_file_name}'
    real_path = f'synthetic/{dataname}/real.csv'

    syn_data = pd.read_csv(syn_path)
    real_data = pd.read_csv(real_path)

    print((real_data[:2]))

    data_dir = f'data/{dataname}' 
    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)

    big_img = plot_density(syn_data, real_data, info)

    save_dir =  f"eval/density_graphs/{dataname}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, sample_file_name.replace('.csv', '.png'))
    big_img.save(save_path)
    print(f"Saved density graph to {save_path}")

def _is_categorical(info, col_idx):
    column_meta = info['metadata']['columns'][str(col_idx)]
    return column_meta['sdtype'] == 'categorical'


def _plot_numeric(ax, real_col, syn_col, col_name):
    real_values = pd.to_numeric(real_col, errors='coerce').dropna()
    syn_values = pd.to_numeric(syn_col, errors='coerce').dropna()

    if real_values.empty and syn_values.empty:
        ax.text(0.5, 0.5, "No numeric values", ha='center', va='center')
        ax.set_title(col_name)
        return

    if sns is not None:
        if not real_values.empty:
            sns.histplot(real_values, stat='density', bins='auto', element='step',
                         fill=False, color='#1f77b4', label='Real', ax=ax)
            if real_values.nunique(dropna=True) > 1:
                sns.kdeplot(real_values, color='#1f77b4', ax=ax)
        if not syn_values.empty:
            sns.histplot(syn_values, stat='density', bins='auto', element='step',
                         fill=False, color='#ff7f0e', label='Synthetic', ax=ax)
            if syn_values.nunique(dropna=True) > 1:
                sns.kdeplot(syn_values, color='#ff7f0e', ax=ax)
    else:
        if not real_values.empty:
            ax.hist(real_values, bins='auto', density=True, histtype='step',
                    color='#1f77b4', label='Real')
        if not syn_values.empty:
            ax.hist(syn_values, bins='auto', density=True, histtype='step',
                    color='#ff7f0e', label='Synthetic')

    ax.set_title(col_name)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)


def _plot_categorical(ax, real_col, syn_col, col_name):
    real_counts = real_col.astype(str).value_counts(normalize=True, dropna=False)
    syn_counts = syn_col.astype(str).value_counts(normalize=True, dropna=False)
    categories = list(real_counts.index)
    categories.extend(cat for cat in syn_counts.index if cat not in real_counts.index)

    positions = np.arange(len(categories))
    width = 0.4
    real_values = [real_counts.get(cat, 0.0) for cat in categories]
    syn_values = [syn_counts.get(cat, 0.0) for cat in categories]

    ax.bar(positions - width / 2, real_values, width=width, label='Real', color='#1f77b4')
    ax.bar(positions + width / 2, syn_values, width=width, label='Synthetic', color='#ff7f0e')
    ax.set_title(col_name)
    ax.set_ylabel("Frequency")
    ax.set_xticks(positions)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8)


def plot_density(syn_data, real_data, info, num_per_row=3):
    column_names = info['column_names']
    num_cat = len(column_names)
    num_col = num_per_row
    num_row = (num_cat-1)//num_col+1

    fig, axes = plt.subplots(num_row, num_col, figsize=(5 * num_col, 3.6 * num_row), squeeze=False)
    axes = axes.flatten()

    for i, col in tqdm(enumerate(column_names), total=len(column_names)):
        ax = axes[i]
        if _is_categorical(info, i):
            _plot_categorical(ax, real_data[col], syn_data[col], col)
        else:
            _plot_numeric(ax, real_data[col], syn_data[col], col)

    for ax in axes[num_cat:]:
        ax.axis('off')

    fig.tight_layout()
    img_bytes = BytesIO()
    fig.savefig(img_bytes, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    img_bytes.seek(0)
    return Image.open(img_bytes).convert('RGB')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataname', type=str, default='adult')
    parser.add_argument('--sample_file_name', type=str, default='tabsyn.csv')

    args = parser.parse_args()
    
    main(args)