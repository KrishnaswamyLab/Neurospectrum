import os
import numpy as np
import matplotlib.pyplot as plt
import phate
import torch
import pickle
from numpy import linalg as LA
from pathlib import Path

from config import ALPHA, MARKER_SIZE, T_VALUE

#Basline, scattering, PCA, and ICA comparison
def compare_results(
    cross_validation_path='cross_validation_results.pkl',
    transformed_results_path='comparison_results/transformed_results.pkl',
    output_path='comparison_results/method_comparison.png'
):
    with open(cross_validation_path, 'rb') as f:
        cv_results = pickle.load(f)
    raw_results = cv_results['fnirs']
    scattering_results = cv_results['scattering']
    
    with open(transformed_results_path, 'rb') as f:
        transformed_results = pickle.load(f)
    subject_means = [
        raw_results['subject']['mean_accuracy'],
        scattering_results['subject']['mean_accuracy'],
        transformed_results['pca_fnirs_subject']['mean_accuracy'],
        transformed_results['ica_fnirs_subject']['mean_accuracy']
    ]
    subject_stds = [
        raw_results['subject']['std_accuracy'],
        scattering_results['subject']['std_accuracy'],
        transformed_results['pca_fnirs_subject']['std_accuracy'],
        transformed_results['ica_fnirs_subject']['std_accuracy']
    ]
    time_means = [
        raw_results['time']['mean_accuracy'],
        scattering_results['time']['mean_accuracy'],
        transformed_results['pca_fnirs_time']['mean_accuracy'],
        transformed_results['ica_fnirs_time']['mean_accuracy']
    ]
    time_stds = [
        raw_results['time']['std_accuracy'],
        scattering_results['time']['std_accuracy'],
        transformed_results['pca_fnirs_time']['std_accuracy'],
        transformed_results['ica_fnirs_time']['std_accuracy']
    ]

    methods = ['Raw fNIRS', 'Scattering', 'PCA', 'ICA']
    x = np.arange(len(methods))
    width = 0.6
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    
    #Setup Figure
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12
    })
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', length=0)

    # Subject-withheld 
    bars = axes[0].bar(x, subject_means, yerr=subject_stds, capsize=6,
                       color=colors, edgecolor='black', linewidth=1)
    axes[0].set_title("Subject-Withheld Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=20, ha='right')
    axes[0].set_ylim(0.2, 1.0)

    for bar, val in zip(bars, subject_means):
        axes[0].text(bar.get_x() + bar.get_width()/2., val + 0.015,
                    f"{val:.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold')

    #Time-withheld plot
    bars = axes[1].bar(x, time_means, yerr=time_stds, capsize=6,
                       color=colors, edgecolor='black', linewidth=1)
    axes[1].set_title("Time-Withheld Accuracy")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=20, ha='right')
    axes[1].set_ylim(0.2, 1.0)
    for bar, val in zip(bars, time_means):
        axes[1].text(bar.get_x() + bar.get_width()/2., val + 0.015,
                    f"{val:.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"Figure saved to: {output_path}")
    
    return {
        'subject': {
            'means': subject_means,
            'stds': subject_stds
        },
        'time': {
            'means': time_means,
            'stds': time_stds
        }
    }

# Now incl. latent spaces 
def compare_all_methods(
    cross_validation_path='cross_validation_results.pkl',
    transformed_results_path='comparison_results/transformed_results.pkl',
    latent_results_dir='extended_latent_results',
    output_path='comparison_results/extended_latent_comparison.png'
):
    with open(cross_validation_path, 'rb') as f:
        cv_results = pickle.load(f)
    
    with open(transformed_results_path, 'rb') as f:
        transformed_results = pickle.load(f)
    raw_results = cv_results['fnirs']
    scattering_results = cv_results['scattering']

    # Hardcoded from earlier runs. Change this.
    latent_8d_subject = 54.46 / 100
    latent_8d_time = 61.08 / 100
    latent_48d_subject = 53.53 / 100
    latent_48d_time = 64.26 / 100
    
    #Autoencoders don't have cross-val yet because of training leakage. NEED TO TALK ABT THIS IN LIMITAITON SECTION
    autoencoder_sd = 0
    
    subject_means = [
        raw_results['subject']['mean_accuracy'],
        scattering_results['subject']['mean_accuracy'],
        transformed_results['pca_fnirs_subject']['mean_accuracy'],
        transformed_results['ica_fnirs_subject']['mean_accuracy'],
        latent_8d_subject,
        latent_48d_subject
    ]
    subject_stds = [
        raw_results['subject']['std_accuracy'],
        scattering_results['subject']['std_accuracy'],
        transformed_results['pca_fnirs_subject']['std_accuracy'],
        transformed_results['ica_fnirs_subject']['std_accuracy'],
        autoencoder_sd,  
        autoencoder_sd 
    ]
    time_means = [
        raw_results['time']['mean_accuracy'],
        scattering_results['time']['mean_accuracy'],
        transformed_results['pca_fnirs_time']['mean_accuracy'],
        transformed_results['ica_fnirs_time']['mean_accuracy'],
        latent_8d_time,
        latent_48d_time
    ]
    
    time_stds = [
        raw_results['time']['std_accuracy'],
        scattering_results['time']['std_accuracy'],
        transformed_results['pca_fnirs_time']['std_accuracy'],
        transformed_results['ica_fnirs_time']['std_accuracy'],
        autoencoder_sd, 
        autoencoder_sd  
    ]

    #Setup Figure
    methods = ['Raw fNIRS', 'Scattering', 'PCA', 'ICA', "Neurospectrum (8D)", "Neurospectrum (48D)"]
    x = np.arange(len(methods))
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#e11584']
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12
    })
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', length=0)
    
    # Subject-withheld
    bars = axes[0].bar(x, subject_means, yerr=subject_stds, capsize=6,
                      color=colors, edgecolor='black', linewidth=1)
    axes[0].set_title("Subject-Withheld Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=20, ha='right')
    axes[0].set_ylim(0.2, 1.0)
    
    for bar, val in zip(bars, subject_means):
        axes[0].text(bar.get_x() + bar.get_width()/2., val + 0.015,
                   f"{val:.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    #Time-withheld 
    bars = axes[1].bar(x, time_means, yerr=time_stds, capsize=6,
                      color=colors, edgecolor='black', linewidth=1)
    axes[1].set_title("Time-Withheld Accuracy")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=20, ha='right')
    axes[1].set_ylim(0.2, 1.0)

    for bar, val in zip(bars, time_means):
        axes[1].text(bar.get_x() + bar.get_width()/2., val + 0.015,
                   f"{val:.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold')

    fig.suptitle("Comparison of Brain State Classification Methods", fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Extended latent space comparison saved to: {output_path}")
    
    print("\n== FINAL RESULTS ==")
    print(f"{'Method':<20} {'Subject-withheld':<20} {'Time-withheld':<20}")
    print("-" * 60)
    for i, method in enumerate(methods):
        print(f"{method:<20} {subject_means[i]:.3f} ± {subject_stds[i]:.3f}    {time_means[i]:.3f} ± {time_stds[i]:.3f}")
    
    return {
        'subject': {
            'means': subject_means,
            'stds': subject_stds,
            'methods': methods
        },
        'time': {
            'means': time_means,
            'stds': time_stds,
            'methods': methods
        }
    }

def load_latent_embeddings(results_dir, latent_dims):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embeddings_dict = {}
    
    for latent_dim in latent_dims:
        embeddings_dict[latent_dim] = {}
        
        subject_path = os.path.join(results_dir, f'subject_withheld_model_latent{latent_dim}.pth')
        if os.path.exists(subject_path):
            subject_results = torch.load(subject_path, map_location=device)
            all_subject_embeddings = []
            all_subject_labels = []
            all_subject_ids = []
            
            for subject, embs in subject_results['history']['embeddings'].items():
                all_subject_embeddings.append(np.array(embs))
                all_subject_labels.extend(subject_results['history']['labels'][subject])
                all_subject_ids.extend([subject] * len(subject_results['history']['labels'][subject]))
            
            subject_latent_space = np.vstack(all_subject_embeddings)
            embeddings_dict[latent_dim]['subject'] = {
                'embeddings': subject_latent_space,
                'labels': all_subject_labels,
                'subjects': all_subject_ids
            }
        
        time_path = os.path.join(results_dir, f'time_withheld_model_latent{latent_dim}.pth')
        if os.path.exists(time_path):
            time_results = torch.load(time_path, map_location=device)
            all_time_embeddings = []
            all_time_labels = []
            all_time_ids = []
            
            for subject, embs in time_results['history']['embeddings'].items():
                all_time_embeddings.append(np.array(embs))
                all_time_labels.extend(time_results['history']['labels'][subject])
                all_time_ids.extend([subject] * len(time_results['history']['labels'][subject]))
            
            time_latent_space = np.vstack(all_time_embeddings)
            embeddings_dict[latent_dim]['time'] = {
                'embeddings': time_latent_space,
                'labels': all_time_labels,
                'subjects': all_time_ids
            }
    return embeddings_dict

# Embed with PHATE
def visualize_latent_space(latent_space, labels, subjects, split_type, latent_dim, include_rest=True):
    latent_space = np.array(latent_space)
    labels = np.array(labels)
    subjects = np.array(subjects)
    if not include_rest:
        non_rest_mask = labels != 0
        latent_space = latent_space[non_rest_mask]
        labels = labels[non_rest_mask]
        subjects = subjects[non_rest_mask]
        print(f"Filtered shape (rest excluded): {latent_space.shape}")
    print(f"PHATE embedding for {split_type} split, {latent_dim}D latent space...")
    
    phate_op = phate.PHATE(
        n_components=3,
        t=T_VALUE,
        random_state=42,
        n_jobs=-1,
        verbose=True
    )
    embedding = phate_op.fit_transform(latent_space)
    
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    if include_rest:
        task_colors = ['#2ecc71', '#e74c3c', '#3498db'] 
        task_names = ['Rest', 'Improv', 'Scale']
    else:
        task_colors = ['#e74c3c', '#3498db']
        task_names = ['Improv', 'Scale']

    # By task
    for i, (task, color) in enumerate(zip(task_names, task_colors)):
        label_idx = i if include_rest else i+1
        mask = labels == label_idx
        ax1.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            embedding[mask, 2],
            c=color,
            label=task,
            alpha=ALPHA,
            s=MARKER_SIZE
        )
    title_suffix = " (Rest Excluded)" if not include_rest else ""
    ax1.set_title(f'{split_type.capitalize()} Split: {latent_dim}D Latent Space by Task{title_suffix}', fontsize=14)
    ax1.set_xlabel('PHATE 1', fontsize=12)
    ax1.set_ylabel('PHATE 2', fontsize=12)
    ax1.set_zlabel('PHATE 3', fontsize=12)
    ax1.legend(fontsize=10)
    
    # By subject
    unique_subjects = sorted(np.unique(subjects))
    n_subjects = len(unique_subjects)
    subject_colors = [plt.cm.hsv(i/n_subjects) for i in range(n_subjects)]

    for i, subject in enumerate(unique_subjects):
        mask = subjects == subject
        ax2.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            embedding[mask, 2],
            c=[subject_colors[i]],
            label=f'Subject {subject}',
            alpha=ALPHA,
            s=MARKER_SIZE
        )
    ax2.set_title(f'{split_type.capitalize()} Split: {latent_dim}D Latent Space by Subject{title_suffix}', fontsize=14)
    ax2.set_xlabel('PHATE 1', fontsize=12) 
    ax2.set_ylabel('PHATE 2', fontsize=12)
    ax2.set_zlabel('PHATE 3', fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    for ax in [ax1, ax2]:
        ax.view_init(elev=20, azim=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_dir = 'latent_visualization_results'
    os.makedirs(output_dir, exist_ok=True)
    rest_suffix = "_with_rest" if include_rest else "_no_rest"
    filename = f"{split_type}_latent{latent_dim}d_space{rest_suffix}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')

    plt.show()

def plot_attention_weights(weights, split_type=""):
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(weights)), weights)
    plt.title(f'Node Attention Weights ({split_type.capitalize()} Split)')
    plt.xlabel('Node Index')
    plt.ylabel('Weight')
    plt.xticks(range(0, len(weights), 5))
    plt.grid(True, alpha=0.3)
    
    #Save figure
    output_dir = 'visualization_results'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{split_type}_attention_weights.png'), dpi=300)
    plt.show()

def compare_attention_with_pca(attention_weights, pca_loadings, split_name, output_dir='visualization_results'):
    
    #Only looking at first principal components? Ask Dhanajnay why we do this tho
    first_pc_loadings = pca_loadings[0]
    abs_loadings = np.abs(first_pc_loadings)
    normalized_loadings = abs_loadings / np.sum(abs_loadings)
    correlation = np.corrcoef(attention_weights, normalized_loadings)[0, 1]
    print(f"\nCorrelation between {split_name} and first PC loadings: {correlation:.4f}")
    

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(attention_weights)), attention_weights)
    plt.title(f'{split_name.capitalize()} Attention Weights', fontsize=14)
    plt.xlabel('Node Index', fontsize=12)
    plt.ylabel('Weight', fontsize=12)
    plt.xticks(range(0, len(attention_weights), 5))
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(normalized_loadings)), normalized_loadings)
    plt.title(f'First Principal Component Loadings', fontsize=14)
    plt.xlabel('Node Index', fontsize=12)
    plt.ylabel('Loading', fontsize=12)
    plt.xticks(range(0, len(normalized_loadings), 5))
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{split_name}_vs_first_pc_comparison.png'), dpi=300)
    plt.show()
    
    attention_top = np.argsort(attention_weights)[-5:][::-1]
    pc_top = np.argsort(normalized_loadings)[-5:][::-1]
    common_nodes = set(attention_top).intersection(set(pc_top))
    
    print(f"\nCommon top channels: {sorted(common_nodes)}")
    return correlation, attention_top, pc_top