import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from config import DATA_DIR, OUTPUT_DIR, THRESHOLD, TASK_SEGMENTS, expected_shape
from scattering import construct_adjacency_matrix, create_task_labels, generate_timepoint_feature

def clean_data(df):
    df = df.apply(pd.to_numeric, errors='coerce')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def load_fnirs_data():
    file_pattern = os.path.join(DATA_DIR, "*_Combined_Probe_Difference.csv")
    subject_data_matrices = []
    for file_path in glob.glob(file_pattern):
        subject_id = os.path.basename(file_path).split("_")[0]
        df = pd.read_csv(file_path)
        
        if df.shape != expected_shape:
            print(f"Excluding Subject {subject_id}: shape {df.shape} ≠ {expected_shape}")
            continue
        
        cleaned_df = clean_data(df)
        subject_data_matrices.append((subject_id, cleaned_df.values))

    print(f"\nProcessed {len(subject_data_matrices)} subjects")
    if len(subject_data_matrices) == 20:
        print("All subjects successfully processed")
    else:
        print(f"Only {len(subject_data_matrices)}/20 subjects processed")
    
    return subject_data_matrices

def verify_data_match(subject_channel_coords, subject_data_matrices):
    xyz_subject_ids = [subj_id for subj_id, _ in subject_channel_coords]
    fNIRS_subject_ids = [subject_id for subject_id, _ in subject_data_matrices]
    xyz_subject_ids_sorted = sorted(xyz_subject_ids)
    fNIRS_subject_ids_sorted = sorted(fNIRS_subject_ids)
    if xyz_subject_ids_sorted == fNIRS_subject_ids_sorted:
        print("Subject IDs match between datasets")
        return True
    else:
        print("Subject ID don't match between datasets")
        missing_in_fNIRS = set(xyz_subject_ids) - set(fNIRS_subject_ids)
        missing_in_XYZ = set(fNIRS_subject_ids) - set(xyz_subject_ids)

        if missing_in_fNIRS:
            print(f"Missing in fNIRS data: {missing_in_fNIRS}")
        if missing_in_XYZ:
            print(f"Missing in XYZ data: {missing_in_XYZ}")
        return False

#Deep wavelet transformatin
def generate_scatter_coefficients(subject_channel_coords, subject_data_matrices):
    fnirs_dict = {subj_id: data for subj_id, data in subject_data_matrices}
    xyz_dict = {subj_id: coords for subj_id, coords in subject_channel_coords}
    common_subjects = set(fnirs_dict) & set(xyz_dict)

    for subject_id in sorted(common_subjects):
        print(f"Processing subject {subject_id}")
        fnirs_data = fnirs_dict[subject_id]
        coords = xyz_dict[subject_id]
        adj_matrix = construct_adjacency_matrix(coords, THRESHOLD)
        ro = fnirs_data.T
        normalized_features = ro
        
        features = generate_timepoint_feature(adj_matrix, normalized_features)
        features = features.T 
        output_file = os.path.join(OUTPUT_DIR, f"subject_{subject_id}_scattering_coeffs.csv")
        np.savetxt(output_file, features, delimiter=',')

    print(f"Generated features for {len(common_subjects)} subjects")

def combine_fnirs_data(data_dir, output_file):
    file_pattern = os.path.join(data_dir, "*_Combined_Probe_Difference.csv")
    _, general_labels, timepoints = create_task_labels(TASK_SEGMENTS)
    all_data = []

    for file_path in glob.glob(file_pattern):
        subject_id = os.path.basename(file_path).split("_")[0]
        df = pd.read_csv(file_path)
        
        if df.shape != expected_shape:
            print(f"Excluding Subject {subject_id}: shape {df.shape} ≠ {expected_shape}")
            continue
        cleaned_df = clean_data(df)
        subject_data = pd.DataFrame(cleaned_df.values)
        subject_data['subject_id'] = subject_id
        subject_data['time_index'] = timepoints
        subject_data['task'] = general_labels
        all_data.append(subject_data)
    combined_df = pd.concat(all_data, ignore_index=True)
    
    #Rename columns
    all_columns = combined_df.columns.tolist()
    node_cols = [f'node_{i}' for i in range(len(all_columns) - 3)]
    new_columns = node_cols + ['subject_id', 'time_index', 'task']
    combined_df.columns = new_columns
    print(f"\nDataset: {len(combined_df)} samples, {len(node_cols)} features")    

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined_df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
    
    return combined_df

def combine_scattering_data(scattering_dir, output_file):
    scattering_dir = Path(scattering_dir)
    scattering_files = list(scattering_dir.glob('subject_*_scattering_coeffs.csv'))
    _, general_labels, timepoints = create_task_labels(TASK_SEGMENTS)
    all_data = []
    print(f"Processing {len(scattering_files)} subjects...")
    
    for file_path in scattering_files:
        subject_id = file_path.stem.split('_')[1]
        print(f"Processing subject {subject_id}...")
        
        subject_coeffs = np.loadtxt(file_path, delimiter=',')
        subject_df = pd.DataFrame(subject_coeffs)
        subject_df['subject_id'] = subject_id
        subject_df['time_index'] = timepoints
        subject_df['task'] = general_labels
        all_data.append(subject_df)

    combined_df = pd.concat(all_data, ignore_index=True)
    feature_cols = [f'feature_{i}' for i in range(768)]
    combined_df.columns = feature_cols + ['subject_id', 'time_index', 'task']
    
    print(f"\nDataset: {len(combined_df)} samples, {len(feature_cols)} features")
    print(f"Task distribution:\n{combined_df['task'].value_counts()}")
    combined_df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
    
    return combined_df

#Setup Dataset definiton (just for pepare data function below)
class BrainStateDataset(Dataset):
    def __init__(self, features, labels, subject_ids=None, time_indices=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.subject_ids = subject_ids
        self.time_indices = None if time_indices is None else torch.LongTensor(time_indices)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        if self.subject_ids is None and self.time_indices is None:
            return self.features[idx], self.labels[idx]
        elif self.time_indices is None:
            return self.features[idx], self.labels[idx], self.subject_ids[idx]
        else:
            return (self.features[idx], 
                    self.labels[idx], 
                    self.subject_ids[idx],
                    self.time_indices[idx])

# Gets data ready for trianing (two cross-validation types tested)
def prepare_data(
    data_path,
    feature_prefix,
    include_metadata=False,
    test_size=0.2,
    random_state=42,
    split_type="random"
):
    df = pd.read_csv(data_path)
    df['subject_id'] = df['subject_id'].astype(str)
    df['time_index'] = df['time_index'].astype(int)
    df = df[df['task'].isin(['Rest', 'Improv', 'Scale'])].reset_index(drop=True)
    df['label'] = df['task'].map({'Rest':0, 'Improv':1, 'Scale':2})

    subject_ids = df['subject_id'].values
    time_indices = df['time_index'].values
    feature_cols = [c for c in df.columns if c.startswith(feature_prefix)]
    X = df[feature_cols].values
    y = df['label'].values

    if split_type == "subject":
        subs = np.unique(subject_ids)
        np.random.seed(random_state)
        n_test = max(1, int(len(subs) * test_size))
        test_subs = np.random.choice(subs, n_test, replace=False)
        mask_test = np.isin(subject_ids, test_subs)
        mask_train = ~mask_test

    elif split_type == "time":
        mask_train = np.zeros(len(df), dtype=bool)
        mask_test = np.zeros(len(df), dtype=bool)
        for s in np.unique(subject_ids):
            idxs = np.where(subject_ids == s)[0]
            sorted_idxs = idxs[np.argsort(time_indices[idxs])]
            cutoff = int(len(sorted_idxs) * (1 - test_size))
            mask_train[sorted_idxs[:cutoff]] = True
            mask_test[sorted_idxs[cutoff:]] = True

    if split_type in ("subject", "time"):
        X_train, X_test = X[mask_train], X[mask_test]
        y_train, y_test = y[mask_train], y[mask_test]
        subj_train, subj_test = subject_ids[mask_train], subject_ids[mask_test]
        time_train, time_test = time_indices[mask_train], time_indices[mask_test]

    # Fix missing data - AFTER split to avoid data leakage!
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Handle class imbalance w/ weighted sampler
    class_wts = 1.0 / np.bincount(y_train)
    sampler = torch.utils.data.WeightedRandomSampler(
                weights=class_wts[y_train],
                num_samples=len(y_train),
                replacement=True
            )
            
    if include_metadata:
        train_ds = BrainStateDataset(X_train, y_train, subj_train, time_train)
        test_ds = BrainStateDataset(X_test, y_test, subj_test, time_test)
    else:
        train_ds = BrainStateDataset(X_train, y_train)
        test_ds = BrainStateDataset(X_test, y_test)

    # Make dataloaders
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    print(f"\nSplit: {split_type} — train {len(train_ds)}, test {len(test_ds)}")
    return train_loader, test_loader

def prepare_data_for_autoencoder(data_path, split_type="random", test_size=0.2, random_state=42):
    df = pd.read_csv(data_path)
    df = df[df['task'].isin(['Rest', 'Improv', 'Scale'])]
    task_to_label = {'Rest': 0, 'Improv': 1, 'Scale': 2}
    df['label'] = df['task'].map(task_to_label)
    df['subject_id'] = df['subject_id'].astype(str)
    df['time_index'] = df['time_index'].astype(int)
    
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    X = df[feature_cols].values
    y = df['label'].values
    subject_ids = df['subject_id'].values
    time_indices = df['time_index'].values
    
    if split_type == "subject":
        subs = np.unique(subject_ids)
        np.random.seed(random_state)
        n_test = max(1, int(len(subs) * test_size))
        test_subs = np.random.choice(subs, n_test, replace=False)
        mask_test = np.isin(subject_ids, test_subs)
        mask_train = ~mask_test
        
        X_train, X_test = X[mask_train], X[mask_test]
        y_train, y_test = y[mask_train], y[mask_test]
        subj_train, subj_test = subject_ids[mask_train], subject_ids[mask_test]
        time_train, time_test = time_indices[mask_train], time_indices[mask_test]
       
    elif split_type == "time":
        mask_train = np.zeros(len(df), dtype=bool)
        mask_test = np.zeros(len(df), dtype=bool)
        for s in np.unique(subject_ids):
            idxs = np.where(subject_ids == s)[0]
            sorted_idxs = idxs[np.argsort(time_indices[idxs])]
            cutoff = int(len(sorted_idxs) * (1 - test_size))
            mask_train[sorted_idxs[:cutoff]] = True
            mask_test[sorted_idxs[cutoff:]] = True
        
        X_train, X_test = X[mask_train], X[mask_test]
        y_train, y_test = y[mask_train], y[mask_test]
        subj_train, subj_test = subject_ids[mask_train], subject_ids[mask_test]
        time_train, time_test = time_indices[mask_train], time_indices[mask_test]
        
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    #Make datasets 
    train_dataset = BrainStateDataset(X_train, y_train, subj_train, time_train)
    test_dataset = BrainStateDataset(X_test, y_test, subj_test, time_test)
    
    # Handle class imbalance
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Make dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader, (subj_train, subj_test)