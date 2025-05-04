import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
from dataprep import prepare_data
from models import BrainStateClassifier, BrainStateFullModel, AttentionFullModel

#MLP training and eval
def train_and_evaluate(train_loader, test_loader, model, num_epochs=50, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    train_losses = []
    train_accs = []
    
    print("Started training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            if len(batch) == 2:
                batch_X, batch_y = batch
            else:
                batch_X, batch_y = batch[0], batch[1]
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100 * correct / total
        train_losses.append(avg_loss)
        train_accs.append(accuracy)
    
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%')
    
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:
                batch_X, batch_y = batch
            else:
                batch_X, batch_y = batch[0], batch[1]
            outputs = model(batch_X.to(device))
            _, predicted = torch.max(outputs.data, 1)
    
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    print("\nTest Set Results:")
    print(classification_report(all_labels, all_preds, 
                              target_names=['Rest', 'Improv', 'Scale'],
                              digits=3))
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(train_accs)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.tight_layout()
    
    return model, all_preds, all_labels, {
        'losses': train_losses,
        'accuracies': train_accs
    }

def run_cross_validation(data_path, feature_prefix, split_type, num_trials=5, input_dim=48):
    accuracies = []
    confusion_matrices = []
    classification_reports = []
    for trial in range(num_trials):
        print(f"\n=== Trial {trial+1}/{num_trials} ===")
        random_seed = 42 + trial
        
        train_loader, test_loader = prepare_data(
            data_path=data_path,
            feature_prefix=feature_prefix,
            include_metadata=False,
            split_type=split_type,
            random_state=random_seed
        )
        model = BrainStateClassifier(input_dim=input_dim)
        model, preds, labels, metrics = train_and_evaluate(
            train_loader=train_loader,
            test_loader=test_loader,
            model=model
        )

        accuracy = accuracy_score(labels, preds)
        accuracies.append(accuracy)
        cm = confusion_matrix(labels, preds)
        confusion_matrices.append(cm)
        
        cr = classification_report(labels, preds, 
                                  target_names=['Rest', 'Improv', 'Scale'],
                                  output_dict=True)
        classification_reports.append(cr)
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    avg_cm = np.mean(confusion_matrices, axis=0)

    results = {
        'accuracies': accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'confusion_matrices': confusion_matrices,
        'avg_confusion_matrix': avg_cm,
        'classification_reports': classification_reports
    }
    
    print(f"\n=== Cross-Validation Results ({split_type} split) ===")
    print(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    
    return results

def train_autoencoder(train_loader, test_loader, num_epochs=50, latent_dim=5, split_name=""):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device with latent dimension {latent_dim}")
    model = BrainStateFullModel(latent_dim=latent_dim).to(device)
    reconstruction_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = {
        'train_total_loss': [],
        'train_recon_loss': [],
        'train_class_loss': [],
        'train_acc': [],
        'embeddings': defaultdict(list),
        'labels': defaultdict(list),
        'time_indices': defaultdict(list)
    }

    for epoch in range(num_epochs):
        model.train()
        epoch_total_loss = 0
        epoch_recon_loss = 0 
        epoch_class_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y, batch_subjects, batch_times in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            x_recon, z, logits = model(batch_X)
            recon_loss = reconstruction_criterion(x_recon, batch_X)
            class_loss = classification_criterion(logits, batch_y)
            total_loss = recon_loss + class_loss
        
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()


            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_class_loss += class_loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        train_acc = 100 * correct / total
        avg_total = epoch_total_loss / len(train_loader)
        avg_recon = epoch_recon_loss / len(train_loader)
        avg_class = epoch_class_loss / len(train_loader)
        history['train_total_loss'].append(avg_total)
        history['train_recon_loss'].append(avg_recon)
        history['train_class_loss'].append(avg_class)
        history['train_acc'].append(train_acc)

        if (epoch + 1) % 10 == 0:
            print(f"{split_name} Epoch {epoch+1}/{num_epochs}: Loss={avg_total:.4f} "
                  f"(R={avg_recon:.4f}, C={avg_class:.4f}), Acc={train_acc:.2f}%")
    
    model.eval()
    test_correct = 0
    test_total = 0
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y, _, _ in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            _, _, logits = model(batch_X)
            _, predicted = torch.max(logits.data, 1)
            test_total += batch_y.size(0)
            test_correct += (predicted == batch_y).sum().item()

            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(batch_y.cpu().numpy())
    
    #Final accuracy
    test_acc = 100 * test_correct / test_total
    print(f"{split_name} Test Accuracy: {test_acc:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=['Rest', 'Improv', 'Scale']))

    model.eval()
    embeddings_dict = defaultdict(list)
    labels_dict = defaultdict(list)
    times_dict = defaultdict(list)
    
    with torch.no_grad():
        for loader in [train_loader, test_loader]:
            for batch_X, batch_y, batch_subjects, batch_times in loader:
                _, z, _ = model(batch_X.to(device))
                
                for i, subject in enumerate(batch_subjects):
                    embeddings_dict[subject].append(z[i].cpu().numpy())
                    labels_dict[subject].append(batch_y[i].item())
                    times_dict[subject].append(batch_times[i].item())

    for subject in embeddings_dict:
        times = np.array(times_dict[subject])
        sort_idx = np.argsort(times)
        
        history['embeddings'][subject] = np.array(embeddings_dict[subject])[sort_idx]
        history['labels'][subject] = np.array(labels_dict[subject])[sort_idx]
        history['time_indices'][subject] = times[sort_idx]

    history['test_accuracy'] = test_acc
    history['classification_report'] = classification_report(test_labels, test_preds, 
                                                          target_names=['Rest', 'Improv', 'Scale'],
                                                          output_dict=True)

    return model, history


def train_attention_model(train_loader, test_loader, num_epochs=50, latent_dim=48, split_name=""):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionFullModel(latent_dim=latent_dim).to(device)
    recon_loss_fn = nn.MSELoss()
    class_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = {
        'total_loss': [],
        'recon_loss': [],
        'class_loss': [],
        'accuracy': [],
        'attention_weights': [],
        'embeddings': defaultdict(list),
        'labels': defaultdict(list),
        'times': defaultdict(list)
    }
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = {'total': 0, 'recon': 0, 'class': 0}
        correct, total = 0, 0
        
        for batch_X, batch_y, _, _ in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            x_recon, z, logits = model(batch_X)
            recon_loss = recon_loss_fn(x_recon, batch_X)
            class_loss = class_loss_fn(logits, batch_y)
            total_loss = recon_loss + class_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_losses['total'] += total_loss.item()
            epoch_losses['recon'] += recon_loss.item()
            epoch_losses['class'] += class_loss.item()
            
            _, predicted = torch.max(logits, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        n_batches = len(train_loader)
        epoch_avg = {k: v/n_batches for k, v in epoch_losses.items()}
        accuracy = 100 * correct / total
        
        history['total_loss'].append(epoch_avg['total'])
        history['recon_loss'].append(epoch_avg['recon'])
        history['class_loss'].append(epoch_avg['class'])
        history['accuracy'].append(accuracy)
        history['attention_weights'].append(
            model.autoencoder.node_attention.alphas.detach().cpu().numpy()
        )
        
        if (epoch + 1) % 10 == 0:
            print(f"{split_name} Epoch {epoch+1}/{num_epochs} - Loss: {epoch_avg['total']:.4f} "
                  f"(R: {epoch_avg['recon']:.4f}, C: {epoch_avg['class']:.4f}), Acc: {accuracy:.2f}%")
    
    model.eval()
    test_correct = 0
    test_total = 0
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y, _, _ in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            _, _, logits = model(batch_X)
            _, predicted = torch.max(logits, 1)
            test_total += batch_y.size(0)
            test_correct += (predicted == batch_y).sum().item()
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(batch_y.cpu().numpy())
    
    test_acc = 100 * test_correct / test_total
    print(f"{split_name} Test Accuracy: {test_acc:.2f}%")
    history['test_accuracy'] = test_acc

    model.eval()
    with torch.no_grad():
        for loader in [train_loader, test_loader]:
            for batch_X, batch_y, batch_subjects, batch_times in loader:
                _, z, _ = model(batch_X.to(device))
    
                for i, subject in enumerate(batch_subjects):
                    history['embeddings'][subject].append(z[i].cpu().numpy())
                    history['labels'][subject].append(batch_y[i].item())
                    history['times'][subject].append(batch_times[i].item())

    for subject in history['embeddings']:
        times = np.array(history['times'][subject])
        sort_idx = np.argsort(times)
        history['embeddings'][subject] = np.array(history['embeddings'][subject])[sort_idx]
        history['labels'][subject] = np.array(history['labels'][subject])[sort_idx]
        history['times'][subject] = times[sort_idx]
    
    return model, history