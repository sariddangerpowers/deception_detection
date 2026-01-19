import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import numpy as np
from pathlib import Path

from src.config import ModelConfig, TrainingConfig, PathConfig
from src.models.fusion_model import build_multimodal_model
from src.preprocessing.loader import get_user_independent_splits, get_cv_loaders, get_final_loaders

def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    from torch.cuda.amp import autocast
    
    pbar = tqdm(loader, desc="  Training", leave=False)
    for batch in pbar:
        video = batch['video'].to(device)
        audio = batch['audio'].to(device)
        text = batch['text'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        if scaler:
            with autocast():
                outputs = model(video, audio, text)
                loss = loss_fn(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(video, audio, text)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix(loss=total_loss/(total/labels.size(0)), acc=correct/total)
        sys.stdout.flush()
        
    return total_loss / len(loader), correct / total

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            text = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(video, audio, text)
            loss = loss_fn(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return total_loss / len(loader), correct / total

def run_training(train_loader, val_loader, model_config, train_config, device, fold_name="final"):
    """
    Core training logic with early stopping.
    """
    model = build_multimodal_model(model_config).to(device)
    
    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=train_config.learning_rate, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    best_val_acc = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    os.makedirs("checkpoints", exist_ok=True)
    model_path = f"checkpoints/best_model_{fold_name}.pt"
    
    for epoch in range(train_config.max_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler=scaler)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"[{fold_name}] Epoch {epoch+1}/{train_config.max_epochs}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"  --> Saved new best model checkpoint.")
        else:
            patience_counter += 1
            if patience_counter >= train_config.patience: # Patience from paper
                print(f"Early stopping at epoch {epoch+1}")
                break
        sys.stdout.flush()
                
    return best_val_acc, history

def main():
    # Configurations
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig() # epochs: 100, lr: 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    path_cfg = PathConfig()
    META = str(path_cfg.metadata_csv)
    
    # 1. Split Users (70/30)
    df, train_val_idx, test_idx = get_user_independent_splits(META)
    
    # 2. 5-Fold Cross Validation
    cv_histories = {}
    fold_results = []
    print("\n--- Starting 5-Fold Cross Validation ---")
    for fold, tl, vl in get_cv_loaders(META, train_val_idx, batch_size=train_cfg.batch_size):
        best_acc, history = run_training(tl, vl, model_cfg, train_cfg, device, fold_name=f"fold_{fold}")
        fold_results.append(best_acc)
        cv_histories[f"fold_{fold}"] = history
    
    avg_cv_acc = np.mean(fold_results)
    print(f"\nAverage CV Accuracy: {avg_cv_acc:.4f}")
    
    # 3. Final Retrain on full Train+Val and test on held-out Test
    print("\n--- Performing Final Retrain and Test ---")
    final_train_loader, test_loader = get_final_loaders(META, train_val_idx, test_idx, batch_size=train_cfg.batch_size)
    
    final_best_acc, final_history = run_training(final_train_loader, test_loader, model_cfg, train_cfg, device, fold_name="final")
    
    # Load best and evaluate on test set
    best_model = build_multimodal_model(model_cfg).to(device)
    best_model.load_state_dict(torch.load("checkpoints/best_model_final.pt"))
    test_loss, test_acc = evaluate(best_model, test_loader, nn.CrossEntropyLoss(), device)
    
    print(f"\nFinal Test Results:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Loss: {test_loss:.4f}")
    
    # Save results
    report = {
        'split_info': {
            'train_val_samples': len(train_val_idx),
            'test_samples': len(test_idx),
            'train_users': list(map(int, set(df.iloc[train_val_idx]['usernum']))),
            'test_users': list(map(int, set(df.iloc[test_idx]['usernum'])))
        },
        'cv_results': {
            'fold_accuracies': fold_results,
            'avg_accuracy': avg_cv_acc,
            'histories': cv_histories
        },
        'final_results': {
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'history': final_history
        }
    }
    
    report_path = path_cfg.project_root / "detailed_training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\nDetailed statistics saved to {report_path}")

if __name__ == "__main__":
    main()
