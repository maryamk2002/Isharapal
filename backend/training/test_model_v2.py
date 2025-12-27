# backend/training/test_model_v2.py
import torch
from pathlib import Path
from dataset_v2 import create_dataloaders
from train_pipeline_v2 import create_model_v2
import json

def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model config
    config_path = Path('saved_models/v2/psl_model_v2.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load model
    model = create_model_v2(
        input_dim=config['input_dim'],
        num_classes=config['num_classes'],
        num_channels=config['architecture']['num_channels'],
        kernel_size=config['architecture']['kernel_size'],
        dropout=config['architecture']['dropout']
    )
    
    # Load weights
    model_path = Path('saved_models/v2/psl_model_v2.pth')
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle both checkpoint formats (full checkpoint or just state_dict)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Load test data
    _, _, test_loader, _ = create_dataloaders(
        batch_size=32,
        sequence_length=config['sequence_length'],
        augment_train=False
    )
    
    # Test
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct}/{total}")

if __name__ == '__main__':
    test_model()