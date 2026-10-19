import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader

from classifier_components import build_classifier_model, build_training_transform


def train_model(data_dir="dataset", save_path="models/pattern_classifier.pth", epochs=15, batch_size=8, lr=0.001):
    # Ensure models directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load Dataset
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset folder '{data_dir}' not found. Please create it and add subclass folders.")
        
    dataset = datasets.ImageFolder(
        root=data_dir, transform=build_training_transform()
    )
    num_classes = len(dataset.classes)
    print(f"Loaded dataset with {len(dataset)} images across {num_classes} classes: {dataset.classes}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Backbone: MobileNetV3-Small
    model = build_classifier_model(num_classes, pretrained=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("\nStarting Training...")
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
        
    # Save checkpoint along with class mapping
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'classes': dataset.classes,
        'class_to_idx': dataset.class_to_idx,
        'num_classes': num_classes
    }
    torch.save(checkpoint, save_path)
    print(f"\nModel training complete! Saved to '{save_path}'.")

if __name__ == "__main__":
    print("Initializing training routine...")
    train_model()