import random
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import wandb
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

from cnn import CNN

def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

def save_checkpoint(state, filename='./data/checkpoint.pth.tar'):
    print('Saving checkpoint...')
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print('Loading checkpoint...')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def start():
    # Constants Values
    seed = 42
    checkpoint = True
    early_stopping = True
    data_normalization_tuple = ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    validation_split = 0.1

    batch_size = 128
    total_epoch = 50
    learning_rate = 1e-3

    patience = max(3, total_epoch//10)
    epochs_no_improve = 0

    # Set Seed
    set_seed(seed)

    # Wandb initialize
    wandb.init(project="csci646-assignment-1")

    # Create trasform for Train & Test Dataset
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(*data_normalization_tuple)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*data_normalization_tuple)
    ])

    # Load CIFAR Dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)

    train_size = int((1 - validation_split) * len(trainset))
    valid_size = len(trainset) - train_size
    
    generator = torch.Generator().manual_seed(seed)
    train_set, valid_set = torch.utils.data.random_split(trainset, [train_size, valid_size], generator)

    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    validloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Train Datapoint Count: {len(train_set)}")
    print(f"Validation Datapoint Count: {len(valid_set)}")
    print(f"Test Datapoint Count: {len(testset)}")

    # Create Model
    model = CNN()

    # Check Cuda Availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Device: {device}")

    # Loss Function and Optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_loss = float('inf')
    # Training the Model
    for epoch in range(total_epoch):
        model.train()

        total, correct = 0, 0
        running_loss = 0.0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            running_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Training accuracy and loss
        train_accuracy = 100 * correct / total
        train_loss = running_loss / len(trainloader)

        # Learning rate decay step
        scheduler.step()

        # Validating the Model
        model.eval()
        total, correct = 0, 0
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = loss_func(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            val_accuracy = 100 * correct / total
            val_loss = val_loss / len(validloader)

        print(f'Epoch [{epoch+1}/{total_epoch}] - Training Accuracy: {train_accuracy:.2f}%, Training Loss: {train_loss:.4f} - Validation Accuracy: {val_accuracy:.2f}%, Validation Loss: {val_loss:.4f}') 
        
        # Log training accuracy and loss
        wandb.log({"epoch": epoch+1, "train_accuracy": train_accuracy, "train_loss": train_loss, "val_accuracy": val_accuracy, "val_loss": val_loss})

        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            # Save checkpoint if model improves
            if checkpoint:
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                })
        else:
            if early_stopping:
                epochs_no_improve += 1
                if epochs_no_improve > patience:
                    print("Early stopping triggered!")
                    break

    # Testing the Model
    model.eval()
    total, correct = 0, 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Append predictions and labels for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate final accuracy
    test_accuracy = 100 * correct / total

    print(f"\n\nTest Accuracy: {test_accuracy:.2f}%\n")

    # Classification report for each category
    class_report = classification_report(all_labels, all_preds, target_names=testset.classes, digits=4)
    print("\nClassification Report:\n", class_report)

    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:\n", conf_matrix)

    # Log test metrics
    wandb.log({
        "test_accuracy": test_accuracy
    })

    wandb.finish()

    print('Training and Testing Complete')

if __name__ == '__main__':
    start()
