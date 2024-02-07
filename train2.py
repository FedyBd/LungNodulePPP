import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from vit_pytorch import ViT
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, auc
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import defaultdict

# Define dataset class
class LungNoduleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            for image_path in os.listdir(label_dir):
                self.image_paths.append(os.path.join(label_dir, image_path))
                self.labels.append(int(label == 'NOD'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# Define data transformations for original images
orig_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Define data transformations for augmented images
augment_transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Define data transformations for augmented images
augment_transform1 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Define data transformations for augmented images
augment_transform3 = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Define data transformations for augmented images with small Gaussian noise
augment_transform2 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1)  # adding small Gaussian noise with std=0.1
])

# Load original dataset
root_dir = r'C:\Users\MED Fedi BOUABID\Desktop\PPP\DATA_LIDC_IDRI'
orig_dataset = LungNoduleDataset(root_dir, transform=orig_transform)

# Augment dataset2
augment_dataset2 = LungNoduleDataset(root_dir, transform=augment_transform2)

# Augment dataset
augment_dataset = LungNoduleDataset(root_dir, transform=augment_transform)

# Augment dataset1
augment_dataset1 = LungNoduleDataset(root_dir, transform=augment_transform1)

# Augment dataset3
augment_dataset3 = LungNoduleDataset(root_dir, transform=augment_transform3)


#  data augmentation 4 transform
augment_transform4 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(45),
    transforms.ToTensor(),
transforms.Normalize(mean=[0.5], std=[0.5])
])

# Augment dataset4
augment_dataset4 = LungNoduleDataset(root_dir, transform=augment_transform4)

# Concatenate datasets
full_dataset = torch.utils.data.ConcatDataset(
    [orig_dataset, augment_dataset, augment_dataset1, augment_dataset2, augment_dataset3, augment_dataset4])

print(len(full_dataset))

from sklearn.model_selection import train_test_split
from collections import defaultdict

# Obtain the labels and corresponding indices from the full dataset
labels = [sample[1] for sample in full_dataset]
indices = list(range(len(full_dataset)))

# Create a dictionary to store the indices of samples for each class
class_indices = defaultdict(list)
for i, label in enumerate(labels):
    class_indices[label].append(indices[i])

# Split the indices for each class using stratified sampling
train_indices = []
val_indices = []
test_indices = []
for class_label, indices in class_indices.items():
    train_class, test_class = train_test_split(indices, test_size=0.05, random_state=42)
    train_class, val_class = train_test_split(train_class, test_size=0.1, random_state=42)
    train_indices.extend(train_class)
    val_indices.extend(val_class)
    test_indices.extend(test_class)

# Create the train, validation, and test datasets using the sampled indices
train_set = torch.utils.data.Subset(full_dataset, train_indices)
val_set = torch.utils.data.Subset(full_dataset, val_indices)
test_set = torch.utils.data.Subset(full_dataset, test_indices)

# Count the number of images for each class in the training set
train_counts = defaultdict(int)
for _, label in train_set:
    train_counts[label] += 1

# Count the number of images for each class in the validation set
val_counts = defaultdict(int)
for _, label in val_set:
    val_counts[label] += 1

# Print the number of images for each class in the training set
print("Training Set:")
for label, count in train_counts.items():
    print(f"Class {label}: {count} images")

# Print the number of images for each class in the validation set
print("\nValidation Set:")
for label, count in val_counts.items():
    print(f"Class {label}: {count} images")

# Define data loaders
batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Load ViT-Ti/4 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViT(
    image_size=64,
    patch_size=4,
    num_classes=2,
    dim=128,
    depth=8,
    heads=8,
    mlp_dim=128,
    dropout=0.1,
    emb_dropout=0.1
).to(device)



# Define loss function, optimizer, and learning rate
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

import matplotlib.pyplot as plt

# Lists to store the training and validation metrics
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    train_loss = 0.0
    train_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    train_loss /= len(train_set)
    train_accuracy = 100 * train_correct / total_samples

    # Validate the model
    val_loss = 0.0
    val_correct = 0
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()
    val_loss /= len(val_set)
    val_accuracy = 100 * val_correct / len(val_set)

    # Store metrics for plotting
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
    print(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

    model.train()

# Plot the curves
epochs = range(1, num_epochs + 1)

# Plot loss curves
plt.figure()
plt.plot(epochs, train_losses, 'b', label='Train Loss')
plt.plot(epochs, val_losses, 'r', label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot accuracy curves
plt.figure()
plt.plot(epochs, train_accuracies, 'b', label='Train Accuracy')
plt.plot(epochs, val_accuracies, 'r', label='Val Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()



# Evaluate the model on the test set
test_correct = 0
model.eval()
test_predicted = []
test_ground_truth = []
test_probs = []
with torch.no_grad():
   for images, labels in test_loader:
       images, labels = images.to(device), labels.to(device)
       outputs = model(images)
       probabilities = torch.softmax(outputs, dim=1)
       _, predicted = torch.max(outputs.data, 1)
       test_correct += (predicted == labels).sum().item()
       # Append the predicted and ground truth labels to the respective lists
       test_predicted.extend(predicted.cpu().numpy())
       test_ground_truth.extend(labels.cpu().numpy())
       test_probs.extend(probabilities[:, 1].cpu().numpy())  # Probability for class 1 (positive)
test_accuracy = 100 * test_correct / len(test_set)
print(f'Test Accuracy: {test_accuracy:.2f}%')
test_sensitivity = recall_score(test_ground_truth, test_predicted, pos_label=0)
test_specificity = recall_score(test_ground_truth, test_predicted, pos_label=1)
test_f1_score = f1_score(test_ground_truth, test_predicted)
print(f'Test Sensitivity: {test_sensitivity:.2f}')
print(f'Test Specificity: {test_specificity:.2f}')
print(f'Test F1 Score: {test_f1_score:.2f}')
# confusion matrix
cm = confusion_matrix(test_ground_truth, test_predicted)
# Set the class labels
class_labels = ['NOD','NON_NOD']  # List of class labels in the order they appear in the confusion matrix

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Compute false positive rate (FPR) and true positive rate (TPR)
fpr, tpr, thresholds = roc_curve(test_ground_truth, test_probs)

# Compute area under the ROC curve (AUC)
auc_score = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

torch.save(model.state_dict(), 'my_model.pth')
