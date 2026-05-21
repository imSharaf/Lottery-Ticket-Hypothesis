'''
Inside a large randomly initialized neural network, there exists a small subset of weights that, when trained, can achieve near-perfect performance on a given task. 
This phenomenon is known as the Lottery Ticket Hypothesis.

This script implements the Lottery Ticket Hypothesis on the Wine dataset.
Flow of Program: 
Train Full Network ->
Prune Small Weights ->
Reset Remaining Weights to original initialization ->
Train again ->
Repeat multiple rounds ->
Observe if sparse network still works.  
'''

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import copy
import numpy as np

torch.manual_seed(42)

# --- 1. Data Loading and Preprocessing ---

wine = load_wine()
X, y = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)


# --- 2. Define Model Architecture ---

class WineNet(nn.Module):
    def __init__(self, input_dim):
        super(WineNet, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 3)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(16)

        self.dropout = nn.Dropout(p=0.2)

        self.act = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.dropout(x)
    
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.act(x)

        return self.fc5(x)


# --- 3. Training and Evaluation Modules ---

def train_model(model, X_train, y_train, epochs=200, masks=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        
        # Apply mask to prevent pruned weights from learning
        if masks is not None:
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and name in masks:
                    if module.weight.grad is not None:
                        module.weight.grad *= masks[name]
        
        optimizer.step()
        
        # Safety: Re-zero pruned weights after update
        if masks is not None:
            with torch.no_grad():
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear) and name in masks:
                        module.weight.data *= masks[name]


def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y_test).sum().item()
        accuracy = (correct / len(y_test)) * 100
    return accuracy


# --- 4. BN RESET (IMPORTANT FOR WINNING TICKET) ---

def reset_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.reset_running_stats()
            m.reset_parameters()


# --- 5. Sparsity Analysis ---

def print_sparsity_report(model):
    """
    Analyzes the neural network layer by layer to report total weights,
    pruned weights, layer-specific sparsity, and dead nodes.
    """
    print("\n" + "="*80)
    print(f"{'LAYER NAME':<15} | {'TOTAL WEIGHTS':<15} | {'PRUNED WEIGHTS':<15} | {'SPARSITY %':<12} | {'DEAD NODES':<10}")
    print("="*80)

    grand_total_weights = 0
    grand_total_pruned = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weights = module.weight.data.cpu().numpy()

            total_elements = weights.size
            pruned_elements = np.sum(weights == 0)
            sparsity = (pruned_elements / total_elements) * 100

            grand_total_weights += total_elements
            grand_total_pruned += pruned_elements

            dead_nodes_count = np.sum(np.all(weights == 0, axis=1))

            print(f"{name:<15} | {total_elements:<15} | {pruned_elements:<15} | {sparsity:<11.2f}% | {dead_nodes_count:<10}")

    overall_sparsity = (grand_total_pruned / grand_total_weights) * 100

    print("="*80)
    print(f"{'GLOBAL SUMMARY':<15} | {grand_total_weights:<15} | {grand_total_pruned:<15} | {overall_sparsity:<11.2f}% | (All Layers)")
    print("="*80 + "\n")


# --- 6. WINNING TICKET PIPELINE ---

'''
A fresh neural network is created and its exact starting weights are stored before training begins.
The full dense network is trained for 200 epochs and the accuracy is evaluated on the test set.
The pipeline looks at the trained weights and creates a binary mask marking the smallest percent of weights as weak and the rest as strong. 
The weak weights are then pruned and the network is retrained setting to zero. 
The lightweight model is then trained again. 
The final accuracy is checked. 
'''

input_dim = X_train.shape[1]
model = WineNet(input_dim)

# Step 1: Store original initialization (θ₀)
initial_state = copy.deepcopy(model.state_dict())

# Train the dense unpruned model
print("--- 1. Training Original (Dense) Model ---")
train_model(model, X_train_t, y_train_t, epochs=200)

before_pruning_score = evaluate_model(model, X_test_t, y_test_t)
print(f"✅ Before Pruning Score (Dense Model Accuracy): {before_pruning_score:.2f}%")


# Step 2: Build mask using magnitude pruning
prune_percent = 0.4
masks = {}

for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        weight = module.weight.data.abs()
        threshold = torch.quantile(weight.flatten(), prune_percent)   # Fixed
        masks[name] = (weight > threshold).float()


# Step 3: Create Winning Ticket (m ⊙ θ₀)
winning_model = WineNet(input_dim)
winning_state = winning_model.state_dict()

for name, module in winning_model.named_modules():
    if isinstance(module, nn.Linear) and name in masks:
        mask = masks[name]
        winning_state[f"{name}.weight"] = initial_state[f"{name}.weight"] * mask
        winning_state[f"{name}.bias"] = initial_state[f"{name}.bias"]

winning_model.load_state_dict(winning_state)

# IMPORTANT: reset BatchNorm statistics
reset_bn(winning_model)


# Step 4: Retrain Winning Ticket
print("\n--- 2. Training Winning Ticket (Pruned Subnetwork) ---")
train_model(winning_model, X_train_t, y_train_t, epochs=200, masks=masks)

after_pruning_score = evaluate_model(winning_model, X_test_t, y_test_t)
print(f"✅ After Pruning Score (Winning Ticket Accuracy): {after_pruning_score:.2f}%")


# --- 7. Final Diagnostic Summary ---

print_sparsity_report(winning_model)

print("--- Diagnostic Scoreboard ---")
print(f"Before Pruning Accuracy : {before_pruning_score:.2f}%")
print(f'Prune Percent: {prune_percent}')
print(f"After Pruning Accuracy  : {after_pruning_score:.2f}%")