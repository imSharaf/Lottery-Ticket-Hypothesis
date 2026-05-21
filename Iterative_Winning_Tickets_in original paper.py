'''
Inside a large randomly initialized neural network, there exists a small subset of weights that, when trained, can achieve near-perfect performance on a given task. 
This phenomenon is known as the Lottery Ticket Hypothesis.

This script implements the Lottery Ticket Hypothesis on the Wine dataset.
Flow of Program: 
Train Full Network ->
Prune Small Weights ->
Reset Remaining Weights to original initialization ->
Train again ->
Repeat multiple rounds ->              ← NOW ACTUALLY IMPLEMENTED
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


# --- 6. ITERATIVE WINNING TICKET PIPELINE ---

'''
A fresh neural network is created and its exact starting weights (θ₀) are stored before any training.

Each round of iterative pruning follows this loop:
  1. Train the current (masked) model for 200 epochs
  2. Compute a new pruning mask on the REMAINING (non-zero) weights only
     — prune_percent_per_round % of surviving weights are removed each round
  3. Combine the new mask with the cumulative mask (logical AND)
  4. Reset ALL surviving weights back to θ₀ (the original initialization)
  5. Reset BatchNorm statistics

After NUM_ROUNDS, the final winning ticket is evaluated and a sparsity report is printed.
The key difference from one-shot pruning: each round prunes from the SURVIVORS of the
previous round, so sparsity compounds and the ticket is found more gradually.
'''

input_dim = X_train.shape[1]
model = WineNet(input_dim)

# Step 1: Store original initialization (θ₀) — never changes across rounds
initial_state = copy.deepcopy(model.state_dict())

# Train the dense unpruned model once to get a baseline score
print("--- 1. Training Original (Dense) Model ---")
train_model(model, X_train_t, y_train_t, epochs=200)

before_pruning_score = evaluate_model(model, X_test_t, y_test_t)
print(f"✅ Before Pruning Score (Dense Model Accuracy): {before_pruning_score:.2f}%")


# --- Iterative Pruning Config ---
# Each round prunes 20% of the SURVIVING weights.
# After 5 rounds, cumulative sparsity ≈ 1 - (0.9^5) ≈ 41% — much more gradual than one-shot 40%.
NUM_ROUNDS          = 5
prune_percent_per_round = 0.1

# Cumulative mask: starts as all-ones (keep everything), gets ANDed each round
cumulative_masks = {}
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        cumulative_masks[name] = torch.ones_like(module.weight.data)

round_scores = []  # track accuracy after each round

print(f"\n--- 2. Iterative Pruning: {NUM_ROUNDS} rounds × {int(prune_percent_per_round*100)}% per round ---")

for round_num in range(1, NUM_ROUNDS + 1):

    # Step 2a: Build new mask on SURVIVING weights only
    # We look at the current model's trained weights but only consider non-pruned ones.
    new_masks = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight_abs = module.weight.data.abs()

            # Only score weights that are still alive (mask == 1)
            alive_mask = cumulative_masks[name].bool()

            # Among alive weights, find the bottom prune_percent_per_round threshold
            alive_values = weight_abs[alive_mask]
            threshold = torch.quantile(alive_values, prune_percent_per_round)

            # New mask: alive AND above threshold
            new_masks[name] = ((weight_abs > threshold) & alive_mask).float()

    # Step 2b: Update cumulative mask (AND with new mask)
    for name in cumulative_masks:
        cumulative_masks[name] = cumulative_masks[name] * new_masks[name]

    # Step 3: Reset model to original initialization (θ₀), apply cumulative mask
    winning_state = model.state_dict()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in cumulative_masks:
            mask = cumulative_masks[name]
            winning_state[f"{name}.weight"] = initial_state[f"{name}.weight"] * mask
            winning_state[f"{name}.bias"]   = initial_state[f"{name}.bias"]

    model.load_state_dict(winning_state)

    # Step 4: Reset BatchNorm statistics before retraining
    reset_bn(model)

    # Step 5: Retrain the winning ticket for this round
    print(f"\n  [Round {round_num}/{NUM_ROUNDS}] Retraining sparse subnetwork...")
    train_model(model, X_train_t, y_train_t, epochs=200, masks=cumulative_masks)

    round_score = evaluate_model(model, X_test_t, y_test_t)
    round_scores.append(round_score)

    # Compute current global sparsity for logging
    total_w, pruned_w = 0, 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data.cpu().numpy()
            total_w  += w.size
            pruned_w += np.sum(w == 0)
    global_sparsity = (pruned_w / total_w) * 100

    print(f"  ✅ Round {round_num} Accuracy: {round_score:.2f}%  |  Global Sparsity: {global_sparsity:.2f}%")


# --- 7. Final Diagnostic Summary ---

print_sparsity_report(model)

print("--- Diagnostic Scoreboard ---")
print(f"Before Pruning Accuracy : {before_pruning_score:.2f}%")
print(f"Prune % Per Round       : {int(prune_percent_per_round*100)}%  ×  {NUM_ROUNDS} rounds")
for i, score in enumerate(round_scores, 1):
    print(f"  Round {i} Accuracy     : {score:.2f}%")
print(f"Final Winning Ticket    : {round_scores[-1]:.2f}%")
