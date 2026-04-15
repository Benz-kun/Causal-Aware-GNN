import pandas as pd
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from model import TradeSAGE

# --- CONFIGURATION ---
NODES_PATH = os.path.join("data", "processed", "nodes.pkl")
EDGES_PATH = os.path.join("data", "processed", "edges.csv")
MODEL_SAVE_PATH = os.path.join("models", "trade_predictor.pth")

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Hyperparameters
EPOCHS = 2000          # Training iterations
LEARNING_RATE = 0.01   # How fast the model learns
HIDDEN_DIM = 64        # Size of the "hidden thought vector"

def load_data():
    print("📂 Loading Processed Data...")
    
    # 1. Load DataFrames
    try:
        df_nodes = pd.read_pickle(NODES_PATH)
        df_edges = pd.read_csv(EDGES_PATH)
    except FileNotFoundError:
        print("❌ Error: Processed data not found.")
        print("   Did you run 'src/4_export_for_training.py'?")
        return None

    print(f"   Loaded {len(df_nodes)} nodes and {len(df_edges)} edges.")

    # 2. Prepare Features (X)
    # Stack the list of embeddings into a single Tensor matrix
    # Shape: [Num_Nodes, 128]
    features = np.stack(df_nodes['embedding'].values)
    x = torch.tensor(features, dtype=torch.float)

    # 3. Prepare Target (y)
    # We predict Gross Output. We use Log1p transformation to handle the massive range of values.
    # (Predicting $10B vs $100B is easier in Log scale: 23.02 vs 25.32)
    raw_output = df_nodes['gross_output'].values
    y = torch.tensor(np.log1p(raw_output), dtype=torch.float).view(-1, 1)

    # 4. Prepare Edges (Connectivity)
    # We need to map UID strings ('AUS_A01') to Integer Indices (0, 1, 2...)
    uid_to_idx = {uid: i for i, uid in enumerate(df_nodes['uid'])}
    
    src = [uid_to_idx[u] for u in df_edges['source']]
    dst = [uid_to_idx[u] for u in df_edges['target']]
    
    # PyTorch Geometric Edge Index Format: [[sources], [targets]]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, y=y)

def train():
    data = load_data()
    if data is None: return

    # Split: 80% Training (Learn), 20% Testing (Validate)
    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    # Create Boolean Masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    
    # Initialize Model
    # Input: 128 (FastRP Embedding Size)
    # Hidden: 64
    # Output: 1 (Log Gross Output)
    model = TradeSAGE(in_channels=128, hidden_channels=HIDDEN_DIM, out_channels=1)
    
    # Optimizer (Adam is standard for GNNs)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Loss Function (Mean Squared Error)
    criterion = torch.nn.MSELoss()

    print(f"🚀 Starting Training (Epochs: {EPOCHS})...")
    
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()            # Clear previous gradients
        out = model(data.x, data.edge_index) # Forward pass
        
        # Calculate loss ONLY on Training Nodes
        loss = criterion(out[train_mask], data.y[train_mask])
        
        loss.backward()                  # Backpropagation (Learn)
        optimizer.step()                 # Update weights
        
        if epoch % 200 == 0:
            print(f"   Epoch {epoch:04d} | Loss: {loss.item():.4f}")

    # --- FINAL EVALUATION ---
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index)
        test_loss = criterion(pred[test_mask], data.y[test_mask])
        print(f"🏁 Training Complete.")
        print(f"   Final Test Loss: {test_loss.item():.4f}")
        print("   (Lower is better. < 2.0 is usually good for this dataset)")

    # Save the trained brain
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"💾 Model Saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()