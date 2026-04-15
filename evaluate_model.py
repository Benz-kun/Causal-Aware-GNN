import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch_geometric.data import HeteroData

# ==========================================
# 1. Predictive Fidelity Metrics
# ==========================================
def get_predictive_metrics(model, loader, device='cpu'):
    """
    Runs the model on the test set and calculates standard regression metrics.
    Returns: dictionary of metrics and the raw true/pred arrays for plotting.
    """
    model.eval()
    model.to(device)
    
    y_true = []
    y_pred = []

    print("\n[Evaluation] Running Predictive Fidelity Check...")
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            # Forward pass 
            out = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
            
            # Handle dictionary output vs tensor output
            if isinstance(out, dict):
                pred = out['sector']
            else:
                pred = out 
            
            target = data['sector'].y 
            
            y_pred.append(pred.cpu().numpy())
            y_true.append(target.cpu().numpy())

    # Flatten arrays
    y_pred = np.concatenate(y_pred).ravel()
    y_true = np.concatenate(y_true).ravel()

    # Calculate Metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }
    
    print(f"   -> RMSE: {rmse:.4f}")
    print(f"   -> MAE:  {mae:.4f}")
    print(f"   -> R2:   {r2:.4f}")
    
    return metrics, y_true, y_pred

# ==========================================
# 2. Causal Sensitivity (The Shock Simulation)
# ==========================================
def simulate_shock(model, data, target_edge_type, edge_mask_idx, shock_factor=0.6, device='cpu'):
    """
    Performs the SUTVA Violation Test.
    Returns the Delta (Impact).
    """
    model.eval()
    data = data.to(device)
    
    print(f"\n[Simulation] Simulating {((1-shock_factor)*100):.0f}% Blockade...")

    # 1. Baseline Run
    with torch.no_grad():
        baseline_out = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        if isinstance(baseline_out, dict): baseline_out = baseline_out['sector']

    # 2. Apply Shock (Intervention)
    shocked_data = data.clone()
    
    # Apply scalar reduction to the specific edge weights
    # Assuming edge_attr is [num_edges, 1] (the weight)
    shocked_data.edge_attr_dict[target_edge_type][edge_mask_idx] *= shock_factor

    # 3. Shocked Run
    with torch.no_grad():
        shocked_out = model(shocked_data.x_dict, shocked_data.edge_index_dict, shocked_data.edge_attr_dict)
        if isinstance(shocked_out, dict): shocked_out = shocked_out['sector']

    # 4. Calculate Causal Impact (Delta)
    delta = baseline_out - shocked_out
    
    return delta.cpu().numpy(), baseline_out.cpu().numpy(), shocked_out.cpu().numpy()

# ==========================================
# 3. Visualization Suite
# ==========================================
def plot_evaluation_results(y_true, y_pred, impact_vector, country_labels):
    """
    Generates and saves two separate graphs for the Q1 Paper.
    """
    # --- Graph 1: Predictive Fidelity (Scatter Plot) ---
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, color="#2c3e50", edgecolor='w', s=80)
    
    # Diagonal Ideal Line
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label="Ideal Prediction ($y=x$)")
    
    plt.title("Model Predictive Fidelity (Actual vs Predicted)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Actual Log-Trade Volume (Normalized)", fontsize=14)
    plt.ylabel("Predicted Log-Trade Volume (Normalized)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save Graph 1
    plt.tight_layout()
    plt.savefig("predictive_fidelity.png", dpi=300)
    print("\n[Output] Saved 'predictive_fidelity.png'")
    plt.close() # Close to free memory

    # --- Graph 2: The "Lithuania Anomaly" (Bar Chart) ---
    plt.figure(figsize=(12, 8))
    
    # Aggregate Impact by Country
    df_impact = pd.DataFrame({'Country': country_labels, 'Loss': impact_vector.ravel()})
    df_country = df_impact.groupby('Country')['Loss'].sum().reset_index()
    df_country = df_country.sort_values(by='Loss', ascending=False).head(10) # Top 10

    # Gradient Bar Plot
    sns.barplot(x='Loss', y='Country', data=df_country, palette="viridis")
    
    plt.title("Structural Shock Propagation (Top 10 Impacted Nations)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Projected Economic Loss (Latent Space $\Delta$)", fontsize=14)
    plt.ylabel("Nation ISO Code", fontsize=14)
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    # Save Graph 2
    plt.tight_layout()
    plt.savefig("shock_propagation.png", dpi=300)
    print("[Output] Saved 'shock_propagation.png'")
    plt.close()

# ==========================================
# 4. Dummy Test Runner (For verification)
# ==========================================
if __name__ == "__main__":
    # This block allows you to run this file IMMEDIATELY to see the graphs.
    print("--- Running Dummy Evaluation for Testing ---")

    # A. Create Dummy Data (Simulating your HeteroData)
    num_nodes = 100
    y_true_dummy = np.random.normal(10, 2, num_nodes) # Real trade values
    y_pred_dummy = y_true_dummy + np.random.normal(0, 0.5, num_nodes) # Predictions with some error

    # B. Create Dummy Shock Results (Simulating the Lithuania finding)
    # Most countries have low impact (0.1), but some have high impact (0.8)
    impact_dummy = np.random.exponential(0.1, num_nodes) 
    # Force "Lithuania" and "Germany" to be high to mimic your finding
    countries = ['USA', 'CHN', 'DEU', 'LTU', 'JPN', 'KOR', 'FRA', 'GBR', 'ITA', 'BRA'] * 10
    impact_dummy[3] = 0.85 # Make LTU high
    impact_dummy[2] = 0.95 # Make DEU high

    # C. Run Plotting
    # Note: save_path argument removed to match function definition
    plot_evaluation_results(
        y_true_dummy, 
        y_pred_dummy, 
        impact_dummy, 
        countries
    )

    # D. Mock Metrics Printout
    print("\n--- Mock Metrics Output ---")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true_dummy, y_pred_dummy)):.4f}")
    print(f"R2 Score: {r2_score(y_true_dummy, y_pred_dummy):.4f}")