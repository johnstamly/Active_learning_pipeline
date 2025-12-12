"""
Visualize Composite Patches Script (3D Staircase + 2D Top View)

This script generates:
1. 3D Staircase visualizations (HTML)
2. 2D Transparent Top-View plots (PNG)
   - Order: Outer Ply (drawn first) -> Inner Ply (drawn last/on top)
   - FORCED Title Mapping based on Row Index.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.lines as mlines
from datetime import datetime
import logging

from config_manager import ConfigManager 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Resolution for 3D meshes
N_ELLIPSE_POINTS = 150 

# --- HELPER: 3D POINTS ---
def create_ellipse_points(a, b, theta_deg):
    """Generates x and y coordinates for a rotated ellipse (for 3D Plotly)."""
    t = np.linspace(0, 2 * np.pi, N_ELLIPSE_POINTS)
    x_unrotated = a * np.cos(t)
    y_unrotated = b * np.sin(t)
    theta_rad = np.deg2rad(theta_deg)
    
    rotation_matrix = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad),  np.cos(theta_rad)]
    ])
    
    rotated_points = rotation_matrix @ np.array([x_unrotated, y_unrotated])
    rotated_points[:, -1] = rotated_points[:, 0] # Close loop
    return rotated_points[0, :], rotated_points[1, :]

# --- HELPER: FORCED TITLE MAPPING ---
def get_run_title(category_val, row_idx):
    """
    Returns the specific Run Title based STRICTLY on the Row Index order.
    """
    try:
        r_idx = int(row_idx)
    except:
        return f"Row {row_idx}"

    # Convert category to string for safe comparison
    cat_str = str(category_val).strip()

    # --- CATEGORY 0 MAPPING ---
    if cat_str == '0' or cat_str == '0.0':
        mapping = {
            0: "Run: 8",          # Row 0
            1: "Initial Specimen: 16",    # Row 1
            2: "Run: 47",         # Row 2
            3: "Run: 40"          # Row 3
        }
        return mapping.get(r_idx, f"Cat0 Row {r_idx}")

    # --- CATEGORY 1 MAPPING ---
    elif cat_str == '1' or cat_str == '1.0':
        mapping = {
            0: "Run: 42",         # Row 0
            1: "Run: 36",         # Row 1
            2: "Run: 23",         # Row 2
            3: "Initial Specimen: 11"     # Row 3
        }
        return mapping.get(r_idx, f"Cat1 Row {r_idx}")
    
    return f"Row {r_idx}"

# ==========================================
# 1. 3D STAIRCASE VISUALIZATION (Plotly)
# ==========================================
def create_3d_staircase_viz(row_data, category, config_manager, title_text):
    # [Data Extraction]
    base_features = config_manager.get('COLUMNS.BASE_INPUT_FEATURES')
    n_half = len(base_features) // 2 
    a_cols = base_features[:n_half] 
    b_cols = base_features[n_half:] 
    
    try:
        a_values = np.array([row_data[col] for col in a_cols])
        b_values = np.array([row_data[col] for col in b_cols])
    except KeyError:
        return None
    
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan']
    elips1_rotations = [0, 90, 0, 90, -45, 45] 
    
    fig = go.Figure()

    for i in range(len(a_values)):
        a, b = a_values[i], b_values[i]
        color = colors[i % len(colors)]
        z_bottom, z_top = float(i), float(i + 1)
        
        if category == 'Elips1':
            ply_orientation = elips1_rotations[i] if i < len(elips1_rotations) else 0
        else:
            ply_orientation = 0
            
        x_points, y_points = create_ellipse_points(a, b, ply_orientation)
        
        # Walls
        wall_x = np.concatenate([x_points, x_points])
        wall_y = np.concatenate([y_points, y_points])
        wall_z = np.concatenate([np.full_like(x_points, z_bottom), np.full_like(x_points, z_top)])
        
        M = N_ELLIPSE_POINTS
        tri_i, tri_j, tri_k = [], [], []
        for p in range(M - 1):
            tri_i.append(p); tri_j.append(p + 1); tri_k.append(p + M)
            tri_i.append(p + 1); tri_j.append(p + 1 + M); tri_k.append(p + M)

        # Mesh Block
        fig.add_trace(go.Mesh3d(
            x=wall_x, y=wall_y, z=wall_z, i=tri_i, j=tri_j, k=tri_k,
            color=color, opacity=0.9, flatshading=True, showscale=False,
            name=f'Ply {i+1}', hoverinfo='name'
        ))
        
        # Black Outlines
        for z_h in [z_bottom, z_top]:
            fig.add_trace(go.Scatter3d(
                x=x_points, y=y_points, z=np.full_like(x_points, z_h),
                mode='lines', line=dict(color='black', width=4), showlegend=False, hoverinfo='skip'
            ))

    fig.update_layout(
        title_text=title_text,
        scene=dict(
            xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Ply Layer',
            aspectmode='data', camera=dict(eye=dict(x=1.6, y=1.6, z=1.4))
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    return fig

# ==========================================
# 2. 2D TOP VIEW VISUALIZATION (Matplotlib)
# ==========================================
def create_2d_ply_visualization(row_data, category, config_manager, title_text, output_path):
    """
    Creates a transparent 2D top-down plot.
    Order: Outer Ply (Bottom, drawn first) -> Inner Ply (Top, drawn last)
    """
    # 1. Data Setup
    base_features = config_manager.get('COLUMNS.BASE_INPUT_FEATURES')
    n_half = len(base_features) // 2 
    a_cols = base_features[:n_half] 
    b_cols = base_features[n_half:] 
    
    try:
        a_values = np.array([row_data[col] for col in a_cols])
        b_values = np.array([row_data[col] for col in b_cols])
    except KeyError:
        return

    # 2. Plot Setup
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan']
    elips1_rotations = [0, 90, 0, 90, -45, 45] 
    
    max_dim = 0
    legend_handles = []

    # 3. Draw Plies (REVERSED: Outer 6 -> Inner 1)
    for i in reversed(range(len(a_values))):
        a, b = a_values[i], b_values[i]
        color = colors[i % len(colors)]
        
        if category == 'Elips1':
            angle = elips1_rotations[i] if i < len(elips1_rotations) else 0
        else:
            angle = 0
            
        max_dim = max(max_dim, a, b)
        
        # Draw Ellipse
        e = Ellipse(xy=(0, 0), width=2*a, height=2*b, angle=angle, 
                    edgecolor=color, facecolor=color, 
                    alpha=0.4, # Transparent
                    linewidth=2)
        ax.add_patch(e)

    # 4. Legend (Normal Order 1 -> 6)
    for i in range(len(a_values)):
        color = colors[i % len(colors)]
        angle = elips1_rotations[i] if (category == 'Elips1' and i < len(elips1_rotations)) else 0
        handle = mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                              markersize=10, alpha=0.6, label=f"Ply {i+1} ({angle}°)")
        legend_handles.append(handle)

    # 5. Formatting
    limit = max_dim * 1.2
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    
    ax.set_xlabel("X-axis (mm)", fontsize=12)
    ax.set_ylabel("Y-axis (mm)", fontsize=12)
    ax.set_title(title_text, fontsize=14, fontweight='bold', pad=15)
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='black', linewidth=0.5, alpha=0.3)
    
    ax.legend(handles=legend_handles, loc='upper right', framealpha=0.9, title="Ply Stack")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig) 
    return output_path

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def generate_candidate_visualizations(config_manager: ConfigManager):
    logger.info("--- Starting Visualizations (3D Solid + 2D Planar) ---")
    
    processed_dir = config_manager.get('PATHS.PROCESSED_DIR')
    
    dir_3d = os.path.join(os.path.dirname(processed_dir), "figures/staircase_3d")
    dir_2d = os.path.join(os.path.dirname(processed_dir), "figures/top_view_2d")
    os.makedirs(dir_3d, exist_ok=True)
    os.makedirs(dir_2d, exist_ok=True)
    
    candidate_files = {
        'Elips0': os.path.join(processed_dir, 'new_candidate_point_cat_0.csv'),
        'Elips1': os.path.join(processed_dir, 'new_candidate_point_cat_1.csv')
    }
    
    try:
        for category, file_path in candidate_files.items():
            if not os.path.exists(file_path): continue

            df = pd.read_csv(file_path)
            if df.empty: continue

            logger.info(f"Processing {len(df)} candidates in {category}...")

            for idx, row in df.iterrows():
                # 1. Get Category
                cat_val = row.get(config_manager.get('COLUMNS.CATEGORY'), 'Unknown')
                
                # 2. GET FORCED TITLE (Based on Row Index)
                # Ensure cat_val is passed correctly from the row
                run_title = get_run_title(cat_val, idx)
                
                # Safe Filename
                specimen_id = row.get('Specimen_ID', idx)
                file_suffix = f"Cat{cat_val}_Row{idx}"
                
                logger.info(f"  > Row {idx} (Cat {cat_val}) -> Title: '{run_title}'")

                # --- 3D HTML ---
                fig_3d = create_3d_staircase_viz(row, category, config_manager, run_title)
                if fig_3d:
                    path_3d = os.path.join(dir_3d, f"staircase_{file_suffix}.html")
                    fig_3d.write_html(path_3d)

                # --- 2D PNG ---
                path_2d = os.path.join(dir_2d, f"top_view_{file_suffix}.png")
                create_2d_ply_visualization(row, category, config_manager, run_title, path_2d)
                
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if os.path.exists("config.yaml"):
        cm = ConfigManager("config.yaml")
        generate_candidate_visualizations(cm)
    else:
        logger.error("config.yaml not found.")