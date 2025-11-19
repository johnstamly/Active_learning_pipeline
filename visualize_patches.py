"""
Visualize Composite Patches Script

This script reads ply parameters from an Excel file (Data_v4.xlsx by default) and creates 
3D visualizations of composite patches with predicted strength information.
It is designed to be run standalone after the main pipeline to visualize new candidates or existing data.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import logging

# --- IMPORT NECESSARY COMPONENTS ---
from config_manager import ConfigManager 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- GLOBAL CONFIG CONSTANT ---
CONFIG_FILE = "config.yaml"

def create_ellipse_points(a, b, theta_deg):
    """
    Generates x and y coordinates for a rotated ellipse centered at the origin.
    """
    t = np.linspace(0, 2 * np.pi, 200)  # 200 points for a smooth curve
    x_unrotated = a * np.cos(t)
    y_unrotated = b * np.sin(t)
    theta_rad = np.deg2rad(theta_deg)
    rotation_matrix = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad),  np.cos(theta_rad)]
    ])
    rotated_points = rotation_matrix @ np.array([x_unrotated, y_unrotated])
    return rotated_points[0, :], rotated_points[1, :]

def read_excel_data(file_path):
    """
    Reads ply parameters from both Elips0 and Elips1 sheets in the Excel file.
    """
    try:
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names
        data = {}
        
        # NOTE: Using 'Elips0' and 'Elips1' as sheet names is standard practice.
        if 'Elips0' in sheet_names:
            data['Elips0'] = pd.read_excel(xls, sheet_name='Elips0')
        if 'Elips1' in sheet_names:
            data['Elips1'] = pd.read_excel(xls, sheet_name='Elips1')
            
        if not data:
            raise ValueError("No 'Elips0' or 'Elips1' sheets found in the Excel file.")
            
        return data
        
    except Exception as e:
        logger.error(f"Error reading Excel file: {e}")
        raise

def create_composite_patch_visualization(row_data, category, config_manager, experiment_id=None):
    """
    Creates a 3D visualization of a composite patch based on ply parameters, 
    reading feature columns and prediction columns from the config.
    """
    # --- 1. CONFIGURATION READOUT ---
    base_features = config_manager.get('COLUMNS.BASE_INPUT_FEATURES')
    pred_mean_col = config_manager.get('COLUMNS.PREDICTED_MEAN')
    
    n_half = len(base_features) // 2 # Should be 6
    a_cols = base_features[:n_half] # a_ply1 to a_ply6
    b_cols = base_features[n_half:] # b_ply1 to b_ply6
    
    # --- 2. DATA EXTRACTION ---
    
    # Extract ply parameters using config-driven column names
    try:
        a_values = np.array([row_data[col] for col in a_cols])
        b_values = np.array([row_data[col] for col in b_cols])
    except KeyError as e:
        raise KeyError(f"Missing ply feature column in data: {e}. Check Excel column names.")
    
    # Extract prediction data (using .get for safety on missing columns in old data)
    predicted_strength = row_data.get(pred_mean_col, np.nan)
    # Variance is not calculated/saved in our simplified pipeline, so we use a placeholder:
    prediction_variance = row_data.get('Prediction_Variance', np.nan) # Placeholder
    
    # --- 3. ORIENTATION LOGIC ---
    
    # Define colors for each ply
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan']
    
    # Define the default specific ply angles for Category 1 (Elips1)
    # NOTE: These are assumptions based on your previous code snippet, as we don't have
    # the theta_ply columns in our final simplified pipeline.
    elips1_rotations = [0, 90, 0, 90, -45, 45] 
    
    fig = go.Figure()
    
    for i in range(6):
        a, b = a_values[i], b_values[i]
        color = colors[i]
        z_level = i # Ply stack height
        
        # --- Determine the angle based on category (Elips0 or Elips1) ---
        if category == 'Elips0':
            # Assuming Elips0 is a monolithic orientation (e.g., 0 degrees overall)
            ply_orientation = 0
            orientation_info = f"Orientation: 0°"
        elif category == 'Elips1':
            # Use the defined ply-specific rotations for Category 1
            ply_orientation = elips1_rotations[i]
            orientation_info = f"Default θ_ply{i+1}: {ply_orientation}°"
        else:
            # Fallback
            ply_orientation = 0
            orientation_info = f"Orientation: 0°"
        
        # 4. Create Ellipse Geometry
        x_points, y_points = create_ellipse_points(a, b, ply_orientation)
        z_points = np.full_like(x_points, z_level)
        
        # 5. Add Traces (Outline and Fill)
        
        # Outline (for clear stacking visibility)
        fig.add_trace(go.Scatter3d(
            x=x_points, y=y_points, z=z_points,
            mode='lines',
            line=dict(color=color, width=7),
            name=f'Ply {i+1} ({orientation_info})',
            hoverinfo='none',
            legendgroup=f'ply{i+1}'
        ))
        
        # Mesh Fill (for interactive hovering)
        fig.add_trace(go.Mesh3d(
            x=x_points, y=y_points, z=z_points,
            color=color,
            opacity=0.2,
            # Indices for triangulating the circle (assuming 0 is center)
            i=np.full(len(x_points)-2, 0),
            j=np.arange(1, len(x_points)-1),
            k=np.arange(2, len(x_points)),
            showscale=False,
            name=f'Ply {i+1} (Fill)',
            legendgroup=f'ply{i+1}',
            hovertemplate=f'<b>Ply {i+1}</b><br>' +
                          f'a: {a:.2f} mm<br>' +
                          f'b: {b:.2f} mm<br>' +
                          f'{orientation_info}<br>' +
                          f'Z-level: {z_level}<extra></extra>'
        ))
    
    # 6. Final Layout
    title = f'Composite Patch Visualization - {category}'
    
    # If prediction data is available, add it to the title
    if pd.notna(predicted_strength):
        title += f'<br>Predicted Strength: {predicted_strength:.2f}'
    
    fig.update_layout(
        title_text=title,
        scene=dict(
            xaxis_title='X-axis (mm)',
            yaxis_title='Y-axis (mm)',
            zaxis_title='Ply Level',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=80),
        legend=dict(
            x=0, y=1,
            bgcolor='rgba(255, 255, 255, 0.5)',
            bordercolor='rgba(0, 0, 0, 0)',
            itemsizing='constant'
        )
    )
    
    return fig

def save_visualization(fig, category, experiment_id, output_dir):
    """Saves the visualization as an HTML file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine the file name based on experiment_id
    safe_id = f"ID_{experiment_id}" 
        
    filename = f"composite_patch_{category.lower()}_{safe_id}.html"
    filepath = os.path.join(output_dir, filename)
    
    logger.info(f"Saving HTML visualization for {category} - {safe_id}")
    
    fig.write_html(filepath)
    
    return filepath

# ... (Keep imports and helper functions create_ellipse_points, create_composite_patch_visualization, etc.) ...

# RENAME 'main' to this:
def generate_candidate_visualizations(config_manager: ConfigManager):
    """
    Orchestrator-friendly function to generate visualizations for newly found candidates.
    """
    logger.info("--- Starting Visualization of New Candidates ---")
    
    processed_dir = config_manager.get('PATHS.PROCESSED_DIR')
    # Save figures in a 'figures' folder next to 'processed'
    output_dir = os.path.join(os.path.dirname(processed_dir), "figures/composite_patches")
    
    # Filenames for the new candidates (Category 0 and 1)
    candidate_files = {
        'Elips0': os.path.join(processed_dir, 'new_candidate_point_cat_0.csv'),
        'Elips1': os.path.join(processed_dir, 'new_candidate_point_cat_1.csv')
    }
    
    try:
        for category, file_path in candidate_files.items():
            if not os.path.exists(file_path):
                logger.warning(f"No candidate file found for {category}. Skipping.")
                continue

            df = pd.read_csv(file_path)
            
            # Loop through the single candidate (or multiple if configured differently)
            for idx, row in df.iterrows():
                # Create ID
                cat_val = row.get(config_manager.get('COLUMNS.CATEGORY'), 'Unknown')
                experiment_id = f"Candidate_Cat{cat_val}" 
                
                logger.info(f"Visualizing {category} ({experiment_id})...")
                
                # Generate Figure
                fig = create_composite_patch_visualization(row, category, config_manager, experiment_id)
                
                # Save
                saved_path = save_visualization(fig, category, experiment_id, output_dir)
                logger.info(f"Saved to: {saved_path}")
                
    except Exception as e:
        logger.error(f"Error during visualization: {e}")

# Keep this for standalone testing
if __name__ == "__main__":
    cm = ConfigManager("config.yaml")
    generate_candidate_visualizations(cm)