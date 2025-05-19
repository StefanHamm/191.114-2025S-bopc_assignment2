# Script for scalability analysis of Julia set computation

import subprocess
import numpy as np
# import matplotlib.pyplot as plt # Not used for the 3D plot anymore
import pandas as pd
from tqdm import tqdm
import random
import plotly.graph_objects as go # Moved import to the top

# Parameters for the scalability analysis
problem_sizes = [1500]

# --- Refined block_sizes generation ---
# Ensure i*j is always a multiple of 32 AND i and j are within reasonable GPU limits
# Max threads per block usually 1024 for CUDA.
# blockDim.x <= 1024, blockDim.y <= 1024, blockDim.z <= 64
# blockDim.x * blockDim.y * blockDim.z <= 1024
block_sizes_candidates = []
# Iterate through potential block dimensions for X and Y
# Let's assume blockDim.z is 1 for 2D blocks
for i in range(1, 1024): # block_size_x, e.g., 1 to 32 (max 1024/1)
    for j in range(1, 1024): # block_size_y, e.g., 1 to 32 (max 1024/1)
        if (i * j) <= 1024 and (i * j) % 32 == 0 and i * j > 0: # Total threads constraint and warp multiple
             block_sizes_candidates.append((i, j))

# If you need larger individual dimensions but still <=1024 total threads:
# for i in range(1, 1025):
#     for j in range(1, 1025):
#         if (i * j) <= 1024 and (i * j) % 32 == 0 and i*j > 0:
#             # To avoid too many candidates, maybe add more constraints or sample differently
#             if (i <= 64 and j <= 64): # Example to keep individual dims reasonable
#                 block_sizes_candidates.append((i,j))
# block_sizes_candidates = list(set(block_sizes_candidates)) # Remove duplicates if any

# Sample from block_sizes_candidates
num_samples = 250 # You had 10, for a good surface plot, you might need more like 50-100+
                 # depending on how many unique x and y coordinates they produce.
if len(block_sizes_candidates) == 0:
    print("Error: No valid block_sizes_candidates generated. Exiting.")
    exit()
elif len(block_sizes_candidates) < num_samples:
    print(f"Warning: Only {len(block_sizes_candidates)} valid block size candidates found. Using all of them.")
    block_sizes = block_sizes_candidates
else:
    block_sizes = random.sample(block_sizes_candidates, num_samples)

print("Selected block_sizes for testing:", block_sizes)

results = [] # This list seems unused if df_raw is the main focus for this plot
raw_results = []

print(f"Number of block_size configurations to test: {len(block_sizes)}")
print("-" * 70)

for block_config in tqdm(block_sizes): # Renamed for clarity
    block_dim_x, block_dim_y = block_config
    for size in problem_sizes:
        cmd = f"./juliaset_gpu -r {size} {size} -n 3 -b {block_dim_x} {block_dim_y}"
        try:
            output = subprocess.check_output(cmd, shell=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {cmd}")
            print(f"Output: {e.output}")
            print(f"Stderr: {e.stderr}")
            continue # Skip this iteration

        # Split output into lines, removing empty lines
        lines_output = [line.strip() for line in output.strip().split('\n') if line.strip()]
        
        runtimes_for_current_config = [] # Runtimes for this specific block_config and size

        for line_str in lines_output:
            parts = line_str.split(';')
            if len(parts) != 7:
                print(f"Warning: Malformed line '{line_str}' from command '{cmd}'. Skipping.")
                continue
            
            rep_str, res_x_str, res_y_str, scale_str, gb_x_str, gb_y_str, runtime_str = parts
            
            try:
                # --- CRITICAL FIX: Convert to numeric types ---
                raw_results.append({
                    'rep': int(rep_str),
                    'res_x': int(res_x_str),
                    'res_y': int(res_y_str),
                    'scale': float(scale_str), # Assuming scale can be float
                    'global_block_x': int(gb_x_str), # These are crucial for grouping and plotting
                    'global_block_y': int(gb_y_str), # These are crucial
                    'runtime': float(runtime_str)
                })
                runtimes_for_current_config.append(float(runtime_str))
            except ValueError as e:
                print(f"Error converting data: {e} for line: {parts} from command: {cmd}")
                continue
        
        # This 'results' list appends aggregated data per (block_config, size) pair
        # It's separate from raw_results. Ensure it's what you intend.
        if runtimes_for_current_config: # if any valid runtimes were collected
            mean_runtime = np.mean(runtimes_for_current_config)
            std_runtime = np.std(runtimes_for_current_config)
            max_runtime = np.max(runtimes_for_current_config)

            # Use the actual gb_x and gb_y from the last valid parsed line,
            # or better, assume they are constant for a given cmd run.
            # For 'results', it might be more about the requested block_dim_x, block_dim_y
            results.append({
                'requested_block_x': block_dim_x,
                'requested_block_y': block_dim_y,
                'problem_size': size, # Assuming this is constant for this inner loop
                'mean_runtime': mean_runtime,
                'std_runtime': std_runtime,
                'max_runtime': max_runtime,
            })


if not raw_results:
    print("Error: No raw results collected. Cannot proceed to plotting.")
    exit()

# Convert results to DataFrame
df_raw = pd.DataFrame(raw_results)

print("\n--- df_raw sample ---")
print(df_raw.head())
print("\n--- df_raw info ---")
df_raw.info()

# Group by the ACTUAL global block dimensions reported by the executable
df_grouped = df_raw.groupby(['global_block_x', 'global_block_y']).agg(
    mean_runtime=('runtime', 'mean'),
    max_runtime=('runtime', 'max')
).reset_index() # Reset index to make global_block_x/y columns again for pivoting

print("\n--- df_grouped sample (after reset_index) ---")
print(df_grouped.head())
print("\n--- df_grouped info ---")
df_grouped.info()


# ---- START: Plotly 3D Surface Plot ----
if df_grouped.empty:
    print("df_grouped is empty. Skipping Plotly 3D surface plot.")
else:
    # Prepare data for Plotly Surface plot
    # We need a grid of Z values.
    try:
        # Pivot the table: global_block_x as index, global_block_y as columns, mean_runtime as values
        df_pivot = df_grouped.pivot(index='global_block_x', columns='global_block_y', values='mean_runtime')
    except Exception as e:
        print(f"Error during pivoting: {e}")
        print("This can happen if there are duplicate (global_block_x, global_block_y) pairs after grouping, which reset_index should prevent.")
        print("Original df_grouped before trying to pivot (if you removed reset_index):")
        # If you had not used reset_index() above, and instead did unstack:
        # df_pivot = df_grouped['mean_runtime'].unstack(level='global_block_y')
        # print(df_grouped)
        df_pivot = pd.DataFrame() # Ensure df_pivot exists to avoid later errors

    if df_pivot.empty:
        print("df_pivot is empty after attempting pivot. Cannot create surface plot.")
    else:
        print("\n--- df_pivot sample ---")
        print(df_pivot.head())
        print("\n--- df_pivot shape ---")
        print(df_pivot.shape)

        # Sort the index (global_block_x) and columns (global_block_y)
        df_pivot = df_pivot.sort_index(axis=0) # Sort rows
        df_pivot = df_pivot.sort_index(axis=1) # Sort columns

        x_coords = df_pivot.columns.to_list()
        y_coords = df_pivot.index.to_list()
        z_coords = df_pivot.values # This will be a 2D numpy array, potentially with NaNs

        print("\n--- Plotly Surface Data ---")
        print(f"x_coords (global_block_y values, {len(x_coords)} unique): {x_coords[:10]}...") # Print first 10
        print(f"y_coords (global_block_x values, {len(y_coords)} unique): {y_coords[:10]}...") # Print first 10
        print(f"z_coords shape: {z_coords.shape}")

        # Check if we have enough data for a surface (at least 2x2 grid)
        if len(x_coords) < 2 or len(y_coords) < 2:
            print("\nWarning: Not enough unique x or y coordinates to form a surface plot.")
            print("You need at least 2 unique global_block_x values and 2 unique global_block_y values after grouping.")
            print("Consider increasing the number of samples in 'block_sizes'.")
        else:
            fig_plotly = go.Figure(data=[go.Surface(
                z=z_coords,
                x=x_coords,
                y=y_coords,
                colorscale='Viridis',
                colorbar=dict(title='Mean Runtime (s)'),
                contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True),
                name='Mean Runtime',
                connectgaps=True # Set to True if you want to try to connect over NaN gaps
            )])

            fig_plotly.update_layout(
                title='3D Surface Plot of Mean Runtime vs Global Block Configuration',
                scene=dict(
                    xaxis_title='Global Block Y (Threads)',
                    yaxis_title='Global Block X (Threads)',
                    zaxis_title='Mean Runtime (s)',
                    aspectratio=dict(x=1, y=1, z=0.7)
                ),
                autosize=False,
                width=1000,
                height=800,
                margin=dict(l=65, r=50, b=65, t=90)
            )

            plot_filename_html = 'blocksize_analysis_3d_surface.html'
            fig_plotly.write_html(plot_filename_html)
            print(f"\nSaved Plotly 3D surface plot to {plot_filename_html}")

# ---- END: Plotly 3D Surface Plot ----

df_raw.to_csv('blocksize_analysis_runs.csv', index=False)
print("Saved raw run data to blocksize_analysis_runs.csv")

print("\nScript finished.")