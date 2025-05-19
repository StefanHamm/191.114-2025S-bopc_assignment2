# Script for scalability analysis of Julia set computation

import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import random

# Parameters for the scalability analysis
problem_sizes = [1500]
block_sizes = [(i, j) for i in range(1, 1025, 1) for j in range(16, 1025, 1) if i * j % 32 == 0]
#sample from block_sizes into block_sizes 200 samples without replacement
block_sizes = random.sample(block_sizes, 10)
print(block_sizes)
results = []
raw_results = []

    
print("Problem size | Processors | Mean runtime (s) | Speedup | Efficiency")
print(len(block_sizes))
print("-" * 70)
for block_size in tqdm(block_sizes):
    for size in problem_sizes:
        cmd = f"./juliaset_gpu -r {size} {size} -n 3 -b {block_size[0]} {block_size[1]}"
        output = subprocess.check_output(cmd, shell=True, text=True)

        values = [line.strip().split(';') for line in output.strip().split('\n')]
        runtimes = []
        for line in values:
            rep, res_x, res_y, scale, global_block_x, global_block_y, runtime_str = line
            raw_results.append({
            'rep': rep,
            'res_x': res_x,
            'res_y': res_y,
            'scale': scale,
            'global_block_x': global_block_x,
            'global_block_y': global_block_y,
            'runtime': float(runtime_str)
            })
            runtimes.append(float(runtime_str))
            
        # Calculate mean and std of runtimes
        mean_runtime = np.mean(runtimes)
        std_runtime = np.std(runtimes)
        max_runtime = np.max(runtimes)

        results.append({
            'res_x': res_x,
            'res_y': res_y,
            'mean_runtime': mean_runtime,
            'std_runtime': std_runtime,
            'max_runtime': max_runtime,
        })

# Convert results to DataFrame
df_raw = pd.DataFrame(raw_results)

print(df_raw)

df_grouped = df_raw.groupby(['global_block_x', 'global_block_y']).agg(mean_runtime=('runtime', 'mean'),
                                                                      max_runtime=('runtime', 'max'))
# ---- START: Plotly 3D Surface Plot ----
import plotly.graph_objects as go
import pandas as pd # Ensure pandas is imported if this snippet is run standalone

# Assuming df_grouped is already created and populated as per your script:
# df_grouped has a MultiIndex: ('global_block_x', 'global_block_y')
# and columns: 'mean_runtime', 'max_runtime'

if df_grouped.empty:
    print("df_grouped is empty. Skipping Plotly 3D surface plot.")
else:
    # Prepare data for Plotly Surface plot
    # We need a grid of Z values.
    # Let 'global_block_x' define the rows (y-axis of the surface plot).
    # Let 'global_block_y' define the columns (x-axis of the surface plot).
    # The values in the grid will be 'mean_runtime'.
    
    # Unstack to pivot 'global_block_y' to columns.
    # The index will be 'global_block_x'.
    df_pivot = df_grouped['mean_runtime'].unstack(level='global_block_y')

    # Sort the index (global_block_x) and columns (global_block_y)
    # This ensures the surface is plotted correctly.
    df_pivot = df_pivot.sort_index(axis=0) # Sort rows by global_block_x
    df_pivot = df_pivot.sort_index(axis=1) # Sort columns by global_block_y

    # Extract coordinates for the surface plot
    # x_coords will be the column names (global_block_y values)
    # y_coords will be the index names (global_block_x values)
    # z_coords will be the 2D array of mean_runtime values
    x_coords = df_pivot.columns.to_list()
    y_coords = df_pivot.index.to_list()
    z_coords = df_pivot.values

    # Create Plotly 3D Surface plot
    fig_plotly = go.Figure(data=[go.Surface(
        z=z_coords,  # 2D array of Z values
        x=x_coords,  # 1D array for X-axis (corresponds to columns of Z)
        y=y_coords,  # 1D array for Y-axis (corresponds to rows of Z)
        colorscale='Viridis',       # Choose a colorscale
        colorbar=dict(title='Mean Runtime (s)'), # Colorbar title
        contours_z=dict( # Add contours on the Z-axis projection
            show=True, 
            usecolormap=True, 
            highlightcolor="limegreen", 
            project_z=True
        ),
        name='Mean Runtime'
    )])

    # Update layout for titles and axis labels
    fig_plotly.update_layout(
        title='3D Surface Plot of Mean Runtime vs Global Block Configuration',
        scene=dict(
            xaxis_title='Global Block Y (Threads)', # Corresponds to x_coords
            yaxis_title='Global Block X (Threads)', # Corresponds to y_coords
            zaxis_title='Mean Runtime (s)',
            # You can adjust camera and aspect ratio if needed
            # camera=dict(eye=dict(x=1.8, y=1.8, z=0.7)),
            aspectratio=dict(x=1, y=1, z=0.6) 
        ),
        autosize=False,
        width=1000, # Adjust width
        height=800, # Adjust height
        margin=dict(l=65, r=50, b=65, t=90) # Adjust margins
    )

    # Save to an interactive HTML file
    plot_filename_html = 'blocksize_analysis_3d_surface.html'
    fig_plotly.write_html(plot_filename_html)
    print(f"Saved Plotly 3D surface plot to {plot_filename_html}")


df_raw.to_csv('blocksize_analysis_runs.csv', index=False)




#df = pd.DataFrame(results)

# """ 
# for workload_type in df['c'].unique():
#     print(f"\n{'='*60}")
#     print(f"Analysis for Workload: {workload_type}")
#     print(f"{'='*60}")

#     # Filter data for the current workload
#     workload_df = df[df['c'] == workload_type].sort_values(by=['problem_size', 'nprocs'])

#     # --- 1. Generate Table ---
#     print(f"\nTable for Workload '{workload_type}': Mean Runtime, Speedup, and Efficiency")
#     print("-" * 75)
#     print(f"{'Size':>6} | {'Processors':>10} | {'Runtime (s)':>15} | {'Speedup':>10} | {'Efficiency':>12}")
#     print("-" * 75)
#     for _, row in workload_df.iterrows():
#         print(
#             f"{int(row['problem_size']):>6} | {int(row['nprocs']):>10} | {row['mean_runtime']:>15.6f} | "
#             f"{row['speedup']:>10.2f} | {row['efficiency']:>12.2f}"
#         )
#     print("-" * 75)


#     # --- 2. Generate Plots (Runtime, Speedup, Efficiency) ---
#     fig, axes = plt.subplots(1, 3, figsize=(21, 6)) # 1 row, 3 columns for the plots
#     fig.suptitle(f'Performance Analysis for Workload: {workload_type}', fontsize=16)

#     # a) Absolute Running Time Plot
#     ax = axes[0]
#     for size in problem_sizes:
#         size_df = workload_df[workload_df['problem_size'] == size]
#         ax.plot(size_df['nprocs'], size_df['mean_runtime'],
#                 marker='o', linestyle='-', label=f'Size {size}')
#     ax.set_xlabel('Number of Cores')
#     ax.set_ylabel('Runtime (seconds)')
#     ax.set_title('Absolute Running Time')
#     ax.set_xticks(processor_counts) # Ensure ticks match actual core counts

#     ax.grid(True, which='both', linestyle='--', linewidth=0.5)
#     ax.legend()


#     # b) Relative Speed-up Plot
#     ax = axes[1]
#     ideal_speedup_line_plotted = False
#     for size in problem_sizes:
#         size_df = workload_df[workload_df['problem_size'] == size]
#         ax.plot(size_df['nprocs'], size_df['speedup'],
#                 marker='s', linestyle='-', label=f'Size {size}')

#         # Add ideal speedup line (only once)
#         if not ideal_speedup_line_plotted:
#              # Ensure the line covers the full range of processor counts
#             ax.plot(processor_counts, processor_counts, 'r--', label='Ideal Speedup')
#             ideal_speedup_line_plotted = True

#     ax.set_xlabel('Number of Cores')
#     ax.set_ylabel('Relative Speed-up (T_1 / T_N)')
#     ax.set_title('Relative Speed-up')
#     ax.set_xticks(processor_counts)
#     # Set y-limit maybe slightly larger than max procs for ideal line visibility
#     ax.set_ylim(bottom=0, top=32 * 1.1)
#     ax.grid(True, which='both', linestyle='--', linewidth=0.5)
#     ax.legend()


#     # c) Parallel Efficiency Plot
#     ax = axes[2]
#     ideal_efficiency_line_plotted = False
#     for size in problem_sizes:
#         size_df = workload_df[workload_df['problem_size'] == size]
#         ax.plot(size_df['nprocs'], size_df['efficiency'],
#                 marker='^', linestyle='-', label=f'Size {size}')

#         # Add ideal efficiency line (only once)
#         if not ideal_efficiency_line_plotted:
#             ax.axhline(y=1.0, color='r', linestyle='--', label='Ideal Efficiency (1.0)')
#             ideal_efficiency_line_plotted = True

#     ax.set_xlabel('Number of Cores')
#     ax.set_ylabel('Parallel Efficiency (Speed-up / N)')
#     ax.set_title('Parallel Efficiency')
#     ax.set_xticks(processor_counts)
#     ax.set_ylim(bottom=0, top=1.1) # Efficiency typically between 0 and 1
#     ax.grid(True, which='both', linestyle='--', linewidth=0.5)
#     ax.legend()
#     #  # Add discussion point placeholder - replace with actual findings

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for suptitle
#     plt.savefig(f'plots/scalability_analysis_{workload_type}_2_2.png')
#     plt.show()

# Save results to CSV
