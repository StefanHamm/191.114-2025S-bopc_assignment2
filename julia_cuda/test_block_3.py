import subprocess
import numpy as np
import matplotlib.pyplot as plt # Import once at the top
import pandas as pd
from tqdm import tqdm
# import random # Not used as block_sizes is fixed in this example

# Parameters for the scalability analysis
problem_sizes = [20000]
block_sizes = [(1,1),(32,1),(1,32),(128,1),(1024,1)] # Fixed for this run
# block_sizes = [(i, j) for i in range(1, 1025, 1) for j in range(16, 1025, 1) if i * j % 32 == 0 and i * j <= 1024]
# if using the full list and sampling:
# import random
# block_sizes = random.sample(block_sizes, 50)

print(block_sizes)
results = []        # For summarized results (mean, std, max per setting)
raw_results = []    # For raw results from every run/repetition
reps = 5
    
print("Problem size | Processors | Mean runtime (s) | Speedup | Efficiency") # This header seems for a different table format
print(len(block_sizes))
print("-" * 70)

for block_size_tuple in tqdm(block_sizes):
    configured_block_x, configured_block_y = block_size_tuple
    for size in problem_sizes:
        cmd = f"./juliaset_gpu -r {size} {size} -n {reps} -b {configured_block_x} {configured_block_y}"
        output = subprocess.check_output(cmd, shell=True, text=True)

        # Parse output lines
        # Example line format: rep;res_x;res_y;scale;global_block_x;global_block_y;runtime_str
        parsed_lines = [line.strip().split(';') for line in output.strip().split('\n') if line.strip()]
        
        runtimes_for_current_setting = []
        for fields in parsed_lines:
            if len(fields) != 7: # Basic validation for number of fields
                print(f"Warning: Skipping malformed line: {';'.join(fields)}")
                continue
            
            rep_str, res_x_str, res_y_str, scale_str, gbx_str, gby_str, runtime_str = fields
            
            # Convert strings to appropriate numeric types
            try:
                runtime_val = float(runtime_str)
                raw_results.append({
                    'rep': int(rep_str),
                    'res_x': int(res_x_str),
                    'res_y': int(res_y_str),
                    'scale': float(scale_str),
                    'global_block_x': int(gbx_str), # Block dim reported by the executable
                    'global_block_y': int(gby_str), # Block dim reported by the executable
                    'runtime': runtime_val
                })
                runtimes_for_current_setting.append(runtime_val)
            except ValueError as e:
                print(f"Warning: Could not parse line: {';'.join(fields)}. Error: {e}")
                continue
            
        if not runtimes_for_current_setting: # If no valid runtimes were parsed for this setting
            print(f"Warning: No valid runtimes recorded for size {size}, block {configured_block_x}x{configured_block_y}")
            continue

        # Calculate mean, std, max of runtimes for the current setting
        mean_runtime = np.mean(runtimes_for_current_setting)
        std_runtime = np.std(runtimes_for_current_setting)
        max_runtime = np.max(runtimes_for_current_setting)

        # Store summarized results. Use 'size' (int) and configured block dimensions.
        results.append({
            'res_x': size, 
            'res_y': size,
            'mean_runtime': mean_runtime,
            'std_runtime': std_runtime,
            'max_runtime': max_runtime,
            'block_size_x_config': configured_block_x, # Configured block dim
            'block_size_y_config': configured_block_y  # Configured block dim
        })

# Convert raw results to DataFrame (all columns will have appropriate numeric types)
df_raw = pd.DataFrame(raw_results)
print("Raw DataFrame sample:\n", df_raw.head())

# Analysis grouped by actual global block sizes used by GPU
# (assuming global_block_x/y are now integers in df_raw)
df_block_analysis = df_raw.groupby(['global_block_x', 'global_block_y']).agg(
    mean_runtime=('runtime', 'mean'),
    max_runtime=('runtime', 'max')
)
df_block_analysis = df_block_analysis.sort_values(by='mean_runtime', ascending=True)
print("\nBlock Size Analysis (sorted by mean_runtime):\n", df_block_analysis)

# Save block analysis results. To keep global_block_x/y as columns:
df_block_analysis.reset_index().to_csv('block_size_analysis3.csv', index=False)

# Save summarized results (df_summary)
df_summary = pd.DataFrame(results)
df_summary.to_csv('results3.csv', index=False)

# Save raw results
df_raw.to_csv('raw_results3.csv', index=False)

import matplotlib.pyplot as plt

df_plot_data = df_raw.groupby(["global_block_x","global_block_y"]).agg(
    mean_runtime=('runtime', 'mean'),
    max_runtime=('runtime', 'max')
)
# Convert MultiIndex to string labels for plotting
df_plot_data.index = [f'({x}, {y})' for x, y in df_plot_data.index]

plt.figure(figsize=(10, 5))
plt.plot(df_plot_data.index, df_plot_data['mean_runtime'], label='Mean Runtime', marker='o')
plt.plot(df_plot_data.index, df_plot_data['max_runtime'], label='Max Runtime', marker='o')
plt.xlabel('Block Size (X, Y)')
plt.xticks(rotation=45)
plt.ylabel('Runtime (ms)')

plot_title = 'Runtime vs Block Size'
if len(block_sizes) == 1:
    plot_title += f' (Runtimes for fixed problem size 20000)'
plt.title(plot_title)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('runtime_vs_input_size.png')
print("\nPlot 'runtime_vs_input_size.png' saved.")