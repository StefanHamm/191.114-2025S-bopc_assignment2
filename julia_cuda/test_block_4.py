import subprocess
import numpy as np
import matplotlib.pyplot as plt # Import once at the top
import pandas as pd
from tqdm import tqdm
# import random # Not used as block_sizes is fixed in this example

# Parameters for the scalability analysis
problem_sizes = [1000, 2000, 4000, 8000, 16000]
block_sizes = [(128,1)] # Fixed for this run
# block_sizes = [(i, j) for i in range(1, 1025, 1) for j in range(16, 1025, 1) if i * j % 32 == 0 and i * j <= 1024]
# if using the full list and sampling:
# import random
# block_sizes = random.sample(block_sizes, 50)

print(block_sizes)
results = []        # For summarized results (mean, std, max per setting)
results2 = []       # For summarized results (mean, std, max per setting)
raw_results = []    # For raw results from every run/repetition
raw_results2 = []   # For raw results from every run/repetition
reps = 5
nvprof = ""    
print("Problem size | Processors | Mean runtime (s) | Speedup | Efficiency") # This header seems for a different table format
print(len(block_sizes))
print("-" * 70)

for block_size_tuple in tqdm(block_sizes):
    configured_block_x, configured_block_y = block_size_tuple
    for size in problem_sizes:
        cmd = f"./juliaset_gpu -r {size} {size} -n {reps} -b {configured_block_x} {configured_block_y}"
        output = subprocess.check_output(cmd, shell=True, text=True)
        cmd2 = f"./juliaset_cpu -r {size} {size} -n {reps}"
        output2 = subprocess.check_output(cmd2, shell=True, text=True)
        
        # Parse output lines
        # Example line format: rep;res_x;res_y;scale;global_block_x;global_block_y;runtime_str
        parsed_lines = [line.strip().split(';') for line in output.strip().split('\n') if line.strip()]
        parsed_lines2 = [line.strip().split(';') for line in output2.strip().split('\n') if line.strip()]
        
        runtimes_for_current_setting = []
        runtimes_for_current_setting2 = []
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
        # Parse CPU output lines
        for fields in parsed_lines2:
            rep_str, res_x_str, res_y_str, scale_str, gbx_str, gby_str, runtime_str = fields
             try:
                runtime_val = float(runtime_str)
                raw_results2.append({
                    'rep': int(rep_str),
                    'res_x': int(res_x_str),
                    'res_y': int(res_y_str),
                    'scale': float(scale_str),
                    'global_block_x': int(gbx_str), # Block dim reported by the executable
                    'global_block_y': int(gby_str), # Block dim reported by the executable
                    'runtime': runtime_val
                })
                runtimes_for_current_setting2.append(runtime_val)
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
        
        # Calculate mean, std, max of runtimes for the current setting
        mean_runtime2 = np.mean(runtimes_for_current_setting2)
        std_runtime2 = np.std(runtimes_for_current_setting2)
        max_runtime2 = np.max(runtimes_for_current_setting2)

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
        
        results2.append({
            'res_x': size, 
            'res_y': size,
            'mean_runtime': mean_runtime2,
            'std_runtime': std_runtime2,
            'max_runtime': max_runtime2,
            'block_size_x_config': -1, # Configured block dim
            'block_size_y_config': -1  # Configured block dim
        })

# UPDATE this part to integrate the CPU results aswell

# # Convert raw results to DataFrame (all columns will have appropriate numeric types)
# df_raw = pd.DataFrame(raw_results)
# print("Raw DataFrame sample:\n", df_raw.head())
# #write nvprof output to file

# # Analysis grouped by actual global block sizes used by GPU
# # (assuming global_block_x/y are now integers in df_raw)
# df_block_analysis = df_raw.groupby(['global_block_x', 'global_block_y']).agg(
#     mean_runtime=('runtime', 'mean'),
#     max_runtime=('runtime', 'max')
# )
# df_block_analysis = df_block_analysis.sort_values(by='mean_runtime', ascending=True)
# print("\nBlock Size Analysis (sorted by mean_runtime):\n", df_block_analysis)

# # Save block analysis results. To keep global_block_x/y as columns:
# df_block_analysis.reset_index().to_csv('block_size_analysis3.csv', index=False)

# # Save summarized results (df_summary)
# df_summary = pd.DataFrame(results)
# df_summary.to_csv('results3.csv', index=False)

# # Save raw results
# df_raw.to_csv('raw_results3.csv', index=False)

# import matplotlib.pyplot as plt

# df_plot_data = df_raw.groupby(["global_block_x","global_block_y"]).agg(
#     mean_runtime=('runtime', 'mean'),
#     max_runtime=('runtime', 'max')
# )
# # Convert MultiIndex to string labels for plotting
# df_plot_data.index = [f'({x}, {y})' for x, y in df_plot_data.index]

# plt.figure(figsize=(10, 5))
# plt.plot(df_plot_data.index, df_plot_data['mean_runtime'], label='Mean Runtime', marker='o')
# plt.plot(df_plot_data.index, df_plot_data['max_runtime'], label='Max Runtime', marker='o')
# plt.xlabel('Block Size (X, Y)')
# plt.xticks(rotation=45)
# plt.ylabel('Runtime (micro seconds)')
# #plt.yscale('log')
# plot_title = 'Runtime fixed Problem Size vs Block Size'
# if len(block_sizes) == 1:
#     plot_title += f' (Runtimes for fixed problem size 20000)'
# plt.title(plot_title)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('runtime_vs_block_size_3.png')
# print("\nPlot 'runtime_vs_block_size_3.png' saved.")


# [Your existing code from the top down to the end of the main loop]
# ... (subprocess calls, parsing, appending to results, results2, raw_results, raw_results2) ...

# Make sure these lists are populated from the loop above:
# results (GPU summary)
# results2 (CPU summary)
# raw_results (GPU raw)
# raw_results2 (CPU raw)

# --- UPDATED BOTTOM PART STARTS HERE ---

# Convert GPU raw results to DataFrame
df_raw_gpu = pd.DataFrame(raw_results)
if not df_raw_gpu.empty:
    print("GPU Raw DataFrame sample:\n", df_raw_gpu.head())
else:
    print("GPU Raw DataFrame (df_raw_gpu) is empty.")

# Convert CPU raw results to DataFrame
df_raw_cpu = pd.DataFrame(raw_results2)
if not df_raw_cpu.empty:
    print("\nCPU Raw DataFrame sample:\n", df_raw_cpu.head())
else:
    print("CPU Raw DataFrame (df_raw_cpu) is empty.")

# GPU Analysis grouped by actual global block sizes used by GPU
if not df_raw_gpu.empty:
    df_block_analysis_gpu = df_raw_gpu.groupby(['global_block_x', 'global_block_y']).agg(
        mean_runtime=('runtime', 'mean'),
        max_runtime=('runtime', 'max')
    )
    df_block_analysis_gpu = df_block_analysis_gpu.sort_values(by='mean_runtime', ascending=True)
    print("\nGPU Block Size Analysis (sorted by mean_runtime):\n", df_block_analysis_gpu)
    # Save GPU block analysis results
    df_block_analysis_gpu.reset_index().to_csv('block_size_analysis_gpu3.csv', index=False)
    print("Saved 'block_size_analysis_gpu3.csv'")
else:
    print("\nSkipping GPU Block Size Analysis as df_raw_gpu is empty.")


# Save summarized GPU results
df_summary_gpu = pd.DataFrame(results)
if not df_summary_gpu.empty:
    df_summary_gpu.to_csv('results_gpu4.csv', index=False)
    print("Saved 'results_gpu3.csv'")
else:
    print("GPU Summary DataFrame (df_summary_gpu) is empty. Cannot save 'results_gpu3.csv'.")

# Save summarized CPU results
df_summary_cpu = pd.DataFrame(results2)
if not df_summary_cpu.empty:
    df_summary_cpu.to_csv('results_cpu4.csv', index=False)
    print("Saved 'results_cpu3.csv'")
else:
    print("CPU Summary DataFrame (df_summary_cpu) is empty. Cannot save 'results_cpu3.csv'.")

# Save raw GPU results
if not df_raw_gpu.empty:
    df_raw_gpu.to_csv('raw_results_gpu4.csv', index=False)
    print("Saved 'raw_results_gpu3.csv'")
else:
    print("Skipping save of 'raw_results_gpu3.csv' as df_raw_gpu is empty.")

# Save raw CPU results
if not df_raw_cpu.empty:
    df_raw_cpu.to_csv('raw_results_cpu4.csv', index=False)
    print("Saved 'raw_results_cpu3.csv'")
else:
    print("Skipping save of 'raw_results_cpu3.csv' as df_raw_cpu is empty.")


# --- Plotting ---
# Matplotlib should be imported at the top of the script.

# Plot 1: GPU Runtime vs. Block Size (for a fixed problem size)
# This plot is meaningful if `block_sizes` has multiple entries and `problem_sizes` is fixed or one is chosen.
if not df_raw_gpu.empty and problem_sizes:
    target_problem_size_for_block_plot = problem_sizes[-1] # Use largest problem size
    df_gpu_for_block_plot = df_raw_gpu[df_raw_gpu['res_x'] == target_problem_size_for_block_plot]

    # Check if there are multiple unique block sizes for the chosen problem size
    unique_block_configs_in_plot_data = []
    if not df_gpu_for_block_plot.empty:
        unique_block_configs_in_plot_data = df_gpu_for_block_plot[['global_block_x', 'global_block_y']].drop_duplicates().values

    if len(unique_block_configs_in_plot_data) > 1:
        df_plot_data_gpu_block = df_gpu_for_block_plot.groupby(["global_block_x", "global_block_y"]).agg(
            mean_runtime=('runtime', 'mean'),
            max_runtime=('runtime', 'max')
        ).sort_values(by='mean_runtime')

        df_plot_data_gpu_block.index = [f'({x}, {y})' for x, y in df_plot_data_gpu_block.index]

        plt.figure(figsize=(max(12, len(df_plot_data_gpu_block.index) * 0.5), 7)) # Adjusted height
        plt.plot(df_plot_data_gpu_block.index, df_plot_data_gpu_block['mean_runtime'], label='GPU Mean Runtime', marker='o')
        plt.plot(df_plot_data_gpu_block.index, df_plot_data_gpu_block['max_runtime'], label='GPU Max Runtime', marker='s', linestyle='--')
        plt.xlabel('GPU Block Size (Actual X, Y)')
        plt.xticks(rotation=45, ha="right")
        plt.ylabel('Runtime (microseconds)') # Assuming runtime_str was in microseconds
        plot_title = f'GPU Runtime vs. Actual Block Size (Problem Size: {target_problem_size_for_block_plot}x{target_problem_size_for_block_plot})'
        plt.title(plot_title)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('gpu_runtime_vs_block_size_detailed3.png')
        print(f"\nPlot 'gpu_runtime_vs_block_size_detailed3.png' saved.")
    elif not df_gpu_for_block_plot.empty:
        print(f"\nSkipping 'gpu_runtime_vs_block_size_detailed3.png' plot: Only one unique block size data found for problem size {target_problem_size_for_block_plot}.")
    else:
        print(f"\nSkipping 'gpu_runtime_vs_block_size_detailed3.png' plot: No GPU data found for problem size {target_problem_size_for_block_plot}.")
else:
    print("\nSkipping 'gpu_runtime_vs_block_size_detailed3.png' plot: df_raw_gpu is empty or problem_sizes not defined.")


# Plot 2: Runtime vs. Problem Size (comparing CPU and GPU)
if not df_summary_cpu.empty or not df_summary_gpu.empty:
    plt.figure(figsize=(10, 6))
    
    # Plot CPU results
    if not df_summary_cpu.empty:
        df_summary_cpu_sorted = df_summary_cpu.sort_values(by='res_x')
        plt.plot(df_summary_cpu_sorted['res_x'], df_summary_cpu_sorted['mean_runtime'], label='CPU Mean Runtime', marker='x', linestyle='--')
    else:
        print("No CPU summary data to plot for Runtime vs. Problem Size.")

    # Plot GPU results for each configured block size
    if not df_summary_gpu.empty:
        df_summary_gpu_sorted = df_summary_gpu.sort_values(by=['block_size_x_config', 'block_size_y_config', 'res_x'])
        unique_gpu_blocks_config = df_summary_gpu_sorted[['block_size_x_config', 'block_size_y_config']].drop_duplicates().values
        
        for b_x, b_y in unique_gpu_blocks_config:
            df_gpu_specific_block = df_summary_gpu_sorted[
                (df_summary_gpu_sorted['block_size_x_config'] == b_x) &
                (df_summary_gpu_sorted['block_size_y_config'] == b_y)
            ]
            if not df_gpu_specific_block.empty:
                plt.plot(df_gpu_specific_block['res_x'], df_gpu_specific_block['mean_runtime'],
                         label=f'GPU Mean Runtime (Config Block: {b_x}x{b_y})', marker='o')
    else:
        print("No GPU summary data to plot for Runtime vs. Problem Size.")

    plt.xlabel('Problem Size (resolution N for NxN image)')
    plt.ylabel('Mean Runtime (microseconds)') # Assuming runtime_str was in microseconds
    plt.title('Runtime vs. Problem Size: CPU vs. GPU')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.7)
    plt.xscale('log', base=2)
    plt.yscale('log')
    if problem_sizes:
        plt.xticks(ticks=problem_sizes, labels=[str(ps) for ps in problem_sizes])
        plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter()) # Ensure numbers are displayed as numbers, not scientific
        plt.minorticks_off() # Can be enabled if desired: plt.minorticks_on()
    
    plt.tight_layout()
    plt.savefig('runtime_vs_problem_size_comparison3.png')
    print("\nPlot 'runtime_vs_problem_size_comparison3.png' saved.")
else:
    print("\nSkipping 'runtime_vs_problem_size_comparison3.png': Both CPU and GPU summary data are empty.")


# Plot 3: Speedup (T_cpu / T_gpu) vs. Problem Size
if not df_summary_cpu.empty and not df_summary_gpu.empty:
    plt.figure(figsize=(10, 6))
    any_speedup_plotted = False
    
    # Ensure CPU data is sorted for merging
    df_summary_cpu_sorted_for_speedup = df_summary_cpu.sort_values(by='res_x')
    
    # Iterate through unique GPU block configurations from summary_gpu
    df_summary_gpu_sorted_for_speedup = df_summary_gpu.sort_values(by=['block_size_x_config', 'block_size_y_config', 'res_x'])
    unique_gpu_blocks_config_for_speedup = df_summary_gpu_sorted_for_speedup[['block_size_x_config', 'block_size_y_config']].drop_duplicates().values

    for b_x, b_y in unique_gpu_blocks_config_for_speedup:
        df_gpu_specific_block = df_summary_gpu_sorted_for_speedup[
            (df_summary_gpu_sorted_for_speedup['block_size_x_config'] == b_x) &
            (df_summary_gpu_sorted_for_speedup['block_size_y_config'] == b_y)
        ]

        if df_gpu_specific_block.empty:
            continue

        # Merge with CPU data for this specific GPU block configuration
        df_merged_for_speedup = pd.merge(
            df_summary_cpu_sorted_for_speedup[['res_x', 'mean_runtime']],
            df_gpu_specific_block[['res_x', 'mean_runtime']], # Already filtered and sorted
            on='res_x',
            suffixes=('_cpu', '_gpu')
        )

        if not df_merged_for_speedup.empty and 'mean_runtime_gpu' in df_merged_for_speedup and 'mean_runtime_cpu' in df_merged_for_speedup:
            # Avoid division by zero if GPU runtime is 0 (highly unlikely for actual measurements)
            df_merged_for_speedup = df_merged_for_speedup[df_merged_for_speedup['mean_runtime_gpu'] > 0]
            if not df_merged_for_speedup.empty:
                df_merged_for_speedup['speedup'] = df_merged_for_speedup['mean_runtime_cpu'] / df_merged_for_speedup['mean_runtime_gpu']
                df_merged_for_speedup = df_merged_for_speedup.sort_values(by='res_x') # Ensure sorted for plotting
                
                plt.plot(df_merged_for_speedup['res_x'], df_merged_for_speedup['speedup'],
                         label=f'Speedup (GPU Config Block: {b_x}x{b_y})', marker='s')
                any_speedup_plotted = True
    
    if any_speedup_plotted:
        plt.xlabel('Problem Size (resolution N for NxN image)')
        plt.ylabel('Speedup (T_cpu / T_gpu)')
        plt.title('Speedup vs. Problem Size')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.7)
        plt.axhline(y=1, color='grey', linestyle=':', linewidth=1) # Line for speedup = 1
        plt.xscale('log', base=2)
        if problem_sizes:
            plt.xticks(ticks=problem_sizes, labels=[str(ps) for ps in problem_sizes])
            plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
            plt.minorticks_off()
        # Y-axis for speedup is typically linear. Can be log if speedups vary over orders of magnitude.
        # plt.yscale('log')
        plt.tight_layout()
        plt.savefig('speedup_vs_problem_size_detailed3.png')
        print("\nPlot 'speedup_vs_problem_size_detailed3.png' saved.")
    else:
        print("\nNo speedup data was plotted. Check for matching problem sizes, non-zero GPU runtimes, and non-empty CPU/GPU results.")
else:
    print("\nSkipping speedup plot: CPU or GPU summary data is empty.")

print("\nAnalysis and plotting complete.")