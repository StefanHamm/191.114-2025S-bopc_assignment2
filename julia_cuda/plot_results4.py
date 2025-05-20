# plot_results.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Parameters that might be needed for plotting (e.g., for x-axis ticks)
# Must match problem_sizes used in benchmark scripts for consistent plotting.
problem_sizes_const = [1000, 2000, 4000, 8000, 16000] 

# --- Load Data ---
print("Loading data for plotting...")
try:
    df_summary_cpu = pd.read_csv('results_cpu4.csv')
    print("Loaded 'results_cpu4.csv'")
except FileNotFoundError:
    print("Error: 'results_cpu4.csv' not found. CPU data will be missing from plots.")
    df_summary_cpu = pd.DataFrame() 

try:
    df_summary_gpu = pd.read_csv('results_gpu4.csv')
    print("Loaded 'results_gpu4.csv'")
except FileNotFoundError:
    print("Error: 'results_gpu4.csv' not found. GPU data will be missing from plots.")
    df_summary_gpu = pd.DataFrame()

try:
    df_raw_gpu = pd.read_csv('raw_results_gpu4.csv')
    print("Loaded 'raw_results_gpu4.csv'")
except FileNotFoundError:
    print("Error: 'raw_results_gpu4.csv' not found. GPU block plot will be skipped.")
    df_raw_gpu = pd.DataFrame()

# --- Plotting ---

# Plot 1: GPU Runtime vs. Block Size (for a fixed problem size)
# This plot is most useful if benchmark_gpu.py was run with multiple block_sizes.
if not df_raw_gpu.empty and problem_sizes_const:
    target_problem_size_for_block_plot = problem_sizes_const[-1] # Use largest problem size as example
    
    df_gpu_for_block_plot = df_raw_gpu[df_raw_gpu['res_x'] == target_problem_size_for_block_plot]

    if not df_gpu_for_block_plot.empty and \
       'global_block_x' in df_gpu_for_block_plot.columns and \
       'global_block_y' in df_gpu_for_block_plot.columns:
        
        unique_block_configs_in_plot_data = df_gpu_for_block_plot[['global_block_x', 'global_block_y']].drop_duplicates().values

        if len(unique_block_configs_in_plot_data) > 1: # Only plot if there's something to compare
            df_plot_data_gpu_block = df_gpu_for_block_plot.groupby(["global_block_x", "global_block_y"]).agg(
                mean_runtime=('runtime', 'mean'),
                max_runtime=('runtime', 'max')
            ).sort_values(by='mean_runtime')

            df_plot_data_gpu_block.index = [f'({int(x)}, {int(y)})' for x, y in df_plot_data_gpu_block.index]

            plt.figure(figsize=(max(12, len(df_plot_data_gpu_block.index) * 0.6), 7))
            plt.plot(df_plot_data_gpu_block.index, df_plot_data_gpu_block['mean_runtime'], label='GPU Mean Runtime', marker='o')
            plt.plot(df_plot_data_gpu_block.index, df_plot_data_gpu_block['max_runtime'], label='GPU Max Runtime', marker='s', linestyle='--')
            plt.xlabel('GPU Block Size (Actual X, Y)')
            plt.xticks(rotation=45, ha="right")
            plt.ylabel('Runtime (microseconds)')
            plot_title = f'GPU Runtime vs. Actual Block Size (Problem Size: {target_problem_size_for_block_plot}x{target_problem_size_for_block_plot})'
            plt.title(plot_title)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig('gpu_runtime_vs_block_size_detailed4.png')
            print(f"\nPlot 'gpu_runtime_vs_block_size_detailed4.png' saved.")
        elif not df_gpu_for_block_plot.empty:
            print(f"\nSkipping 'gpu_runtime_vs_block_size_detailed4.png' plot: Only one or no unique actual block size data found for problem size {target_problem_size_for_block_plot} in raw GPU data. This plot requires multiple block sizes to be benchmarked and recorded.")
        else: # df_gpu_for_block_plot is empty
             print(f"\nSkipping 'gpu_runtime_vs_block_size_detailed4.png' plot: No GPU data found for problem size {target_problem_size_for_block_plot}.")
    elif not df_raw_gpu.empty : # df_raw_gpu not empty but columns missing
         print(f"\nSkipping 'gpu_runtime_vs_block_size_detailed4.png' plot: 'global_block_x' or 'global_block_y' columns missing in raw GPU data.")
    # else df_raw_gpu is empty or problem_sizes_const not defined
    

# Plot 2: Runtime vs. Problem Size (comparing CPU and GPU)
if not df_summary_cpu.empty or not df_summary_gpu.empty:
    plt.figure(figsize=(10, 6))
    plotted_anything_rt_vs_ps = False
    
    if not df_summary_cpu.empty and 'res_x' in df_summary_cpu.columns and 'mean_runtime' in df_summary_cpu.columns:
        df_summary_cpu_plot = df_summary_cpu.dropna(subset=['res_x', 'mean_runtime'])
        if not df_summary_cpu_plot.empty:
            df_summary_cpu_sorted = df_summary_cpu_plot.sort_values(by='res_x')
            plt.plot(df_summary_cpu_sorted['res_x'], df_summary_cpu_sorted['mean_runtime'], label='CPU Mean Runtime', marker='x', linestyle='--')
            plotted_anything_rt_vs_ps = True
    
    if not df_summary_gpu.empty and all(col in df_summary_gpu.columns for col in ['block_size_x_config', 'block_size_y_config', 'res_x', 'mean_runtime']):
        df_summary_gpu_plot = df_summary_gpu.dropna(subset=['res_x', 'mean_runtime'])
        if not df_summary_gpu_plot.empty:
            df_summary_gpu_sorted = df_summary_gpu_plot.sort_values(by=['block_size_x_config', 'block_size_y_config', 'res_x'])
            unique_gpu_blocks_config = df_summary_gpu_sorted[['block_size_x_config', 'block_size_y_config']].drop_duplicates().values
            
            for b_x, b_y in unique_gpu_blocks_config:
                df_gpu_specific_block = df_summary_gpu_sorted[
                    (df_summary_gpu_sorted['block_size_x_config'] == b_x) &
                    (df_summary_gpu_sorted['block_size_y_config'] == b_y)
                ]
                if not df_gpu_specific_block.empty:
                    plt.plot(df_gpu_specific_block['res_x'], df_gpu_specific_block['mean_runtime'],
                             label=f'GPU Mean Runtime (Config Block: {int(b_x)}x{int(b_y)})', marker='o')
                    plotted_anything_rt_vs_ps = True

    if plotted_anything_rt_vs_ps:
        plt.xlabel('Problem Size (resolution N for NxN image)')
        plt.ylabel('Mean Runtime (microseconds)')
        plt.title('Runtime vs. Problem Size: CPU vs. GPU')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.7)
        plt.xscale('log', base=2)
        plt.yscale('log')
        
        all_res_x_values = set()
        if not df_summary_cpu.empty and 'res_x' in df_summary_cpu: all_res_x_values.update(df_summary_cpu['res_x'].dropna().unique())
        if not df_summary_gpu.empty and 'res_x' in df_summary_gpu: all_res_x_values.update(df_summary_gpu['res_x'].dropna().unique())
        
        valid_ticks = sorted([ps for ps in problem_sizes_const if ps in all_res_x_values])
        
        if valid_ticks:
            plt.xticks(ticks=valid_ticks, labels=[str(ps) for ps in valid_ticks])
        else: # Fallback
            plt.xticks(ticks=problem_sizes_const, labels=[str(ps) for ps in problem_sizes_const])

        plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter()) # Ensure numbers are displayed as numbers
        plt.minorticks_off()
        
        plt.tight_layout()
        plt.savefig('runtime_vs_problem_size_comparison4.png')
        print("\nPlot 'runtime_vs_problem_size_comparison4.png' saved.")
    else:
        print("\nSkipping 'runtime_vs_problem_size_comparison4.png': No valid data available to plot for either CPU or GPU after checks.")


# Plot 3: Speedup (T_cpu / T_gpu) vs. Problem Size
if not df_summary_cpu.empty and not df_summary_gpu.empty and \
   all(col in df_summary_cpu.columns for col in ['res_x', 'mean_runtime']) and \
   all(col in df_summary_gpu.columns for col in ['block_size_x_config', 'block_size_y_config', 'res_x', 'mean_runtime']):
    
    plt.figure(figsize=(10, 6))
    any_speedup_plotted = False
    
    df_summary_cpu_for_speedup = df_summary_cpu.dropna(subset=['res_x', 'mean_runtime']).sort_values(by='res_x')
    df_summary_gpu_for_speedup = df_summary_gpu.dropna(subset=['res_x', 'mean_runtime']).sort_values(by=['block_size_x_config', 'block_size_y_config', 'res_x'])

    if not df_summary_cpu_for_speedup.empty and not df_summary_gpu_for_speedup.empty:
        unique_gpu_blocks_config_for_speedup = df_summary_gpu_for_speedup[['block_size_x_config', 'block_size_y_config']].drop_duplicates().values

        for b_x, b_y in unique_gpu_blocks_config_for_speedup:
            df_gpu_specific_block = df_summary_gpu_for_speedup[
                (df_summary_gpu_for_speedup['block_size_x_config'] == b_x) &
                (df_summary_gpu_for_speedup['block_size_y_config'] == b_y)
            ]

            if df_gpu_specific_block.empty:
                continue

            df_merged_for_speedup = pd.merge(
                df_summary_cpu_for_speedup[['res_x', 'mean_runtime']],
                df_gpu_specific_block[['res_x', 'mean_runtime']],
                on='res_x',
                suffixes=('_cpu', '_gpu')
            )

            if not df_merged_for_speedup.empty:
                # Ensure mean_runtime_gpu is not zero or NaN
                df_merged_for_speedup = df_merged_for_speedup[
                    (df_merged_for_speedup['mean_runtime_gpu'] > 0) & 
                    (df_merged_for_speedup['mean_runtime_cpu'].notna())
                ] 
                if not df_merged_for_speedup.empty:
                    df_merged_for_speedup['speedup'] = df_merged_for_speedup['mean_runtime_cpu'] / df_merged_for_speedup['mean_runtime_gpu']
                    df_merged_for_speedup = df_merged_for_speedup.sort_values(by='res_x')
                    
                    plt.plot(df_merged_for_speedup['res_x'], df_merged_for_speedup['speedup'],
                             label=f'Speedup (GPU Config Block: {int(b_x)}x{int(b_y)})', marker='s')
                    any_speedup_plotted = True
        
        if any_speedup_plotted:
            plt.xlabel('Problem Size (resolution N for NxN image)')
            plt.ylabel('Speedup (T_cpu / T_gpu)')
            plt.title('Speedup vs. Problem Size')
            plt.legend()
            plt.grid(True, which="both", ls="-", alpha=0.7)
            plt.axhline(y=1, color='grey', linestyle=':', linewidth=1) # Line for speedup = 1
            plt.xscale('log', base=2)
            
            all_res_x_values_speedup = set()
            # Re-fetch merged data for accurate ticks
            if 'df_merged_for_speedup' in locals() and not df_merged_for_speedup.empty and 'res_x' in df_merged_for_speedup:
                 all_res_x_values_speedup.update(df_merged_for_speedup['res_x'].dropna().unique())

            valid_ticks_speedup = sorted([ps for ps in problem_sizes_const if ps in all_res_x_values_speedup])

            if valid_ticks_speedup:
                 plt.xticks(ticks=valid_ticks_speedup, labels=[str(ps) for ps in valid_ticks_speedup])
            else: # Fallback
                 plt.xticks(ticks=problem_sizes_const, labels=[str(ps) for ps in problem_sizes_const])

            plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
            plt.minorticks_off()
            
            plt.tight_layout()
            plt.savefig('speedup_vs_problem_size_detailed4.png')
            print("\nPlot 'speedup_vs_problem_size_detailed4.png' saved.")
        else:
            print("\nNo speedup data was plotted. Check for matching problem sizes between CPU and GPU runs, non-zero/NaN GPU runtimes, and non-NaN CPU runtimes.")
    else:
        print("\nSkipping speedup plot: Cleaned CPU or GPU summary data is empty.")
else:
    print("\nSkipping speedup plot: CPU or GPU summary data (or required columns) is missing, or failed to load.")

print("\nAnalysis and plotting complete.")