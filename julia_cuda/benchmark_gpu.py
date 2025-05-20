# benchmark_gpu.py
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
# import random # Uncomment if using random sampling for block_sizes

# Parameters
problem_sizes = [1000, 2000, 4000, 8000, 16000]
reps = 5

# GPU Block Sizes
# Fixed as per the example in the problem description
block_sizes = [(128,1)] 

# Example of generating a larger list of block sizes (commented out)
# block_sizes_generated = [(i, j) for i in range(1, 1025, 1) for j in range(16, 1025, 1) if i * j % 32 == 0 and i * j <= 1024]
# if using the full list and sampling:
# block_sizes = random.sample(block_sizes_generated, 50) # Sample 50 block sizes

summary_results_gpu = []
raw_results_gpu = []

# GPU Timeout settings
gpu_timeout_seconds = 300 # 5 minutes per run
gpu_timeout_runtime_usec = gpu_timeout_seconds * 1000 * 1000

print("Starting GPU Benchmark...")
print(f"Using block_sizes: {block_sizes}")
print(f"Problem sizes: {problem_sizes}")
print("-" * 70)

for block_size_tuple in tqdm(block_sizes, desc="GPU Block Sizes"):
    configured_block_x, configured_block_y = block_size_tuple
    for size in tqdm(problem_sizes, desc=f"Problem Sizes (Block {configured_block_x}x{configured_block_y})", leave=False):
        print(f"Running GPU: problem {size}x{size}, block {configured_block_x}x{configured_block_y}")
        cmd = f"./juliaset_gpu -r {size} {size} -n {reps} -b {configured_block_x} {configured_block_y}"
        
        try:
            output = subprocess.check_output(cmd, shell=True, text=True, timeout=gpu_timeout_seconds)
        except subprocess.TimeoutExpired:
            print(f"GPU command timed out for size {size}, block {configured_block_x}x{configured_block_y}. Recording {gpu_timeout_runtime_usec/1e6:.2f}s.")
            output_lines = []
            for i in range(reps):
                # Format: rep;res_x;res_y;scale;global_block_x;global_block_y;runtime_str
                # Use configured blocks as global_block_x/y if actual are unknown due to timeout
                output_lines.append(f"{i+1};{size};{size};0.5;{configured_block_x};{configured_block_y};{gpu_timeout_runtime_usec}")
            output = "\n".join(output_lines)
        except subprocess.CalledProcessError as e:
            print(f"Error running GPU command for size {size}, block {configured_block_x}x{configured_block_y}: {e}. Recording {gpu_timeout_runtime_usec/1e6:.2f}s.")
            output_lines = []
            for i in range(reps):
                output_lines.append(f"{i+1};{size};{size};0.5;{configured_block_x};{configured_block_y};{gpu_timeout_runtime_usec}")
            output = "\n".join(output_lines)

        parsed_lines = [line.strip().split(';') for line in output.strip().split('\n') if line.strip()]
        runtimes_for_current_setting = []

        for fields in parsed_lines:
            if len(fields) != 7:
                print(f"Warning: Skipping malformed GPU line: {';'.join(fields)}")
                continue
            
            rep_str, res_x_str, res_y_str, scale_str, gbx_str, gby_str, runtime_str = fields
            try:
                runtime_val = float(runtime_str) # Assuming runtime_str is in microseconds
                raw_results_gpu.append({
                    'rep': int(rep_str),
                    'res_x': int(res_x_str),
                    'res_y': int(res_y_str),
                    'scale': float(scale_str),
                    'global_block_x': int(gbx_str), # Actual block dim used by executable
                    'global_block_y': int(gby_str), # Actual block dim used by executable
                    'runtime': runtime_val,
                    'block_size_x_config': configured_block_x, # Configured block dim
                    'block_size_y_config': configured_block_y  # Configured block dim
                })
                runtimes_for_current_setting.append(runtime_val)
            except ValueError as e:
                print(f"Warning: Could not parse GPU line: {';'.join(fields)}. Error: {e}")
                continue
        
        if not runtimes_for_current_setting:
            print(f"Warning: No valid GPU runtimes recorded for size {size}, block {configured_block_x}x{configured_block_y}. Using NaN.")
            mean_runtime = np.nan
            std_runtime = np.nan
            max_runtime = np.nan
        else:
            mean_runtime = np.mean(runtimes_for_current_setting)
            std_runtime = np.std(runtimes_for_current_setting)
            max_runtime = np.max(runtimes_for_current_setting)

        summary_results_gpu.append({
            'res_x': size,
            'res_y': size,
            'mean_runtime': mean_runtime,
            'std_runtime': std_runtime,
            'max_runtime': max_runtime,
            'block_size_x_config': configured_block_x,
            'block_size_y_config': configured_block_y
        })
        if not np.isnan(mean_runtime):
            print(f"GPU {size}x{size} (Block {configured_block_x}x{configured_block_y}): Mean Runtime {mean_runtime/1e6:.4f}s")
        else:
            print(f"GPU {size}x{size} (Block {configured_block_x}x{configured_block_y}): Mean Runtime Not Available")


# Save GPU results
df_summary_gpu = pd.DataFrame(summary_results_gpu)
if not df_summary_gpu.empty:
    df_summary_gpu.to_csv('results_gpu4.csv', index=False)
    print("\nSaved 'results_gpu4.csv'")
else:
    print("\nGPU Summary DataFrame is empty. Not saving 'results_gpu4.csv'.")

df_raw_gpu = pd.DataFrame(raw_results_gpu)
if not df_raw_gpu.empty:
    df_raw_gpu.to_csv('raw_results_gpu4.csv', index=False)
    print("Saved 'raw_results_gpu4.csv'")
else:
    print("GPU Raw DataFrame is empty. Not saving 'raw_results_gpu4.csv'.")

# GPU Block Size Analysis (from raw results)
if not df_raw_gpu.empty:
    if 'global_block_x' in df_raw_gpu.columns and 'global_block_y' in df_raw_gpu.columns:
        df_block_analysis_gpu = df_raw_gpu.groupby(['global_block_x', 'global_block_y']).agg(
            mean_runtime=('runtime', 'mean'),
            max_runtime=('runtime', 'max')
        ).sort_values(by='mean_runtime', ascending=True)
        
        print("\nGPU Block Size Analysis (sorted by mean_runtime):\n", df_block_analysis_gpu.head())
        df_block_analysis_gpu.reset_index().to_csv('block_size_analysis_gpu4.csv', index=False)
        print("Saved 'block_size_analysis_gpu4.csv'")
    else:
        print("\nSkipping GPU Block Size Analysis: 'global_block_x' or 'global_block_y' not found in raw GPU data.")
else:
    print("\nSkipping GPU Block Size Analysis as df_raw_gpu is empty.")

print("\nGPU benchmarking complete.")