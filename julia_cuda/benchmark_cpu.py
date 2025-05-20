# benchmark_cpu.py
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm

# Parameters
problem_sizes = [1000, 2000, 4000, 8000, 16000]
reps = 5

# Define timeout settings for CPU benchmarks
# If a size takes longer than its timeout, a fixed runtime is recorded.
# Original script used timeout=15s for check_output, and hardcoded 150,000,000 us (150s) as runtime on timeout.
# This dictionary maps problem size to subprocess timeout (seconds).
timeout_seconds_map = {
    1000: 180,  # 3 minutes
    2000: 300,  # 5 minutes
    4000: 600,  # 10 minutes
    8000: 120,   # 15 seconds (as per original script's quick timeout for larger sizes)
    16000: 300   # 15 seconds (as per original script's quick timeout for larger sizes)
}
# Runtime to record (in microseconds) if a timeout occurs.
# Original script used 150,000,000 us.
timeout_runtime_usec = 150 * 1000 * 1000 

summary_results_cpu = []
raw_results_cpu = []

print("Starting CPU Benchmark...")
print("Problem size | Mean runtime (us)")
print("-" * 40)

for size in tqdm(problem_sizes, desc="CPU Benchmarking"):
    print(f"Running CPU version for problem size {size}x{size}")
    
    current_timeout_seconds = timeout_seconds_map.get(size, 180) # Default to 180s if size not in map

    try:
        cmd = f"./juliaset_cpu -r {size} {size} -n {reps}"
        output = subprocess.check_output(cmd, shell=True, text=True, timeout=current_timeout_seconds)
    except subprocess.TimeoutExpired:
        print(f"CPU command timed out for size {size} (timeout: {current_timeout_seconds}s). Recording {timeout_runtime_usec/1e6:.2f}s as runtime.")
        output_lines = []
        for i in range(reps): # Create 'reps' lines of timeout data
            # Format: rep;res_x;res_y;scale;global_block_x;global_block_y;runtime_str
            output_lines.append(f"{i+1};{size};{size};0.5;-1;-1;{timeout_runtime_usec}")
        output = "\n".join(output_lines)
    except subprocess.CalledProcessError as e:
        print(f"Error running CPU command for size {size}: {e}. Recording {timeout_runtime_usec/1e6:.2f}s as runtime.")
        output_lines = []
        for i in range(reps):
            output_lines.append(f"{i+1};{size};{size};0.5;-1;-1;{timeout_runtime_usec}")
        output = "\n".join(output_lines)

    parsed_lines = [line.strip().split(';') for line in output.strip().split('\n') if line.strip()]
    runtimes_for_current_setting = []

    for fields in parsed_lines:
        if len(fields) != 7:
            print(f"Warning: Skipping malformed CPU line: {';'.join(fields)}")
            continue
        
        rep_str, res_x_str, res_y_str, scale_str, gbx_str, gby_str, runtime_str = fields
        try:
            runtime_val = float(runtime_str) # Assuming runtime_str is in microseconds
            raw_results_cpu.append({
                'rep': int(rep_str),
                'res_x': int(res_x_str),
                'res_y': int(res_y_str),
                'scale': float(scale_str),
                'global_block_x': int(gbx_str), # Typically -1 or 1 for CPU
                'global_block_y': int(gby_str), # Typically -1 or 1 for CPU
                'runtime': runtime_val
            })
            runtimes_for_current_setting.append(runtime_val)
        except ValueError as e:
            print(f"Warning: Could not parse CPU line: {';'.join(fields)}. Error: {e}")
            continue
    
    if not runtimes_for_current_setting:
        print(f"Warning: No valid CPU runtimes recorded for size {size}. Using NaN.")
        mean_runtime = np.nan 
        std_runtime = np.nan
        max_runtime = np.nan
    else:
        mean_runtime = np.mean(runtimes_for_current_setting)
        std_runtime = np.std(runtimes_for_current_setting)
        max_runtime = np.max(runtimes_for_current_setting)

    summary_results_cpu.append({
        'res_x': size,
        'res_y': size,
        'mean_runtime': mean_runtime,
        'std_runtime': std_runtime,
        'max_runtime': max_runtime,
        'block_size_x_config': -1, # Placeholder for CPU
        'block_size_y_config': -1  # Placeholder for CPU
    })
    if not np.isnan(mean_runtime):
        print(f"CPU {size}x{size}: Mean Runtime {mean_runtime/1e6:.4f}s")
    else:
        print(f"CPU {size}x{size}: Mean Runtime Not Available")


# Save CPU results
df_summary_cpu = pd.DataFrame(summary_results_cpu)
if not df_summary_cpu.empty:
    df_summary_cpu.to_csv('results_cpu4.csv', index=False)
    print("\nSaved 'results_cpu4.csv'")
else:
    print("\nCPU Summary DataFrame is empty. Not saving 'results_cpu4.csv'.")

df_raw_cpu = pd.DataFrame(raw_results_cpu)
if not df_raw_cpu.empty:
    df_raw_cpu.to_csv('raw_results_cpu4.csv', index=False)
    print("Saved 'raw_results_cpu4.csv'")
else:
    print("CPU Raw DataFrame is empty. Not saving 'raw_results_cpu4.csv'.")

print("\nCPU benchmarking complete.")