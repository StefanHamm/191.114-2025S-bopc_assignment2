# Script for scalability analysis of Julia set computation

import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import random

# Parameters for the scalability analysis
problem_sizes = [1000, 2000, 4000, 8000, 16000]
#block_sizes = [(i, j) for i in range(1, 1025, 1) for j in range(16, 1025, 1) if i * j % 32 == 0 and i * j <= 1024]
block_sizes = [(6,16)]
#sample from block_sizes into block_sizes 200 samples without replacement
#block_sizes = random.sample(block_sizes, 50)
print(block_sizes)
results = []
raw_results = []
reps = 5
    
print("Problem size | Processors | Mean runtime (s) | Speedup | Efficiency")
print(len(block_sizes))
print("-" * 70)
for block_size in tqdm(block_sizes):
    for size in problem_sizes:
        cmd = f"./juliaset_gpu -r {size} {size} -n {reps} -b {block_size[0]} {block_size[1]}"
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
#print sorted values of mean_runtime
df_grouped = df_grouped.sort_values(by='mean_runtime', ascending=True)
print(df_grouped)

#save to csv 
df_grouped.to_csv('block_size_analysis2.csv', index=False)
# Save the results to a CSV file
df = pd.DataFrame(results)
df.to_csv('results2.csv', index=False)
# save raw results to csv
df_raw.to_csv('raw_results2.csv', index=False)
# # print sorted values of mean_runtime

# Here, use your default thread block size. Repeat the kernel launch fives times (–nrep 5).
# • Plot the average and maximum running time (y-axis) for each input size (x-axis).
# • Discuss your findings

# now do the plotting:

# Plotting the results
import matplotlib.pyplot as plt

# x = input sizes
x = problem_sizes
# y = mean and max runtimes
# do this by grouping by res_x and res_y
df_grouped = df_raw.groupby(['res_x', 'res_y']).agg(mean_runtime=('runtime', 'mean'),
                                                                      max_runtime=('runtime', 'max'))

# Plot mean runtime
plt.figure(figsize=(10, 5))
plt.plot(df_grouped.index, df_grouped['mean_runtime'], label='Mean Runtime', marker='o')
plt.plot(df_grouped.index, df_grouped['max_runtime'], label='Max Runtime', marker='o')
plt.xlabel('Input Size')
plt.ylabel('Runtime (s)')
plt.title('Runtime vs Input Size')
plt.legend()
plt.grid()
plt.savefig('runtime_vs_input_size.png')



