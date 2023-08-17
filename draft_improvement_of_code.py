import os
import numpy as np
from tqdm import tqdm

def create_directory(directory):
    os.makedirs(directory, exist_ok=True)

def remove_files_with_extensions(directory, extensions):
    for filename in os.listdir(directory):
        if any(filename.endswith(ext) for ext in extensions):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)

def run_simulation(comb, time, target_dir):
    simulation_command = (
        f'python Simulation.py 10000 {comb[0]} {comb[1]} {comb[2]} {comb[3]} {time} {target_dir}'
    )
    os.system(simulation_command)

def main():
    output_dir = "your_output_directory_path"
    decreasing_combinations = [...]  # Define your combinations
    times = np.arange(0.5, 8, 0.05)

    for comb in tqdm(decreasing_combinations, desc="outer", position=0):
        target_dir = os.path.join(
            output_dir, f'temp_{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}'
        )
        
        create_directory(target_dir)

        extensions_to_remove = (".csv", ".png")
        remove_files_with_extensions(target_dir, extensions_to_remove)

        for time in tqdm(times, desc="inner loop", position=1, leave=False):
            run_simulation(comb, time, target_dir)

        os.system(f'python JitterAndHistograms.py {target_dir}')
        os.system(f'python output_assemble.py {target_dir}')

if __name__ == "__main__":
    main()
