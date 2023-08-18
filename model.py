import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
import glob
import sys
import os
from tqdm import tqdm
from itertools import product
from matplotlib.pyplot import cm
import gc

# Add your output_dir here!! 
output_dir = '/Users/thomasmoore/Desktop/' + 'Kilonova_model/' 

def is_decreasing(combination):
    for i in range(1, len(combination)):
        if combination[i] > combination[i - 1]:
            return False
    return True

array1 = np.arange(6300, 6300 + 4 * 250, 250)
array2 = np.arange(4700, 4700 + 4 * 250, 250)
array3 = np.arange(4200, 4200 + 4 * 250, 250)
array4 = np.arange(3500, 3500 + 4 * 250, 250)

all_combinations = product(*[array1, array2, array3, array4])

decreasing_combinations = [comb for comb in all_combinations if is_decreasing(comb)]
decreasing_combinations = []
for comb in product(*[array1, array2, array3, array4]):
    if is_decreasing(comb):
        decreasing_combinations.append(comb)

times = np.arange(0.5,8,0.5)

for comb in tqdm(decreasing_combinations, desc=" outer", position=0):
    target_dir = output_dir +'/temp_'+str(comb[0])+'_'+str(comb[1])+'_'+str(comb[2])+'_'+str(comb[3])

    if os.path.isdir(target_dir) == False:
        print("No output directory - making one!")
        os.makedirs(target_dir)

    if os.path.isdir(output_dir) == True:
        for filename in os.listdir(target_dir):
            if filename.endswith(".csv"):
                file_path = os.path.join(target_dir, filename)
                os.remove(file_path)
            if filename.endswith(".png"):
                file_path = os.path.join(target_dir, filename)
                os.remove(file_path)
                
            if os.path.isdir(output_dir) == False:
                print("No output directory - making one!")
                os.makedirs(target_dir)
    for time in tqdm(times, desc=" inner loop", position=1, leave=False):
        os.system(f'python Simulation.py 10000 {comb[0]} {comb[1]} {comb[2]} {comb[3]} {time} {target_dir}')    
    os.system(f'python JitterAndHistograms.py {target_dir}')
    os.system(f'python output_assemble.py {target_dir}')