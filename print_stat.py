import os
import numpy as np

stat_dir = "/content/VAE_experiment/stats"
trial_range = 5

def calc_stat(stat_dir, stat_name, trial_range):
    mcd_dict = {"f2f": [], "m2f": [], "f2m": [], "m2m": [], "total": []}
    msd_dict = {"f2f": [], "m2f": [], "f2m": [], "m2m": [], "total": []}

    for trial in range(trial_range):
        stat_path = f"{stat_dir}/{stat_name}_{trial}.txt"
        if not os.path.exists(stat_path):
            continue
        
        with open(stat_path, 'r') as f:
            for line in f:
                if line[0] == '-':
                    for i in range(5):
                        line = f.readline()
                        pair_type = line.split("  ")[0]
                        mean = line.split("  ")[2]
                        std = line.split("  ")[4]
                        mcd_dict[pair_type].append(float(mean))
                    break
            for line in f:
                if line[0] == '-':
                    for i in range(5):
                        line = f.readline()
                        pair_type = line.split("  ")[0]
                        mean = line.split("  ")[2]
                        std = line.split("  ")[4]
                        msd_dict[pair_type].append(float(mean))
                    break

    print(stat_name, "RESULT")
    print("---------------- MCD ----------------")
    for pair_type, all_stat in mcd_dict.items():
        mean = np.round(np.mean(all_stat), 3)
        std = np.round(np.std(all_stat), 3)
        print(pair_type,":", mean, std)
    print("---------------- MSD ----------------")
    for pair_type, all_stat in msd_dict.items():
        mean = np.round(np.mean(all_stat), 3)
        std = np.round(np.std(all_stat), 3)
        print(pair_type,":", mean, std)
    print("-------------------------------------")

# 원하는 파일에 대해서만 통계 계산
calc_stat(stat_dir, "VAE3", trial_range)
calc_stat(stat_dir, "VAE3_CC", trial_range)
'''
import os
import numpy as np
stat_dir = "stats"

vae_list = ["VAE1", "VAE2", "VAE3", "MD"]
exp_list = ["SI", "I", "LI", "AC", "SC", "CC", "D1", "D2", "HCP2", "HCP3", "HCP4", "DEC1", "DEC2", "DEC3", "DEC4"]

trial_range = 5

def calc_stat(stat_dir, stat_name, trial_range):
    mcd_dict = {"f2f": [], "m2f": [], "f2m": [], "m2m": [], "total": []}
    msd_dict = {"f2f": [], "m2f": [], "f2m": [], "m2m": [], "total": []}

    # for trial in range(trial_range):
    for trial in range(trial_range):
        stat_path = stat_dir+"/"+stat_name+"_"+str(trial)+".txt"
        with open(stat_path, 'r') as f:
            for line in f:
                if line[0] == '-':
                    for i in range(5):
                        line = f.readline()
                        pair_type = line.split("  ")[0]
                        mean = line.split("  ")[2]
                        std = line.split("  ")[4]
                        mcd_dict[pair_type].append(float(mean))
                        
                    break
            for line in f:
                if line[0] == '-':
                    for i in range(5):
                        line = f.readline()
                        pair_type = line.split("  ")[0]
                        mean = line.split("  ")[2]
                        std = line.split("  ")[4]
                        msd_dict[pair_type].append(float(mean))
                        
                    break

    print(stat_name, "RESULT")
    
    print("---------------- MCD ----------------")
    for pair_type, all_stat in mcd_dict.items():
        mean = np.round(np.mean(all_stat), 3)
        std = np.round(np.std(all_stat), 3)
        print(pair_type,":", mean, std)
    print("---------------- MSD ----------------")
    for pair_type, all_stat in msd_dict.items():
        mean = np.round(np.mean(all_stat), 3)
        std = np.round(np.std(all_stat), 3)
        print(pair_type,":", mean, std)
    print("-------------------------------------")

for vae_type in vae_list:
    calc_stat(stat_dir, vae_type, trial_range)
for exp_type in exp_list:
    calc_stat(stat_dir, "VAE3_"+exp_type, trial_range)
'''

        
