#!/bin/bash

log_dir="/content/VAE_experiment/model"
out_dir="/content/VAE_experiment/graph"
mkdir -p $out_dir

# Base model log to graph
if [ -f "$log_dir/VAE3/log.txt" ]; then
    python3 /content/VAE_experiment/print_graph.py --log_path=$log_dir/VAE3/log.txt --out_path=$out_dir/VAE3.png
else
    echo "Log file for VAE3 not found"
fi

# Further experiments log to graph
for exp in SI I LI AC SC CC; do
    if [ -f "$log_dir/VAE3_$exp/log.txt" ]; then
        python3 /content/VAE_experiment/print_graph.py --log_path=$log_dir/VAE3_$exp/log.txt --out_path=$out_dir/VAE3_$exp.png
    else
        echo "Log file for VAE3_$exp not found"
    fi
done
