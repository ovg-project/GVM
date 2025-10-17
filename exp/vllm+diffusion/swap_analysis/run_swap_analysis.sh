#!/bin/bash

pushd ..
./start_vllm_server.sh none true 0.9 2>&1 | tee swap_analysis/vllm.log &

sleep 60

nsys profile --force-overwrite true --trace=cuda,nvtx --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true -o swap_analysis/diffusion_profile bash start_diffusion.sh gvm 2 2>&1 | tee swap_analysis/diffusion.log

pkill -f vllm

popd

# After the run, the profiling data will be saved in diffusion_profile.nsys-rep
rm -f report.sqlite
nsys export -t sqlite -o report.sqlite diffusion_profile.nsys-rep

python plot_swap_breakdown.py report.sqlite --out swap_breakdown.png --dump-json