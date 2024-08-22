#!/bin/bash

# Function to kill all monitoring processes and optionally the C++ program
cleanup() {
    echo "Terminating monitoring processes..."
    for PID in "${MONITOR_PIDS[@]}"
    do
        kill $PID 2>/dev/null
    done
    echo "Monitoring processes terminated."
    # Uncomment the following line if you also want to kill the C++ program on Ctrl+C
    # kill $CPP_PID 2>/dev/null
    echo "Exiting script."
    exit 0
}

# List of GPU IDs to monitor
GPU_IDS=(0)  # Add more GPU IDs as needed, e.g. (0 1 2 3)

# Run the C++ executable in the background
python tutorial/train.py &
CPP_PID=$!

# Array to hold the PIDs of the nvidia-smi monitoring processes
declare -a MONITOR_PIDS

# Loop through each GPU ID and start a monitoring process in the background
for ID in "${GPU_IDS[@]}"
do
    nvidia-smi --id=$ID --query-gpu=memory.used --format=csv -l 1 > "./gpu_usage_${ID}.csv" &
    MONITOR_PIDS+=($!)
done

# Start monitoring CPU memory usage every 1 second and output to a CSV file
(
    echo "timestamp, total_GB, used_GB, free_GB, available_GB" > "./cpu_memory_usage.csv"
    while true; do
        echo "$(date +%s), $(free | grep Mem | awk '{printf "%.2f, %.2f, %.2f, %.2f", $2/1024/1024, $3/1024/1024, $4/1024/1024, $7/1024/1024}')" >> "./cpu_memory_usage.csv"
        sleep 1
    done
) &
MONITOR_PIDS+=($!)

# Trap SIGINT and SIGTERM to clean up properly
trap cleanup SIGINT SIGTERM

# Wait for the C++ program to finish
wait $CPP_PID

# Execute cleanup after the C++ program finishes naturally
cleanup
