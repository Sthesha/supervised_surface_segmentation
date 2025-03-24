#!/bin/bash

# Kill any existing Jupyter process
pkill -f jupyter-lab
sleep 2

# Start Jupyter in background with nohup
echo "Starting new Jupyter instance..."
nohup jupyter lab --ip 0.0.0.0 --no-browser --allow-root --port 8888  > /workspace/jupyter.log 2>&1 &

# Wait a moment for startup
sleep 3

# Show the log
tail /workspace/jupyter.log
