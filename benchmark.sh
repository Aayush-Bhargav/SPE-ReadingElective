#!/bin/bash

# Configuration
IMAGE_CAT="./larry.jpeg"
INTERNAL_PORT=8000 

benchmark_lane() {
    local NAME=$1
    local IMAGE=$2
    local HOST_PORT=$3
    
    echo "=========================================="
    echo "Benchmarking Lane: $NAME"
    echo "=========================================="
    
    # 1. Start the container
    start_time=$(date +%s%N)
    docker run -d -p $HOST_PORT:$INTERNAL_PORT --name "bench_$NAME" $IMAGE > /dev/null
    
    # 2. Polling with RAW BINARY (--data-binary)
    # This sends the image bytes directly without form-data wrappers
    COUNT=0
    until curl -s -f --data-binary "@$IMAGE_CAT" "http://localhost:$HOST_PORT/predict" > /dev/null || [ $COUNT -eq 50 ]; do
        sleep 0.2
        ((COUNT++))
    done
    
    if [ $COUNT -eq 50 ]; then
        echo ">> ERROR: Lane $NAME failed to start in time or rejected the binary format."
        docker logs "bench_$NAME" | tail -n 5
        docker stop "bench_$NAME" > /dev/null && docker rm "bench_$NAME" > /dev/null
        return
    fi
    
    end_time=$(date +%s%N)
    cold_start=$(( (end_time - start_time) / 1000000 ))
    echo ">> Cold Start Latency: ${cold_start}ms"

    # 3. Steady State - Using 'hey' (Now that you use binary, 'hey' works perfectly!)
    # We use -D to send the file body and -T to set the content type.
    echo ">> Running Steady State Load Test (200 requests)..."
    hey -n 200 -c 10 -m POST -D "$IMAGE_CAT" -T "application/octet-stream" "http://localhost:$HOST_PORT/predict"
    
    # Cleanup
    docker stop "bench_$NAME" > /dev/null && docker rm "bench_$NAME" > /dev/null
    echo ">> Done."
    echo ""
}

# Run the benchmark
benchmark_lane "python" "secure-python:latest" 8500
benchmark_lane "rust"   "secure-rust:latest"   8501
benchmark_lane "wasm"   "secure-wasm:latest"   8502