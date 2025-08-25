#!/bin/bash
set -e

# Docker entrypoint script for Diabetic Retinopathy API

echo "🏥 Starting Diabetic Retinopathy Detection API"
echo "============================================="

# Check if model file exists
MODEL_PATH=${MODEL_PATH:-"/app/models/best_model.pth"}
if [ -f "$MODEL_PATH" ]; then
    echo "✅ Model found at: $MODEL_PATH"
else
    echo "⚠️  Model not found at: $MODEL_PATH"
    echo "   The API will start but predictions will fail until a model is available."
fi

# Check CUDA availability if requested
if [ "$USE_CUDA" = "true" ] || [ "$USE_CUDA" = "auto" ]; then
    python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
        echo "🚀 GPU acceleration available"
    else
        echo "💻 Using CPU for inference"
    fi
fi

# Create necessary directories
mkdir -p /app/logs
mkdir -p /app/temp

# Set up logging
export PYTHONUNBUFFERED=1

# Health check function
health_check() {
    local max_attempts=30
    local attempt=1
    
    echo "🔍 Waiting for API to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            echo "✅ API is ready!"
            return 0
        fi
        
        echo "⏳ Attempt $attempt/$max_attempts - API not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "❌ API failed to become ready within timeout"
    return 1
}

# Function to run API server
run_server() {
    echo "🚀 Starting API server..."
    
    # Start server in background for health check
    if [ "$1" = "--with-health-check" ]; then
        python -m uvicorn backend.app.main:app \
            --host ${API_HOST:-0.0.0.0} \
            --port ${API_PORT:-8000} \
            --workers ${API_WORKERS:-1} \
            --log-level ${LOG_LEVEL:-info} &
        
        SERVER_PID=$!
        
        # Wait for server to be ready
        if health_check; then
            # Bring server to foreground
            wait $SERVER_PID
        else
            kill $SERVER_PID 2>/dev/null || true
            exit 1
        fi
    else
        # Run server normally
        exec python -m uvicorn backend.app.main:app \
            --host ${API_HOST:-0.0.0.0} \
            --port ${API_PORT:-8000} \
            --workers ${API_WORKERS:-1} \
            --log-level ${LOG_LEVEL:-info}
    fi
}

# Function to run training
run_training() {
    echo "🎯 Starting model training..."
    
    # Check if training data exists
    if [ ! -f "/app/data/train.csv" ]; then
        echo "❌ Training data not found at /app/data/train.csv"
        exit 1
    fi
    
    python -m ml.train \
        --labels-csv /app/data/train.csv \
        --img-dir /app/data/train_images \
        --epochs ${EPOCHS:-20} \
        --batch-size ${BATCH_SIZE:-16} \
        --lr ${LEARNING_RATE:-0.001}
}

# Function to run evaluation
run_evaluation() {
    echo "📊 Starting model evaluation..."
    
    # Check if test data exists
    if [ ! -f "/app/data/test.csv" ]; then
        echo "❌ Test data not found at /app/data/test.csv"
        exit 1
    fi
    
    python -m ml.evaluate_model \
        --model-path ${MODEL_PATH:-/app/models/best_model.pth} \
        --test-csv /app/data/test.csv \
        --img-dir /app/data/test_images \
        --output-dir /app/evaluation_results
}

# Function to run tests
run_tests() {
    echo "🧪 Running tests..."
    python -m pytest tests/ -v --tb=short
}

# Main command handling
case "$1" in
    "server")
        run_server
        ;;
    "server-with-health-check")
        run_server --with-health-check
        ;;
    "train")
        run_training
        ;;
    "evaluate")
        run_evaluation
        ;;
    "test")
        run_tests
        ;;
    "bash")
        exec /bin/bash
        ;;
    *)
        echo "Usage: $0 {server|server-with-health-check|train|evaluate|test|bash}"
        echo ""
        echo "Commands:"
        echo "  server                    - Start the API server"
        echo "  server-with-health-check  - Start server with health check"
        echo "  train                     - Train the model"
        echo "  evaluate                  - Evaluate the model"
        echo "  test                      - Run tests"
        echo "  bash                      - Open bash shell"
        echo ""
        echo "If no command specified, starting server..."
        run_server
        ;;
esac