#!/bin/bash
# Docker management scripts for Diabetic Retinopathy Project

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
}

# Build functions
build_production() {
    print_info "Building production image..."
    docker build -t diabetic-retinopathy:latest --target production .
    print_success "Production image built successfully"
}

build_development() {
    print_info "Building development image..."
    docker build -t diabetic-retinopathy:dev --target development .
    print_success "Development image built successfully"
}

# Run functions
run_production() {
    print_info "Starting production API server..."
    docker-compose up -d api
    print_success "Production server started"
    print_info "API available at: http://localhost:8000"
    print_info "Health check: http://localhost:8000/health"
}

run_development() {
    print_info "Starting development environment..."
    docker-compose up api
}

run_full_stack() {
    print_info "Starting full stack (API + Frontend)..."
    docker-compose --profile frontend up -d
    print_success "Full stack started"
    print_info "API: http://localhost:8000"
    print_info "Frontend: http://localhost:3000"
}

# Training functions
run_training() {
    print_info "Starting model training..."
    if [ ! -d "./data" ]; then
        print_error "Data directory not found. Please ensure your training data is in ./data/"
        exit 1
    fi
    
    docker-compose --profile training up trainer
}

run_evaluation() {
    print_info "Starting model evaluation..."
    if [ ! -f "./ml/models/best_model.pth" ]; then
        print_error "Model file not found. Please train a model first."
        exit 1
    fi
    
    docker-compose --profile evaluation up evaluator
}

# Monitoring functions
start_monitoring() {
    print_info "Starting monitoring stack..."
    docker-compose --profile monitoring up -d prometheus grafana
    print_success "Monitoring started"
    print_info "Prometheus: http://localhost:9090"
    print_info "Grafana: http://localhost:3001 (admin/admin)"
}

# Utility functions
show_logs() {
    local service=${1:-api}
    print_info "Showing logs for $service..."
    docker-compose logs -f $service
}

show_status() {
    print_info "Container status:"
    docker-compose ps
    echo ""
    
    print_info "System resources:"
    docker system df
}

cleanup() {
    print_warning "Stopping all containers..."
    docker-compose down
    
    print_warning "Removing unused images..."
    docker image prune -f
    
    print_success "Cleanup completed"
}

deep_cleanup() {
    print_warning "This will remove all containers, networks, and unused images!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose down --volumes --remove-orphans
        docker system prune -a -f --volumes
        print_success "Deep cleanup completed"
    else
        print_info "Cleanup cancelled"
    fi
}

run_tests() {
    print_info "Running tests in Docker..."
    docker-compose run --rm api python -m pytest tests/ -v
}

# Help function
show_help() {
    echo "🏥 Diabetic Retinopathy Docker Management"
    echo "========================================"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Build Commands:"
    echo "  build-prod         Build production image"
    echo "  build-dev          Build development image"
    echo "  build-all          Build all images"
    echo ""
    echo "Run Commands:"
    echo "  start              Start production API server"
    echo "  dev                Start development environment"
    echo "  full-stack         Start API + Frontend"
    echo "  train              Start model training"
    echo "  evaluate           Start model evaluation"
    echo "  monitor            Start monitoring stack"
    echo ""
    echo "Utility Commands:"
    echo "  logs [service]     Show logs (default: api)"
    echo "  status             Show container status"
    echo "  test               Run tests"
    echo "  shell [service]    Open shell in container"
    echo "  stop               Stop all services"
    echo "  restart            Restart all services"
    echo ""
    echo "Cleanup Commands:"
    echo "  cleanup            Basic cleanup"
    echo "  deep-cleanup       Remove everything (WARNING: destructive)"
    echo ""
    echo "Examples:"
    echo "  $0 build-all       # Build all images"
    echo "  $0 dev             # Start development environment"
    echo "  $0 logs api        # Show API logs"
    echo "  $0 shell api       # Open shell in API container"
}

# Main command handling
main() {
    check_docker
    
    case "${1:-help}" in
        "build-prod")
            build_production
            ;;
        "build-dev")
            build_development
            ;;
        "build-all")
            build_development
            build_production
            ;;
        "start")
            run_production
            ;;
        "dev")
            run_development
            ;;
        "full-stack")
            run_full_stack
            ;;
        "train")
            run_training
            ;;
        "evaluate")
            run_evaluation
            ;;
        "monitor")
            start_monitoring
            ;;
        "logs")
            show_logs $2
            ;;
        "status")
            show_status
            ;;
        "test")
            run_tests
            ;;
        "shell")
            service=${2:-api}
            print_info "Opening shell in $service container..."
            docker-compose exec $service /bin/bash
            ;;
        "stop")
            print_info "Stopping all services..."
            docker-compose down
            ;;
        "restart")
            print_info "Restarting all services..."
            docker-compose down
            docker-compose up -d
            ;;
        "cleanup")
            cleanup
            ;;
        "deep-cleanup")
            deep_cleanup
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"