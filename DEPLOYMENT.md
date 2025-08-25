# 🚀 Deployment Guide

This guide covers how to deploy the Diabetic Retinopathy Detection API in various environments.

## 🐳 Docker Deployment

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- At least 4GB RAM
- Trained model file (`best_model.pth`)

### Quick Start

1. **Clone and prepare the repository:**
```bash
git clone <your-repo>
cd diabetic-retinopathy-project
cp .env.example .env
```

2. **Build and start the production service:**
```bash
# Build production image
docker build -t diabetic-retinopathy:latest --target production .

# Start production API
docker-compose up -d api
```

3. **Verify deployment:**
```bash
curl http://localhost:8000/health
```

### Environment Configuration

Create a `.env` file with your settings:

```env
# Production settings
ENVIRONMENT=production
DEBUG=False
LOG_LEVEL=INFO

# API configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Model configuration
MODEL_PATH=/app/models/best_model.pth
USE_CUDA=auto

# Security
CORS_ORIGINS=https://yourdomain.com
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com

# Performance
MAX_IMAGE_SIZE_MB=10
ENABLE_GRADCAM=True
```

### Docker Compose Profiles

Use different profiles for different scenarios:

```bash
# Production API only
docker-compose up -d api

# Development with auto-reload
docker-compose --profile dev up api-dev

# Full stack with frontend
docker-compose --profile frontend up -d

# With monitoring
docker-compose --profile monitoring up -d prometheus grafana

# Model training
docker-compose --profile training up trainer
```

### Management Scripts

Use the provided Docker management script:

```bash
# Make script executable
chmod +x scripts/docker-commands.sh

# Build all images
./scripts/docker-commands.sh build-all

# Start production
./scripts/docker-commands.sh start

# Start development environment
./scripts/docker-commands.sh dev

# View logs
./scripts/docker-commands.sh logs api

# Run tests
./scripts/docker-commands.sh test

# Cleanup
./scripts/docker-commands.sh cleanup
```

## ☁️ Cloud Deployment

### AWS ECS Deployment

1. **Create ECR repository:**
```bash
aws ecr create-repository --repository-name diabetic-retinopathy-api
```

2. **Build and push image:**
```bash
# Get login token
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-west-2.amazonaws.com

# Build and tag
docker build -t diabetic-retinopathy:latest --target production .
docker tag diabetic-retinopathy:latest 123456789.dkr.ecr.us-west-2.amazonaws.com/diabetic-retinopathy-api:latest

# Push
docker push 123456789.dkr.ecr.us-west-2.amazonaws.com/diabetic-retinopathy-api:latest
```

3. **Create ECS task definition:**
```json
{
  "family": "diabetic-retinopathy-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::123456789:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "123456789.dkr.ecr.us-west-2.amazonaws.com/diabetic-retinopathy-api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "ENVIRONMENT", "value": "production"},
        {"name": "API_WORKERS", "value": "4"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/diabetic-retinopathy-api",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Google Cloud Run Deployment

1. **Build and push to Google Container Registry:**
```bash
# Configure Docker for GCR
gcloud auth configure-docker

# Build and tag
docker build -t diabetic-retinopathy:latest --target production .
docker tag diabetic-retinopathy:latest gcr.io/your-project/diabetic-retinopathy-api:latest

# Push
docker push gcr.io/your-project/diabetic-retinopathy-api:latest
```

2. **Deploy to Cloud Run:**
```bash
gcloud run deploy diabetic-retinopathy-api \
  --image gcr.io/your-project/diabetic-retinopathy-api:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --set-env-vars ENVIRONMENT=production,API_WORKERS=4
```

### Azure Container Instances

1. **Create resource group:**
```bash
az group create --name diabetic-retinopathy-rg --location eastus
```

2. **Deploy container:**
```bash
az container create \
  --resource-group diabetic-retinopathy-rg \
  --name diabetic-retinopathy-api \
  --image diabetic-retinopathy:latest \
  --dns-name-label diabetic-retinopathy-api \
  --ports 8000 \
  --environment-variables ENVIRONMENT=production API_WORKERS=4 \
  --cpu 2 \
  --memory 4
```

## 🔧 Production Configuration

### Reverse Proxy Setup (Nginx)

Create `nginx/nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream api {
        server api:8000;
    }
    
    server {
        listen 80;
        server_name yourdomain.com;
        
        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name yourdomain.com;
        
        # SSL configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        
        # API proxy
        location / {
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Health check
        location /health {
            access_log off;
            proxy_pass http://api/health;
        }
        
        # File upload size limit
        client_max_body_size 25M;
    }
}
```

### Load Balancer Configuration

For high availability, deploy multiple API instances behind a load balancer:

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  api-1:
    build:
      context: .
      target: production
    environment:
      - ENVIRONMENT=production
    volumes:
      - ./ml/models:/app/models:ro
    
  api-2:
    build:
      context: .
      target: production
    environment:
      - ENVIRONMENT=production
    volumes:
      - ./ml/models:/app/models:ro
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx-lb.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - api-1
      - api-2
```

## 📊 Monitoring and Logging

### Health Checks

The API includes comprehensive health checks:

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed API info
curl http://localhost:8000/info
```

### Monitoring with Prometheus

Start monitoring stack:

```bash
docker-compose --profile monitoring up -d prometheus grafana
```

Access:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001 (admin/admin)

### Log Aggregation

For production, set up centralized logging:

```yaml
# In docker-compose.yml
services:
  api:
    logging:
      driver: "fluentd"
      options:
        fluentd-address: localhost:24224
        tag: diabetic-retinopathy.api
```

## 🔒 Security Considerations

### Environment Variables

Never commit sensitive data. Use environment variables:

```bash
# Production secrets
export MODEL_PATH=/secure/path/to/model.pth
export API_SECRET_KEY=your-secret-key
export DATABASE_URL=your-database-url
```

### Network Security

- Use HTTPS in production
- Implement rate limiting
- Use firewall rules to restrict access
- Regularly update base images

### Model Security

- Store models in secure locations (S3, encrypted volumes)
- Implement model versioning
- Monitor for model drift

## 🚨 Troubleshooting

### Common Issues

1. **Model not found:**
```bash
# Check model path
docker-compose exec api ls -la /app/models/

# Verify environment variables
docker-compose exec api env | grep MODEL_PATH
```

2. **CUDA/GPU issues:**
```bash
# Check GPU availability
docker-compose exec api python -c "import torch; print(torch.cuda.is_available())"

# Use CPU fallback
export USE_CUDA=false
```

3. **Memory issues:**
```bash
# Increase Docker memory limit
# Check resource usage
docker stats
```

4. **Performance issues:**
```bash
# Scale API instances
docker-compose up --scale api=3

# Check resource usage
docker-compose exec api top
```

### Log Analysis

```bash
# View API logs
docker-compose logs -f api

# Check health status
curl -s http://localhost:8000/health | jq .

# Monitor performance
docker-compose exec api python -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'Memory: {psutil.virtual_memory().percent}%')
"
```

## 📋 Production Checklist

Before deploying to production:

- [ ] Model file is available and tested
- [ ] Environment variables are properly set
- [ ] HTTPS certificates are configured
- [ ] Database migrations are applied (if applicable)
- [ ] Monitoring is set up
- [ ] Backup strategy is in place
- [ ] CI/CD pipeline is configured
- [ ] Load testing is performed
- [ ] Security scan is completed
- [ ] Documentation is updated

## 🔄 Updates and Maintenance

### Rolling Updates

```bash
# Build new image
docker build -t diabetic-retinopathy:v2 --target production .

# Update with zero downtime
docker-compose up -d --no-deps api
```

### Model Updates

```bash
# Replace model file
cp new_model.pth ml/models/best_model.pth

# Restart API to reload model
docker-compose restart api
```

### Backup Strategy

```bash
# Backup models
docker run --rm -v $(pwd)/ml/models:/source -v $(pwd)/backups:/backup alpine tar czf /backup/models-$(date +%Y%m%d).tar.gz -C /source .

# Backup configuration
cp .env backups/env-$(date +%Y%m%d).bak
```

This deployment guide should help you successfully deploy the Diabetic Retinopathy Detection API in various environments. Adjust configurations based on your specific requirements and infrastructure.