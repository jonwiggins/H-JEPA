# H-JEPA Deployment Guide

Complete guide for deploying H-JEPA models in production environments.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Model Serving API](#model-serving-api)
7. [Performance Optimization](#performance-optimization)
8. [Monitoring and Logging](#monitoring-and-logging)
9. [Security Best Practices](#security-best-practices)
10. [Troubleshooting](#troubleshooting)

## Overview

H-JEPA deployment infrastructure provides:

- **Docker Support**: Containerized training and inference environments
- **Model Serving**: Production-ready REST API with FastAPI
- **Kubernetes**: Scalable cloud-native deployment
- **Optimization**: TorchScript, ONNX export, and INT8 quantization
- **Monitoring**: Prometheus metrics and Grafana dashboards
- **CI/CD**: Automated testing and deployment pipelines

### Architecture

```
┌─────────────────┐
│  Load Balancer  │
└────────┬────────┘
         │
    ┌────┴────┐
    │ Ingress │
    └────┬────┘
         │
    ┌────┴─────────────────┐
    │  H-JEPA Service      │
    │  (Multiple Replicas) │
    └──────────────────────┘
         │
    ┌────┴────┐
    │  Model  │
    │ Storage │
    └─────────┘
```

## Quick Start

### 1. Build Docker Images

```bash
# Build training image
docker build -f Dockerfile.train -t h-jepa:train .

# Build inference image (CPU)
docker build -f Dockerfile.inference --build-arg TORCH_DEVICE=cpu -t h-jepa:inference-cpu .

# Build inference image (GPU)
docker build -f Dockerfile.inference --build-arg TORCH_DEVICE=cuda -t h-jepa:inference-gpu .
```

### 2. Deploy with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f inference-gpu

# Stop services
docker-compose down
```

### 3. Test Deployment

```bash
# Run automated tests
./scripts/test_deployment.sh --host localhost --port 8000 --image /path/to/test.jpg

# Manual test
curl http://localhost:8000/health
```

## Docker Deployment

### Training Environment

The training Dockerfile includes:
- CUDA 11.8 + cuDNN 8
- PyTorch 2.1.0
- All H-JEPA dependencies
- TensorBoard for monitoring

```bash
# Run training container
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/results:/workspace/results \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  h-jepa:train \
  python scripts/train.py --config configs/default.yaml
```

### Inference Environment

Optimized for production serving:
- Minimal base image (runtime only)
- FastAPI server
- Health checks and monitoring
- Security hardening

```bash
# Run inference server (CPU)
docker run -d \
  --name h-jepa-server \
  -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/models:ro \
  -e MODEL_PATH=/app/models/checkpoint_best.pt \
  -e DEVICE=cpu \
  h-jepa:inference-cpu

# Run inference server (GPU)
docker run -d \
  --name h-jepa-server \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/models:ro \
  -e MODEL_PATH=/app/models/checkpoint_best.pt \
  -e DEVICE=cuda \
  h-jepa:inference-gpu
```

### Docker Compose

Multi-container setup with:
- Training service
- Inference service (CPU/GPU)
- TensorBoard
- Prometheus
- Grafana

```bash
# Start specific services
docker-compose up -d inference-gpu tensorboard

# Scale inference replicas
docker-compose up -d --scale inference-cpu=3

# View resource usage
docker-compose top
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (v1.24+)
- kubectl configured
- GPU nodes (for GPU deployment)
- Persistent storage

### Deploy to Kubernetes

```bash
# Create namespace
kubectl apply -f kubernetes/namespace.yaml

# Create ConfigMap and Secrets
kubectl apply -f kubernetes/configmap.yaml

# Deploy model server
kubectl apply -f kubernetes/deployment.yaml

# Create service
kubectl apply -f kubernetes/service.yaml

# Check deployment status
kubectl get pods -n h-jepa -l app=h-jepa

# View logs
kubectl logs -n h-jepa -l app=h-jepa --tail=100 -f
```

### Scaling

```bash
# Manual scaling
kubectl scale deployment h-jepa-inference --replicas=5 -n h-jepa

# Auto-scaling is configured via HPA
kubectl get hpa -n h-jepa

# View scaling events
kubectl describe hpa h-jepa-hpa -n h-jepa
```

### Monitoring

```bash
# Check pod metrics
kubectl top pods -n h-jepa

# Check node metrics
kubectl top nodes

# Access Prometheus
kubectl port-forward svc/prometheus 9090:9090 -n monitoring

# Access Grafana
kubectl port-forward svc/grafana 3000:3000 -n monitoring
```

## Cloud Deployment

### AWS (EKS)

1. **Create EKS Cluster**:

```bash
eksctl create cluster \
  --name h-jepa-cluster \
  --region us-west-2 \
  --nodegroup-name gpu-nodes \
  --node-type p3.2xlarge \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 5 \
  --managed
```

2. **Install NVIDIA Device Plugin**:

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

3. **Deploy H-JEPA**:

```bash
kubectl apply -f kubernetes/
```

### GCP (GKE)

1. **Create GKE Cluster**:

```bash
gcloud container clusters create h-jepa-cluster \
  --zone us-central1-a \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --machine-type n1-standard-4 \
  --num-nodes 2 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 5
```

2. **Install NVIDIA Driver**:

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

3. **Deploy H-JEPA**:

```bash
kubectl apply -f kubernetes/
```

### Azure (AKS)

1. **Create AKS Cluster**:

```bash
az aks create \
  --resource-group h-jepa-rg \
  --name h-jepa-cluster \
  --node-count 2 \
  --node-vm-size Standard_NC6s_v3 \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 5
```

2. **Install NVIDIA Device Plugin**:

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

3. **Deploy H-JEPA**:

```bash
kubectl apply -f kubernetes/
```

## Model Serving API

### Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "version": "1.0.0"
}
```

#### Model Info
```bash
curl http://localhost:8000/info
```

Response:
```json
{
  "model_type": "H-JEPA",
  "device": "cuda",
  "image_size": 224,
  "num_hierarchies": 3,
  "embed_dim": 768,
  "num_patches": 196,
  "patch_size": 16
}
```

#### Feature Extraction (Single Image)
```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@image.jpg" \
  -F "hierarchy_level=0"
```

Response:
```json
{
  "features": [[0.123, -0.456, ...]],
  "shape": [1, 196, 768],
  "hierarchy_level": 0,
  "inference_time_ms": 15.3
}
```

#### Feature Extraction (Batch)
```bash
curl -X POST http://localhost:8000/extract_batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "hierarchy_level=0"
```

Response:
```json
{
  "features": [[[...]], [[...]]],
  "shapes": [[1, 196, 768], [1, 196, 768]],
  "hierarchy_level": 0,
  "batch_size": 2,
  "total_inference_time_ms": 25.6,
  "average_inference_time_ms": 12.8
}
```

#### Metrics
```bash
curl http://localhost:8000/metrics
```

### Python Client Example

```python
import requests
from pathlib import Path

# API endpoint
url = "http://localhost:8000"

# Health check
response = requests.get(f"{url}/health")
print(response.json())

# Extract features
with open("image.jpg", "rb") as f:
    files = {"file": f}
    data = {"hierarchy_level": 0}
    response = requests.post(f"{url}/extract", files=files, data=data)

features = response.json()
print(f"Shape: {features['shape']}")
print(f"Inference time: {features['inference_time_ms']} ms")
```

## Performance Optimization

### TorchScript Export

```python
from src.inference.optimized_model import export_to_torchscript

# Export model to TorchScript
traced_model = export_to_torchscript(
    model,
    output_path="model.torchscript.pt",
    hierarchy_level=0,
    optimize=True,
)

# Use TorchScript model
features = traced_model(images)
```

### ONNX Export

```python
from src.inference.optimized_model import export_to_onnx

# Export model to ONNX
export_to_onnx(
    model,
    output_path="model.onnx",
    hierarchy_level=0,
    opset_version=14,
    dynamic_axes=True,
)
```

### INT8 Quantization

```python
from src.inference.optimized_model import quantize_model

# Dynamic quantization
quantized_model = quantize_model(
    model,
    output_path="model.quantized.pt",
    hierarchy_level=0,
    quantization_type='dynamic',
)
```

### Batch Inference

```python
from src.inference.optimized_model import BatchInference

# Create batch inference engine
batch_engine = BatchInference(
    model=model,
    device='cuda',
    batch_size=32,
    hierarchy_level=0,
)

# Extract features in batches
features = batch_engine.extract_features(images)

# Benchmark performance
results = batch_engine.benchmark(
    num_images=1000,
    num_runs=10,
)
print(f"Throughput: {results['throughput_images_per_sec']:.2f} images/sec")
```

### Benchmarking

```bash
# Run comprehensive benchmark
./scripts/benchmark.sh \
  --model checkpoints/checkpoint_best.pt \
  --device cuda \
  --batch-sizes "1 4 8 16 32" \
  --num-runs 100 \
  --output benchmark_results.json

# View results
cat benchmark_results.json | jq .
```

## Monitoring and Logging

### Prometheus Metrics

H-JEPA exposes the following metrics:

- `hjepa_requests_total`: Total number of requests
- `hjepa_request_duration_seconds`: Request latency histogram
- `hjepa_inference_duration_seconds`: Inference latency histogram

### Grafana Dashboard

1. Access Grafana:
```bash
# Docker Compose
open http://localhost:3000

# Kubernetes
kubectl port-forward svc/grafana 3000:3000 -n monitoring
```

2. Login (default: admin/admin)

3. Import H-JEPA dashboard (create custom dashboard with metrics)

### Logging

```bash
# View Docker logs
docker logs h-jepa-server -f

# View Kubernetes logs
kubectl logs -n h-jepa -l app=h-jepa -f

# Filter by level
kubectl logs -n h-jepa -l app=h-jepa | grep ERROR
```

## Security Best Practices

### Container Security

1. **Non-root user**: Inference container runs as non-root user
2. **Read-only filesystem**: Model directory mounted read-only
3. **Minimal base image**: Runtime-only base image reduces attack surface
4. **Security scanning**: Trivy scans for vulnerabilities

### Network Security

1. **TLS/SSL**: Use HTTPS in production (configure in Ingress)
2. **API authentication**: Add authentication middleware if needed
3. **Rate limiting**: Configure rate limiting in Ingress
4. **Network policies**: Restrict pod-to-pod communication

### Secrets Management

```bash
# Kubernetes secrets
kubectl create secret generic hjepa-secrets \
  --from-literal=api-key=your-api-key \
  -n h-jepa

# Use in deployment
env:
  - name: API_KEY
    valueFrom:
      secretKeyRef:
        name: hjepa-secrets
        key: api-key
```

## Troubleshooting

### Common Issues

#### 1. Model Loading Fails

```bash
# Check model path
ls -lh /path/to/model.pt

# Check logs
docker logs h-jepa-server | grep "Loading model"

# Verify checkpoint
python -c "import torch; print(torch.load('model.pt').keys())"
```

#### 2. GPU Not Detected

```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Verify Kubernetes GPU
kubectl describe node | grep nvidia.com/gpu
```

#### 3. Out of Memory

```bash
# Reduce batch size
export BATCH_SIZE=8

# Use CPU offloading
export DEVICE=cpu

# Monitor memory
docker stats h-jepa-server
```

#### 4. Slow Inference

```bash
# Benchmark model
./scripts/benchmark.sh --model model.pt --device cuda

# Check GPU utilization
nvidia-smi -l 1

# Enable TorchScript
python -m src.inference.optimized_model export --format torchscript
```

### Debug Mode

```bash
# Run with debug logging
docker run -e LOG_LEVEL=debug h-jepa:inference-cpu

# Interactive debugging
docker run -it --entrypoint /bin/bash h-jepa:inference-cpu
```

## CI/CD Pipeline

### GitHub Actions

The repository includes automated workflows:

1. **Testing** (`.github/workflows/test.yml`):
   - Runs on push/PR
   - Tests on Python 3.8-3.11
   - Code quality checks
   - Coverage reporting

2. **Docker Build** (`.github/workflows/docker.yml`):
   - Builds on main branch
   - Pushes to GitHub Container Registry
   - Security scanning with Trivy

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Set up hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Performance Benchmarks

Typical performance on different hardware:

| Hardware | Batch Size | Throughput (img/s) | Latency (ms/img) |
|----------|------------|-------------------|------------------|
| CPU (8 cores) | 1 | 5 | 200 |
| CPU (8 cores) | 32 | 25 | 40 |
| T4 GPU | 1 | 100 | 10 |
| T4 GPU | 32 | 500 | 2 |
| V100 GPU | 1 | 200 | 5 |
| V100 GPU | 32 | 1000 | 1 |

*Note: Performance varies based on model size and configuration*

## Production Checklist

Before deploying to production:

- [ ] Model checkpoint tested and validated
- [ ] Docker images built and scanned for vulnerabilities
- [ ] Health checks configured and working
- [ ] Monitoring and alerting set up
- [ ] Auto-scaling configured
- [ ] Backup and disaster recovery plan
- [ ] Security review completed
- [ ] Load testing performed
- [ ] Documentation updated
- [ ] CI/CD pipeline tested

## Support

For issues and questions:

- GitHub Issues: [H-JEPA Issues](https://github.com/yourusername/H-JEPA/issues)
- Documentation: [README.md](README.md)
- Examples: [examples/](examples/)

## License

MIT License - see [LICENSE](LICENSE) file for details.
