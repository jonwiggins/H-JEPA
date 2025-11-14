# H-JEPA Deployment Quickstart

Get H-JEPA deployed in 5 minutes!

## Option 1: Docker (Fastest)

### Prerequisites
- Docker installed
- Trained model checkpoint
- (Optional) NVIDIA Docker runtime for GPU

### Steps

1. **Build the image**:
```bash
docker build -f Dockerfile.inference --build-arg TORCH_DEVICE=cpu -t h-jepa:inference .
```

2. **Run the server**:
```bash
docker run -d \
  --name h-jepa-server \
  -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/models:ro \
  -e MODEL_PATH=/app/models/checkpoint_best.pt \
  -e DEVICE=cpu \
  h-jepa:inference
```

3. **Test it**:
```bash
curl http://localhost:8000/health
```

**That's it!** API is available at http://localhost:8000

---

## Option 2: Docker Compose (Recommended)

### Prerequisites
- Docker and Docker Compose
- Trained model checkpoint

### Steps

1. **Start services**:
```bash
docker-compose up -d inference-cpu
```

2. **Check status**:
```bash
docker-compose ps
```

3. **Test**:
```bash
curl http://localhost:8000/health
```

**Bonus**: Also starts Prometheus (`:9090`) and Grafana (`:3000`)!

---

## Option 3: Kubernetes (Production)

### Prerequisites
- Kubernetes cluster
- kubectl configured
- Model stored in PersistentVolume

### Steps

1. **Deploy**:
```bash
kubectl apply -f kubernetes/
```

2. **Check status**:
```bash
kubectl get pods -n h-jepa
```

3. **Access**:
```bash
kubectl port-forward svc/h-jepa-service-internal 8000:8000 -n h-jepa
curl http://localhost:8000/health
```

---

## Option 4: Local Development

### Prerequisites
- Python 3.8+
- Trained model checkpoint

### Steps

1. **Install dependencies**:
```bash
pip install -r requirements.txt
pip install fastapi uvicorn
```

2. **Set environment variables**:
```bash
export MODEL_PATH=checkpoints/checkpoint_best.pt
export DEVICE=cpu
```

3. **Run server**:
```bash
python -m uvicorn src.serving.model_server:app --host 0.0.0.0 --port 8000
```

4. **Test**:
```bash
curl http://localhost:8000/health
```

---

## Quick Deploy Script

Use our automated deployment script:

```bash
# Docker deployment
./scripts/deploy.sh --type docker --device cpu --model checkpoints/checkpoint_best.pt

# Kubernetes deployment
./scripts/deploy.sh --type kubernetes --device cuda --model /path/to/model.pt

# Local deployment
./scripts/deploy.sh --type local --device cpu --model checkpoints/checkpoint_best.pt
```

---

## Testing Your Deployment

### Health Check
```bash
curl http://localhost:8000/health
```

### Extract Features
```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@test_image.jpg" \
  -F "hierarchy_level=0"
```

### Run Full Tests
```bash
./scripts/test_deployment.sh --host localhost --port 8000 --image test_image.jpg
```

---

## Optimization

### Export Optimized Model

```bash
python scripts/export_model.py \
  --checkpoint checkpoints/checkpoint_best.pt \
  --output-dir exported_models \
  --formats torchscript onnx quantized
```

### Benchmark Performance

```bash
./scripts/benchmark.sh \
  --model checkpoints/checkpoint_best.pt \
  --device cuda \
  --batch-sizes "1 8 16 32"
```

---

## Common Issues

### Docker: Port already in use
```bash
# Use different port
docker run -p 8001:8000 ...
```

### Kubernetes: Pod not starting
```bash
# Check logs
kubectl logs -n h-jepa -l app=h-jepa

# Check events
kubectl get events -n h-jepa
```

### Model not loading
```bash
# Verify checkpoint
python -c "import torch; print(torch.load('model.pt').keys())"

# Check path in container
docker exec h-jepa-server ls -lh /app/models/
```

---

## Next Steps

- **Full Documentation**: See [DEPLOYMENT.md](DEPLOYMENT.md)
- **API Reference**: Visit http://localhost:8000/docs
- **Monitoring**: Set up Prometheus and Grafana
- **Production**: Configure auto-scaling and load balancing

---

## Performance Expectations

| Setup | Hardware | Throughput (img/s) |
|-------|----------|-------------------|
| Docker CPU | 8 cores | ~25 |
| Docker GPU | T4 | ~500 |
| Kubernetes | 3x T4 | ~1500 |

*With batch_size=32

---

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/H-JEPA/issues)
- **Docs**: [README.md](README.md)
- **Deployment**: [DEPLOYMENT.md](DEPLOYMENT.md)

Happy deploying! 🚀
