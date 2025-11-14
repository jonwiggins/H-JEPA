# Kubernetes Deployment Guide

## Quick Start

### 1. Prerequisites

- Kubernetes cluster (v1.24+)
- kubectl configured
- GPU nodes (optional, for GPU deployment)
- Storage class for PersistentVolumes

### 2. Deploy

```bash
# Create namespace and resources
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Verify deployment
kubectl get pods -n h-jepa
kubectl get svc -n h-jepa
```

### 3. Access

```bash
# Get service endpoint
kubectl get svc h-jepa-service -n h-jepa

# Port forward for local access
kubectl port-forward svc/h-jepa-service-internal 8000:8000 -n h-jepa

# Test
curl http://localhost:8000/health
```

## Files Overview

- `namespace.yaml`: Namespace, ResourceQuota, and LimitRange
- `configmap.yaml`: Configuration and secrets
- `deployment.yaml`: Main deployment with HPA and PDB
- `service.yaml`: Service and Ingress configuration

## Configuration

### Update Model Path

Edit `configmap.yaml`:

```yaml
data:
  MODEL_PATH: "/models/your-model.pt"
```

### Change Device

For CPU deployment, edit `configmap.yaml`:

```yaml
data:
  DEVICE: "cpu"
```

And update `deployment.yaml` to remove GPU requests.

### Scale Replicas

```bash
# Manual scaling
kubectl scale deployment h-jepa-inference --replicas=5 -n h-jepa

# Update HPA
kubectl edit hpa h-jepa-hpa -n h-jepa
```

## Monitoring

### Check Logs

```bash
# All pods
kubectl logs -n h-jepa -l app=h-jepa --tail=100

# Specific pod
kubectl logs -n h-jepa POD_NAME -f

# Previous logs (if pod crashed)
kubectl logs -n h-jepa POD_NAME --previous
```

### Check Metrics

```bash
# Pod metrics
kubectl top pods -n h-jepa

# HPA status
kubectl get hpa -n h-jepa

# Events
kubectl get events -n h-jepa --sort-by='.lastTimestamp'
```

## Troubleshooting

### Pod Not Starting

```bash
# Describe pod
kubectl describe pod POD_NAME -n h-jepa

# Check events
kubectl get events -n h-jepa | grep POD_NAME

# Check logs
kubectl logs POD_NAME -n h-jepa
```

### Out of Resources

```bash
# Check node resources
kubectl top nodes

# Check resource quotas
kubectl describe resourcequota -n h-jepa

# Reduce resource requests in deployment.yaml
```

### PVC Not Binding

```bash
# Check PVC status
kubectl get pvc -n h-jepa

# Check PV
kubectl get pv

# Describe PVC
kubectl describe pvc h-jepa-model-pvc -n h-jepa
```

## Production Recommendations

1. **High Availability**: Set `minReplicas: 3` in HPA
2. **Resource Limits**: Adjust based on actual usage
3. **Storage**: Use fast SSD storage for models
4. **Monitoring**: Set up Prometheus and Grafana
5. **Logging**: Configure centralized logging
6. **Security**: Enable RBAC and network policies
7. **Backups**: Regular backups of model storage
8. **Updates**: Use rolling updates with proper health checks
