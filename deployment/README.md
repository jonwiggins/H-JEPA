# Deployment Configuration

This directory contains configuration files for monitoring and deployment.

## Files

### prometheus.yml

Prometheus configuration for metrics collection.

**Usage:**
```bash
# With Docker Compose
docker-compose up -d prometheus

# Access Prometheus
open http://localhost:9090
```

**Custom metrics:**
- H-JEPA inference metrics
- System metrics
- GPU metrics (if available)

### grafana-datasources.yml

Grafana datasource configuration.

**Usage:**
```bash
# With Docker Compose
docker-compose up -d grafana

# Access Grafana
open http://localhost:3000
# Default credentials: admin/admin
```

## Monitoring Setup

### 1. Start Monitoring Stack

```bash
docker-compose up -d prometheus grafana
```

### 2. Access Dashboards

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

### 3. Create Grafana Dashboard

1. Login to Grafana (admin/admin)
2. Add Prometheus datasource (auto-configured)
3. Create new dashboard
4. Add panels with queries:

```promql
# Request rate
rate(hjepa_requests_total[5m])

# Request latency (95th percentile)
histogram_quantile(0.95, rate(hjepa_request_duration_seconds_bucket[5m]))

# Inference latency
histogram_quantile(0.95, rate(hjepa_inference_duration_seconds_bucket[5m]))
```

## Metrics Reference

### H-JEPA Metrics

- `hjepa_requests_total{endpoint, status}`: Total requests by endpoint and status
- `hjepa_request_duration_seconds`: Request duration histogram
- `hjepa_inference_duration_seconds`: Inference duration histogram

### System Metrics (if using node-exporter)

- CPU usage
- Memory usage
- Disk I/O
- Network I/O

### GPU Metrics (if using nvidia-gpu-exporter)

- GPU utilization
- GPU memory usage
- GPU temperature
- GPU power consumption

## Alerting

### Prometheus Alerts

Create `alerts.yml`:

```yaml
groups:
  - name: h-jepa
    rules:
      - alert: HighErrorRate
        expr: rate(hjepa_requests_total{status="error"}[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(hjepa_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High request latency detected"
```

## Cloud Provider Monitoring

### AWS CloudWatch

```bash
# Export metrics to CloudWatch
# Use CloudWatch agent or custom exporter
```

### GCP Cloud Monitoring

```bash
# Export metrics to Cloud Monitoring
# Use GCP monitoring agent
```

### Azure Monitor

```bash
# Export metrics to Azure Monitor
# Use Azure monitoring agent
```

## Best Practices

1. **Retention**: Configure appropriate retention period for metrics
2. **Sampling**: Use appropriate scrape intervals (15s recommended)
3. **Alerts**: Set up alerts for critical metrics
4. **Dashboards**: Create dashboards for key metrics
5. **Backup**: Regular backup of monitoring data
6. **Security**: Secure Prometheus and Grafana with authentication
