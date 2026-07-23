# Deployment Guide - Reservoir History Matching

## Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python train.py

EXPOSE 5006

CMD ["python", "app.py"]
```

### Build and Run
```bash
docker build -t reservoir-history-matching .
docker run -p 5006:5006 reservoir-history-matching
```

### Docker Compose
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "5006:5006"
    environment:
      - FLASK_DEBUG=0
    volumes:
      - ./outputs:/app/outputs
    restart: unless-stopped
```

```bash
docker-compose up -d
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| FLASK_DEBUG | Enable debug mode | 1 |
| PORT | Server port | 5006 |
| HOST | Server host | 0.0.0.0 |

## Manual Deployment

### Prerequisites
- Python 3.8+
- pip

### Steps
```bash
git clone https://github.com/kcabrera83/reservoir-history-matching.git
cd reservoir-history-matching
pip install -r requirements.txt
python train.py
python test_api.py  # optional - run tests
python app.py
```

## Production Considerations

### Gunicorn (Recommended)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5006 app:app
```

### Security
- Set `DEBUG=False` in production
- Use HTTPS with a reverse proxy
- Add authentication for prediction endpoints
- Validate input feature ranges

### Monitoring
- Monitor `/api/health` for model availability
- Track prediction accuracy over time
- Monitor forecast drift from actuals
- Log all API requests

### Performance
- Models loaded eagerly at startup
- Consider model versioning for A/B testing
- Use feature caching for repeated reservoir properties

### Model Management
- Retrain periodically with new production data
- Compare model versions using metrics from `/api/models`
- Store prediction history for model validation

## CI/CD
GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push.

## API Self-Documentation
Access OpenAPI docs at: `http://localhost:5006/api/docs`
