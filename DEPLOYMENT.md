# 🚀 Deployment Guide - Diggyy Fresh

Multiple deployment options for different use cases.

---

## 📊 Option 1: Streamlit Cloud (Recommended - FREE)

**Best for:** Quick demo, hackathon presentation, free hosting

### Steps:

1. **Push code to GitHub** (already done!)

2. **Go to Streamlit Cloud**
   - Visit: https://streamlit.io/cloud
   - Sign in with GitHub
   - Click "New app"

3. **Configure deployment**
   - Repository: `your-username/diggyy-fresh`
   - Branch: `main`
   - Main file path: `dashboard/app.py`
   - Click "Deploy"

4. **Wait 2-3 minutes** for deployment

5. **Get your public URL!**
   - Format: `https://your-app-name.streamlit.app`
   - Share with judges/investors

### Auto-deploy on push:
- Every push to `main` automatically redeploys
- No manual intervention needed

---

## 🐳 Option 2: Docker Deployment

**Best for:** Production, custom infrastructure, full control

### Local Docker:

```bash
# Build and run with docker-compose
docker-compose up

# Or build individually:
docker build -f docker/Dockerfile.dashboard -t diggyy-fresh-dashboard .
docker run -p 8501:8501 diggyy-fresh-dashboard

# Access at: http://localhost:8501
```

### Cloud Docker Deployment:

#### AWS ECS/Fargate:
```bash
# Build image
docker build -f docker/Dockerfile.dashboard -t diggyy-fresh .

# Tag for ECR
docker tag diggyy-fresh:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/diggyy-fresh

# Push to ECR
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/diggyy-fresh

# Deploy via ECS console or CLI
```

#### GCP Cloud Run:
```bash
# Build and deploy in one command
gcloud run deploy diggyy-fresh \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Azure Container Instances:
```bash
# Create container instance
az container create \
  --resource-group diggyy-fresh-rg \
  --name diggyy-fresh-dashboard \
  --image diggyy-fresh:latest \
  --dns-name-label diggyy-fresh \
  --ports 8501
```

---

## 💻 Option 3: Manual Deployment

**Best for:** Local testing, development

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python src/synthetic_data_generator.py

# Train agent (optional, takes ~10 min)
python src/agent.py

# Run dashboard
streamlit run dashboard/app.py
```

Access at: http://localhost:8501

---

## 🤖 Option 4: Hugging Face Spaces

**Best for:** ML community exposure, free GPU

1. Create account at https://huggingface.co
2. Create new Space (Streamlit)
3. Push code to Space repo
4. Auto-deploys with free compute

---

## 📋 Pre-Deployment Checklist

Before deploying, ensure:

- [ ] ✅ All tests passing (`pytest tests/`)
- [ ] ✅ Code formatted (`black src/ tests/`)
- [ ] ✅ Synthetic data generated
- [ ] ✅ Agent trained (or auto-trains in dashboard)
- [ ] ✅ Environment variables set (if using LLM features)

---

## 🔧 Environment Variables

For LLM features (optional):

```bash
# .env file
OLLAMA_HOST=http://localhost:11434  # For local Ollama
# or
OPENAI_API_KEY=sk-...  # If using OpenAI instead
```

---

## 🎯 Post-Deployment

### Monitor:
- **Streamlit Cloud**: Built-in logs and metrics
- **Docker**: `docker logs -f container-name`
- **Cloud providers**: CloudWatch, Stackdriver, Application Insights

### Scale:
- Streamlit Cloud: Automatic scaling (limited on free tier)
- Docker: Use orchestration (Kubernetes, ECS, etc.)

### Update:
- Push to main branch → auto-deploys (Streamlit Cloud)
- Rebuild Docker image → redeploy container

---

## 🆘 Troubleshooting

### Dashboard won't load:
- Check if all dependencies installed
- Generate synthetic data: `python src/synthetic_data_generator.py`
- Check logs for errors

### Port already in use:
```bash
# Kill process on port 8501
lsof -ti:8501 | xargs kill -9

# Or use different port
streamlit run dashboard/app.py --server.port 8502
```

### Memory issues:
- Reduce episode length in training
- Use smaller model
- Deploy to instance with more RAM

---

## 📊 Recommended: Streamlit Cloud

For this hackathon, **Streamlit Cloud is the best option**:
- ✅ Free
- ✅ Automatic HTTPS
- ✅ No server management
- ✅ Auto-deploy on push
- ✅ Shareable URL for judges
- ✅ 2-minute setup

Deploy now: https://streamlit.io/cloud

---

**Questions?** Check the [Streamlit docs](https://docs.streamlit.io) or open an issue.
