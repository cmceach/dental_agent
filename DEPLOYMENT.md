# Deployment Guide

This guide covers multiple deployment options for the Dental Guideline Agent Streamlit app.

## Prerequisites

- API Keys:
  - `GOOGLE_API_KEY` - Google Gemini API key
  - `EXA_API_KEY` - Exa API key for web search
- Python 3.9+ (for local/self-hosted deployments)

## Option 1: Streamlit Cloud (Recommended - Easiest)

Streamlit Cloud is the easiest way to deploy Streamlit apps.

### Steps:

1. **Push your code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/dental_agent.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io/
   - Sign in with GitHub
   - Click "New app"
   - Select your repository: `YOUR_USERNAME/dental_agent`
   - Main file path: `streamlit_app.py`
   - Branch: `main`

3. **Set Environment Variables**
   - In Streamlit Cloud dashboard, go to Settings → Secrets
   - Add your environment variables:
     ```toml
     GOOGLE_API_KEY=your_google_api_key_here
     EXA_API_KEY=your_exa_api_key_here
     ```
   - Optional: Add other config variables (see `.env.example`)

4. **Deploy**
   - Click "Deploy"
   - Your app will be available at: `https://YOUR_APP_NAME.streamlit.app`

### Notes:
- Streamlit Cloud provides free hosting
- Automatic HTTPS
- Auto-updates on git push
- File size limits apply (check Streamlit Cloud limits)

---

## Option 2: Docker Deployment

Deploy using Docker containers for flexibility and portability.

### Files Needed:

1. **Dockerfile** (create this file):
   ```dockerfile
   FROM python:3.11-slim

   WORKDIR /app

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       gcc \
       && rm -rf /var/lib/apt/lists/*

   # Copy requirements and install Python dependencies
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   # Copy application code
   COPY . .

   # Expose Streamlit port
   EXPOSE 8501

   # Health check
   HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

   # Run Streamlit
   ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **.dockerignore** (create this file):
   ```
   __pycache__
   *.pyc
   .git
   .env
   .venv
   venv
   *.md
   .gitignore
   ```

### Build and Run:

```bash
# Build the Docker image
docker build -t dental-agent .

# Run the container
docker run -d \
  -p 8501:8501 \
  -e GOOGLE_API_KEY=your_google_api_key \
  -e EXA_API_KEY=your_exa_api_key \
  --name dental-agent \
  dental-agent
```

### Docker Compose (Optional):

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  streamlit:
    build: .
    ports:
      - "8501:8501"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - EXA_API_KEY=${EXA_API_KEY}
      - MODEL=${MODEL:-gemini-2.5-flash}
      - AUTO_UPLOAD_PDFS=${AUTO_UPLOAD_PDFS:-false}
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

### Deploy to Cloud Platforms:

**AWS (ECS/Fargate):**
- Build and push to ECR
- Create ECS task definition
- Set environment variables in task definition

**Google Cloud (Cloud Run):**
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT/dental-agent
gcloud run deploy dental-agent \
  --image gcr.io/YOUR_PROJECT/dental-agent \
  --platform managed \
  --region us-central1 \
  --set-env-vars GOOGLE_API_KEY=your_key,EXA_API_KEY=your_key \
  --allow-unauthenticated
```

**Azure (Container Instances):**
- Push to Azure Container Registry
- Create container instance with environment variables

---

## Option 3: Self-Hosted Server

Deploy on your own server (VPS, EC2, etc.).

### Steps:

1. **Install Python and dependencies**
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-venv
   ```

2. **Clone repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/dental_agent.git
   cd dental_agent
   ```

3. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Set environment variables**
   ```bash
   # Create .env file
   nano .env
   # Add your API keys (see .env.example)
   ```

5. **Run with systemd (production)**
   
   Create `/etc/systemd/system/dental-agent.service`:
   ```ini
   [Unit]
   Description=Dental Guideline Agent Streamlit App
   After=network.target

   [Service]
   Type=simple
   User=your_username
   WorkingDirectory=/path/to/dental_agent
   Environment="PATH=/path/to/dental_agent/venv/bin"
   ExecStart=/path/to/dental_agent/venv/bin/streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

   Enable and start:
   ```bash
   sudo systemctl enable dental-agent
   sudo systemctl start dental-agent
   ```

6. **Set up reverse proxy (Nginx)**
   
   Install Nginx:
   ```bash
   sudo apt install nginx
   ```

   Create `/etc/nginx/sites-available/dental-agent`:
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://127.0.0.1:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
           proxy_cache_bypass $http_upgrade;
       }
   }
   ```

   Enable site:
   ```bash
   sudo ln -s /etc/nginx/sites-available/dental-agent /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   ```

7. **Set up SSL (Let's Encrypt)**
   ```bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d your-domain.com
   ```

---

## Option 4: Railway.app

Simple cloud deployment platform.

### Steps:

1. **Install Railway CLI**
   ```bash
   npm i -g @railway/cli
   railway login
   ```

2. **Deploy**
   ```bash
   railway init
   railway up
   ```

3. **Set environment variables**
   - In Railway dashboard, add `GOOGLE_API_KEY` and `EXA_API_KEY`

4. **Configure port**
   - Railway automatically detects Streamlit
   - Or set `PORT` environment variable

---

## Environment Variables for Production

Make sure to set these in your deployment platform:

**Required:**
- `GOOGLE_API_KEY` - Google Gemini API key
- `EXA_API_KEY` - Exa API key

**Optional:**
- `MODEL` - Gemini model (default: `gemini-2.5-flash`)
- `DENTAL_GUIDELINE_DOMAINS` - Comma-separated domain list
- `MIN_DATE_YEARS_AGO` - Date filter (default: 5)
- `SEARCH_RESULTS_COUNT` - Results per search (default: 8)
- `MAX_CHARACTERS_PER_RESULT` - Characters per result (default: 3000)
- `RECURSION_LIMIT` - Agent iteration limit (default: 25)
- `AUTO_UPLOAD_PDFS` - Enable PDF auto-upload (default: false)
- `MAX_PDF_SIZE_MB` - Max PDF size (default: 25)

---

## Security Considerations

1. **Never commit `.env` file** - Already in `.gitignore`
2. **Use secrets management** - Use platform-specific secrets (Streamlit Cloud Secrets, AWS Secrets Manager, etc.)
3. **Limit API access** - Consider IP whitelisting if possible
4. **Monitor usage** - Set up alerts for API quota limits
5. **Rate limiting** - Consider adding rate limiting for production

---

## Troubleshooting

### Common Issues:

1. **Import errors**
   - Ensure all dependencies in `requirements.txt` are installed
   - Check Python version (3.9+)

2. **API key errors**
   - Verify environment variables are set correctly
   - Check for typos in variable names

3. **Port issues**
   - Ensure port 8501 (or configured port) is open
   - Check firewall settings

4. **File upload issues**
   - Check file size limits on hosting platform
   - Verify Gemini File API quota

---

## Monitoring and Maintenance

- Monitor API usage and costs
- Set up health checks
- Regular dependency updates
- Backup important configurations
- Monitor error logs

