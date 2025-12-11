# Berlin Archive RAG System

Production-grade Retrieval-Augmented Generation (RAG) system for the Berlin Media Archive with strict attribution requirements.

## 🎯 Features

- **Multi-modal Search**: Query across audio transcripts and text documents
- **Strict Attribution**: Every answer includes precise source citations
- **Speaker Diarization**: Filter by speaker in audio content
- **Metadata Filtering**: Filter by content type, source, or speaker
- **Observability**: Comprehensive logging and tracing
- **Docker Support**: Containerized for easy deployment

## 📋 Prerequisites

- Python 3.9+
- Docker & Docker Compose (optional, for containerization)
- VS Code (recommended)
- API Keys:
  - Google AI (Gemini) API key
  - Pinecone API key

## 🚀 Quick Start

### Option 1: Local Setup (VS Code)

```bash
# 1. Clone or create project directory
mkdir berlin-archive-rag
cd berlin-archive-rag

# 2. Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Create .env file with your API keys
cp .env.example .env
# Edit .env with your actual keys

# 5. Run the application
python app.py

# 6. In a new terminal, test the API
python test_local.py
```

### Option 2: Docker Setup

```bash
# 1. Build the image
docker build -t berlin-archive-rag .

# 2. Run with environment variables
docker run -p 8080:8080 \
  -e GOOGLE_API_KEY=your_key \
  -e GEMINI_API_KEY=your_key \
  -e PINECONE_API_KEY=your_key \
  berlin-archive-rag

# OR use Docker Compose (easier)
docker-compose up
```

## 📁 Project Structure

```
berlin-archive-rag/
├── app.py                  # Main Flask application
├── test_local.py          # Local testing script
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create from .env.example)
├── .env.example          # Example environment file
├── Dockerfile            # Docker container definition
├── docker-compose.yml    # Docker Compose configuration
├── .dockerignore         # Docker ignore patterns
├── .gitignore           # Git ignore patterns
├── render.yaml          # Render deployment config
├── README.md            # This file
└── logs/                # Application logs (auto-created)
```

## 🔧 Configuration

Create a `.env` file in the project root:

```env
# Google AI Configuration
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_API_KEY=your_google_api_key_here

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=assess

# Server Configuration
PORT=8080
FLASK_ENV=development
```

## 🧪 Testing

### Using the Test Script

```bash
# Make sure the server is running first
python app.py

# In another terminal, run tests
python test_local.py
```

This will:
- Check server health
- Run sample queries
- Save results to `test_output.json` and `test_results.json`

### Manual Testing with curl

```bash
# Health check
curl http://localhost:8080/health

# Query the archive
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the primary definition of success?",
    "top_k": 5
  }'
```

## 📊 API Endpoints

### GET `/` or `/health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "service": "Berlin Media Archive RAG System",
  "version": "1.0.0"
}
```

### GET `/status`
Detailed status information

### POST `/query`
Main query endpoint

**Request:**
```json
{
  "question": "Your research question",
  "top_k": 5,
  "filter_by_type": "audio",
  "filter_by_source": "filename.mp3",
  "filter_by_speaker": "SPEAKER_01",
  "include_raw_chunks": false
}
```

**Response:**
```json
{
  "question": "...",
  "answer": "... (Source: file.mp3 at 04:20) ...",
  "citations": [
    {
      "citation_id": 1,
      "source": "file.mp3",
      "type": "audio",
      "timestamp": "04:20",
      "speaker_id": "SPEAKER_01",
      "text_snippet": "...",
      "relevance_score": 0.95
    }
  ],
  "query_metadata": {
    "duration_seconds": 1.23,
    "chunks_retrieved": 5,
    "citations_count": 3
  }
}
```

## 🐳 Docker Commands

```bash
# Build image
docker build -t berlin-archive-rag .

# Run container
docker run -p 8080:8080 --env-file .env berlin-archive-rag

# Using Docker Compose
docker-compose up          # Start services
docker-compose down        # Stop services
docker-compose logs -f     # View logs
docker-compose restart     # Restart services

# Rebuild and start
docker-compose up --build
```

## 📝 Output Files

### test_output.json
Contains the main query result with full response structure:
- Question
- Answer with citations
- Citation details
- Query metadata
- Retrieval trace

### test_results.json
Contains results from all test cases:
- Test summary
- Individual test results
- Pass/fail status
- Timestamps

### archive_rag.log
Application logs with detailed execution traces

## 🔍 Troubleshooting

### Server won't start
```bash
# Check if port is in use
# Windows:
netstat -ano | findstr :8080
# Mac/Linux:
lsof -i :8080

# Use different port
PORT=8081 python app.py
```

### Import errors
```bash
# Reinstall dependencies
pip install --upgrade --force-reinstall -r requirements.txt
```

### API key errors
- Ensure `.env` file exists in project root
- Check for extra spaces or quotes in API keys
- Verify keys are active in respective platforms

### Docker build fails
```bash
# Clear cache and rebuild
docker system prune -a
docker build --no-cache -t berlin-archive-rag .
```

### Server shows "initializing"
- Wait 30-60 seconds for initialization
- Check logs: `docker-compose logs -f`
- Verify Pinecone index exists and has data

## 🚢 Deployment

### Render
1. Push code to GitHub
2. Connect repository in Render
3. Set environment variables in dashboard
4. Deploy using `render.yaml` configuration

### Other Platforms
The Docker container is compatible with:
- Google Cloud Run
- AWS ECS/Fargate
- Azure Container Instances
- DigitalOcean App Platform

## 📚 Development

### VS Code Configuration

Install recommended extensions:
- Python (Microsoft)
- Docker (Microsoft)
- Python Debugger

### Debug Configuration

Create `.vscode/launch.json`:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Flask",
      "type": "debugpy",
      "request": "launch",
      "module": "flask",
      "env": {
        "FLASK_APP": "app.py",
        "FLASK_DEBUG": "1"
      },
      "args": ["run", "--port", "8080"]
    }
  ]
}
```

## 📖 Architecture

### Components
1. **Embedding Service**: Google Generative AI embeddings
2. **Vector Store**: Pinecone for semantic search
3. **LLM**: Gemini 1.5 Flash for answer generation
4. **Attribution Engine**: Ensures strict source citation
5. **Flask API**: RESTful interface

### Query Flow
1. User submits question
2. Question is embedded using Google embeddings
3. Pinecone retrieves relevant chunks
4. LLM generates attributed answer
5. Citations extracted and validated
6. Results returned with full metadata

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## 🆘 Support

For issues or questions:
1. Check troubleshooting section
2. Review logs in `archive_rag.log`
3. Check Docker logs: `docker-compose logs`
4. Open an issue on GitHub

## 🎓 Learning Resources

- [Pinecone Documentation](https://docs.pinecone.io/)
- [Google AI Documentation](https://ai.google.dev/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Docker Documentation](https://docs.docker.com/)
