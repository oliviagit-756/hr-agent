# HR Agent Backend

**HR Agent Backend** is a Django-based REST API for **job matching** and **resume parsing**. The API leverages **transfer learning** models to match job descriptions (JDs) with candidates' resumes. It provides two core functionalities:

1. **Job-to-Resume** search: Given a resume, find the most relevant job descriptions.
2. **Resume-to-Job** search: Given a job description, find the most relevant resumes.

The API uses **FAISS** for fast semantic search and **MS MARCO MiniLM** cross-encoder for improved re-ranking.

## Features:
- **Bi-Encoder (MiniLM)** for semantic search with **FAISS** for fast retrieval.
- **Cross-Encoder (MS MARCO MiniLM)** for accurate job/resume matching.
- **RESTful API endpoints** for:
  - Job-to-Resume matching (`POST /api/search/candidates`)
  - Resume-to-Job matching (`POST /api/search/jobs`)
  - Health check (`GET /api/health`)

## Setup & Installation

### Clone the Repository

```bash
git clone https://github.com/oliviagit-756/hr-agent-backend.git
cd hr-agent-backend

