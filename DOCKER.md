# Docker Development

This project uses Docker Compose for the frontend, backend, and Redis.
Ollama and PostgreSQL stay on the host machine and are reached from the backend container.

## Prerequisites

- Docker Desktop
- Ollama running on the host machine
- PostgreSQL running on the host machine
- A local Ollama model created from `models/Modelfile`

## Start

```bash
docker compose -f docker-compose.dev.yml up --build
```

Frontend: `http://localhost:5173`

Backend health check: `http://localhost:8000/api/v1/health`

## Ollama

The backend container uses:

- `OLLAMA_BASE_URL=http://host.docker.internal:11434`
- `OLLAMA_MODEL=text2sql-local`
- `DATABASE_URL=postgresql://...@host.docker.internal:5432/...`

Create the Ollama model on the host before starting Compose.

Example:

```bash
ollama create text2sql-local -f ./models/Modelfile
ollama run text2sql-local
```

## Notes

- `models/` is excluded from Docker build context on purpose.
- The frontend dev server proxies `/api/v1` to the backend container.
- Redis still runs in Docker by default.
- PostgreSQL is not started by Compose. Update `DATABASE_URL` in `docker-compose.dev.yml` if your host DB name, port, or credentials differ.
