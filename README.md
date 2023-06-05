# Jarvis

## Development

### Running Scylla in docker

start Scylla:
```bash
docker run -d -p 9042:9042 --name scylla scylladb/scylla:5.2.1 --smp 2
```

check Scylla status:
```bash
docker exec -it scylla nodetool status
```

start cqlsh and then run cql schemas:
```bash
docker exec -it scylla cqlsh
```

restart Scylla:
```bash
docker exec -it scylla supervisorctl restart scylla
```

### Running Qdrant vector search engine in docker

start Qdrant:
```bash
docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant:latest
```

### Create a config.toml file

```bash
cp config/default.toml config.toml
```

then update it with your own configuration.

### Run Jarvis

```bash
make run-dev
```
