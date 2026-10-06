#!/bin/bash
set -e
if [ -z "$NEO4J_PASSWORD" ]; then
  echo "NEO4J_PASSWORD is not set" >&2
  exit 1
fi
export NEO4J_AUTH="neo4j/${NEO4J_PASSWORD}"
exec /startup/docker-entrypoint.sh "$@"
