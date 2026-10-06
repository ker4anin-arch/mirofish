#!/bin/bash
set -e
if [ -z "$NEO4J_PASSWORD" ]; then
  echo "NEO4J_PASSWORD is not set" >&2
  exit 1
fi
export NEO4J_AUTH="neo4j/${NEO4J_PASSWORD}"
# The official entrypoint turns every NEO4J_* variable into a config setting;
# NEO4J_PASSWORD is not one and would make Neo4j refuse to start.
unset NEO4J_PASSWORD
exec /startup/docker-entrypoint.sh "$@"
