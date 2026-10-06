# Neo4j for Render/VPS: builds NEO4J_AUTH from NEO4J_PASSWORD so the password can
# be shared with the MiroFish service (env group on Render, .env on a VPS).
FROM neo4j:5.26
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["tini", "-g", "--", "/entrypoint.sh"]
CMD ["neo4j"]
