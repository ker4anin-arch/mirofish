# Neo4j for Render: builds NEO4J_AUTH from NEO4J_PASSWORD so the password can
# be shared with the MiroFish web service through an environment group.
FROM neo4j:5.26
COPY render-entrypoint.sh /render-entrypoint.sh
RUN chmod +x /render-entrypoint.sh
ENTRYPOINT ["tini", "-g", "--", "/render-entrypoint.sh"]
CMD ["neo4j"]
