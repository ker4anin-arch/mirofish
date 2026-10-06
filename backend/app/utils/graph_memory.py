"""Self-hosted knowledge-graph memory built on Graphiti + Neo4j.

This module replaces Zep Cloud. It exposes the small subset of the
``zep_cloud`` client surface that MiroFish uses (``graph.*``, ``graph.node.*``,
``graph.edge.*``, ``graph.episode.*`` and ``batch.*``) so the services keep
their existing call shapes, while the data lives in your own Neo4j instance.

Design notes
------------
* Graphiti is async; Flask is threaded. A single background event loop owns
  the Graphiti instance and the Neo4j async driver, and synchronous callers
  submit coroutines to it.
* Like Zep, writes are asynchronous: ``graph.add`` and ``batch.process`` save
  the episode immediately and return, extraction then runs in a per-graph
  queue (Graphiti requires sequential ingestion within one graph). Callers poll
  ``graph.episode.get(...).processed`` / ``batch.get(...).status``.
* Ingestion state (queued/processing/failed) and batches are process-local.
  Episodes that already exist in Neo4j but are unknown to this process (for
  example after a restart) are reported as processed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import uuid as uuid_lib
from collections.abc import Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, TypeVar

# Graphiti sends anonymous usage telemetry by default; keep self-hosted data local.
os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")

from pydantic import BaseModel, Field, create_model  # noqa: E402

from ..config import Config  # noqa: E402
from .logger import get_logger  # noqa: E402

logger = get_logger("mirofish.graph_memory")
# Graphiti's queries reference optional properties; Neo4j reports each one as a
# "property key does not exist" notification, which floods the logs.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)


class _DropConcurrentIndexErrors(logging.Filter):
    """Graphiti creates indices in parallel; on a fresh database the
    ``IF NOT EXISTS`` statements race and Neo4j reports harmless
    EquivalentSchemaRuleAlreadyExists errors."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "EquivalentSchemaRuleAlreadyExists" not in record.getMessage()


logging.getLogger("graphiti_core.driver.neo4j_driver").addFilter(_DropConcurrentIndexErrors())

T = TypeVar("T")

GRAPH_LABEL = "MiroFishGraph"
# Simulation memory is always ingested one episode at a time. Document chunks
# are ingested in bulk slices of this size; 1 switches documents to
# sequential ingestion too (slower, usually extracts more relations).
BULK_SLICE_SIZE = int(os.environ.get("GRAPH_BULK_SLICE_SIZE", "10"))


# ---------------------------------------------------------------------------
# Errors (mirror the parts of zep_cloud's error model the services rely on)
# ---------------------------------------------------------------------------


class GraphMemoryError(Exception):
    """Base error carrying an HTTP-like status code for retry classification."""

    def __init__(
        self,
        message: str | None = None,
        status_code: int | None = None,
        *,
        body: Any = None,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(message or str(body or status_code))
        self.status_code = status_code
        self.headers: dict[str, str] = dict(headers or {})
        self.body = body if body is not None else message


class NotFoundError(GraphMemoryError):
    def __init__(self, message: str):
        super().__init__(message, status_code=404)


class BadRequestError(GraphMemoryError):
    def __init__(self, message: str):
        super().__init__(message, status_code=400)


# ---------------------------------------------------------------------------
# Ontology model base classes (replacement for zep_cloud.external_clients.ontology)
# ---------------------------------------------------------------------------

EntityText = str


class EntityModel(BaseModel):
    """Base class for custom entity types."""


class EdgeModel(BaseModel):
    """Base class for custom edge types."""


@dataclass
class EntityEdgeSourceTarget:
    source: str = "Entity"
    target: str = "Entity"


def _model_fields(model: type[BaseModel]) -> list[dict[str, str]]:
    return [
        {"name": name, "description": info.description or name}
        for name, info in model.model_fields.items()
    ]


def serialize_ontology(
    entities: dict[str, type[BaseModel]] | None,
    edges: dict[str, tuple[type[BaseModel], list[EntityEdgeSourceTarget]]] | None,
) -> dict[str, Any]:
    return {
        "entity_types": [
            {
                "name": name,
                "description": (model.__doc__ or "").strip(),
                "attributes": _model_fields(model),
            }
            for name, model in (entities or {}).items()
        ],
        "edge_types": [
            {
                "name": name,
                "description": (model.__doc__ or "").strip(),
                "attributes": _model_fields(model),
                "source_targets": [
                    {"source": st.source, "target": st.target} for st in source_targets
                ],
            }
            for name, (model, source_targets) in (edges or {}).items()
        ],
    }


def _build_model(name: str, description: str, attributes: list[dict[str, str]]) -> type[BaseModel]:
    fields = {
        attr["name"]: (Optional[str], Field(default=None, description=attr["description"]))
        for attr in attributes
    }
    model = create_model(name, __base__=BaseModel, **fields)
    model.__doc__ = description or f"A {name}."
    return model


@dataclass
class GraphitiOntology:
    entity_types: dict[str, type[BaseModel]] | None = None
    edge_types: dict[str, type[BaseModel]] | None = None
    edge_type_map: dict[tuple[str, str], list[str]] | None = None


def ontology_to_graphiti(ontology: dict[str, Any] | None) -> GraphitiOntology:
    if not ontology:
        return GraphitiOntology()
    entity_types = {
        e["name"]: _build_model(e["name"], e.get("description", ""), e.get("attributes", []))
        for e in ontology.get("entity_types", [])
    }
    edge_types: dict[str, type[BaseModel]] = {}
    edge_type_map: dict[tuple[str, str], list[str]] = {}
    for e in ontology.get("edge_types", []):
        edge_types[e["name"]] = _build_model(
            "".join(part.capitalize() for part in e["name"].split("_")) or e["name"],
            e.get("description", ""),
            e.get("attributes", []),
        )
        for st in e.get("source_targets") or [{"source": "Entity", "target": "Entity"}]:
            key = (st.get("source") or "Entity", st.get("target") or "Entity")
            edge_type_map.setdefault(key, []).append(e["name"])
    return GraphitiOntology(
        entity_types=entity_types or None,
        edge_types=edge_types or None,
        edge_type_map=edge_type_map or None,
    )


# ---------------------------------------------------------------------------
# Plain result records (attribute names match the zep_cloud SDK objects)
# ---------------------------------------------------------------------------


@dataclass
class GraphRecord:
    graph_id: str
    name: str | None = None
    description: str | None = None
    created_at: str | None = None


@dataclass
class NodeRecord:
    uuid_: str
    name: str
    labels: list[str]
    summary: str
    attributes: dict[str, Any]
    created_at: str | None = None


@dataclass
class EdgeRecord:
    uuid_: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    attributes: dict[str, Any]
    episodes: list[str]
    created_at: str | None = None
    valid_at: str | None = None
    invalid_at: str | None = None
    expired_at: str | None = None


@dataclass
class EpisodeRecord:
    uuid_: str
    processed: bool
    content: str | None = None
    source_description: str | None = None
    created_at: str | None = None
    status: str | None = None
    error: str | None = None


@dataclass
class SearchResults:
    edges: list[EdgeRecord] = field(default_factory=list)
    nodes: list[NodeRecord] = field(default_factory=list)


@dataclass
class RawResponse:
    data: list[Any]
    headers: dict[str, str]


@dataclass
class BatchItemRecord:
    sequence_index: int
    episode_uuid: str
    source_uuid: str
    status: str = "pending"
    error: str | None = None


@dataclass
class BatchProgress:
    percent_complete: float
    succeeded_items: int
    failed_items: int
    total_items: int


@dataclass
class BatchRecord:
    batch_id: str
    metadata: dict[str, Any]
    status: str
    progress: BatchProgress


@dataclass
class BatchListPage:
    batches: list[BatchRecord]
    next_cursor: int | None = None


@dataclass
class BatchItemPage:
    items: list[BatchItemRecord]
    next_cursor: int | None = None


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            parsed = datetime.now(timezone.utc)
    else:
        parsed = datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _node_record(node: Any) -> NodeRecord:
    return NodeRecord(
        uuid_=node.uuid,
        name=node.name or "",
        labels=list(node.labels or []),
        summary=node.summary or "",
        attributes=dict(node.attributes or {}),
        created_at=_iso(node.created_at),
    )


def _edge_record(edge: Any) -> EdgeRecord:
    return EdgeRecord(
        uuid_=edge.uuid,
        name=edge.name or "",
        fact=edge.fact or "",
        source_node_uuid=edge.source_node_uuid,
        target_node_uuid=edge.target_node_uuid,
        attributes=dict(edge.attributes or {}),
        episodes=[str(e) for e in (edge.episodes or [])],
        created_at=_iso(edge.created_at),
        valid_at=_iso(edge.valid_at),
        invalid_at=_iso(edge.invalid_at),
        expired_at=_iso(edge.expired_at),
    )


# ---------------------------------------------------------------------------
# Embedding / reranking clients
# ---------------------------------------------------------------------------


def _build_embedder():
    from graphiti_core.embedder.client import EmbedderClient

    if Config.EMBEDDING_API_KEY:
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

        return OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key=Config.EMBEDDING_API_KEY,
                base_url=Config.EMBEDDING_BASE_URL,
                embedding_model=Config.EMBEDDING_MODEL_NAME,
                embedding_dim=Config.EMBEDDING_DIM,
            )
        )

    class LocalEmbedder(EmbedderClient):
        """In-process ONNX embeddings via fastembed (no API key required)."""

        def __init__(self, model_name: str):
            from fastembed import TextEmbedding

            logger.info("Loading local embedding model %s", model_name)
            self._model = TextEmbedding(model_name)
            self._lock = threading.Lock()

        def _embed(self, texts: list[str]) -> list[list[float]]:
            with self._lock:
                return [vector.tolist() for vector in self._model.embed(texts)]

        async def create(self, input_data):
            if isinstance(input_data, str):
                texts = [input_data]
            else:
                texts = [str(item) for item in input_data]
            return (await asyncio.to_thread(self._embed, texts))[0]

        async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
            if not input_data_list:
                return []
            return await asyncio.to_thread(self._embed, list(input_data_list))

    return LocalEmbedder(Config.EMBEDDING_MODEL_NAME)


def _build_cross_encoder():
    from graphiti_core.cross_encoder.client import CrossEncoderClient

    class PassthroughReranker(CrossEncoderClient):
        """Keeps the hybrid-search order; avoids per-passage LLM calls."""

        async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
            total = len(passages)
            return [(passage, float(total - index)) for index, passage in enumerate(passages)]

    return PassthroughReranker()


def _build_llm_client():
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

    from pydantic import ValidationError

    class TolerantJSONClient(OpenAIGenericClient):
        """Repairs common json_object-mode mistakes before Graphiti parses them.

        In json_object mode the schema is only described in the prompt, and
        some providers (DeepSeek) occasionally answer with the schema's shape,
        e.g. ``{"properties": {...actual fields...}}``. Unwrap that, and turn
        any remaining schema mismatch into a JSONDecodeError so Graphiti's
        built-in retry re-asks instead of failing the whole episode.
        """

        async def _generate_response(self, messages, response_model=None, *args, **kwargs):
            result = await super()._generate_response(messages, response_model, *args, **kwargs)
            if response_model is None or not isinstance(result, dict):
                return result
            fields = set(response_model.model_fields)
            nested = result.get("properties")
            if (
                "properties" not in fields
                and isinstance(nested, dict)
                and not (result.keys() & fields)
            ):
                result = nested
            # Graphiti stores some responses (entity attributes) verbatim as
            # node properties, so undeclared keys must not leak through.
            result = {key: value for key, value in result.items() if key in fields}
            try:
                response_model.model_validate(result)
            except ValidationError as error:
                raise json.JSONDecodeError(
                    f"response does not match {response_model.__name__}: {error.errors()[:3]}",
                    json.dumps(result, ensure_ascii=False)[:200],
                    0,
                ) from error
            return result

    model = Config.GRAPH_LLM_MODEL_NAME or Config.LLM_MODEL_NAME
    return TolerantJSONClient(
        config=LLMConfig(
            api_key=Config.LLM_API_KEY,
            base_url=Config.LLM_BASE_URL,
            model=model,
            small_model=model,
            max_tokens=Config.GRAPH_LLM_MAX_TOKENS,
        ),
        max_tokens=Config.GRAPH_LLM_MAX_TOKENS,
        structured_output_mode=Config.GRAPH_LLM_STRUCTURED_OUTPUT,
    )


# ---------------------------------------------------------------------------
# Background engine
# ---------------------------------------------------------------------------


@dataclass
class _Job:
    graph_id: str
    episode_uuids: list[str]
    bulk: bool


class _GraphEngine:
    """Owns the event loop, Graphiti instance and per-graph ingestion queues."""

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="graph-memory-loop", daemon=True
        )
        self._thread.start()
        self._graphiti = None
        self._init_lock: asyncio.Lock | None = None
        self._queues: dict[str, asyncio.Queue] = {}
        self._workers: dict[str, asyncio.Task] = {}
        self._state_lock = threading.Lock()
        # episode uuid -> (status, error)
        self._episode_state: dict[str, tuple[str, str | None]] = {}
        self._batches: dict[str, dict[str, Any]] = {}

    # -- loop plumbing -----------------------------------------------------

    def run(self, coro: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    async def graphiti(self):
        if self._graphiti is not None:
            return self._graphiti
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
        async with self._init_lock:
            if self._graphiti is None:
                from graphiti_core import Graphiti
                from graphiti_core.driver.neo4j_driver import Neo4jDriver

                driver = Neo4jDriver(
                    uri=Config.NEO4J_URI,
                    user=Config.NEO4J_USER,
                    password=Config.NEO4J_PASSWORD,
                    database=Config.NEO4J_DATABASE,
                )
                graphiti = Graphiti(
                    graph_driver=driver,
                    llm_client=_build_llm_client(),
                    embedder=_build_embedder(),
                    cross_encoder=_build_cross_encoder(),
                    max_coroutines=Config.GRAPH_MAX_COROUTINES,
                )
                await graphiti.build_indices_and_constraints()
                await driver.execute_query(
                    f"CREATE CONSTRAINT mirofish_graph_id IF NOT EXISTS "
                    f"FOR (g:{GRAPH_LABEL}) REQUIRE g.graph_id IS UNIQUE"
                )
                self._graphiti = graphiti
                logger.info("Graphiti connected to %s", Config.NEO4J_URI)
        return self._graphiti

    async def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        graphiti = await self.graphiti()
        records, _, _ = await graphiti.driver.execute_query(cypher, **params)
        return [dict(record) for record in records]

    # -- graph metadata ----------------------------------------------------

    async def get_graph_row(self, graph_id: str) -> dict[str, Any]:
        rows = await self.query(
            f"MATCH (g:{GRAPH_LABEL} {{graph_id: $graph_id}}) RETURN g {{.*}} AS g",
            graph_id=graph_id,
        )
        if not rows:
            raise NotFoundError(f"graph not found: {graph_id}")
        return rows[0]["g"]

    async def get_ontology(self, graph_id: str) -> GraphitiOntology:
        row = await self.get_graph_row(graph_id)
        raw = row.get("ontology")
        return ontology_to_graphiti(json.loads(raw) if raw else None)

    # -- episode state -----------------------------------------------------

    def set_state(self, episode_uuid: str, status: str, error: str | None = None) -> None:
        with self._state_lock:
            self._episode_state[episode_uuid] = (status, error)

    def get_state(self, episode_uuid: str) -> tuple[str, str | None] | None:
        with self._state_lock:
            return self._episode_state.get(episode_uuid)

    async def save_episode(
        self,
        graph_id: str,
        episode_uuid: str,
        content: str,
        source_description: str,
        reference_time: datetime,
    ) -> None:
        from graphiti_core.nodes import EpisodeType, EpisodicNode

        graphiti = await self.graphiti()
        episode = EpisodicNode(
            uuid=episode_uuid,
            name=f"episode-{episode_uuid[:8]}",
            group_id=graph_id,
            labels=[],
            source=EpisodeType.text,
            content=content,
            source_description=source_description,
            created_at=datetime.now(timezone.utc),
            valid_at=reference_time,
        )
        await episode.save(graphiti.driver)

    def enqueue(self, job: _Job) -> None:
        for episode_uuid in job.episode_uuids:
            self.set_state(episode_uuid, "queued")
        self._loop.call_soon_threadsafe(self._enqueue_in_loop, job)

    def _enqueue_in_loop(self, job: _Job) -> None:
        queue = self._queues.get(job.graph_id)
        if queue is None:
            queue = asyncio.Queue()
            self._queues[job.graph_id] = queue
        queue.put_nowait(job)
        worker = self._workers.get(job.graph_id)
        if worker is None or worker.done():
            self._workers[job.graph_id] = self._loop.create_task(
                self._worker(job.graph_id, queue)
            )

    async def _worker(self, graph_id: str, queue: asyncio.Queue) -> None:
        while not queue.empty():
            job: _Job = await queue.get()
            try:
                await self._process(job)
            except Exception as error:  # noqa: BLE001 - recorded per episode
                logger.exception("Graph ingestion failed for graph %s", graph_id)
                for episode_uuid in job.episode_uuids:
                    self.set_state(episode_uuid, "failed", f"{type(error).__name__}: {error}")
            finally:
                queue.task_done()

    async def _process(self, job: _Job) -> None:
        from graphiti_core.nodes import EpisodeType, EpisodicNode
        from graphiti_core.utils.bulk_utils import RawEpisode

        graphiti = await self.graphiti()
        try:
            ontology = await self.get_ontology(job.graph_id)
        except NotFoundError:
            for episode_uuid in job.episode_uuids:
                self.set_state(episode_uuid, "failed", "graph was deleted")
            return

        episodes = await EpisodicNode.get_by_uuids(graphiti.driver, job.episode_uuids)
        by_uuid = {episode.uuid: episode for episode in episodes}
        ordered = [by_uuid[u] for u in job.episode_uuids if u in by_uuid]
        for missing in set(job.episode_uuids) - set(by_uuid):
            self.set_state(missing, "failed", "episode not found")

        if job.bulk and BULK_SLICE_SIZE > 1 and len(ordered) > 1:
            for start in range(0, len(ordered), BULK_SLICE_SIZE):
                chunk = ordered[start:start + BULK_SLICE_SIZE]
                for episode in chunk:
                    self.set_state(episode.uuid, "processing")
                try:
                    await graphiti.add_episode_bulk(
                        [
                            RawEpisode(
                                name=episode.name,
                                uuid=episode.uuid,
                                content=episode.content,
                                source_description=episode.source_description,
                                source=EpisodeType.text,
                                reference_time=episode.valid_at,
                            )
                            for episode in chunk
                        ],
                        group_id=job.graph_id,
                        entity_types=ontology.entity_types,
                        edge_types=ontology.edge_types,
                        edge_type_map=ontology.edge_type_map,
                    )
                except Exception as error:  # noqa: BLE001
                    logger.exception("Bulk ingestion slice failed for graph %s", job.graph_id)
                    for episode in chunk:
                        self.set_state(episode.uuid, "failed", f"{type(error).__name__}: {error}")
                    continue
                for episode in chunk:
                    self.set_state(episode.uuid, "succeeded")
            return

        for episode in ordered:
            self.set_state(episode.uuid, "processing")
            try:
                previous = await graphiti.retrieve_episodes(
                    episode.valid_at, group_ids=[job.graph_id]
                )
                await graphiti.add_episode(
                    name=episode.name,
                    episode_body=episode.content,
                    source_description=episode.source_description,
                    reference_time=episode.valid_at,
                    source=EpisodeType.text,
                    group_id=job.graph_id,
                    uuid=episode.uuid,
                    previous_episode_uuids=[p.uuid for p in previous if p.uuid != episode.uuid],
                    entity_types=ontology.entity_types,
                    edge_types=ontology.edge_types,
                    edge_type_map=ontology.edge_type_map,
                )
            except Exception as error:  # noqa: BLE001
                logger.exception("Episode ingestion failed: %s", episode.uuid)
                self.set_state(episode.uuid, "failed", f"{type(error).__name__}: {error}")
                continue
            self.set_state(episode.uuid, "succeeded")

    # -- batches -------------------------------------------------------------

    def new_batch(self, metadata: dict[str, Any]) -> str:
        batch_id = uuid_lib.uuid4().hex
        with self._state_lock:
            self._batches[batch_id] = {
                "metadata": dict(metadata or {}),
                "items": [],
                "status": "draft",
            }
        return batch_id

    def batch(self, batch_id: str) -> dict[str, Any]:
        with self._state_lock:
            batch = self._batches.get(batch_id)
        if batch is None:
            raise NotFoundError(f"batch not found: {batch_id}")
        return batch

    def batch_ids(self) -> list[str]:
        with self._state_lock:
            return list(self._batches)


_engine: _GraphEngine | None = None
_engine_lock = threading.Lock()


def _get_engine() -> _GraphEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = _GraphEngine()
        return _engine


# ---------------------------------------------------------------------------
# Zep-shaped client facade
# ---------------------------------------------------------------------------


def _translate_errors(error: Exception) -> Exception:
    from graphiti_core.errors import (
        EdgeNotFoundError,
        GroupsEdgesNotFoundError,
        GroupsNodesNotFoundError,
        NodeNotFoundError,
    )

    if isinstance(error, (NodeNotFoundError, EdgeNotFoundError)):
        return NotFoundError(str(error))
    if isinstance(error, (GroupsEdgesNotFoundError, GroupsNodesNotFoundError)):
        return NotFoundError(str(error))
    return error


class _Base:
    def __init__(self, engine: _GraphEngine, timeout: float):
        self._engine = engine
        self._timeout = timeout

    def _run(self, coro: Coroutine[Any, Any, T]) -> T:
        try:
            return self._engine.run(coro, timeout=self._timeout)
        except GraphMemoryError:
            raise
        except Exception as error:
            translated = _translate_errors(error)
            if translated is error:
                raise
            raise translated from error


class _PagedLister(_Base):
    """Implements ``get_by_graph_id`` with uuid cursors (Zep header style)."""

    def __init__(self, engine: _GraphEngine, timeout: float, kind: str):
        super().__init__(engine, timeout)
        self._kind = kind
        self.with_raw_response = self

    def get_by_graph_id(self, graph_id: str, limit: int = 100, cursor: str | None = None):
        items = self._run(self._list(graph_id, limit, cursor))
        headers = {}
        if len(items) == limit and items:
            headers["zep-next-cursor"] = items[-1].uuid_
        return RawResponse(data=items, headers=headers)

    async def _list(self, graph_id: str, limit: int, cursor: str | None):
        from graphiti_core.edges import EntityEdge
        from graphiti_core.errors import GroupsEdgesNotFoundError
        from graphiti_core.nodes import EntityNode

        await self._engine.get_graph_row(graph_id)
        graphiti = await self._engine.graphiti()
        if self._kind == "node":
            nodes = await EntityNode.get_by_group_ids(
                graphiti.driver, [graph_id], limit=limit, uuid_cursor=cursor
            )
            return [_node_record(node) for node in nodes]
        try:
            edges = await EntityEdge.get_by_group_ids(
                graphiti.driver, [graph_id], limit=limit, uuid_cursor=cursor
            )
        except GroupsEdgesNotFoundError:
            return []
        return [_edge_record(edge) for edge in edges]


class _NodeNamespace(_PagedLister):
    def __init__(self, engine: _GraphEngine, timeout: float):
        super().__init__(engine, timeout, "node")

    def get(self, uuid_: str) -> NodeRecord:
        async def _get():
            from graphiti_core.nodes import EntityNode

            graphiti = await self._engine.graphiti()
            return _node_record(await EntityNode.get_by_uuid(graphiti.driver, uuid_))

        return self._run(_get())

    def get_edges(self, node_uuid: str) -> list[EdgeRecord]:
        async def _get():
            from graphiti_core.edges import EntityEdge

            graphiti = await self._engine.graphiti()
            edges = await EntityEdge.get_by_node_uuid(graphiti.driver, node_uuid)
            return [_edge_record(edge) for edge in edges]

        return self._run(_get())


class _EdgeNamespace(_PagedLister):
    def __init__(self, engine: _GraphEngine, timeout: float):
        super().__init__(engine, timeout, "edge")


class _EpisodeNamespace(_Base):
    def get(self, uuid_: str) -> EpisodeRecord:
        state = self._engine.get_state(uuid_)

        async def _get():
            from graphiti_core.nodes import EpisodicNode

            graphiti = await self._engine.graphiti()
            return await EpisodicNode.get_by_uuid(graphiti.driver, uuid_)

        try:
            episode = self._run(_get())
        except Exception as error:
            if state is None:
                raise NotFoundError(f"episode not found: {uuid_}") from error
            episode = None

        if state is None:
            status, error_message = "succeeded", None
        else:
            status, error_message = state
        if status == "failed":
            # Surface ingestion failures instead of letting pollers spin forever.
            raise GraphMemoryError(f"episode {uuid_} ingestion failed: {error_message}")
        return EpisodeRecord(
            uuid_=uuid_,
            processed=status == "succeeded",
            content=getattr(episode, "content", None),
            source_description=getattr(episode, "source_description", None),
            created_at=_iso(getattr(episode, "created_at", None)),
            status=status,
            error=error_message,
        )


class _GraphNamespace(_Base):
    def __init__(self, engine: _GraphEngine, timeout: float):
        super().__init__(engine, timeout)
        self.node = _NodeNamespace(engine, timeout)
        self.edge = _EdgeNamespace(engine, timeout)
        self.episode = _EpisodeNamespace(engine, timeout)

    def create(self, graph_id: str, name: str | None = None, description: str | None = None):
        from graphiti_core.helpers import validate_group_id

        validate_group_id(graph_id)

        async def _create():
            rows = await self._engine.query(
                f"MATCH (g:{GRAPH_LABEL} {{graph_id: $graph_id}}) RETURN g.graph_id AS id",
                graph_id=graph_id,
            )
            if rows:
                raise BadRequestError(f"graph already exists: {graph_id}")
            created_at = datetime.now(timezone.utc).isoformat()
            await self._engine.query(
                f"CREATE (g:{GRAPH_LABEL} {{graph_id: $graph_id, name: $name, "
                f"description: $description, created_at: $created_at}})",
                graph_id=graph_id,
                name=name or graph_id,
                description=description or "",
                created_at=created_at,
            )
            return GraphRecord(graph_id, name, description, created_at)

        return self._run(_create())

    def get(self, graph_id: str) -> GraphRecord:
        row = self._run(self._engine.get_graph_row(graph_id))
        return GraphRecord(
            graph_id=row["graph_id"],
            name=row.get("name"),
            description=row.get("description"),
            created_at=row.get("created_at"),
        )

    def delete(self, graph_id: str) -> None:
        async def _delete():
            from graphiti_core.nodes import Node

            await self._engine.get_graph_row(graph_id)
            graphiti = await self._engine.graphiti()
            await Node.delete_by_group_id(graphiti.driver, graph_id)
            await self._engine.query(
                f"MATCH (g:{GRAPH_LABEL} {{graph_id: $graph_id}}) DETACH DELETE g",
                graph_id=graph_id,
            )

        self._run(_delete())

    def set_ontology(
        self,
        graph_ids: list[str],
        entities: dict[str, type[BaseModel]] | None = None,
        edges: dict[str, tuple[type[BaseModel], list[EntityEdgeSourceTarget]]] | None = None,
    ) -> None:
        payload = json.dumps(serialize_ontology(entities, edges), ensure_ascii=False)
        # Fail before writing anything if Graphiti would reject the types.
        from graphiti_core.utils.ontology_utils.entity_types_utils import validate_entity_types

        validate_entity_types(ontology_to_graphiti(json.loads(payload)).entity_types)

        async def _set():
            for graph_id in graph_ids:
                await self._engine.get_graph_row(graph_id)
                await self._engine.query(
                    f"MATCH (g:{GRAPH_LABEL} {{graph_id: $graph_id}}) SET g.ontology = $ontology",
                    graph_id=graph_id,
                    ontology=payload,
                )

        self._run(_set())

    def add(
        self,
        graph_id: str,
        data: str,
        type: str = "text",  # noqa: A002 - zep_cloud keyword
        created_at: str | None = None,
        source_description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EpisodeRecord:
        if not data or not data.strip():
            raise BadRequestError("episode data must not be empty")
        episode_uuid = str(uuid_lib.uuid4())

        async def _add():
            await self._engine.get_graph_row(graph_id)
            await self._engine.save_episode(
                graph_id,
                episode_uuid,
                data,
                source_description or "MiroFish episode",
                _parse_time(created_at),
            )

        self._run(_add())
        self._engine.enqueue(_Job(graph_id, [episode_uuid], bulk=False))
        return EpisodeRecord(uuid_=episode_uuid, processed=False, content=data, status="queued")

    def search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
        reranker: str | None = None,  # accepted for compatibility; RRF is used
        **_: Any,
    ) -> SearchResults:
        from graphiti_core.search.search_config_recipes import (
            EDGE_HYBRID_SEARCH_RRF,
            NODE_HYBRID_SEARCH_RRF,
        )

        if scope not in {"edges", "nodes"}:
            raise BadRequestError(f"unsupported search scope: {scope}")
        recipe = EDGE_HYBRID_SEARCH_RRF if scope == "edges" else NODE_HYBRID_SEARCH_RRF
        config = recipe.model_copy(deep=True)
        config.limit = limit

        async def _search():
            await self._engine.get_graph_row(graph_id)
            graphiti = await self._engine.graphiti()
            results = await graphiti.search_(query, config=config, group_ids=[graph_id])
            return SearchResults(
                edges=[_edge_record(edge) for edge in results.edges],
                nodes=[_node_record(node) for node in results.nodes],
            )

        return self._run(_search())


class _BatchNamespace(_Base):
    """Process-local stand-in for Zep's Batch API used by document ingestion."""

    def create(self, metadata: dict[str, Any] | None = None) -> BatchRecord:
        batch_id = self._engine.new_batch(metadata or {})
        return self.get(batch_id)

    def add(self, batch_id: str, items: list[Any]) -> list[BatchItemRecord]:
        batch = self._engine.batch(batch_id)
        if batch["status"] != "draft":
            raise BadRequestError(f"batch {batch_id} is not a draft")
        added = []
        for item in items:
            episode_uuid = str(uuid_lib.uuid4())
            record = BatchItemRecord(
                sequence_index=len(batch["items"]),
                episode_uuid=episode_uuid,
                source_uuid=episode_uuid,
            )
            batch["items"].append((record, item))
            added.append(record)
        return added

    def process(self, batch_id: str) -> BatchRecord:
        batch = self._engine.batch(batch_id)
        if batch["status"] != "draft":
            return self.get(batch_id)
        entries = list(batch["items"])
        by_graph: dict[str, list[str]] = {}

        async def _save_all():
            for record, item in entries:
                graph_id = item.graph_id
                await self._engine.get_graph_row(graph_id)
                await self._engine.save_episode(
                    graph_id,
                    record.episode_uuid,
                    item.data,
                    item.source_description or "MiroFish document chunk",
                    _parse_time(getattr(item, "created_at", None)),
                )
                by_graph.setdefault(graph_id, []).append(record.episode_uuid)

        # Saving many episodes can exceed the per-request read timeout.
        self._engine.run(_save_all(), timeout=None)
        batch["status"] = "queued"
        for graph_id, episode_uuids in by_graph.items():
            self._engine.enqueue(_Job(graph_id, episode_uuids, bulk=True))
        return self.get(batch_id)

    def _refresh(self, batch: dict[str, Any]) -> tuple[str, BatchProgress]:
        records = [record for record, _ in batch["items"]]
        if batch["status"] != "draft":
            for record in records:
                state = self._engine.get_state(record.episode_uuid)
                if state is not None:
                    record.status, record.error = state
        total = len(records)
        succeeded = sum(1 for r in records if r.status == "succeeded")
        failed = sum(1 for r in records if r.status == "failed")
        status = batch["status"]
        if status != "draft" and total:
            if succeeded == total:
                status = "succeeded"
            elif succeeded + failed == total:
                status = "failed" if succeeded == 0 else "partial"
            elif any(r.status == "processing" for r in records) or succeeded or failed:
                status = "processing"
            else:
                status = "queued"
            batch["status"] = status
        percent = (100.0 * (succeeded + failed) / total) if total else 0.0
        return status, BatchProgress(percent, succeeded, failed, total)

    def get(self, batch_id: str) -> BatchRecord:
        batch = self._engine.batch(batch_id)
        status, progress = self._refresh(batch)
        return BatchRecord(batch_id, dict(batch["metadata"]), status, progress)

    def list(self, limit: int = 100, cursor: int | None = None) -> BatchListPage:
        ids = self._engine.batch_ids()
        start = cursor or 0
        page = ids[start:start + limit]
        next_cursor = start + limit if start + limit < len(ids) else None
        return BatchListPage([self.get(batch_id) for batch_id in page], next_cursor)

    def list_items(self, batch_id: str, limit: int = 100, cursor: int | None = None) -> BatchItemPage:
        batch = self._engine.batch(batch_id)
        self._refresh(batch)
        records = [record for record, _ in batch["items"]]
        start = cursor or 0
        page = records[start:start + limit]
        next_cursor = start + limit if start + limit < len(records) else None
        return BatchItemPage(page, next_cursor)


@dataclass
class BatchAddItem:
    """Same shape as ``zep_cloud.BatchAddItem`` for the fields MiroFish uses."""

    graph_id: str
    data: str
    type: str = "graph_episode"
    data_type: str = "text"
    source_description: str | None = None
    created_at: str | None = None
    metadata: dict[str, Any] | None = None


class GraphMemoryClient:
    """Entry point: ``client.graph.*`` and ``client.batch.*``."""

    def __init__(self, timeout: float = 60.0):
        engine = _get_engine()
        self.graph = _GraphNamespace(engine, timeout)
        self.batch = _BatchNamespace(engine, timeout)


def check_connection(timeout: float = 30.0) -> None:
    """Connect to Neo4j (and build indices) or raise."""

    engine = _get_engine()
    engine.run(engine.query("RETURN 1 AS ok"), timeout=timeout)
