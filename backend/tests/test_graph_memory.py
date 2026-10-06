import pytest

from app.utils import graph_memory
from app.utils.graph_memory import (
    BatchAddItem,
    EntityEdgeSourceTarget,
    _BatchNamespace,
    _build_model,
    ontology_to_graphiti,
    serialize_ontology,
)


def test_ontology_round_trip_builds_graphiti_types():
    person = _build_model("Person", "A human being.", [
        {"name": "role", "description": "Public role"},
    ])
    works_for = _build_model("WorksFor", "Employment.", [])
    serialized = serialize_ontology(
        {"Person": person},
        {"WORKS_FOR": (works_for, [EntityEdgeSourceTarget("Person", "Company")])},
    )

    assert serialized["entity_types"] == [{
        "name": "Person",
        "description": "A human being.",
        "attributes": [{"name": "role", "description": "Public role"}],
    }]

    ontology = ontology_to_graphiti(serialized)
    assert set(ontology.entity_types) == {"Person"}
    model = ontology.entity_types["Person"]
    assert model.__doc__ == "A human being."
    assert model().role is None
    assert ontology.edge_type_map == {("Person", "Company"): ["WORKS_FOR"]}


def test_empty_ontology_passes_none_to_graphiti():
    ontology = ontology_to_graphiti(None)
    assert ontology.entity_types is None
    assert ontology.edge_types is None
    assert ontology.edge_type_map is None


class FakeEngine:
    def __init__(self):
        self.states = {}
        self.batches = {}
        self.enqueued = []

    def new_batch(self, metadata):
        batch_id = f"batch-{len(self.batches)}"
        self.batches[batch_id] = {"metadata": metadata, "items": [], "status": "draft"}
        return batch_id

    def batch(self, batch_id):
        if batch_id not in self.batches:
            raise graph_memory.NotFoundError(batch_id)
        return self.batches[batch_id]

    def batch_ids(self):
        return list(self.batches)

    def get_state(self, episode_uuid):
        return self.states.get(episode_uuid)

    def run(self, coro, timeout=None):
        coro.close()

    def enqueue(self, job):
        self.enqueued.append(job)


@pytest.fixture
def batches():
    engine = FakeEngine()
    return engine, _BatchNamespace(engine, timeout=1)


def test_batch_status_tracks_item_states(batches):
    engine, api = batches
    batch = api.create(metadata={"graph_id": "g"})
    items = api.add(
        batch_id=batch.batch_id,
        items=[BatchAddItem(graph_id="g", data="a"), BatchAddItem(graph_id="g", data="b")],
    )
    assert [item.sequence_index for item in items] == [0, 1]
    assert api.get(batch.batch_id).status == "draft"

    engine.batches[batch.batch_id]["status"] = "queued"
    assert api.get(batch.batch_id).status == "queued"

    engine.states[items[0].episode_uuid] = ("succeeded", None)
    engine.states[items[1].episode_uuid] = ("processing", None)
    summary = api.get(batch.batch_id)
    assert summary.status == "processing"
    assert summary.progress.succeeded_items == 1

    engine.states[items[1].episode_uuid] = ("failed", "boom")
    assert api.get(batch.batch_id).status == "partial"
    listed = api.list_items(batch_id=batch.batch_id).items
    assert [item.status for item in listed] == ["succeeded", "failed"]
    assert listed[1].error == "boom"

    engine.states[items[1].episode_uuid] = ("succeeded", None)
    assert api.get(batch.batch_id).status == "succeeded"


def test_batch_add_after_processing_is_rejected(batches):
    engine, api = batches
    batch = api.create(metadata={})
    engine.batches[batch.batch_id]["status"] = "queued"
    with pytest.raises(graph_memory.BadRequestError):
        api.add(batch_id=batch.batch_id, items=[BatchAddItem(graph_id="g", data="x")])


def test_unknown_batch_is_not_found(batches):
    _, api = batches
    with pytest.raises(graph_memory.NotFoundError):
        api.get("missing")
