"""
Audience crowd generator.

The knowledge graph only contains the people and organisations named in the
uploaded documents (usually 10-40). To model how an *audience* reacts, this
service adds ordinary participants: an LLM first plans audience segments from
the scenario and the graph entities (reader archetypes described in the
documents become segment templates), then creates varied members for each
segment in small parallel batches.

Crowd members become regular OASIS agents. They have no node in the knowledge
graph; their ``source_entity_uuid`` is synthetic (``crowd_...``).
"""

from __future__ import annotations

import concurrent.futures
import json
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI

from ..config import Config
from ..utils.locale import get_language_instruction, get_locale, set_locale
from ..utils.logger import get_logger
from ..utils.openai_chat_compat import create_chat_completion, extract_chat_completion_text
from .oasis_profile_generator import OasisAgentProfile, OasisProfileGenerator
from .zep_entity_reader import EntityNode

logger = get_logger("mirofish.crowd")

CROWD_ENTITY_TYPE = "AudienceMember"
MAX_CROWD_SIZE = 1000
BATCH_SIZE = 8


@dataclass
class AudienceSegment:
    name: str
    description: str
    count: int


def _json_from_response(response: Any) -> Dict[str, Any]:
    content = extract_chat_completion_text(response).strip()
    if content.startswith("```"):
        content = content.strip("`")
        content = content[content.find("{"):]
    return json.loads(content)


class CrowdGenerator:
    def __init__(self, profile_generator: Optional[OasisProfileGenerator] = None):
        self.client = OpenAI(api_key=Config.LLM_API_KEY, base_url=Config.LLM_BASE_URL)
        self.model_name = Config.LLM_MODEL_NAME
        self.profile_generator = profile_generator or OasisProfileGenerator()

    # ------------------------------------------------------------------ LLM

    def _chat_json(self, system: str, prompt: str, temperature: float = 0.8) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                response = create_chat_completion(
                    self.client,
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=max(0.3, temperature - attempt * 0.2),
                )
                return _json_from_response(response)
            except Exception as error:  # noqa: BLE001 - retried, then raised
                last_error = error
                logger.warning("Crowd LLM call failed (attempt %s): %s", attempt + 1, str(error)[:200])
        raise RuntimeError(f"crowd generation LLM call failed: {last_error}")

    # ------------------------------------------------------------- planning

    @staticmethod
    def _entity_digest(entities: List[EntityNode], limit: int = 60) -> str:
        rows = []
        for entity in entities[:limit]:
            summary = (entity.summary or "").replace("\n", " ")[:300]
            rows.append(f"- {entity.name} ({entity.get_entity_type() or 'Entity'}): {summary}")
        return "\n".join(rows)

    def plan_segments(
        self,
        simulation_requirement: str,
        entities: List[EntityNode],
        total: int,
        context_text: Optional[str] = None,
    ) -> List[AudienceSegment]:
        system = (
            "You design audiences for social-media opinion simulations. Return valid JSON only.\n"
            + get_language_instruction()
        )
        audience_block = (
            f"\nAudience description provided by the user (follow its segments and shares when given):\n"
            f"{context_text[:15000]}\n"
            if context_text else ""
        )
        prompt = f"""Scenario / question to simulate:
{simulation_requirement}

Entities found in the source documents:
{self._entity_digest(entities) or "(none)"}
{audience_block}
Plan the ordinary audience that would see and react to this topic online: {total} people in total.
- Split them into 3-8 segments of ordinary people (readers, customers, citizens...).
- If some entities above describe typical audience members or reader types (composite
  personas, "typical reader", segments), base segments on them and keep their traits.
- Do NOT make segments out of specific named public figures or organisations; those
  already take part in the simulation as themselves.
- Segment sizes should reflect how large each group realistically is in this audience.

Return JSON:
{{"segments": [{{"name": "short segment name", "description": "who they are, situation, what they care about, typical attitude to the topic (2-4 sentences)", "count": <int>}}]}}
The counts must add up to {total}."""
        data = self._chat_json(system, prompt, temperature=0.5)
        segments = [
            AudienceSegment(
                name=str(item.get("name") or "Audience").strip()[:80],
                description=str(item.get("description") or "").strip(),
                count=max(0, int(item.get("count") or 0)),
            )
            for item in data.get("segments", [])
            if isinstance(item, dict)
        ]
        segments = [segment for segment in segments if segment.count > 0] or [
            AudienceSegment("Audience", "Ordinary people interested in the topic.", total)
        ]
        return self._rescale(segments, total)

    @staticmethod
    def _rescale(segments: List[AudienceSegment], total: int) -> List[AudienceSegment]:
        current = sum(segment.count for segment in segments)
        if current == total:
            return segments
        scaled = [max(1, round(segment.count * total / current)) for segment in segments]
        # Fix rounding drift on the largest segment.
        drift = total - sum(scaled)
        largest = max(range(len(scaled)), key=lambda i: scaled[i])
        scaled[largest] = max(1, scaled[largest] + drift)
        for segment, count in zip(segments, scaled):
            segment.count = count
        return segments

    # ----------------------------------------------------------- generation

    def _generate_batch(
        self,
        simulation_requirement: str,
        segment: AudienceSegment,
        count: int,
        batch_number: int,
    ) -> List[Dict[str, Any]]:
        system = (
            "You create realistic, diverse social-media users for an opinion simulation. "
            "Return valid JSON only; string values must not contain raw newlines.\n"
            + get_language_instruction()
        )
        prompt = f"""Scenario:
{simulation_requirement}

Audience segment: {segment.name}
{segment.description}

Create {count} different people from this segment (batch #{batch_number}; make them clearly
different from typical first picks: vary age, gender, city/region, exact occupation and
business size, income, family situation, online habits and temperament).
Their attitudes to the topic must vary realistically within the segment: some supportive,
some neutral or indifferent, some skeptical or critical — not everyone agrees.
Use realistic full names that fit the country and language of the scenario.

Return JSON:
{{"people": [{{
  "name": "full name",
  "age": <int>,
  "gender": "male" or "female",
  "mbti": "e.g. ISTJ",
  "country": "country",
  "profession": "occupation",
  "bio": "social-media bio, up to 200 characters",
  "persona": "who they are, situation, how they use social media, writing style, and their attitude to this topic with reasons (500-900 characters)",
  "interested_topics": ["topic", "..."]
}}]}}"""
        data = self._chat_json(system, prompt)
        people = data.get("people") or []
        return [person for person in people if isinstance(person, dict) and person.get("name")][:count]

    def generate(
        self,
        simulation_requirement: str,
        entities: List[EntityNode],
        total: int,
        start_user_id: int,
        parallel_count: int = 5,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        context_text: Optional[str] = None,
    ) -> Tuple[List[OasisAgentProfile], List[EntityNode]]:
        total = max(0, min(int(total), MAX_CROWD_SIZE))
        if total == 0:
            return [], []

        segments = self.plan_segments(simulation_requirement, entities, total, context_text=context_text)
        logger.info(
            "Crowd plan: %s",
            ", ".join(f"{segment.name}={segment.count}" for segment in segments),
        )

        jobs: List[Tuple[AudienceSegment, int, int]] = []
        for segment in segments:
            remaining, batch_number = segment.count, 1
            while remaining > 0:
                size = min(BATCH_SIZE, remaining)
                jobs.append((segment, size, batch_number))
                remaining -= size
                batch_number += 1

        locale = get_locale()
        results: List[Tuple[AudienceSegment, List[Dict[str, Any]]]] = []
        done = 0

        def run(job):
            set_locale(locale)
            segment, size, batch_number = job
            try:
                return segment, self._generate_batch(simulation_requirement, segment, size, batch_number)
            except Exception as error:  # noqa: BLE001 - one batch must not sink the crowd
                logger.warning("Crowd batch failed for segment %s: %s", segment.name, error)
                return segment, []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, parallel_count)) as pool:
            for segment, people in pool.map(run, jobs):
                results.append((segment, people))
                done += len(people)
                if progress_callback:
                    progress_callback(min(done, total), total, segment.name)

        profiles: List[OasisAgentProfile] = []
        crowd_entities: List[EntityNode] = []
        seen_names: Dict[str, int] = {}
        user_id = start_user_id
        for segment, people in results:
            for person in people:
                name = str(person.get("name")).strip()[:80]
                seen_names[name] = seen_names.get(name, 0) + 1
                if seen_names[name] > 1:
                    name = f"{name} {seen_names[name]}"
                entity_uuid = f"crowd_{uuid.uuid4().hex[:16]}"
                bio = str(person.get("bio") or "")[:300]
                try:
                    age = int(person.get("age")) if person.get("age") is not None else None
                except (TypeError, ValueError):
                    age = None
                profiles.append(
                    OasisAgentProfile(
                        user_id=user_id,
                        user_name=self.profile_generator._generate_username(name),
                        name=name,
                        bio=bio or segment.name,
                        persona=f"[{segment.name}] {person.get('persona') or segment.description}",
                        age=age,
                        gender=person.get("gender"),
                        mbti=person.get("mbti"),
                        country=person.get("country"),
                        profession=person.get("profession"),
                        interested_topics=person.get("interested_topics") or [],
                        source_entity_uuid=entity_uuid,
                        source_entity_type=CROWD_ENTITY_TYPE,
                    )
                )
                crowd_entities.append(
                    EntityNode(
                        uuid=entity_uuid,
                        name=name,
                        labels=["Entity", CROWD_ENTITY_TYPE],
                        summary=f"{segment.name}: {bio}",
                        attributes={"segment": segment.name},
                    )
                )
                user_id += 1

        logger.info("Crowd generated: %s of %s requested members", len(profiles), total)
        return profiles, crowd_entities
