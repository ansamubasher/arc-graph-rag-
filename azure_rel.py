from openai import AzureOpenAI

import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("ENDPOINT_URL")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

from cosmos_reader import CosmosReader
from chunker import Chunker
from azure_extraction import EntityExtractor


class RelationshipExtractor:

    # Fixed, reusable relationship vocabulary — same rationale as
    # ALLOWED_TYPES in the entity extractor: an open-ended relationship
    # field means the same connection gets named differently across
    # chunks (RENTS vs LEASES vs OCCUPIES), which breaks graph merging.
    ALLOWED_RELATIONSHIPS = [
        "HAS_TENANT",
        "HAS_LANDLORD",
        "LEASES",
        "LOCATED_IN",       # Property -> Location
        "PART_OF",          # e.g. Suite 130 -> The Building
        "HAS_RATE",         # Property/Duration -> Rate
        "HAS_VALUE",        # Money/MoneyTerm -> Money
        "HAS_DEADLINE",     # Clause/Property -> Date/DateTerm
        "EFFECTIVE_ON",     # Clause/Term -> Date
        "APPLIES_TO",       # Duration/Rate -> Property
        "REFERENCES",       # generic fallback link between two named things
        "OTHER",
    ]

    def __init__(self):

        missing = [
            name
            for name, value in {
                "AZURE_OPENAI_ENDPOINT/ENDPOINT_URL": AZURE_OPENAI_ENDPOINT,
                "AZURE_OPENAI_KEY/AZURE_OPENAI_API_KEY": AZURE_OPENAI_KEY,
                "AZURE_OPENAI_DEPLOYMENT/DEPLOYMENT_NAME": AZURE_OPENAI_DEPLOYMENT,
            }.items()
            if not value
        ]

        if missing:
            raise ValueError(
                "Missing required environment variables: " + ", ".join(missing)
            )

        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION
        )

    def _build_prompt(self, chunk, entities):

        rel_list = "\n".join(f"- {r}" for r in self.ALLOWED_RELATIONSHIPS)

        return f"""You are given a list of entities already extracted from a piece of text,
and the original text itself.

Identify relationships between these entities.

Rules:
Rules:
1. Use ONLY these relationship types:
{rel_list}
   Do NOT use "OTHER" or invent new relationship names. If none of these
   types accurately and specifically describes the connection, omit the
   relationship entirely rather than forcing a weak fit.

2. "source" and "target" must EXACTLY match an "entity" value from the
   Entities list below (same spelling/casing). Do not introduce new
   entities that aren't in that list.

3. source and target must never be the same entity. Never output a
   relationship where source == target.

4. Only extract a relationship if the text explicitly and directly states
   that connection between THESE TWO SPECIFIC entities. Two entities
   appearing in the same chunk, sentence, or paragraph is NOT sufficient
   evidence of a relationship — the text must grammatically or
   semantically connect them (e.g. "X applies during Y," "X is part of
   Y," "X equals Y"). Do not infer a relationship just because one entity
   is salient/frequent in the chunk. If you are not confident, omit it
   rather than including a weak guess.

5. If an entity could plausibly take multiple HAS_VALUE or HAS_RATE
   relationships from this chunk (e.g. restated, superseded, or
   alternative figures), emit only the one the text identifies as
   current/operative for the stated period or condition. Do not emit
   more than one HAS_VALUE and more than one HAS_RATE per (source,
   target-context) unless the text clearly distinguishes them as applying
   to different time periods or conditions — in that case, make the
   distinction explicit by including the qualifying period/condition as
   part of the relationship (e.g. target should read "$2.15 (Months
   97-108)" not just "$2.15").

6. Normalize numeric/percentage phrasing to a single consistent form
   (e.g. "3%" not "three percent"; pick the numeral form). Do not emit
   the same fact twice in two different textual forms.

7. Do not duplicate a relationship that is already implied by the
   "related_to" field on an entity (e.g. if entity A already has
   related_to: B, you don't need to also output A REFERENCES B unless the
   text specifies a more precise relationship type than that).

8. Avoid duplicate (source, relationship, target) triples in your output.

Return ONLY a valid JSON array, no markdown fences, no commentary. Example:

[
  {{"source": "John Smith", "relationship": "LEASES", "target": "Apartment 2B"}},
  {{"source": "Lease001", "relationship": "HAS_TENANT", "target": "John Smith"}}
]

Entities:

{json.dumps(entities, indent=2)}

Text:

{chunk["text"]}
"""

    @staticmethod
    def _clean_json(raw):
        """Strip markdown code fences / stray text some models add despite instructions."""
        raw = raw.strip()
        match = re.search(r"```(?:json)?\s*(\[.*\])\s*```", raw, re.DOTALL)
        if match:
            return match.group(1)
        match = re.search(r"(\[.*\])", raw, re.DOTALL)
        if match:
            return match.group(1)
        return raw

    def extract(self, chunk, entities):

        if not entities:
            # nothing to relate — skip the API call entirely
            return []

        prompt = self._build_prompt(chunk, entities)

        try:
            response = self.client.chat.completions.create(

                model=AZURE_OPENAI_DEPLOYMENT,

                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You extract relationships between entities in "
                            "legal/real-estate text. Return only a valid JSON "
                            "array matching the requested schema. No markdown "
                            "fences, no commentary."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=0

            )
        except Exception as e:
            print(f"    [warn] Azure OpenAI API call failed for chunk {chunk.get('chunk_id')}: {e}")
            return []

        raw = response.choices[0].message.content
        cleaned = self._clean_json(raw)

        try:
            relationships = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"    [warn] Failed to parse JSON for chunk {chunk.get('chunk_id')}: {e}")
            print(f"    [warn] Raw response: {raw[:200]}...")
            return []

        if not isinstance(relationships, list):
            print(f"    [warn] Expected a JSON array, got {type(relationships)} for chunk {chunk.get('chunk_id')}")
            return []

        # build a lookup of valid entity names from this chunk so we can
        # drop any hallucinated source/target that isn't in the entity list
        valid_entity_names = {e.get("entity") for e in entities if isinstance(e, dict)}

        normalized = []
        seen = set()

        for r in relationships:
            if not isinstance(r, dict):
                continue

            source = r.get("source")
            target = r.get("target")
            relationship = r.get("relationship")

            if not source or not target or not relationship:
                continue

            if source not in valid_entity_names or target not in valid_entity_names:
                # hallucinated entity not in our extracted list — drop it
                continue

            if relationship not in self.ALLOWED_RELATIONSHIPS:
                relationship = "OTHER"

            key = (source, relationship, target)
            if key in seen:
                continue
            seen.add(key)

            normalized.append({
                "source": source,
                "relationship": relationship,
                "target": target,
            })

        return normalized


def main():

    reader = CosmosReader()
    chunker = Chunker()
    entity_extractor = EntityExtractor()
    relationship_extractor = RelationshipExtractor()

    for organization_id in reader.get_organization_ids():

        print(f"\nProcessing organization: {organization_id}")

        documents = reader.load_documents_by_organization(organization_id)

        for document in documents:

            chunks = chunker.chunk_document(document)

            print(f"\n  Document: {document['id']} — {len(chunks)} chunks")

            for chunk in chunks:

                entities = entity_extractor.extract(chunk)

                relationships = relationship_extractor.extract(chunk, entities)

                print(f"\n    Chunk: {chunk['chunk_id']}")
                print(f"    Entities extracted: {len(entities)}")
                print(f"    Relationships extracted: {len(relationships)}")

                for relationship in relationships:
                    print(
                        f"      - {relationship['source']} "
                        f"--[{relationship['relationship']}]--> "
                        f"{relationship['target']}"
                    )


if __name__ == "__main__":
    main()