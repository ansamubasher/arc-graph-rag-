from openai import AzureOpenAI

import json
import os
import re

from dotenv import load_dotenv

load_dotenv()


def _env(*keys):
    for key in keys:
        value = os.getenv(key)
        if value is None:
            continue
        value = value.strip()
        if value:
            return value
    return None


AZURE_OPENAI_ENDPOINT = _env("AZURE_OPENAI_ENDPOINT", "ENDPOINT_URL")
AZURE_OPENAI_KEY = _env("AZURE_OPENAI_KEY", "AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = _env("AZURE_OPENAI_DEPLOYMENT", "DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = _env("AZURE_OPENAI_API_VERSION") or "2025-01-01-preview"

from cosmos_reader import CosmosReader
from chunker import Chunker


class EntityExtractor:

    # Fixed, reusable type vocabulary so the same kind of thing always
    # gets the same label across chunks/documents. Extend as needed,
    # but keep it closed-ish — an open-ended type field is how you end
    # up with "Important Date" / "Date" / "Key Date" all meaning the
    # same thing.
    ALLOWED_TYPES = [
        "Party",          # person/org named in the doc (tenant, landlord, etc.)
        "Property",       # a physical space/unit/premises (e.g. "Suite 130", "Building 5")
        "Location",       # a city/state/address-level place, not a specific premises (e.g. "Newark, California")
        "Measurement",    # a quantity with a unit that is NOT money (e.g. "35,115 sqft", "148,848 square feet")
        "Date",           # an actual calendar date (must be a real value, not a label)
        "DateTerm",       # a *named* date/deadline referenced but not given a value in this chunk
        "Money",          # a dollar amount — only use if an actual numeric value is present in the text
        "MoneyTerm",      # a *named* money concept referenced without a value in this chunk (e.g. "Base Rent" with no figure given)
        "Rate",           # a per-unit value (e.g. $/sqft, %), not a flat amount
        "Duration",       # a span of time (e.g. "18 months", "Months 1-12")
        "Clause",         # a named contractual term/section/fee type that isn't a date or money concept
        "Other",
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

        # AzureOpenAI expects the resource endpoint (e.g. https://<name>.openai.azure.com/).
        if "services.ai.azure.com/api/projects" in AZURE_OPENAI_ENDPOINT:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT appears to be an AI Foundry project URL. "
                "Use the Azure OpenAI resource endpoint instead, e.g. "
                "https://<resource-name>.openai.azure.com/"
            )

        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION
        )

    def _build_prompt(self, chunk):

        types_list = "\n".join(f"- {t}" for t in self.ALLOWED_TYPES)

        return f"""Extract entities from the text below.

Rules:
1. Use ONLY these entity types: 
{types_list}

2. CRITICAL: Reject bare generic nouns as entity names.
     - NEVER use "Property", "Premises", "the Property", or other undefined terms as entity names.
     - ALWAYS extract the specific address, suite, building name, or defined-term label instead.
     - Example: if text says "the Property is located at 123 Main St" and later "the Property was built in 1990",
         extract entity name as "123 Main Street" (the specific referent), NOT "Property".
     - If a document defines "125 Main Street (herein called 'the Property')", capture:
         - entity: "125 Main Street"
         - aliases: ["the Property", "Property"]  (NEW FIELD — see rule 3)
     - This ensures distinct locations don't collapse into one node.

3. NEW: Add "aliases" field to capture alternative names/labels for the same entity.
     - Optional field; use only if the text explicitly defines synonyms or defined terms.
     - Example: if text says "Suite 130 (the Leased Premises)", emit:
         {{"entity": "Suite 130", "aliases": ["the Leased Premises", "Leased Premises"], ...}}
     - Aliases are hints for the graph builder to merge duplicate nodes later.

4. "entity" must be the canonical name of the thing itself (e.g. "2nd Floor Premises"),
   never a label that embeds a fact about it (e.g. NOT "Price per Square Foot (2nd Floor)").
   If a value belongs to an entity, put it in "value" and link it with "related_to",
   not in the entity name.

5. For Money and Rate types, always populate "value" (number) and "unit"
   (e.g. "USD", "USD/sqft", "%"). Do not put the number only in "entity".
   Only use Money/Rate if an actual numeric figure appears in the text. If a
   money-related concept is mentioned without a figure in this chunk (e.g.
   "Base Rent commences on..." with no dollar amount), use type "MoneyTerm"
   instead — do not fabricate a value.

   For percentages: "value" must be the plain number as written (e.g. "3%"
   -> value: 3, unit: "%"). Do NOT convert to a decimal fraction (e.g. do
   NOT use value: 0.03 for "3%"). The value should always match the digits
   shown in "entity".

6. For Date: only use this type if an actual date/value appears in the text
   (e.g. "January 1, 2027"). If the text only *names* a deadline or date term
   without giving its value (e.g. "TI Deadline", "Term Commencement Date"),
   use type "DateTerm" instead — do not fabricate or infer a date.

7. For Measurement: use this for any quantity + unit that is NOT money
   (e.g. "35,115 sqft", "148,848 square feet"). Do not type these as
   "Property" — Property is for named/addressable spaces, Measurement is
   for the quantity describing one.

7b. For allocations of a countable resource (parking spaces, parking
    passes, storage units, signage rights, etc.), extract entity as a
    clean label naming the resource — e.g. "Parking Spaces" or "Parking
    Passes" — and NEVER embed the count in the entity name (do NOT use
    "112 parking passes" as the entity name).
    - type: "Measurement"
    - value: the number (e.g. 112)
    - unit: the counting word as written in the text (e.g. "spaces", "passes")
    - related_to: REQUIRED — must be the specific Party or Property that
      the allocation belongs to. If the text does not make this
      unambiguous, still extract the entity but set related_to to null
      rather than guessing.

8. For Location: use this for general places (city, state, address-level
   text like "Newark, California"). Use "Property" only for a specific
   named premises/unit/building within the document (e.g. "Suite 130",
   "Building 5").

9. "related_to": required whenever an entity is clearly an attribute/value
   tied to another entity also extracted from this text — not optional
   when the link is obvious. In particular, for tabular data (e.g. a rent
   schedule with columns like Duration / Rate / Total), every Rate and
   Money entity in a row MUST be related_to the Duration or Property entity
   for that same row, so the row's entities can be reconstructed later.
   If unsure, prefer relating to the most specific entity in the same row
   over the most general one (e.g. relate a Money figure to a specific
   Duration like "Months 1-12", not just to the overall Property).

   IMPORTANT: base "related_to" on what the value actually modifies
   grammatically/semantically in the sentence — NOT on which entity is
   physically nearest in the text. Re-read the exact clause containing the
   number and identify what it is describing before choosing related_to.
   For example, if the text says "TI work must be completed within 18
   months of the Term Commencement Date", the Duration "18 months" relates
   to "Term Commencement Date" specifically, not to a different date term
   (like "Execution Date") that merely appears elsewhere in the same
   chunk. If the chunk contains multiple plausible targets and you cannot
   tell from the text which one is correct, set "related_to" to null rather
   than guessing based on proximity.

10. Do not invent entities not present in the text. Do not deduplicate across
   chunks — just extract what's in this text.

11. FILTER: Do NOT emit any entity with type "Other" — they are noise phrases.
    If an entity would be typed "Other", omit it entirely from the output.

Return ONLY a valid JSON array, no markdown fences, no commentary. Example:

[
    {{"entity": "123 Main Street", "type": "Property", "value": null, "unit": null, "related_to": null, "aliases": ["the Property"]}},
  {{"entity": "35,115 sqft", "type": "Measurement", "value": 35115, "unit": "sqft", "related_to": "2nd Floor Premises"}},
  {{"entity": "Newark, California", "type": "Location", "value": null, "unit": null, "related_to": null}},
  {{"entity": "$125/sqft", "type": "Rate", "value": 125, "unit": "USD/sqft", "related_to": "2nd Floor Premises"}},
  {{"entity": "TI Deadline", "type": "DateTerm", "value": null, "unit": null, "related_to": null}},
  {{"entity": "Base Rent", "type": "MoneyTerm", "value": null, "unit": null, "related_to": null}},
  {{"entity": "Months 1-12", "type": "Duration", "value": null, "unit": null, "related_to": null}},
  {{"entity": "$2.25", "type": "Rate", "value": 2.25, "unit": "USD/sqft", "related_to": "Months 1-12"}},
  {{"entity": "John Smith", "type": "Party", "value": null, "unit": null, "related_to": null}}
  {{"entity": "Parking Passes", "type": "Measurement", "value": 112, "unit": "passes", "related_to": "ABC Corp"}}
]

Text:

{chunk["text"]}
"""

    @staticmethod
    def _clean_json(raw):
        """Strip markdown code fences / stray text some models add despite instructions."""
        raw = raw.strip()
        # remove ```json ... ``` or ``` ... ``` wrappers
        match = re.search(r"```(?:json)?\s*(\[.*\])\s*```", raw, re.DOTALL)
        if match:
            return match.group(1)
        # fall back: grab the first [...] block in case of stray preamble
        match = re.search(r"(\[.*\])", raw, re.DOTALL)
        if match:
            return match.group(1)
        return raw

    def extract(self, chunk):

        prompt = self._build_prompt(chunk)

        response = self.client.chat.completions.create(

            model=AZURE_OPENAI_DEPLOYMENT,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract entities from legal/real-estate text. "
                        "Return only a valid JSON array matching the requested schema. "
                        "No markdown fences, no commentary."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0

        )

        raw = response.choices[0].message.content
        cleaned = self._clean_json(raw)

        try:
            entities = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"    [warn] Failed to parse JSON for chunk {chunk.get('chunk_id')}: {e}")
            print(f"    [warn] Raw response: {raw[:200]}...")
            return []

        if not isinstance(entities, list):
            print(f"    [warn] Expected a JSON array, got {type(entities)} for chunk {chunk.get('chunk_id')}")
            return []

        # normalize / guard against missing keys so downstream code doesn't break
        normalized = []
        for e in entities:
            if not isinstance(e, dict) or "entity" not in e:
                continue
            entity_type = e.get("type")
            if entity_type not in self.ALLOWED_TYPES:
                entity_type = "Other"

            # FILTER: Drop "Other" typed entities entirely — they are noise.
            if entity_type == "Other":
                continue

            value = e.get("value")
            unit = e.get("unit")

            # Safety net: some models still emit "3%" as value=0.03 despite
            # the prompt rule. If unit is "%" and value looks like a decimal
            # fraction (< 1), assume it was meant as a fraction and rescale
            # to match the percentage convention used elsewhere (value
            # should match the digits shown in "entity").
            if unit == "%" and isinstance(value, (int, float)) and 0 < value < 1:
                value = value * 100

            aliases = e.get("aliases", [])
            if not isinstance(aliases, list):
                aliases = [aliases] if aliases else []

            normalized.append({
                "entity": e.get("entity"),
                "type": entity_type,
                "value": value,
                "unit": unit,
                "related_to": e.get("related_to"),
                "aliases": aliases,
            })

        return normalized


def main():

    reader = CosmosReader()
    chunker = Chunker()
    extractor = EntityExtractor()

    for organization_id in reader.get_organization_ids():

        print(f"\nProcessing organization: {organization_id}")

        documents = reader.load_documents_by_organization(
            organization_id
        )

        for document in documents:

            chunks = chunker.chunk_document(document)

            print(
                f"\n  Document: {document['id']} — {len(chunks)} chunks"
            )

            for chunk in chunks:

                entities = extractor.extract(chunk)

                print(f"\n    Chunk: {chunk['chunk_id']}")
                print(f"    Text:  {chunk['text'][:80]}...")
                print(f"    Entities extracted: {len(entities)}")

                for entity in entities:
                    value_str = ""
                    if entity.get("value") is not None:
                        unit = entity.get("unit") or ""
                        value_str = f" = {entity['value']}{(' ' + unit) if unit else ''}"
                    rel_str = f" (related_to: {entity['related_to']})" if entity.get("related_to") else ""

                    print(
                        f"      - [{entity.get('type')}] {entity.get('entity')}{value_str}{rel_str}"
                    )


if __name__ == "__main__":
    main()