import json
import os
import math
from concurrent.futures import ThreadPoolExecutor

from openai import AzureOpenAI
from config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
)


class GlobalSearch:
    """
    Global search over community reports (map-reduce), for questions that
    span the whole document/graph rather than being anchored to a specific
    entity — e.g. "summarize the key lease terms" or "what are the main
    risks in this document?" GraphQA's local search would fail on these
    because there's no single entity to seed a keyword match from.

    Map step: ask the LLM for a partial answer + relevance score (0-100)
    from each community report, in small batches.
    Reduce step: keep the highest-scoring partial answers and ask the LLM
    to synthesize them into one final answer.

    Requires community_builder.py to have been run first for the
    organization (produces graphs/{organization_id}_communities.json).
    """

    def __init__(self, graph_folder="graphs", batch_size=20, top_k=5, max_reports=120, filter_limit=30):
        self.graph_folder = graph_folder
        self.batch_size = batch_size
        self.top_k = top_k
        self.max_reports = max_reports
        self.filter_limit = filter_limit
        self.reports = []
        self._current_graph_id = None
        self.report_embeddings = None

        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version="2024-02-01",
        )

    def load_reports(self, graph_id):
        self._current_graph_id = graph_id

        path = os.path.join(
            self.graph_folder,
            f"{graph_id}_communities.json"
        )

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No community reports found at {path}. "
                f"Run community_builder.py for this organization first."
            )

        with open(path) as f:
            self.reports = json.load(f)

        print(f"Loaded {len(self.reports)} community reports.")

        # Backward compatibility: extract document_ids from graphml if missing
        if any("document_ids" not in r for r in self.reports):
            try:
                print("Older community JSON loaded without document_ids. Attempting to parse them from the graphml file...")
                import networkx as nx
                graph_path = os.path.join(self.graph_folder, f"{graph_id}.graphml")
                if os.path.exists(graph_path):
                    graph = nx.read_graphml(graph_path)
                    if not graph.is_multigraph():
                        graph = nx.MultiDiGraph(graph)
                    
                    node_to_doc = {}
                    for n, data in graph.nodes(data=True):
                        disp = data.get("display_name", n)
                        chunks_val = data.get("source_chunks")
                        doc_ids = set()
                        if chunks_val:
                            chunk_list = []
                            if isinstance(chunks_val, list):
                                chunk_list = chunks_val
                            elif isinstance(chunks_val, str):
                                chunk_list = [c.strip() for c in chunks_val.split(",") if c.strip()]
                            for chunk_id in chunk_list:
                                if "_" in chunk_id:
                                    doc_ids.add(chunk_id.rsplit("_", 1)[0])
                                else:
                                    doc_ids.add(chunk_id)
                        node_to_doc[disp] = doc_ids
                        node_to_doc[n] = doc_ids

                    for r in self.reports:
                        if "document_ids" not in r:
                            doc_ids = set()
                            for m in r.get("members", []):
                                if m in node_to_doc:
                                    doc_ids.update(node_to_doc[m])
                            r["document_ids"] = sorted(list(doc_ids))
            except ImportError:
                print("[warn] networkx is not installed in this Python environment. Please run the script using the virtual environment: .venv\\Scripts\\python global_search.py")
            except Exception as e:
                print(f"[warn] Failed to parse document_ids from graphml dynamically: {e}")

    def _require_reports(self):
        if not self.reports:
            raise ValueError("Community reports not loaded. Call load_reports() first.")

    def _get_embeddings(self, texts):
        """Fetches embeddings for a list of texts in a single batch request."""
        if not AZURE_OPENAI_EMBEDDING_DEPLOYMENT:
            raise ValueError("AZURE_OPENAI_EMBEDDING_DEPLOYMENT is not configured.")
        try:
            response = self.client.embeddings.create(
                model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"    [warn] Failed to fetch embeddings: {e}")
            raise e

    @staticmethod
    def _cosine_similarity(v1, v2):
        dot = sum(x * y for x, y in zip(v1, v2))
        norm1 = math.sqrt(sum(x * x for x in v1))
        norm2 = math.sqrt(sum(x * x for x in v2))
        if norm1 > 0 and norm2 > 0:
            return dot / (norm1 * norm2)
        return 0.0

    @staticmethod
    def _compute_tf_idf_similarity(query, documents):
        """
        Computes a simple TF-IDF similarity between a query string and a list of document strings.
        No external dependencies required.
        """
        def tokenize(text):
            return [word.lower().strip(".,;:!?\"'()[]{}") for word in text.split() if len(word.strip(".,;:!?\"'()[]{}")) > 2]

        query_tokens = tokenize(query)
        if not query_tokens:
            return [0.0] * len(documents)

        doc_tokens_list = [tokenize(doc) for doc in documents]
        doc_count = len(documents)
        df = {}
        for doc_tokens in doc_tokens_list:
            unique_tokens = set(doc_tokens)
            for token in unique_tokens:
                df[token] = df.get(token, 0) + 1

        idf = {}
        for token in set(query_tokens):
            idf[token] = math.log((doc_count + 1) / (df.get(token, 0) + 0.5)) + 1.0

        query_tf = {}
        for token in query_tokens:
            query_tf[token] = query_tf.get(token, 0) + 1
        query_vector = {token: tf * idf[token] for token, tf in query_tf.items()}
        query_norm = math.sqrt(sum(val ** 2 for val in query_vector.values()))

        scores = []
        for doc_tokens in doc_tokens_list:
            if not doc_tokens:
                scores.append(0.0)
                continue
            
            dot_product = 0.0
            doc_sum_sq = 0.0
            for token in set(doc_tokens):
                tf = doc_tokens.count(token)
                token_idf = idf.get(token, 1.0)
                doc_weight = tf * token_idf
                doc_sum_sq += doc_weight ** 2
                if token in query_vector:
                    dot_product += query_vector[token] * doc_weight

            doc_norm = math.sqrt(doc_sum_sq)
            if query_norm > 0 and doc_norm > 0:
                scores.append(dot_product / (query_norm * doc_norm))
            else:
                scores.append(0.0)

        return scores

    def _filter_reports(self, question, reports, limit=30):
        """
        Filters reports to the top `limit` most relevant to the question.
        Tries semantic search first (using Azure OpenAI embeddings if configured),
        falling back to TF-IDF keyword similarity on failure or if not configured.
        """
        if not reports:
            return []
        
        doc_texts = [
            f"{r.get('title', '')} {r.get('summary', '')}"
            for r in reports
        ]

        use_semantic = bool(AZURE_OPENAI_EMBEDDING_DEPLOYMENT)

        if use_semantic:
            try:
                print("Using Azure OpenAI semantic embedding filter...")
                q_emb = self._get_embeddings([question])[0]
                
                # Fetch community report embeddings (only if not cached already)
                if getattr(self, "_cached_embeddings_graph_id", None) != self._current_graph_id or getattr(self, "report_embeddings", None) is None:
                    print(f"Generating embeddings for {len(reports)} community reports...")
                    self.report_embeddings = self._get_embeddings(doc_texts)
                    self._cached_embeddings_graph_id = self._current_graph_id

                scores = [self._cosine_similarity(q_emb, r_emb) for r_emb in self.report_embeddings]
                
            except Exception as e:
                print(f"[warn] Semantic filtering failed, falling back to keyword filter. Error: {e}")
                use_semantic = False

        if not use_semantic:
            print("Using pure-Python TF-IDF keyword filter...")
            scores = self._compute_tf_idf_similarity(question, doc_texts)

        scored_reports = list(zip(scores, reports))
        scored_reports.sort(key=lambda x: x[0], reverse=True)

        print(f"Top 5 matched communities:")
        for score, r in scored_reports[:5]:
            print(f"  - [Score: {score:.3f}] Community {r.get('community_id')}: {r.get('title')}")

        return [r for score, r in scored_reports[:limit]]
 
    @staticmethod
    def _batches(items, size):
        for i in range(0, len(items), size):
            yield items[i:i + size]

    def _rank_reports(self, reports):
        """Prefer higher-rated, larger communities first for large-graph questions."""

        def _sort_key(report):
            return (
                report.get("rating", 0),
                report.get("size", 0),
                len(report.get("members", [])),
            )

        return sorted(reports, key=_sort_key, reverse=True)

    def _map_batch(self, question, batch):
        """
        Sends one batch of community reports to the LLM, asking for a
        partial answer per report plus a 0-100 relevance score. Batching
        keeps this cheap even with many communities.
        """

        reports_text = "\n\n".join(
            f"[Community {r['community_id']}] {r.get('title', '')}\n"
            f"Rating: {r.get('rating')}\n"
            f"Size: {r.get('size')}\n"
            f"Members: {', '.join(str(m) for m in r.get('members', []))}\n"
            f"Summary: {r.get('summary', '')}"
            for r in batch
        )

        prompt = f"""You are answering a question using summaries of different clusters
    ("communities") of a knowledge graph. For EACH community below, decide
    whether it contains information relevant to the question.

    This is a LARGE-GRAPH question. Be selective: only return a partial_answer
    for communities that are clearly relevant and high-value. If a community is
    only weakly related, return an empty partial_answer and score it 0.
    Prefer fewer, stronger partial answers over many weak ones.

Question: {question}

Community summaries:

{reports_text}

IMPORTANT NAME-PRESERVATION RULES:
- If you include a fact in "partial_answer", include the specific entity
    names from the source summary (tenant names, lease IDs, property/suite
    names, named clauses/terms, etc.).
- Do NOT replace named entities with generic wording like "the tenant",
    "a property", "multiple parties", or "several terms" when names are present.
- Use only names present in the provided community summary; do not invent names.

DETAIL-COVERAGE RULES (MANDATORY):
- Use ALL available fields for each community: title, summary, and members.
- If the question asks about entities (for example tenants/properties/leases),
    include every matching named entity found in that community.
- For each included entity, extract every available attribute from the input
    (values, rates, dates, durations, sections, obligations, related terms).
- If a detail is missing in the source, explicitly say "not specified in source"
    instead of guessing.
- Prefer concrete facts over short generic statements.

LARGE-GRAPH ANSWERING RULES:
- Do not try to summarize every community in the batch.
- Only keep communities that materially help answer the question.
- When the graph is large, prioritize the top entities/terms and ignore
    low-importance details unless the user explicitly asks for exhaustive coverage.

Return ONLY a valid JSON array, one object per community, no markdown
fences, no commentary:

[
  {{"community_id": <int>, "partial_answer": "<answer using only this community's info, or empty string if not relevant>", "score": <0-100 relevance score>}}
]
"""

        try:
            response = self.client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You extract partial answers from cluster "
                            "summaries of a knowledge graph. Return only "
                            "valid JSON. Preserve specific entity names in "
                            "partial answers; do not generalize them away. "
                            "Use title, summary, and members to produce "
                            "dense, specific partial answers with all "
                            "available attributes from each relevant "
                            "community."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
        except Exception as e:
            print(f"    [warn] map step failed for batch: {e}")
            return []

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"    [warn] failed to parse map step JSON: {e}")
            return []

        if not isinstance(parsed, list):
            return []

        return [
            p for p in parsed
            if isinstance(p, dict) and p.get("partial_answer") and p.get("score", 0) > 0
        ]

    def _reduce(self, question, partial_answers):
        """Synthesizes the top-scoring partial answers into one final answer."""

        top = sorted(partial_answers, key=lambda p: p.get("score", 0), reverse=True)[: self.top_k]

        if not top:
            return "I couldn't find information relevant to this question across the document's communities."

        report_map = {r["community_id"]: r for r in self.reports}
        combined_parts = []
        for p in top:
            cid = p.get("community_id")
            report = report_map.get(cid)
            doc_ids = report.get("document_ids", []) if report else []
            if doc_ids:
                doc_source_label = ", ".join(doc_ids)
            else:
                doc_source_label = f"Community {cid}"
            
            combined_parts.append(
                f"[Document: {doc_source_label}] (relevance {p['score']}) {p['partial_answer']}"
            )
        combined = "\n\n".join(combined_parts)

        prompt = f"""Using ONLY the partial answers below (each drawn from a different
part of the document), write one coherent final answer to the question.
Resolve any contradictions by preferring higher-relevance partial answers.
If the partial answers don't actually answer the question, say so.

This is a LARGE-GRAPH answer: keep the response focused on the most important
entities, terms, and differences. Do not attempt to cover every minor detail
unless the question explicitly asks for exhaustive coverage.

OUTPUT FORMATTING REQUIREMENTS (MANDATORY):
- Return clean Markdown only (no code fences unless explicitly useful).
- CITATIONS: Throughout your answer, you must cite the source Document IDs (e.g., [Document: doc_id_1], [Document: doc_id_2]) at the end of sentences where facts from those documents are referenced. Use the exact Document IDs provided in the sources.
- Start with a short direct answer section.
- Then provide a "Key Points" section using bullet points.
- When multiple entities/terms are being compared (e.g., tenants,
    properties, leases, clauses, risks), include a comparison table.
- After the table, provide a "Detailed Breakdown" section with one
    subsection per entity so each item is covered in detail.
- Use clear headings, concise wording, and readable spacing.

LARGE-GRAPH FINAL-ANSWER RULES:
- Prefer a concise executive summary first.
- Include only the top relevant entities or terms.
- If the question is broad, group results by document, tenant, or property
    instead of listing every low-value community.
- If there are too many items, explicitly say that you are focusing on the
    most important ones.

TENANT-SPECIFIC REQUIREMENT:
- If the question is about tenants (or tenant-related terms), include
    EVERY tenant name that appears in the partial answers.
- Do not omit any tenant found in the source partial answers.
- For each tenant, include all available details from the source (for
    example lease IDs, property/suite names, dates, obligations, financial
    terms, risks/issues, and any notable clauses).

IMPORTANT NAME-PRESERVATION RULES:
- Retain and cite specific entity names from the partial answers in the
    final answer (e.g., tenant names, lease IDs, property/suite names).
- Do NOT collapse specific names into abstract/general language when the
    names are available.
- If multiple specific entities are relevant, list them explicitly.

Question: {question}

Partial answers:

{combined}

Final answer:"""

        try:
            response = self.client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You synthesize partial answers drawn from a "
                            "document's knowledge graph into one final "
                            "answer. Do not use outside knowledge. Preserve "
                            "specific entity names from the source partial "
                            "answers; do not generalize them away. Cite the "
                            "source documents using [Document: <doc_id>] inline "
                            "where applicable. Return well-structured Markdown "
                            "with headings, bullet points, and comparison "
                            "tables when applicable. If tenant-related, "
                            "include every tenant present in the provided "
                            "partial answers with detailed coverage for each."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
        except Exception as e:
            return f"[error] Azure OpenAI call failed during reduce step: {e}"

        final_text = response.choices[0].message.content.strip()

        return final_text

    def answer(self, question, graph_id=None):

        if graph_id:
            self.load_reports(graph_id)

        self._require_reports()

        # Step 1: Pre-filter reports to only the most relevant ones
        filtered_reports = self._filter_reports(question, self.reports, limit=self.filter_limit)

        # Step 2: Run batch mapping in parallel using ThreadPoolExecutor
        partial_answers = []
        batches = list(self._batches(filtered_reports, self.batch_size))
        
        print(f"Mapping {len(filtered_reports)} reports in {len(batches)} batch(es) concurrently...")

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._map_batch, question, batch)
                for batch in batches
            ]
            for future in futures:
                try:
                    res = future.result()
                    if res:
                        partial_answers.extend(res)
                except Exception as e:
                    print(f"    [warn] Batch mapping failed: {e}")

        final_answer = self._reduce(question, partial_answers)

        return {
            "answer": final_answer,
            "partial_answers": partial_answers,
        }


def main():

    graph_id = input("Graph ID: ").strip()

    search = GlobalSearch()
    search.load_reports(graph_id)

    while True:

        question = input("\nQuestion (or 'exit'): ").strip()

        if question.lower() == "exit":
            break

        result = search.answer(question)

        print(f"\nUsed {len(result['partial_answers'])} relevant community(ies).")
        print(f"\nAnswer:\n{result['answer']}")
        print("-" * 40)


if __name__ == "__main__":
    main()