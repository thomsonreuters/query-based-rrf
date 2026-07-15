"""
LLM ambiguity-labeling utilities, used by 3-query-features-analysis.py.

Calls a plain OpenAI-compatible chat completions API with a prompt adapted
from CLAMBER (ACL 2024, arXiv:2405.12063) to score each query with an
ambiguity category: 0 = not ambiguous, 1-8 = which CLAMBER category best
applies.

Processing is chunked (default 50 queries/chunk) with concurrency bounded by
an asyncio.Semaphore within each chunk, and each chunk is handed back via an
optional `on_chunk` callback as soon as it completes -- so callers labeling
thousands of queries can checkpoint to disk incrementally instead of losing
all progress if the run fails partway through.

Configuration is entirely via environment variables, so this module has no
dependency on any particular provider or internal service:
    OPENAI_API_KEY   required.
    OPENAI_BASE_URL  optional -- point this at any OpenAI-compatible endpoint
                     (Azure OpenAI, a self-hosted proxy, vLLM, etc.) instead
                     of the default api.openai.com.

`_build_client()` / `_label_chunk()` below are one concrete implementation
(a plain OpenAI-compatible async client) -- not the only correct shape. If
your setup already has its own LLM-calling infra (e.g. this repo's
experiment/llm_backend.py BedrockBackend/LocalQwen3Backend/LocalMistralBackend,
or any other in-house client/gateway), swap those two functions for that
infra instead of routing through OPENAI_API_KEY/OPENAI_BASE_URL. Everything
else in this file (prompt template, chunking/concurrency/checkpointing,
label parsing) is infra-agnostic and shouldn't need to change.
"""
import asyncio
import os
import re
from typing import Callable, Optional

import pandas as pd

AMBIGUITY_PROMPT_TEMPLATE = """Given a query first identify whether the question is ambiguous or not. If it
is ambiguous, identify which of the following categories best describes it.
If it is not ambiguous, the category is 0.

1. Unfamiliar -- Query contains unfamiliar entities or facts.
   Example: "Find the price of Samsung Chromecast"
2. Contradiction -- Query contains self-contradictions.
3. Lexical -- Query contains terms with multiple meanings.
   Example: "Tell me about the source of Nile"
4. Semantic -- Query lacks of context leading multiple interpretations.
   Example: "When did he land on the moon?"
5. Who -- Query output contains confusion due to missing personal elements.
   Example: "Suggest me some gifts for my mother"
6. When -- Query output contains confusion due to missing temporal elements.
   Example: "How many goals did Argentina score in the World Cup?"
7. Where -- Query output contains confusion due to missing spatial elements.
   Example: "Tell me how to reach New York"
8. What -- Query output contains confusion due to missing task-specific
   elements.
   Example: "Real name of gwen stacy in spiderman?"

The response should start with the ambiguity analysis of the question and
then follow by the matching category number (0 through 8) as the final
character of the response, with nothing after it.

like this:
Ambiguity analysis: The query is ambiguous because it lacks context and could be interpreted in multiple
Ambiguity Category: 4

Query: "{query_text}"
"""

# Model IDs accepted by --ambiguity-model. Passed straight through as the
# `model` argument to the chat completions call, so any model your configured
# endpoint (OPENAI_BASE_URL) serves under this name will work.
MODEL_CHOICES = {"gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1", "gpt-4o", "gpt-5-mini", "gpt-5"}

VALID_LABELS = set("012345678")


def parse_ambiguity_label(response_text: str):
    """Extract the digit (0-8) from the LLM response, per the prompt's
    instruction that it be the final character. Tolerates trailing
    punctuation/whitespace the model sometimes adds after the digit (e.g.
    "...category number is 1." or "...is 1.\n"). Falls back to a narrow
    search of just the last 20 characters for a standalone digit (handles
    e.g. "...ambiguity is 5 (Who)." where a short explanatory aside follows
    the digit -- observed to need only ~8 chars of lookback in practice) --
    deliberately not searching the whole response, since an earlier-
    mentioned, non-final category number could otherwise be mistaken for
    the answer. Returns None if no valid digit is found."""
    stripped = (response_text or "").strip().rstrip(".!?\"')]} \t\n")
    if stripped and stripped[-1] in VALID_LABELS:
        return int(stripped[-1])

    tail = (response_text or "")[-20:]
    matches = re.findall(r"(?<!\d)([0-8])(?!\d)", tail)
    if matches:
        return int(matches[-1])
    return None


def _build_client():
    """Build a plain OpenAI-compatible async client from environment variables."""
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it (optionally also set "
            "OPENAI_BASE_URL to route through a different OpenAI-compatible "
            "endpoint) before calling label_queries()."
        )
    return AsyncOpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL"))


def _generation_params(model_name: str) -> dict:
    """gpt-5-family (and other reasoning) models don't support a `temperature`
    parameter and spend part of the token budget on invisible internal
    reasoning before the visible answer, so they get a larger token cap and
    no temperature kwarg; everything else gets temperature=0.0 (deterministic)
    and a smaller cap."""
    if model_name.startswith(("gpt-5", "o1", "o3")):
        return {"max_completion_tokens": 2000}
    return {"temperature": 0.0, "max_tokens": 400}


def _chunk(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


async def _label_chunk(client, model_name: str, chunk_df: pd.DataFrame,
                        semaphore: asyncio.Semaphore) -> pd.DataFrame:
    async def bounded_invoke(query_text: str) -> str:
        prompt = AMBIGUITY_PROMPT_TEMPLATE.format(query_text=query_text)
        async with semaphore:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                **_generation_params(model_name),
            )
        return response.choices[0].message.content

    # return_exceptions=True turns any failed call into an Exception object in
    # `responses` instead of aborting the whole chunk -- caught below and recorded
    # as a None label rather than crashing the run. This assumes failures surface
    # as a raised exception from client.chat.completions.create(), which is how
    # the OpenAI SDK's async client behaves. If _build_client()/bounded_invoke are
    # swapped for a different infra (see module docstring), check that infra's
    # failure mode still raises on error (rather than e.g. returning an error
    # payload with a 200-equivalent success status) -- otherwise failures would
    # silently be treated as successful (non-ambiguous-looking) responses here.
    responses = await asyncio.gather(
        *[bounded_invoke(t) for t in chunk_df["query_text"]], return_exceptions=True
    )

    rows = []
    for (qid, text), response in zip(zip(chunk_df["query_id"], chunk_df["query_text"]), responses):
        if isinstance(response, Exception):
            rows.append({
                "query_id": qid, "query_text": text,
                "ambiguity_category": None, "raw_response": f"ERROR: {response}",
            })
            continue
        rows.append({
            "query_id": qid, "query_text": text,
            "ambiguity_category": parse_ambiguity_label(response),
            "raw_response": response,
        })
    return pd.DataFrame(rows)


async def label_queries_async(
    queries_df: pd.DataFrame,
    model_name: str = "gpt-4o-mini",
    concurrency: int = 5,
    chunk_size: int = 50,
    on_chunk: Optional[Callable[[pd.DataFrame], None]] = None,
) -> pd.DataFrame:
    """Label every row of queries_df (expects query_id, query_text columns).

    Processes in chunks of `chunk_size`, with up to `concurrency` calls in
    flight at once within each chunk. If `on_chunk` is given, it's called
    with each chunk's result DataFrame as soon as that chunk finishes --
    callers can use this to checkpoint to disk incrementally for large runs.
    Returns the full concatenated result regardless.
    """
    client = _build_client()
    semaphore = asyncio.Semaphore(concurrency)

    all_chunks = []
    indices = list(range(len(queries_df)))
    for chunk_indices in _chunk(indices, chunk_size):
        chunk_df = queries_df.iloc[chunk_indices]
        result = await _label_chunk(client, model_name, chunk_df, semaphore)
        all_chunks.append(result)
        if on_chunk is not None:
            on_chunk(result)

    return pd.concat(all_chunks, ignore_index=True) if all_chunks else pd.DataFrame(
        columns=["query_id", "query_text", "ambiguity_category", "raw_response"]
    )


def label_queries(
    queries_df: pd.DataFrame,
    model_name: str = "gpt-4o-mini",
    concurrency: int = 5,
    chunk_size: int = 50,
    on_chunk: Optional[Callable[[pd.DataFrame], None]] = None,
) -> pd.DataFrame:
    """Sync wrapper around label_queries_async, for use from non-async scripts."""
    return asyncio.run(
        label_queries_async(queries_df, model_name, concurrency, chunk_size, on_chunk)
    )
