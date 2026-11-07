"""Core data model for the oMLX benchmark harness.

Everything that is persisted to a run bundle is defined here, together with a
stable ``SCHEMA_VERSION`` so that downstream comparison tools can migrate.

Schema 2.1 is the performance-only bundle shape: per-sample metrics plus
per-context overall/cell aggregates and an environment snapshot; accuracy
fields are gone.  ``overall`` rows are keyed by ``model`` + ``context_length``
so 1K/8K/32K buckets are never mixed into one average.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "2.1"


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class AnswerType(str, Enum):
    MC_LETTER = "mc_letter"          # multiple choice A/B/C/D
    NUMERIC = "numeric"              # final number matching (GSM8K)
    EXACT = "exact"                  # normalized exact match
    CODE = "code"                    # unit test execution
    CONSTRAINTS = "constraints"      # IFEval-style verifiable instructions
    TOOL_CALL = "tool_call"          # function calling structure
    SEARCH_ANSWER = "search_answer"  # answer via local search agent


class CacheMode(str, Enum):
    COLD = "cold"      # clear prefix/SSD cache before every measured request
    WARM = "warm"      # clear once per cell, then measure steady-state reuse
    NONE = "none"      # no management (for diagnostics only)


class RecordStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CONTAMINATED = "contaminated"  # cache hit detected in a cold measurement
    WARMUP = "warmup"


class Phase(str, Enum):
    WARMUP = "warmup"
    MEASURED = "measured"


# --------------------------------------------------------------------------- #
# Dataset / inference
# --------------------------------------------------------------------------- #


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ToolSpec(BaseModel):
    """OpenAI-style tool definition."""

    type: Literal["function"] = "function"
    function: Dict[str, Any]


class Sample(BaseModel):
    """A single dataset sample, normalized across sources."""

    sample_id: str
    dataset: str
    scenario: str
    subtask: Optional[str] = None
    context: Optional[str] = None          # long document / passages (if any)
    question: str
    choices: Optional[List[str]] = None    # e.g. ["A text", "B text", ...]
    answer: Optional[str] = None           # gold answer in canonical form
    answer_type: AnswerType
    tools: Optional[List[ToolSpec]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Statistics computed at load time (advisory; authoritative count is
    # produced at request-build time with the model tokenizer).
    context_chars: Optional[int] = None
    est_context_tokens: Optional[int] = None


class RequestSpec(BaseModel):
    """The exact request sent to the model (kept for reproducibility)."""

    model: str
    messages: List[ChatMessage]
    max_tokens: int
    temperature: float = 0.0
    top_p: Optional[float] = None
    seed: Optional[int] = None
    stream: bool = True
    tools: Optional[List[ToolSpec]] = None
    extra: Dict[str, Any] = Field(default_factory=dict)
    chat_template_kwargs: Optional[Dict[str, Any]] = None

    # Context bookkeeping.  Performance mode never targets a context length:
    # ``target_context_tokens`` stays None and the sample's native prompt is
    # used (truncated only when it exceeds the model window).
    target_context_tokens: Optional[int] = None
    actual_prompt_tokens: Optional[int] = None
    truncated: bool = False
    padded: bool = False
    context_meta: Dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    time_to_first_token: Optional[float] = None
    time_to_first_visible_token: Optional[float] = None
    generation_duration: Optional[float] = None
    prompt_eval_duration: Optional[float] = None
    generation_tokens_per_second: Optional[float] = None
    prompt_tokens_per_second: Optional[float] = None
    total_time: Optional[float] = None


class ToolCall(BaseModel):
    id: Optional[str] = None
    name: str
    arguments: str  # raw JSON string as emitted by the model


class RawResponse(BaseModel):
    text: str = ""
    reasoning_text: str = ""
    tool_calls: List[ToolCall] = Field(default_factory=list)
    finish_reason: Optional[str] = None
    usage: Usage = Field(default_factory=Usage)

    # Stream-level observations
    num_chunks: int = 0
    first_token_ts: Optional[float] = None      # perf_counter, relative to t0
    first_visible_ts: Optional[float] = None
    last_token_ts: Optional[float] = None
    ttft_provider_reported: Optional[float] = None


class PerfMetrics(BaseModel):
    # Latency
    ttft_ms: Optional[float] = None
    ttft_visible_ms: Optional[float] = None
    tpot_ms: Optional[float] = None
    e2e_ms: Optional[float] = None
    # Throughput
    gen_tps: Optional[float] = None
    prefill_tps: Optional[float] = None
    total_tps: Optional[float] = None
    # Input context size (per sample; authoritative prompt token count + chars)
    input_context_tokens: Optional[int] = None
    input_context_chars: Optional[int] = None
    # Resources
    peak_cpu_pct: Optional[float] = None
    mean_cpu_pct: Optional[float] = None
    cpu_seconds: Optional[float] = None
    #: Process-tree RSS peak in MB (diagnostic; kept for compatibility).
    peak_rss_mb: Optional[float] = None
    #: Process-tree ``phys_footprint`` peak in GiB (oMLX-consistent units).
    #: Includes Metal allocations; NOT the same metric as oMLX benchmark
    #: ``PeakMem`` (which is ``mlx.core.get_peak_memory()``).
    peak_memory_gb: Optional[float] = None
    cpu_samples: int = 0
    cpu_sample_window_ms: Optional[float] = None
    # Token accounting (from response usage where available)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None


class Score(BaseModel):
    correct: bool = False
    score: float = 0.0
    detail: Dict[str, Any] = Field(default_factory=dict)


class CacheState(BaseModel):
    mode: CacheMode = CacheMode.COLD
    cleared_before_request: bool = False
    clear_latency_ms: Optional[float] = None
    cached_tokens: Optional[int] = None
    polluted: bool = False
    note: Optional[str] = None


class SampleRecord(BaseModel):
    """One request worth of evidence (the atomic unit of a run)."""

    schema_version: str = SCHEMA_VERSION
    run_id: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    model: str
    # ``model_config`` is reserved by pydantic, hence the explicit suffix.
    model_config_snapshot: Dict[str, Any] = Field(default_factory=dict)
    scenario: str
    context_length: Optional[int] = None      # always None in performance mode
    dataset: str
    sample_id: str
    repeat: int = 0
    phase: Phase = Phase.MEASURED
    status: RecordStatus = RecordStatus.OK
    error: Optional[str] = None

    request: RequestSpec
    response: RawResponse = Field(default_factory=RawResponse)
    metrics: PerfMetrics = Field(default_factory=PerfMetrics)
    # Kept for domain compatibility; performance mode never scores (always None).
    score: Optional[Score] = None
    cache: CacheState = Field(default_factory=CacheState)
    sample_meta: Dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


class Distribution(BaseModel):
    n: int = 0
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    p95: Optional[float] = None


class CellAggregate(BaseModel):
    """Per model x scenario x dataset aggregate."""

    model: str
    scenario: str
    context_length: Optional[int] = None
    dataset: str
    n_samples: int = 0
    n_repeats: int = 0
    n_valid: int = 0
    n_warmup: int = 0
    n_contaminated: int = 0
    n_error: int = 0
    #: Valid measured samples whose generation stopped at ``perf.max_tokens``
    #: (``finish_reason == "length"``).  They are INCLUDED in every aggregate;
    #: decode metrics use the actually generated tokens, not ``max_tokens``.
    n_length_capped: int = 0
    stability_cv: Optional[float] = None      # coefficient of variation of ttft
    unstable: bool = False
    perf: Dict[str, Distribution] = Field(default_factory=dict)


class OverallAggregate(BaseModel):
    """Per-model x per-context averages: the headline result shown to users.

    One row per ``(model, context_length)``.  Context buckets differ by an
    order of magnitude in TTFT/prefill, so a single model-wide mean would be
    dominated by whichever bucket happens to have more samples; never collapse
    these rows.  ``context_length`` is ``None`` for legacy samples without a
    declared bucket.
    """

    model: str
    #: Declared bucket (1024/8192/32768); None = legacy/no declared bucket.
    context_length: Optional[int] = None
    n_datasets: int = 0
    n_samples: int = 0
    n_valid: int = 0
    n_warmup: int = 0
    n_contaminated: int = 0
    n_error: int = 0
    #: Valid measured samples that hit ``perf.max_tokens`` (finish_reason=length).
    #: They are included in the averages; TG TPS is measured over the emitted
    #: decode window, never extrapolated from the cap.
    n_length_capped: int = 0
    stability_cv: Optional[float] = None
    unstable: bool = False
    perf: Dict[str, Distribution] = Field(default_factory=dict)
    #: Max RSS observed across all per-sample peaks (MB, diagnostic).
    peak_rss_mb_max: Optional[float] = None
    #: Max phys_footprint observed across all per-sample peaks (GiB).
    peak_memory_gb_max: Optional[float] = None


class SampleResult(BaseModel):
    """One row of the per-sample result table (retained for every sample)."""

    model: str
    scenario: str
    dataset: str
    #: Declared bucket size in tokens (1024/8192/32768); None for legacy samples.
    context_length: Optional[int] = None
    sample_id: str
    repeat: int = 0
    status: str = "ok"
    input_context_tokens: Optional[int] = None
    input_context_chars: Optional[int] = None
    ttft_ms: Optional[float] = None
    ttft_visible_ms: Optional[float] = None
    tpot_ms: Optional[float] = None
    gen_tps: Optional[float] = None
    e2e_ms: Optional[float] = None
    prefill_tps: Optional[float] = None
    peak_cpu_pct: Optional[float] = None
    peak_rss_mb: Optional[float] = None
    peak_memory_gb: Optional[float] = None
    completion_tokens: Optional[int] = None
    #: Provider finish reason ("stop"/"length"/...); "length" means the output
    #: hit ``perf.max_tokens`` and is still a normal measured sample.
    finish_reason: Optional[str] = None
    #: True when ``finish_reason == "length"`` (generation capped, not filtered).
    length_capped: bool = False
    truncated: bool = False


class RunManifest(BaseModel):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    created_at: str
    project_version: str
    mode: str = "performance"
    batch: Dict[str, Any] = Field(default_factory=dict)
    config_snapshot: Dict[str, Any] = Field(default_factory=dict)
    hardware: Dict[str, Any] = Field(default_factory=dict)
    omlx: Dict[str, Any] = Field(default_factory=dict)
    server: Dict[str, Any] = Field(default_factory=dict)
    models: Dict[str, Any] = Field(default_factory=dict)
    scenarios: List[str] = Field(default_factory=list)
    host: Dict[str, Any] = Field(default_factory=dict)
    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None
    notes: Optional[str] = None
