from pydantic import BaseModel
from typing import List
from urllib.parse import urlparse
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional, Tuple, Set


class StrRequestModel(BaseModel):
    str_request: str

class DialogueParams(BaseModel):
    dialogue_type: str
    message_str: str


class ContentItemModel(BaseModel):
    url:  str
    heading: str
    description: str
    icon_url: str
    hostname: str
    hostname_slug: str
    distance: str
    img_url: str
    date: str
    tags: List[str]

class DialogueModel(BaseModel):
    assistant: List[str]
    user: List[str]

class Message(BaseModel):
    role: str
    utterance: str


####################################


class HealthResponse(BaseModel):
    status: Literal["ok"]


class CreateJobRequest(BaseModel):
    bp_id: str = Field(..., description="ASPICE base practice identifier, e.g., MAN.3.BP1 or SWE.1.BP3")
    top_k: int = Field(5, ge=1, le=20, description="Number of evidence chunks to retrieve")
    backend: Literal["xgrammar", "guidance"] = Field(
        "xgrammar", description="Structured decoding backend"
    )
    notes: Optional[str] = Field(
        None,
        description="Optional free-form notes captured alongside the job metadata",
    )
    manual_keywords: Optional[List[str]] = Field(
        default=None,
        description="Additional keyword hints used to refine retrieval queries.",
    )
    fusion_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional weights for reference indexes (e.g., aspice_normative, aspice_guidelines).",
    )
    include_related_process_evidence: Optional[bool] = Field(
        default=None,
        description="If true, include project evidence for processes referenced by selected rules (e.g., SUP.9)",
    )
    related_process_k: Optional[int] = Field(
        default=None, ge=0, le=10,
        description="Max number of project evidence chunks per related process",
    )
    reasoning_enabled: Optional[bool] = Field(
        default=None,
        description="Enable model-native reasoning if supported by the selected model/profile.",
    )


class CreateJobResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    message: Optional[str]
    progress: Optional[float]
    backend: Optional[str]
    bp_id: Optional[str]
    top_k: Optional[int]
    submitted_at: Optional[str]
    updated_at: Optional[str]
    duration_ms: Optional[int]
    prompt_chars: Optional[int]
    prompt_tokens: Optional[int] = None
    error: Optional[str]
    notes: Optional[str]
    manual_keywords: Optional[List[str]]
    fusion_weights: Optional[Dict[str, float]]
    rerun_of: Optional[str] = None
    #pydantic warning: model_profile, model_id
    model_profile: Optional[str] = None
    model_id: Optional[str] = None
    quantization: Optional[str] = None
    tensor_parallel_size: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None
    kv_cache_memory_bytes: Optional[int] = None
    group_id: Optional[str] = None
    parent_report_id: Optional[str] = None
    process: Optional[str] = None


class JobResultResponse(BaseModel):
    job_id: str
    status: Literal["completed"]
    result: Dict[str, Any]


class JobListResponse(BaseModel):
    jobs: List[JobStatusResponse]


class DeleteJobsRequest(BaseModel):
    job_ids: Optional[List[str]] = None
    delete_all: bool = False
    purge_artifacts: Optional[bool] = None


class DeleteJobsResponse(BaseModel):
    deleted_ids: List[str] = Field(default_factory=list)
    skipped_ids: Optional[List[str]] = None


class BasePractice(BaseModel):
    id: str
    title: Optional[str] = None
    description: Optional[str] = None
    notes: Optional[List[str]] = None
    process_id: Optional[str] = None


class EvidenceItem(BaseModel):
    score: float
    text: str
    metadata: Dict[str, Any]


class EvaluateDocumentsRequest(BaseModel):
    bp_ids: List[str] = Field(..., description="Selected base practice identifiers")
    top_k: int = Field(5, ge=1, le=20)
    manual_keywords: Optional[List[str]] = None
    stop_words: Optional[List[str]] = Field(
        default=None,
        description="Stop words/phrases applied to page titles during retrieval (case-insensitive).",
    )
    fusion_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional weights for reference indexes (aspice_normative, aspice_guidelines, etc.)",
    )
    include_related_process_evidence: Optional[bool] = None
    related_process_k: Optional[int] = None


class EvaluateDocumentsResponse(BaseModel):
    bps: List[Dict[str, Any]]


class ProcessEntry(BaseModel):
    id: str
    name: Optional[str] = None
    bp_ids: List[str]


class ProcessesResponse(BaseModel):
    processes: List[ProcessEntry]


class BulkJobsRequest(BaseModel):
    bp_ids: List[str]
    backend: Literal["xgrammar", "guidance"] = "xgrammar"
    top_k: int = 7
    notes: Optional[str] = None
    manual_keywords: Optional[List[str]] = None
    stop_words: Optional[List[str]] = Field(
        default=None,
        description="Stop words/phrases applied to page titles during retrieval (case-insensitive).",
    )
    fusion_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional weights for reference indexes (aspice_normative, aspice_guidelines, etc.)",
    )
    include_related_process_evidence: Optional[bool] = None
    related_process_k: Optional[int] = None
    reasoning_enabled: Optional[bool] = None
    parent_report_id: Optional[str] = Field(
        default=None,
        description="Optional parent report id to associate all created jobs with. If omitted, a new id is generated.",
    )


class BulkJobsResponse(BaseModel):
    group_id: str
    job_ids: List[str]
    parent_report_id: Optional[str] = None


class ProcessAssessRequest(BaseModel):
    reasoning_enabled: Optional[bool] = None


class ProjectAssessRequest(BaseModel):
    reasoning_enabled: Optional[bool] = None


class BulkGroupStatusResponse(BaseModel):
    group_id: str
    jobs: List[JobStatusResponse]
    completed: int
    total: int


class BulkProcessAggregate(BaseModel):
    process_id: str
    bp_ids: List[str]
    capability_rating: Optional[str] = None
    rationale: Optional[str] = None
    gaps: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    guideline_rules: Optional[List[Dict[str, Any]]] = None
    governance_checks: Optional[List[Dict[str, Any]]] = None
    contextual_warnings: Optional[List[Dict[str, Any]]] = None


class BulkProjectSummary(BaseModel):
    capability_rating: Optional[str] = None
    rationale: Optional[str] = None
    gaps: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    included_processes: List[str] = Field(default_factory=list)
    process_summaries: Optional[List[Dict[str, Any]]] = None
    confidence: Optional[float] = None


class BulkGroupResultResponse(BaseModel):
    group_id: str
    processes: List[BulkProcessAggregate]
    overall_capability_rating: Optional[str] = None
    overall_rationale: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None
    project_summary: Optional[BulkProjectSummary] = None
    combined_report: Optional[Dict[str, Any]] = None
    # Optional: exact performance JSON per README (served separately too)
    performance: Optional[Dict[str, Any]] = None


class RerunJobRequest(BaseModel):
    backend: Optional[Literal["xgrammar", "guidance"]] = Field(
        default=None,
        description="Override structured backend for the rerun.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional notes for the rerun (defaults to original job notes if omitted).",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Override the evidence retrieval depth.",
    )
    manual_keywords: Optional[List[str]] = Field(
        default=None,
        description="Provide a new set of manual keywords (empty list clears them).",
    )
    fusion_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Override fusion weights; omit to reuse the original."  # noqa: E501
    )
    include_related_process_evidence: Optional[bool] = Field(
        default=None,
        description="If true, include project evidence for related processes",
    )
    related_process_k: Optional[int] = Field(
        default=None, ge=0, le=10,
        description="Max number of project evidence chunks per related process",
    )
    reasoning_enabled: Optional[bool] = Field(
        default=None,
        description="Override reasoning enablement for rerun (omit to reuse original).",
    )
    group_id: Optional[str] = Field(
        default=None,
        description="Associate the rerun with this group (for cascades/report scoping).",
    )
    parent_report_id: Optional[str] = Field(
        default=None,
        description="Associate the rerun with this parent report (scoped Assessment Jobs).",
    )

