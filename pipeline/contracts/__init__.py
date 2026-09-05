"""Stable, evidence-first contracts shared by every pipeline stage.

Legacy V3 records remain exported during the staged migration.  New stages
must use the typed contracts below rather than ad-hoc nested dictionaries.
"""

from .legacy import ChapterRecord, ChapterState, StoryState
from .continuity import (
    ContinuityIssue,
    GlobalEntity,
    LocationTransition,
    StateLedgerEntry,
    TimelineEvent,
    WorkContinuity,
)
from .graph import NarrativeGraph, NarrativeGraphEdge, NarrativeGraphNode
from .fact_hydration import (
    FACT_HYDRATION_SCHEMA_VERSION,
    FactHydrationBundle,
    FactRequirement,
    HydratedFact,
)
from .author_skill import AuthorSkillBundle, DistilledAuthorTrait, SkillQualification
from .generation_plan import ChapterContract, EventGraph, EventGraphEvent, StoryStage, WorkStoryPlan
from .outline import (
    OUTLINE_SCHEMA_VERSION,
    CharacterArcOutline,
    HierarchicalOutlineBundle,
    OutlineNode,
)
from .narrative import (
    ChapterAnnotation,
    Entity,
    EventAtom,
    Fact,
    NarrativeMechanism,
    SceneCard,
    SpatialRelation,
    StateChange,
    TemporalRelation,
    TimeAnchor,
)
from .program import ChapterProgram, EntityBinding, EventProgram, FactContract, NarrativeBeat, ParagraphProgram, SceneProgram
from .scene_output import AffordanceIssue, ParagraphDraft, ParagraphIssue, ParagraphValidation, SceneAffordanceValidation, SceneDraft, SceneProseValidation, SceneValidation
from .semantic_audit import SemanticAuditBatch, SemanticAuditIssue, SemanticAuditReport
from .skill import AuthorProfile, PromptProgram, Template
from .source import ChapterDocument, SourceSpan, SourceUnit, build_chapter_document, build_paragraph_units, classify_chapter_content
from .synopsis import (
    SYNOPSIS_SCHEMA_VERSION,
    BoundaryFrame,
    ChapterSummary,
    ChapterSynopsis,
    ChapterSynopsisBundle,
    LocalPlotSegment,
    LocalSynopsis,
    SynopsisIssue,
    SynopsisQuality,
)
from .style import ChapterStyleCard, StyleMetric, StylePatternObservation
from .style_profile import AuthorStyleProfile, StyleBaseline, StyleConstraint
from .template_library import NarrativeTemplate, NarrativeTemplateLibrary
from .validation import ContractValidationError, ValidationIssue, ValidationReport, validate_annotation, validate_document

__all__ = [
    "AuthorProfile",
    "AuthorSkillBundle",
    "AuthorStyleProfile",
    "AffordanceIssue",
    "build_chapter_document",
    "build_paragraph_units",
    "ChapterAnnotation",
    "ChapterDocument",
    "ChapterProgram",
    "ChapterContract",
    "ChapterRecord",
    "ChapterState",
    "ChapterStyleCard",
    "classify_chapter_content",
    "ContractValidationError",
    "ContinuityIssue",
    "Entity",
    "EntityBinding",
    "EventGraph",
    "EventGraphEvent",
    "EventAtom",
    "EventProgram",
    "Fact",
    "FactContract",
    "GlobalEntity",
    "DistilledAuthorTrait",
    "LocationTransition",
    "NarrativeTemplate",
    "NarrativeTemplateLibrary",
    "NarrativeBeat",
    "NarrativeMechanism",
    "NarrativeGraph",
    "NarrativeGraphEdge",
    "NarrativeGraphNode",
    "PromptProgram",
    "ParagraphProgram",
    "ParagraphDraft",
    "ParagraphIssue",
    "ParagraphValidation",
    "SceneCard",
    "SceneAffordanceValidation",
    "SceneDraft",
    "SceneProgram",
    "SceneProseValidation",
    "SceneValidation",
    "SemanticAuditBatch",
    "SemanticAuditIssue",
    "SemanticAuditReport",
    "SpatialRelation",
    "SourceSpan",
    "SourceUnit",
    "StateChange",
    "StateLedgerEntry",
    "TemporalRelation",
    "TimeAnchor",
    "StoryState",
    "StoryStage",
    "StyleMetric",
    "StyleBaseline",
    "StyleConstraint",
    "StylePatternObservation",
    "SkillQualification",
    "Template",
    "TimelineEvent",
    "ValidationIssue",
    "ValidationReport",
    "validate_annotation",
    "validate_document",
    "WorkContinuity",
    "WorkStoryPlan",
    "SYNOPSIS_SCHEMA_VERSION",
    "BoundaryFrame",
    "ChapterSummary",
    "ChapterSynopsis",
    "ChapterSynopsisBundle",
    "LocalPlotSegment",
    "LocalSynopsis",
    "SynopsisIssue",
    "SynopsisQuality",
    "FACT_HYDRATION_SCHEMA_VERSION",
    "FactHydrationBundle",
    "FactRequirement",
    "HydratedFact",
    "OUTLINE_SCHEMA_VERSION",
    "CharacterArcOutline",
    "HierarchicalOutlineBundle",
    "OutlineNode",
]
