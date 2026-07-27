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
from .narrative import (
    ChapterAnnotation,
    Entity,
    EventAtom,
    Fact,
    SceneCard,
    SpatialRelation,
    StateChange,
    TemporalRelation,
    TimeAnchor,
)
from .program import ChapterProgram, EntityBinding, EventProgram, FactContract, NarrativeBeat, ParagraphProgram, SceneProgram
from .scene_output import ParagraphDraft, ParagraphIssue, SceneDraft, SceneProseValidation, SceneValidation
from .skill import AuthorProfile, PromptProgram, Template
from .source import ChapterDocument, SourceSpan, SourceUnit, build_chapter_document, build_paragraph_units, classify_chapter_content
from .style import ChapterStyleCard, StyleMetric, StylePatternObservation
from .style_profile import AuthorStyleProfile, StyleBaseline, StyleConstraint
from .template_library import NarrativeTemplate, NarrativeTemplateLibrary
from .validation import ContractValidationError, ValidationIssue, ValidationReport, validate_annotation, validate_document

__all__ = [
    "AuthorProfile",
    "AuthorStyleProfile",
    "build_chapter_document",
    "build_paragraph_units",
    "ChapterAnnotation",
    "ChapterDocument",
    "ChapterProgram",
    "ChapterRecord",
    "ChapterState",
    "ChapterStyleCard",
    "classify_chapter_content",
    "ContractValidationError",
    "ContinuityIssue",
    "Entity",
    "EntityBinding",
    "EventAtom",
    "EventProgram",
    "Fact",
    "FactContract",
    "GlobalEntity",
    "LocationTransition",
    "NarrativeTemplate",
    "NarrativeTemplateLibrary",
    "NarrativeBeat",
    "NarrativeGraph",
    "NarrativeGraphEdge",
    "NarrativeGraphNode",
    "PromptProgram",
    "ParagraphProgram",
    "ParagraphDraft",
    "ParagraphIssue",
    "SceneCard",
    "SceneDraft",
    "SceneProgram",
    "SceneProseValidation",
    "SceneValidation",
    "SpatialRelation",
    "SourceSpan",
    "SourceUnit",
    "StateChange",
    "StateLedgerEntry",
    "TemporalRelation",
    "TimeAnchor",
    "StoryState",
    "StyleMetric",
    "StyleBaseline",
    "StyleConstraint",
    "StylePatternObservation",
    "Template",
    "TimelineEvent",
    "ValidationIssue",
    "ValidationReport",
    "validate_annotation",
    "validate_document",
    "WorkContinuity",
]
