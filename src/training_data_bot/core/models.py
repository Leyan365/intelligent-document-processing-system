"""
Core data models for training data bot.

This module defines Pydantic models for all data structures used throughout
the application , ensuring type safety and validation.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator # Note: root_validator is typically replaced by @model_validator in V2


class BaseEntity(BaseModel):
    """Base class for all entites with common fields."""
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Pydantic V1/V2 Configuration
    class Config:
        use_enum_values = True
        allow_population_by_field_name = True
        # FIX: Corrected typo
        arbitrary_types_allowed = True 


# Enums
class DocumentType(str, Enum):
    """Supported Documents Types"""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    URL = "url"


class TaskType(str, Enum):
    QA_GENERATION = "qa_generation"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    NER = "named_entity_recognition"
    RED_TEAMING = "red_teaming"
    INSTRUCTION_RESPONSE = "instruction_response"


class QualityMetric(str, Enum):
    """Quality asseseement metrics"""
    TOXICITY = "toxicity"
    BIAS = "bias"
    DIVERSITY = "diversity"
    COHERENCE = "coherence"
    RELEVANCE = "relevance"


class ProcessingStatus(str, Enum):
    """Processing status values"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExportFormat(str, Enum):
    """Export format options"""
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"


# Documents models
class Document(BaseEntity):
    """Represents a source document"""
    title: str
    content: str
    source: str # file path, URL, etc.
    doc_type: DocumentType
    language: Optional[str] = "en"
    encoding: Optional[str] = "utf-8"
    size: int = 0 # bytes
    word_count: int = 0
    char_count: int = 0

    # processing info
    extraction_method: Optional[str] = None
    processing_time: Optional[float] = None # Changed type to float for consistency

    # FIX: Corrected validator signature (v, values) and logic
    @validator("word_count", pre=True, always=True)
    def calculate_word_count(cls, v, values):
        if "content" in values and values.get("content") is not None and v == 0:
            return len(values["content"].split())
        return v
    
    # FIX: Corrected validator signature (v, values) and logic
    @validator("char_count", pre=True, always=True)
    def calculate_char_count(cls, v, values):
        if "content" in values and values.get("content") is not None and v == 0:
            return len(values["content"])
        return v
    

class TextChunk(BaseEntity):
    document_id: UUID
    content: str
    start_index: int
    end_index: int
    chunk_index: int
    token_count: int = 0

    # Context preservation
    preceding_context: Optional[str] = None
    following_context: Optional[str] = None

    # Semantic info
    embeddings: Optional[List[float]] = None
    topics: List[str] = Field(default_factory=list) # Changed hint to List[str]

    # FIX: Corrected validator signature (v, values) and logic
    @validator("token_count", pre=True, always=True)
    def calculate_token_count(cls, v, values):
        if "content" in values and values.get("content") is not None and v == 0:
            # Rough estimation: 1 token = 4 characters
            return len(values["content"]) // 4
        return v
    

# Task Models
class TaskTemplate(BaseEntity):
    """Represents a task template"""
    name: str
    task_type: TaskType
    description: str
    prompt_template: str

    # Task_specific configuration
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Quality requirements
    min_output_length: int = 10
    max_output_length: int = 2000
    quality_thresholds: Dict[QualityMetric, float] = Field(default_factory=dict)

    # Performance settings
    timeout: int = 60
    max_retries: int = 3


class TaskResult(BaseEntity):
    """Result of a task execution"""
    task_id: UUID
    # FIX: Corrected field definition with an assignment
    template_id: UUID 
    input_chunk_id: UUID

    # Output
    output: str
    confidence: Optional[float] = None

    # Quality scores
    # FIX: Corrected field definition with an assignment
    quality_scores: Dict[QualityMetric, float] = Field(default_factory=dict)

    # Processing info
    processing_time: float
    token_usage: int = 0
    cost: Optional[float] = None

    # Status
    status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: Optional[str] = None


# Training Data Models
class TrainingExample(BaseEntity):
    """A single training example"""
    input_text: str
    output_text: str
    task_type: TaskType

    # Source tracking
    source_document_id: UUID
    source_chunk_id: Optional[UUID] = None
    template_id: Optional[UUID] = None

    # Quality assessment
    quality_scores: Dict[QualityMetric, float] = Field(default_factory=dict)
    quality_approved: Optional[bool] = None

    # Additional fields for different formats
    instruction: Optional[str] = None # For instruction-following datasets
    context: Optional[str] = None  # For context-based tasks
    category: Optional[str] = None # For classification tasks


class Dataset(BaseEntity):
    """A collection of training examples"""
    name: str
    description: str
    version: str = "1.0.0"

    # Content
    # FIX: Changed default_factory to list
    examples: List[TrainingExample] = Field(default_factory=list) 

    # Statistics
    total_examples: int = 0
    task_type_counts: Dict[TaskType, int] = Field(default_factory=dict)
    quality_stats: Dict[QualityMetric, Dict[str, float]] = Field(default_factory=dict)

    # Splits
    train_split: float = 0.8
    validation_split: float = 0.1
    test_split: float = 0.1

    # Export info
    export_format: ExportFormat = ExportFormat.JSONL
    exported_at: Optional[datetime] = None
    export_path: Optional[Path] = None

    # FIX: Corrected validator signature (v, values) and logic
    @validator("total_examples", pre=True, always=True)
    def calculate_total_examples(cls, v, values):
        if "examples" in values and v == 0:
            return len(values["examples"])
        return v