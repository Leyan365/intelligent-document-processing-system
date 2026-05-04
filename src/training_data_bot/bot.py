import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from uuid import UUID

from .core.config import settings
from .core.logging import get_logger, LogContext
from .core.exceptions import TrainingDataBotError, ConfigurationError
from .core.models import ( # Corrected import path (was 'core.models' which is relative to the current file not the package root)
    Document,
    Dataset,
    TrainingExample,
    TaskType,
    DocumentType,
    ProcessingJob,
    ProcessingStatus, # Corrected capitalization
    QualityReport,
    ExportFormat
)

# import modules
from .sources import UnifiedLoader # The document reader boss
from .decodo import DecodoClient # For web Scraping
from .ai import AIClient # The AI brain
from .tasks import TaskManager # The work organizer
from .preprocessing import TextPreprocessor # The text cleaner
from .evaluation import QualityEvaluator # The quality inspector
from .storage import DatasetExporter, DatabaseManager


class TrainingDataBot:
    """
    Main Training Data Bot class.

    This class provides a high-level interface for:
    - Loading documents from various sources
    - Processing text with task templates
    - Quality assessment and filtering
    - Dataset creation and export
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        """
        Initialize the Training Data Bot.
        Args:
        config: Optional configuration overrides
        """
        self.logger = get_logger("training_data_bot")
        self.config = config or {}
        self._init_components()
        self.logger.info("Training Data Bot initialized successfully")


    def _init_components(self):

        """Initialize all bot components."""
        try:
            self.loader = UnifiedLoader()
            self.decodo_client = DecodoClient()
            self.ai_client = AIClient()
            self.task_manager = TaskManager()
            self.preprocessor = TextPreprocessor()
            self.evaluator = QualityEvaluator()
            self.exporter = DatasetExporter()
            self.db_manager = DatabaseManager()
            # State (Memory boxes)
            self.documents: Dict[UUID, Document] = {}
            self.datasets: Dict[UUID, Dataset] = {}
            self.jobs: Dict[UUID, ProcessingJob] = {}
        except Exception as e:
            # Note: The original '...' is a placeholder, replaced with a string for error handling
            self.logger.error(f"Failed to initialize bot components: {e}") 
            raise ConfigurationError(f"Failed to initialize bot components: {e}", detail=str(e))
        

#1: Document Loading

    async def load_documents(
        self,
        sources: Union[str, Path, List[Union[str, Path]]],
        doc_types: Optional[List[DocumentType]] = None,
        **kwargs
    ) -> List[Document]:
        
        with LogContext("document_loading", sources=str(sources)):
            try:
                # Ensure sources is a list
                if isinstance(sources, (str, Path)):
                    sources = [sources]
                
                # Documents list initialization must be outside the loop if it's meant to accumulate results
                documents = [] # <- Document list initialized here

                for source in sources:
                    source_path = Path(source)
                    if source_path.is_dir():
                        dir_docs = await self.loader.load_directory(source_path)
                        documents.extend(dir_docs)
                    else:
                        doc = await self.loader.load_single(source)
                        documents.append(doc)

                for doc in documents:
                    self.documents[doc.id] = doc

                self.logger.info(f"Loaded {len(documents)} documents") # <- Fix typo 'documnets'
                return documents
            
            except Exception as e:
                self.logger.error(f"Failed to load documents: {e}")
                raise

#2: Document Processing

    async def process_documents(
        self,
        documents: Optional[List[Document]] = None,
        task_types: Optional[List[TaskType]] = None,
        quality_filter: bool = True,
        **kwargs
    ) -> Dataset:
        

        with LogContext("document_processing"):
            try:
                # use all documents if none specified
                if documents is None:
                    documents = list(self.documents.values())

                if not documents:
                    raise TrainingDataBotError("No documents to process")
                
                # use default task types if none specified
                if task_types is None:
                    task_types = [TaskType.QA_GENERATION, TaskType.CLASSIFICATION]

                # Create processing job
                job = ProcessingJob(
                    name = f"Process {len(documents)} documents",
                    job_type = "document_processing",
                    total_items = len(documents) * len(task_types),
                    input_data = {
                        "document_count": len(documents),
                        "task_types": [t.value for t in task_types],
                        "quality_filter": quality_filter,
                    }
                )

                self.jobs[job.id] = job
                job.status = ProcessingStatus.PROCESSING # Corrected capitalization

                # process documents

                all_examples = [] # <- Fix typo 'all_exmaples'

                for doc in documents:
                    # Process document (chunking, cleaning)
                    chunks = await self.preprocessor.process_documents(doc)

                    # process each chunk with each task type
                    for task_type in task_types:
                        for chunk in chunks:
                            try:
                                # Execute tasks
                                result = await self.task_manager.execute_task(
                                    task_type = task_type,
                                    input_chunk = chunk,
                                    client = self.ai_client
                                )

                                # Create training example
                                example = TrainingExample(
                                    input_text=chunk.content,
                                    output_text=result.output,
                                    task_type=task_type,
                                    source_document_id = doc.id,
                                    source_chunk_id = chunk.id, # <- Added missing comma
                                    template_id = result.template_id, # <- Added missing comma
                                    quality_score = result.quality_scores, # <- Added missing comma
                                )

                                # Apply quality filtering if enabled
                                if quality_filter:
                                    quality_report = await self.evaluator.evaluate(example) # <- Added missing arguments to evaluator.evaluate
                                    if quality_report.passed:
                                        all_examples.append(example)
                                        example.quality_approved = True

                                    else:
                                        example.quality_approved = False
                                        self.logger.debug(f"Example filtered due to low quality: {quality_report.reasons}") # <- Added reason for logging

                                else:
                                    all_examples.append(example)

                                job.processed_items += 1 
                            
                            except Exception as e:
                                self.logger.error(f"Failed to process chunk: {e}") # <- Fix typo 'failed'
                                job.failed_items += 1
                                continue


                # Create dataset
                dataset = Dataset(
                    name = f"Generated Dataset {len(self.datasets) + 1}",
                    description = f"Dataset generated from {len(documents)} document", # <- Added missing comma
                    examples = all_examples, # <- Fix typo 'all_exmaples'
                )

                # Store dataset
                self.datasets[dataset.id] = dataset

                # Update job status
                job.status = ProcessingStatus.COMPLETED # Corrected capitalization
                job.output_data = {
                    "dataset_id": str(dataset.id),
                    "examples_generated": len(all_examples), # <- Fix typo 'all_exmaples'
                    "quality_filtered": quality_filter,
                }

                self.logger.info(f"Processing completed. Generated {len(all_examples)} examples") # <- Fix typo 'Processed' and 'all_exmaples'
                return dataset
            
            except Exception as e:
                if 'job' in locals():
                    job.status = ProcessingStatus.FAILED # Corrected capitalization
                    job.error_message = str(e)

                self.logger.error(f"Document processing failed: {e}") 
                raise # Re-raise the exception to stop execution


#3: Quality Evaluation

    async def evaluate_dataset(
        self,
        dataset: Dataset,
        detailed_report: bool = True
    ) -> QualityReport:
        

        with LogContext("dataset_evaluation", dataset_id = str(dataset.id)):
            try:
                report = await self.evaluator.evaluate_dataset(
                    dataset = dataset,
                    detailed = detailed_report
                )

                self.logger.info(
                    f"Dataset evaluation completed. Overall score: {report.overall_score:.2f}"
                )
                return report
            
            except Exception as e:
                self.logger.error(f"Dataset evaluation failed: {e}")
                raise


#4:ExportDataset

    async def export_dataset(
        self,
        dataset: Dataset,
        output_path: Union[str, Path],
        format: ExportFormat = ExportFormat.JSONL,
        split_data: bool = True,
        **kwargs
    )-> Path:
        
        with LogContext("dataset_export", dataset_id = str(dataset.id)):
            try:
                exported_path = await self.exporter.export_dataset(
                    dataset = dataset,
                    output_path = Path(output_path),
                    format = format,
                    split_data = split_data,
                    **kwargs
                )

                # Update dataset metadata
                dataset.export_format = format
                dataset.export_path = exported_path

                self.logger.info(f"Dataset exported to {exported_path}") # <- Fix typo 'tp'
                return exported_path
            
            except Exception as e:
                self.logger.error(f"Dataset export failed: {e}")
                raise

#  StatisticsReport

    def get_statistics(self)-> Dict[str, Any]:
        
        # Helper function needed to count
        def _count_by_type(items, attr_name):
             counts = {}
             for item in items:
                 attr = getattr(item, attr_name, None)
                 key = attr.value if hasattr(attr, 'value') else str(attr)
                 counts[key] = counts.get(key, 0) + 1
             return counts
             
        def _count_examples_by_task_type():
            counts = {}
            for ds in self.datasets.values():
                for ex in ds.examples:
                    key = ex.task_type.value
                    counts[key] = counts.get(key, 0) + 1
            return counts

        # The main statistics dictionary with corrections
        return {
            "documents": {
                "total": len(self.documents),
                # Note: The original key was "doc_types", corrected to what it should be:
                "by_type": _count_by_type(self.documents.values(), "document_type"), 
                "total_size": sum(doc.size for doc in self.documents.values()),
            },
            "datasets": {
                "total": len(self.datasets),
                "total_examples": sum(len(ds.examples) for ds in self.datasets.values()),
                "by_task_type": _count_examples_by_task_type()
            },
            "jobs": {
                "total": len(self.jobs),
                # Note: Corrected property access
                "by_status": _count_by_type(self.jobs.values(), "status"), 
                "active": len([j for j in self.jobs.values() if j.status == ProcessingStatus.PROCESSING]) # Corrected class access
            },
            "quality": { # <- Added missing comma to close 'jobs' dict
                "approved_examples": sum(
                    # Added 'ds' back to the list comprehension and ensured it has a default value if 'quality_approved' isn't set
                    len([ex for ex in ds.examples if getattr(ex, 'quality_approved', False)]) 
                    for ds in self.datasets.values()
                ),
                "total_examples": sum(len(ds.examples) for ds in self.datasets.values()) # <- Corrected to iterate over values()
            } # <- Added missing closing brace
        } # <- Added missing closing brace

# Cleanup

    async def cleanup(self):
        """Cleanup resources and close connections."""
        try:
            await self.db_manager.close()
            # Conditional calls to close() based on method existence
            if hasattr(self.decodo_client, 'close'):
                await self.decodo_client.close()
            if hasattr(self.ai_client, 'close'):
                await self.ai_client.close()
            self.logger.info("Bot cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}") # <- Fix typo 'cleanuo'


# Context Manager

    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

# Convenience Methods

    async def quick_process(
        self,
        source: Union[str, Path],
        output_path: Union[str, Path],
        task_types: Optional[List[TaskType]] = None,
        export_format: ExportFormat = ExportFormat.JSONL
    )-> Dataset:
        
        # Load documents
        documents = await self.load_documents([source])

        # Process documents
        dataset = await self.process_documents(
            documents=documents,
            task_types=task_types)
        
        # Export dataset
        await self.export_dataset(
            dataset=dataset, 
            output_path=output_path, 
            format=export_format
        )
        
        return dataset