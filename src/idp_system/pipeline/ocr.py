"""PaddleOCR text extraction."""

from pathlib import Path
from typing import Any

from ..core.exceptions import DocumentLoadError
from .preprocessing import ImagePreprocessor


class OCRService:
    """Local PaddleOCR wrapper with lazy model initialization."""

    def __init__(
        self,
        language: str = "en",
        use_angle_cls: bool = True,
        preprocessor: ImagePreprocessor | None = None,
    ) -> None:
        self.language = language
        self.use_angle_cls = use_angle_cls
        self.preprocessor = preprocessor or ImagePreprocessor()
        self._engine: Any | None = None

    def extract_text(self, image: str | Path | Any, preprocess: bool = True) -> str:
        """Extract text from an image path or image array."""
        ocr_input = self.preprocessor.preprocess(image) if preprocess else image
        result = self._get_engine().ocr(ocr_input, cls=self.use_angle_cls)
        return clean_ocr_text(_collect_text_lines(result))

    def _get_engine(self) -> Any:
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise DocumentLoadError(
                    "PaddleOCR is required for OCR extraction. Install package: paddleocr"
                ) from exc

            self._engine = PaddleOCR(
                lang=self.language,
                use_angle_cls=self.use_angle_cls,
                show_log=False,
            )
        return self._engine


def clean_ocr_text(lines: list[str]) -> str:
    """Normalize OCR lines into readable text."""
    return "\n".join(line.strip() for line in lines if line and line.strip()).strip()


def _collect_text_lines(result: Any) -> list[str]:
    """Collect text strings from PaddleOCR's nested result shapes."""
    lines: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, tuple) and len(value) >= 1 and isinstance(value[0], str):
            lines.append(value[0])
            return
        if isinstance(value, list):
            if (
                len(value) >= 2
                and isinstance(value[1], tuple)
                and value[1]
                and isinstance(value[1][0], str)
            ):
                lines.append(value[1][0])
                return
            for item in value:
                visit(item)

    visit(result)
    return lines
