"""PaddleOCR text extraction."""

from pathlib import Path
from time import perf_counter
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
        result = _run_ocr(self._get_engine(), ocr_input, self.use_angle_cls)
        return clean_ocr_text(_collect_text_from_ocr_result(result))

    def _get_engine(self) -> Any:
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise DocumentLoadError(
                    "PaddleOCR is required for OCR extraction. Install package: paddleocr"
                ) from exc

            engine_kwargs = {
                "lang": self.language,
                "use_angle_cls": self.use_angle_cls,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
                "show_log": False,
            }
            start_time = perf_counter()
            print("Initializing PaddleOCR service...")
            self._engine = _init_paddleocr(PaddleOCR, engine_kwargs)
            elapsed = perf_counter() - start_time
            print(f"PaddleOCR service initialized in {elapsed:.2f}s.")
        return self._engine


def _init_paddleocr(paddleocr_cls: Any, engine_kwargs: dict[str, Any]) -> Any:
    """Initialize PaddleOCR while tolerating version-specific keyword support."""
    fallback_keys = (
        ("show_log",),
        (
            "use_doc_orientation_classify",
            "use_doc_unwarping",
            "use_textline_orientation",
        ),
        ("use_angle_cls",),
    )
    current_kwargs = dict(engine_kwargs)

    while True:
        try:
            return paddleocr_cls(**current_kwargs)
        except (TypeError, ValueError) as exc:
            removed = False
            for keys in fallback_keys:
                if any(key in current_kwargs and key in str(exc) for key in keys):
                    for key in keys:
                        current_kwargs.pop(key, None)
                    removed = True
                    break
            if not removed:
                raise


def _run_ocr(engine: Any, image: Any, use_angle_cls: bool) -> Any:
    """Run OCR across PaddleOCR versions with different inference APIs."""
    ocr_input = _normalize_ocr_input(image)

    if hasattr(engine, "ocr"):
        try:
            return engine.ocr(ocr_input, cls=use_angle_cls)
        except TypeError as exc:
            if "cls" not in str(exc):
                raise
            return engine.ocr(ocr_input)

    if hasattr(engine, "predict"):
        try:
            return engine.predict(ocr_input, cls=use_angle_cls)
        except TypeError as exc:
            if "cls" not in str(exc):
                raise
            return engine.predict(ocr_input)

    raise DocumentLoadError("PaddleOCR engine does not expose an ocr or predict method.")


def _normalize_ocr_input(image_or_path: Any) -> Any:
    """Ensure image arrays passed to PaddleOCR/PaddleX have 3 channels."""
    if isinstance(image_or_path, (str, Path)):
        return image_or_path

    shape = getattr(image_or_path, "shape", None)
    if shape is None:
        return image_or_path

    try:
        ndim = image_or_path.ndim
    except AttributeError:
        ndim = len(shape)

    if ndim == 2:
        np = _import_numpy()
        return np.ascontiguousarray(np.stack((image_or_path,) * 3, axis=-1))

    if ndim == 3 and len(shape) >= 3:
        channels = shape[2]
        np = _import_numpy()
        if channels == 1:
            return np.ascontiguousarray(np.repeat(image_or_path, 3, axis=2))
        if channels == 4:
            return np.ascontiguousarray(image_or_path[:, :, :3])
        if channels == 3:
            return image_or_path

    return image_or_path


def _import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise DocumentLoadError(
            "NumPy is required for OCR image normalization. Install package: numpy"
        ) from exc
    return np


def clean_ocr_text(lines: list[str]) -> str:
    """Normalize OCR lines into readable text."""
    return "\n".join(line.strip() for line in lines if line and line.strip()).strip()


def _collect_text_from_ocr_result(result: Any) -> list[str]:
    """Collect recognized text across PaddleOCR/PaddleX result shapes."""
    lines: list[str] = []
    seen: set[int] = set()

    def visit(value: Any) -> None:
        if value is None:
            return

        if isinstance(value, str):
            if _looks_like_text(value):
                lines.append(value)
            return

        if isinstance(value, (int, float, bool)):
            return

        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)

        if isinstance(value, dict):
            for key in ("rec_texts", "texts", "text", "rec_text"):
                if key in value:
                    visit(value[key])
            return

        for attr in ("rec_texts", "texts", "text", "rec_text"):
            try:
                attr_value = getattr(value, attr)
            except Exception:
                continue
            visit(attr_value)

        if isinstance(value, (list, tuple)):
            old_format_text = _old_format_text(value)
            if old_format_text:
                lines.append(old_format_text)
                return
            for item in value:
                visit(item)

    visit(result)
    return lines


def _old_format_text(value: list[Any] | tuple[Any, ...]) -> str | None:
    """Extract text from old [box, (text, score)] PaddleOCR line format."""
    if len(value) < 2:
        return None

    payload = value[1]
    if isinstance(payload, (list, tuple)) and payload and isinstance(payload[0], str):
        return payload[0] if _looks_like_text(payload[0]) else None
    return None


def _looks_like_text(value: str) -> bool:
    """Filter out numeric-only scores and structural strings."""
    cleaned = value.strip()
    if not cleaned:
        return False
    return any(char.isalpha() for char in cleaned)


if __name__ == "__main__":
    np = _import_numpy()
    grayscale = np.zeros((10, 20), dtype=np.uint8)
    normalized = _normalize_ocr_input(grayscale)
    assert normalized.ndim == 3
    assert normalized.shape[2] == 3

    old_shape = [[[[0, 0], [1, 0], [1, 1], [0, 1]], ("Invoice No", 0.98)]]
    dict_shape = {"rec_texts": ["Invoice No", "INV-123"]}
    print(clean_ocr_text(_collect_text_from_ocr_result(old_shape)))
    print(clean_ocr_text(_collect_text_from_ocr_result(dict_shape)))
