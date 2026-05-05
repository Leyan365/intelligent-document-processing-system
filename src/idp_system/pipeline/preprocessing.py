"""OpenCV helpers for OCR image preprocessing."""

from pathlib import Path
from typing import Any

from ..core.exceptions import DocumentLoadError


class ImagePreprocessor:
    """Small, testable OpenCV preprocessing steps used before OCR."""

    def read_image(self, image_path: str | Path) -> Any:
        """Read an image file with OpenCV."""
        cv2 = self._import_cv2()
        image = cv2.imread(str(image_path))
        if image is None:
            raise DocumentLoadError(f"Unable to read image: {image_path}")
        return image

    def to_grayscale(self, image: Any) -> Any:
        """Convert BGR/RGB image data to grayscale."""
        cv2 = self._import_cv2()
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def denoise(self, image: Any) -> Any:
        """Reduce OCR-hostile image noise."""
        cv2 = self._import_cv2()
        return cv2.medianBlur(image, 3)

    def threshold(self, image: Any) -> Any:
        """Apply adaptive thresholding for clearer text foreground."""
        cv2 = self._import_cv2()
        gray = self.to_grayscale(image)
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )

    def deskew(self, image: Any) -> Any:
        """Deskew a text image when OpenCV can estimate a reliable angle."""
        cv2 = self._import_cv2()
        np = self._import_numpy()
        gray = self.to_grayscale(image)
        coords = np.column_stack(np.where(gray < 255))
        if coords.size == 0:
            return image

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.1:
            return image

        height, width = gray.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def preprocess(self, image: str | Path | Any) -> Any:
        """Run the default OCR preprocessing pipeline."""
        source = self.read_image(image) if isinstance(image, (str, Path)) else image
        gray = self.to_grayscale(source)
        denoised = self.denoise(gray)
        thresholded = self.threshold(denoised)
        return self.deskew(thresholded)

    @staticmethod
    def _import_cv2() -> Any:
        try:
            import cv2
        except ImportError as exc:
            raise DocumentLoadError(
                "OpenCV is required for image preprocessing. Install package: opencv-python"
            ) from exc
        return cv2

    @staticmethod
    def _import_numpy() -> Any:
        try:
            import numpy as np
        except ImportError as exc:
            raise DocumentLoadError(
                "NumPy is required for image preprocessing. Install package: numpy"
            ) from exc
        return np
