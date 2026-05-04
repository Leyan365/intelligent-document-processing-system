"""Information extraction placeholder."""


class InformationExtractor:
    """Future spaCy and regex information extractor."""

    def extract(self, text: str, document_type: str | None = None) -> dict[str, object]:
        raise NotImplementedError("Information extraction will be implemented in a later phase.")
