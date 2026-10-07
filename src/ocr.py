"""Text extraction from screenshots using EasyOCR."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

import config

ImageInput = Union[str, Path, bytes, np.ndarray, Image.Image]


@lru_cache(maxsize=4)
def _reader(languages: tuple[str, ...]):
    """Build (and cache) an EasyOCR reader for the given languages."""
    import easyocr

    return easyocr.Reader(list(languages), gpu=False, verbose=False)


def preload(languages: list[str] | None = None) -> None:
    """Build the OCR reader now so the first screenshot does not pay the load cost."""
    _reader(tuple(languages or config.OCR_LANGUAGES))


def extract_text(image: ImageInput, languages: list[str] | None = None) -> str:
    """Run OCR on an image and return the recognised text, or "" if none.

    Args:
        image: File path, raw bytes, numpy array or PIL image.
        languages: EasyOCR language codes; defaults to ``config.OCR_LANGUAGES``.
    """
    reader = _reader(tuple(languages or config.OCR_LANGUAGES))
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    elif isinstance(image, Path):
        image = str(image)
    lines = reader.readtext(image, detail=0, paragraph=True)
    return " ".join(line.strip() for line in lines if line.strip())
