from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class CharBox:
    """A detected character region in a page image.

    Coordinates are pixel positions of the axis-aligned bounding box in the
    source page. `baseline` is the y coordinate of the character baseline,
    used later by the recomposer to align replacements.
    """

    x: int
    y: int
    width: int
    height: int
    label: Optional[str] = None
    confidence: float = 0.0
    baseline: Optional[int] = None

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    def crop_from(self, image: np.ndarray) -> np.ndarray:
        """Return the pixels of `image` enclosed by this box."""
        return image[self.y : self.y2, self.x : self.x2]


@dataclass
class StyledChar:
    """A detected character together with its stylized replacement."""

    box: CharBox
    original: np.ndarray
    styled: Optional[np.ndarray] = None


@dataclass
class DocumentResult:
    """Outcome of running the improvement pipeline over one page."""

    source_path: Path
    boxes: list = field(default_factory=list)
    chars: list = field(default_factory=list)
    output_path: Optional[Path] = None
