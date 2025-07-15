from dataclasses import dataclass
from models.settings.detect_xy import DetectXY

@dataclass
class TrackedPerson:
  id: int
  center_x: int
  center_y: int
  current_score: float = None
  pose: str = None

  def is_mask_area(self, mask_xy: DetectXY) -> bool:
    return (mask_xy.min[0] <= self.center_x <= mask_xy.max[0] and
            mask_xy.min[1] <= self.center_y <= mask_xy.max[1])
