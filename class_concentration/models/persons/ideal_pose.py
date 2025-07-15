from dataclasses import dataclass
import numpy as np
from models.persons.relative_xy import RelativeXY

@dataclass
class IdealPose:
  """
  理想ポーズ
  """
  pose_name: str
  relative_xy: RelativeXY
  cov_i: np.ndarray

  @classmethod
  def from_xys(cls, pose_name: str, relative_xy_list: list["RelativeXY"]) -> "IdealPose":
    ideal_relative_xy: RelativeXY = RelativeXY.from_mean_relativeXYs(relative_xy_list)
    cov_i: np.ndarray = cls._calculate_cov_from_posedict(relative_xy_list)
    return cls(pose_name = pose_name, relative_xy = ideal_relative_xy, cov_i = cov_i)

  @classmethod
  def _calculate_cov_from_posedict(cls, ideal_relative_xy_list: list[RelativeXY]) -> np.ndarray:
    """
    pose_xydict を展開し、ベクトル化して共分散行列を計算する
    """
    vectors = []
    # 存在しない座標はひとまずゼロ埋め
    for relative_xy in ideal_relative_xy_list:
      flatten_np: np.ndarray = relative_xy.flatten_relative_xys()
      vectors.append(flatten_np)


    X = np.array(vectors)
    # 分散共分散行列を計算
    cov = np.cov(X.T)
    # 分散共分散行列の逆行列を計算
    cov_i = np.linalg.pinv(cov)

    return cov_i
  