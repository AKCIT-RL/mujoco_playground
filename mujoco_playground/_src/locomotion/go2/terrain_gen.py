# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Procedural heightfield generator for the Go2 terrain curriculum.

The terrain is a single large heightfield (hfield) tiled as a
``num_rows x num_cols`` grid:

* each **column** is a terrain *type* (e.g. ``rough``, ``slope``, ``stairs``);
* each **row** is a difficulty *level* (row 0 = easiest, last row = hardest).

All parallel environments share the same hfield (the MJX default). The only
per-environment difference is the *spawn tile* ``(row, col)``, which the
curriculum wrapper updates between episodes.

Everything here is pure NumPy so it can run once in ``__init__`` (host side) and
be baked into ``mj_model.hfield_data``.
"""

from typing import Dict, List, Sequence

import numpy as np

# Supported terrain primitives (one per grid column).
TERRAIN_TYPES = ("rough", "slope", "stairs")


def _difficulty_for_row(
    row: int, num_rows: int, difficulty_range: Sequence[float]
) -> float:
  """Linearly maps a row index to a difficulty scalar in ``difficulty_range``."""
  d0, d1 = float(difficulty_range[0]), float(difficulty_range[1])
  if num_rows <= 1:
    return d1
  frac = row / (num_rows - 1)
  return d0 + (d1 - d0) * frac


def _tile_rough(
    tile_px: int, res: float, difficulty: float, rng: np.random.Generator
) -> np.ndarray:
  """Random bumps. Amplitude grows ~1cm -> ~12cm with difficulty."""
  amp = 0.01 + (0.12 - 0.01) * difficulty
  return rng.uniform(0.0, amp, size=(tile_px, tile_px)).astype(np.float32)


def _tile_slope(
    tile_px: int, res: float, difficulty: float, rng: np.random.Generator
) -> np.ndarray:
  """Linear ramp along +x. Angle grows ~5deg -> ~25deg with difficulty."""
  del rng
  angle_deg = 5.0 + (25.0 - 5.0) * difficulty
  slope = np.tan(np.deg2rad(angle_deg))
  x_local = (np.arange(tile_px, dtype=np.float32) + 0.5) * res  # [0, tile_size]
  row = slope * x_local  # height per column.
  return np.tile(row[None, :], (tile_px, 1)).astype(np.float32)


def _tile_stairs(
    tile_px: int, res: float, difficulty: float, rng: np.random.Generator
) -> np.ndarray:
  """Ascending stairs along +x. Step height grows ~3cm -> ~15cm."""
  del rng
  step_h = 0.03 + (0.15 - 0.03) * difficulty
  step_w = 0.3  # meters per tread.
  x_local = (np.arange(tile_px, dtype=np.float32) + 0.5) * res
  row = step_h * np.floor(x_local / step_w)
  return np.tile(row[None, :], (tile_px, 1)).astype(np.float32)


_TILE_FNS = {
    "rough": _tile_rough,
    "slope": _tile_slope,
    "stairs": _tile_stairs,
}


def generate_terrain(
    num_rows: int,
    terrain_types: Sequence[str],
    tile_size: float = 8.0,
    resolution: float = 0.1,
    difficulty_range: Sequence[float] = (0.0, 1.0),
    seed: int = 0,
) -> Dict[str, object]:
  """Builds the curriculum heightfield.

  Args:
    num_rows: number of difficulty levels (grid rows, along +y).
    terrain_types: one terrain primitive per grid column (along +x). Each entry
      must be a key of :data:`TERRAIN_TYPES`.
    tile_size: side length of each square tile, in meters.
    resolution: hfield cell size, in meters.
    difficulty_range: ``(d_min, d_max)`` difficulty for the first / last row.
    seed: RNG seed for the procedural noise.

  Returns:
    A dict with:
      ``heights``: ``(nrow, ncol)`` float32 array, elevation in meters (min 0).
      ``z_top``: max elevation in meters (hfield ``size[2]``).
      ``nrow``/``ncol``: hfield resolution.
      ``radius_x``/``radius_y``: hfield half-extents in meters (``size[0:2]``).
      ``tile_centers``: ``(num_rows, num_cols, 2)`` world ``(x, y)`` of each
        tile center.
      ``spawn_heights``: ``(num_rows, num_cols)`` terrain height at each tile
        center.
      ``terrain_types``: the resolved list of column types.
  """
  terrain_types = list(terrain_types)
  for t in terrain_types:
    if t not in _TILE_FNS:
      raise ValueError(f"Unknown terrain type {t!r}; valid: {TERRAIN_TYPES}")
  num_cols = len(terrain_types)
  if num_cols == 0:
    raise ValueError("terrain_types must contain at least one entry.")

  rng = np.random.default_rng(seed)

  tile_px = int(round(tile_size / resolution))
  res = tile_size / tile_px  # actual resolution after rounding.
  nrow = tile_px * num_rows
  ncol = tile_px * num_cols

  heights = np.zeros((nrow, ncol), dtype=np.float32)
  spawn_heights = np.zeros((num_rows, num_cols), dtype=np.float32)
  tile_centers = np.zeros((num_rows, num_cols, 2), dtype=np.float32)

  radius_x = num_cols * tile_size / 2.0
  radius_y = num_rows * tile_size / 2.0

  center_px = tile_px // 2
  for r in range(num_rows):
    difficulty = _difficulty_for_row(r, num_rows, difficulty_range)
    for c, ttype in enumerate(terrain_types):
      tile = _TILE_FNS[ttype](tile_px, res, difficulty, rng)
      r0, c0 = r * tile_px, c * tile_px
      heights[r0 : r0 + tile_px, c0 : c0 + tile_px] = tile

      x_center = -radius_x + (c + 0.5) * tile_size
      y_center = -radius_y + (r + 0.5) * tile_size
      tile_centers[r, c] = (x_center, y_center)
      spawn_heights[r, c] = tile[center_px, center_px]

  z_top = float(max(heights.max(), 1e-3))

  return {
      "heights": heights,
      "z_top": z_top,
      "nrow": nrow,
      "ncol": ncol,
      "radius_x": radius_x,
      "radius_y": radius_y,
      "tile_centers": tile_centers,
      "spawn_heights": spawn_heights,
      "terrain_types": terrain_types,
  }


def build_scene_xml(
    nrow: int,
    ncol: int,
    radius_x: float,
    radius_y: float,
    z_top: float,
    z_bottom: float = 0.1,
) -> str:
  """Returns a Go2 feetonly scene XML string with a procedural hfield.

  The hfield is declared with ``nrow``/``ncol`` and zero data; the env fills
  ``mj_model.hfield_data`` after compilation.
  """
  size = f"{radius_x:.4f} {radius_y:.4f} {z_top:.4f} {z_bottom:.4f}"
  return f"""<mujoco model="go2 rough curriculum scene">
  <include file="go2_mjx_feetonly.xml"/>

  <statistic center="0 0 0.1" extent="0.8" meansize="0.04"/>

  <visual>
    <rgba force="1 0 0 1"/>
    <global azimuth="120" elevation="-20"/>
    <map force="0.01"/>
    <scale forcewidth="0.3" contactwidth="0.5" contactheight="0.2"/>
    <quality shadowsize="8192"/>
  </visual>

  <asset>
    <texture type="2d" name="groundplane" file="assets/rocky_texture.png"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance=".8"/>
    <hfield name="terrain" nrow="{nrow}" ncol="{ncol}" size="{size}"/>
  </asset>

  <worldbody>
    <geom name="floor" type="hfield" hfield="terrain" material="groundplane" contype="1" conaffinity="0" priority="1"
      friction="1.0"/>
  </worldbody>

  <include file="sensor_feet.xml"/>

  <keyframe>
    <key name="home" qpos="
    0 0 0.35
    1 0 0 0
    0.1 0.9 -1.8
    -0.1 0.9 -1.8
    0.1 0.9 -1.8
    -0.1 0.9 -1.8"
      ctrl="0.1 0.9 -1.8 -0.1 0.9 -1.8 0.1 0.9 -1.8 -0.1 0.9 -1.8"/>
  </keyframe>
</mujoco>
"""


def list_terrain_types() -> List[str]:
  return list(TERRAIN_TYPES)
