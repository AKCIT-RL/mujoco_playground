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
"""Go2 rough-terrain task with an adaptive terrain curriculum.

The world is one large heightfield tiled as a ``num_rows x num_cols`` grid
(see :mod:`terrain_gen`):

* each **column** is a terrain type (``rough`` / ``slope`` / ``stairs``);
* each **row** is a difficulty level (row 0 easiest).

All MJX environments share the same geometry; the only per-environment
difference is the *spawn tile* ``(terrain_level, terrain_col)``. The companion
:class:`CurriculumAutoResetWrapper` reads each environment's episode score
(planar distance travelled from the spawn) and, between episodes:

* **promotes** (``level += 1``) if the agent survived to the time limit and
  travelled past ``promote_distance``;
* **regresses** (``level -= 1``) if the agent fell early without travelling
  ``regress_distance``.

The task / observation / reward logic is inherited from
:class:`go2.joystick.Joystick`.
"""

from typing import Any, Dict, Optional, Tuple, Union

from brax.envs.wrappers import training as brax_training
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go2 import base as go2_base
from mujoco_playground._src.locomotion.go2 import go2_constants as consts
from mujoco_playground._src.locomotion.go2 import joystick as go2_joystick
from mujoco_playground._src.locomotion.go2 import terrain_gen
from mujoco_playground._src.wrapper import Wrapper


def default_config() -> config_dict.ConfigDict:
  """Joystick config extended with terrain-grid and curriculum settings."""
  config = go2_joystick.default_config()
  # Hfield collisions need a larger contact/constraint budget than flat terrain.
  config.naconmax = 16 * 8192
  config.njmax = 80
  # Reward reshaping to avoid the "freeze in place" local optimum on rough
  # terrain. With the base joystick reward clipped to be non-negative, standing
  # still at the default pose earns a guaranteed ~+pose reward with zero fall
  # risk, so on risky terrain the agent prefers not to move (and then never
  # promotes in the curriculum). We shrink the free "pose" reward, reward
  # command tracking a bit more, and add an explicit penalty for the shortfall
  # between the commanded and the actual planar speed.
  config.reward_config.scales.pose = 0.1
  config.reward_config.scales.tracking_lin_vel = 1.5
  config.reward_config.scales.stand_still_moving = -0.5
  config.terrain = config_dict.create(
      num_rows=5,
      terrain_types=["rough", "slope", "stairs"],
      tile_size=8.0,
      resolution=0.1,
      difficulty_range=[0.0, 1.0],
      seed=0,
  )
  config.curriculum = config_dict.create(
      # Min distance (m) travelled to be promoted after surviving an episode.
      promote_distance=3.0,
      # If the agent falls before travelling this far (m), it is regressed.
      regress_distance=0.5,
  )
  return config


class RoughCurriculum(go2_joystick.Joystick):
  """Go2 joystick task on a tiled curriculum heightfield."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    # Bypass Joystick/Go2Env file-based __init__: we build the model from a
    # procedurally generated scene string instead.
    mjx_env.MjxEnv.__init__(self, config, config_overrides)

    self._model_assets = go2_base.get_assets()

    terr = terrain_gen.generate_terrain(
        num_rows=int(self._config.terrain.num_rows),
        terrain_types=list(self._config.terrain.terrain_types),
        tile_size=float(self._config.terrain.tile_size),
        resolution=float(self._config.terrain.resolution),
        difficulty_range=list(self._config.terrain.difficulty_range),
        seed=int(self._config.terrain.seed),
    )

    scene_xml = terrain_gen.build_scene_xml(
        nrow=terr["nrow"],
        ncol=terr["ncol"],
        radius_x=terr["radius_x"],
        radius_y=terr["radius_y"],
        z_top=terr["z_top"],
    )
    self._mj_model = mujoco.MjModel.from_xml_string(
        scene_xml, assets=self._model_assets
    )
    self._mj_model.opt.timestep = self._config.sim_dt
    self._mj_model.opt.ccd_iterations = 20

    # PD gains.
    self._mj_model.dof_damping[6:] = self._config.Kd
    self._mj_model.actuator_gainprm[:, 0] = self._config.Kp
    self._mj_model.actuator_biasprm[:, 1] = -self._config.Kp

    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    # Bake the procedural heightfield (normalized to [0, 1], scaled by size[2]).
    norm = (terr["heights"] / terr["z_top"]).ravel().astype(np.float32)
    self._mj_model.hfield_data[:] = norm

    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._xml_path = "<go2_rough_curriculum>"
    self._imu_site_id = self._mj_model.site("imu").id
    self._feet_floor_found_sensor = [
        self._mj_model.sensor(f"{geom}_floor_found").id
        for geom in consts.FEET_GEOMS
    ]

    self._post_init()

    # Curriculum bookkeeping (host -> device arrays).
    self._tile_centers = jp.asarray(terr["tile_centers"])  # (R, C, 2)
    self._spawn_heights = jp.asarray(terr["spawn_heights"])  # (R, C)
    self._terrain_types = terr["terrain_types"]
    self._num_rows = int(self._config.terrain.num_rows)
    self._num_cols = len(terr["terrain_types"])
    self._promote_distance = float(self._config.curriculum.promote_distance)
    self._regress_distance = float(self._config.curriculum.regress_distance)

  # ----- accessors -------------------------------------------------------- #

  @property
  def num_rows(self) -> int:
    return self._num_rows

  @property
  def num_cols(self) -> int:
    return self._num_cols

  @property
  def xml_path(self) -> str:
    return self._xml_path

  # ----- reset / step ----------------------------------------------------- #

  def reset(self, rng: jax.Array) -> mjx_env.State:
    """Default reset: spawn at difficulty level 0 on a random terrain column."""
    rng, key = jax.random.split(rng)
    col = jax.random.randint(key, (), 0, self._num_cols)
    level = jp.zeros((), dtype=jp.int32)
    return self.reset_to(rng, level, col)

  def reset_to(
      self, rng: jax.Array, level: jax.Array, col: jax.Array
  ) -> mjx_env.State:
    """Resets the robot onto the tile ``(level, col)`` of the grid."""
    state = super().reset(rng)

    level = jp.asarray(level, dtype=jp.int32)
    col = jp.asarray(col, dtype=jp.int32)
    center = self._tile_centers[level, col]  # (2,)
    spawn_h = self._spawn_heights[level, col]  # scalar

    qpos = state.data.qpos
    qpos = qpos.at[0].set(center[0])
    qpos = qpos.at[1].set(center[1])
    qpos = qpos.at[2].set(spawn_h + self._init_q[2])
    data = state.data.replace(qpos=qpos)
    data = mjx.forward(self.mjx_model, data)
    obs = self._get_obs(data, state.info)

    state.info["terrain_level"] = level
    state.info["terrain_col"] = col
    state.info["spawn_xy"] = center
    state.info["max_progress"] = jp.zeros(())

    return state.replace(data=data, obs=obs)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    state = super().step(state, action)
    xy = state.data.qpos[0:2]
    dist = jp.linalg.norm(xy - state.info["spawn_xy"])
    state.info["max_progress"] = jp.maximum(state.info["max_progress"], dist)
    return state

  # ----- reward ----------------------------------------------------------- #

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: Dict[str, Any],
      metrics: Dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,
      contact: jax.Array,
  ) -> Dict[str, jax.Array]:
    rewards = super()._get_reward(
        data, action, info, metrics, done, first_contact, contact
    )
    rewards["stand_still_moving"] = self._cost_stand_still_moving(
        info["command"], self.get_global_linvel(data)
    )
    return rewards

  def _cost_stand_still_moving(
      self, commands: jax.Array, global_linvel: jax.Array
  ) -> jax.Array:
    """Penalizes the shortfall between commanded and actual planar speed.

    Only active when a non-trivial command is issued, so it does not fight the
    ``stand_still`` reward that keeps the robot still on a zero command. This
    directly discourages the "freeze in place" strategy on risky terrain.
    """
    cmd_norm = jp.linalg.norm(commands)
    speed = jp.linalg.norm(global_linvel[:2])
    return jp.clip(cmd_norm - speed, 0.0, None) * (cmd_norm > 0.1)

  # ----- curriculum logic ------------------------------------------------- #

  def compute_next_tile(
      self,
      terrain_level: jax.Array,
      terrain_col: jax.Array,
      max_progress: jax.Array,
      truncation: jax.Array,
  ) -> Tuple[jax.Array, jax.Array]:
    """Computes the next spawn tile from the finished episode's outcome.

    Operates elementwise, so it works on both scalars and batched arrays.

    Args:
      terrain_level: current difficulty row.
      terrain_col: current terrain column (kept unchanged).
      max_progress: max planar distance travelled during the episode.
      truncation: 1.0 if the episode timed out (survived), 0.0 if the agent
        fell.

    Returns:
      ``(new_level, terrain_col)`` with ``new_level`` clipped to
      ``[0, num_rows - 1]``.
    """
    survived = truncation > 0.5
    promote = survived & (max_progress > self._promote_distance)
    regress = (~survived) & (max_progress < self._regress_distance)
    delta = promote.astype(jp.int32) - regress.astype(jp.int32)
    new_level = jp.clip(
        terrain_level.astype(jp.int32) + delta, 0, self._num_rows - 1
    )
    return new_level, terrain_col.astype(jp.int32)


class CurriculumAutoResetWrapper(Wrapper):
  """Auto-reset wrapper that repositions agents according to the curriculum.

  Mirrors brax's auto-reset, but on ``done`` it reads each environment's
  episode outcome from ``state.info`` (``terrain_level``, ``terrain_col``,
  ``max_progress``, ``truncation``) and respawns the robot on the
  curriculum-selected tile, instead of resetting to a fixed cached state.

  Must wrap an ``EpisodeWrapper(VmapWrapper(RoughCurriculum))`` stack so that
  ``truncation`` / ``steps`` are populated and ``reset`` / ``step`` are batched.
  """

  def __init__(self, env: Any):
    super().__init__(env)
    self._info_key = "AutoReset"

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng_key = jax.vmap(jax.random.split)(rng)
    rng, key = rng_key[..., 0], rng_key[..., 1]
    state = self.env.reset(key)
    state.info[f"{self._info_key}_rng"] = rng
    state.info[f"{self._info_key}_done_count"] = jp.zeros(
        key.shape[:-1], dtype=int
    )
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    base = self.env.unwrapped

    rng_key = jax.vmap(jax.random.split)(state.info[f"{self._info_key}_rng"])
    reset_rng, reset_key = rng_key[..., 0], rng_key[..., 1]

    if "steps" in state.info:
      steps = state.info["steps"]
      steps = jp.where(state.done, jp.zeros_like(steps), steps)
      state.info.update(steps=steps)

    state = state.replace(done=jp.zeros_like(state.done))
    state = self.env.step(state, action)

    # Decide the next spawn tile for every (potentially) done environment.
    new_level, new_col = base.compute_next_tile(
        state.info["terrain_level"],
        state.info["terrain_col"],
        state.info["max_progress"],
        state.info["truncation"],
    )

    # Full-structure reset (level 0, random col) reused only for its pytree
    # layout; data / obs / tile info are overridden with the curriculum tile.
    reset_state = self.reset(reset_key)
    repos = jax.vmap(base.reset_to)(reset_key, new_level, new_col)
    rinfo = reset_state.info
    rinfo["terrain_level"] = new_level
    rinfo["terrain_col"] = new_col
    rinfo["spawn_xy"] = repos.info["spawn_xy"]
    rinfo["max_progress"] = repos.info["max_progress"]
    rinfo["command"] = repos.info["command"]
    reset_state = reset_state.replace(
        data=repos.data, obs=repos.obs, info=rinfo
    )

    def where_done(x, y):
      done = state.done
      if done.shape and done.shape[0] != x.shape[0]:
        return y
      if done.shape:
        done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
      return jp.where(done, x, y)

    data = jax.tree.map(where_done, reset_state.data, state.data)
    obs = jax.tree.map(where_done, reset_state.obs, state.obs)
    next_info = jax.tree.map(where_done, reset_state.info, state.info)

    done_count_key = f"{self._info_key}_done_count"
    next_info[done_count_key] = state.info[done_count_key]
    if "steps" in next_info:
      next_info["steps"] = state.info["steps"]
    next_info[done_count_key] += state.done.astype(int)
    next_info[f"{self._info_key}_rng"] = reset_rng

    return state.replace(data=data, obs=obs, info=next_info)


def wrap_for_curriculum_training(
    env: mjx_env.MjxEnv,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[Any] = None,
) -> Wrapper:
  """Brax-compatible wrap function with the curriculum auto-reset.

  Drop-in replacement for ``wrapper.wrap_for_brax_training`` (same signature),
  to be passed as ``wrap_env_fn`` to ``brax`` PPO training. Domain randomization
  is not supported together with the terrain curriculum.
  """
  if randomization_fn is not None:
    raise NotImplementedError(
        "Domain randomization is not supported with the terrain curriculum."
    )
  env = brax_training.VmapWrapper(env)
  env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
  env = CurriculumAutoResetWrapper(env)
  return env
