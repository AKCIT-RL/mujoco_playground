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
"""Push-recovery task for Go2.

This is a config variant of the joystick locomotion task with external
perturbations ("kicks") enabled by default. It adds two dedicated robustness
metrics that are useful when evaluating offline RL policies:

  - ``survival_time``: time (in seconds) elapsed before the robot falls. This is
    the "time-to-fall" metric.
  - ``recovery_time``: time (in seconds) the robot takes to return to an upright
    orientation after a kick has been applied.

The locomotion dynamics, observations and rewards are inherited unchanged from
:class:`mujoco_playground._src.locomotion.go2.joystick.Joystick`. Only the
perturbation schedule and the extra metrics are added here.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go2 import joystick as go2_joystick


def default_config() -> config_dict.ConfigDict:
  """Joystick config with perturbations enabled for push recovery."""
  config = go2_joystick.default_config()
  config.pert_config.enable = True
  # Stronger and more frequent kicks than the joystick defaults so that the
  # recovery behaviour is actually exercised.
  config.pert_config.velocity_kick = [1.0, 4.0]
  config.pert_config.kick_durations = [0.05, 0.2]
  config.pert_config.kick_wait_times = [1.0, 3.0]
  return config


class PushRecovery(go2_joystick.Joystick):
  """Track a joystick command while recovering from external pushes."""

  # Upright when the body z-axis (upvector) is close to the world z-axis.
  _UPRIGHT_THRESHOLD = 0.9

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        task=task, config=config, config_overrides=config_overrides
    )

  def reset(self, rng: jax.Array) -> mjx_env.State:
    state = super().reset(rng)
    state.info["survival_steps"] = jp.zeros(())
    state.info["recovering"] = jp.zeros(())
    state.info["recovery_steps"] = jp.zeros(())
    state.info["recovery_time"] = jp.zeros(())
    state.metrics["survival_time"] = jp.zeros(())
    state.metrics["recovery_time"] = jp.zeros(())
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    state = super().step(state, action)

    # Time-to-fall: count steps survived until the episode terminates.
    not_done = 1.0 - state.done
    survival_steps = (state.info["survival_steps"] + 1.0) * not_done
    state.info["survival_steps"] = survival_steps
    state.metrics["survival_time"] = survival_steps * self.dt

    # A kick is being applied while the perturbation window is active.
    kick_active = (
        state.info["steps_since_last_pert"]
        >= state.info["steps_until_next_pert"]
    )
    upright = self.get_upvector(state.data)[-1] > self._UPRIGHT_THRESHOLD

    # Enter "recovering" mode whenever a kick is active, then count the steps it
    # takes (after the kick ends) to return to an upright orientation.
    recovering = jp.maximum(state.info["recovering"], kick_active.astype(float))
    counting = recovering * (1.0 - kick_active.astype(float))
    recovery_steps = state.info["recovery_steps"] + counting
    finished = (recovering > 0) & upright & jp.logical_not(kick_active)
    recovery_time = jp.where(
        finished, recovery_steps * self.dt, state.info["recovery_time"]
    )
    recovery_steps = jp.where(finished, 0.0, recovery_steps)
    recovering = jp.where(finished | (state.done > 0), 0.0, recovering)

    state.info["recovering"] = recovering
    state.info["recovery_steps"] = recovery_steps
    state.info["recovery_time"] = recovery_time
    state.metrics["recovery_time"] = recovery_time

    return state
