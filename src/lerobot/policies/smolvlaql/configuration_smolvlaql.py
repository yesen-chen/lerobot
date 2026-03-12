# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("smolvlaql")
@dataclass
class SmolVLAQLConfig(SmolVLAConfig):
    """SmolVLA config extended with QC-FQL (Chunked Flow Q-Learning) parameters.

    Adds a value head on top of the shared Action Expert backbone for Q-value estimation.
    During critic training, expert hidden states are detached so only the value head is updated.
    During policy training, the standard flow matching loss updates the expert backbone + action head.
    """

    # Value Head architecture
    qc_value_hidden_dims: tuple[int, ...] = (512, 512)
    qc_num_critics: int = 2

    # Target network EMA coefficient
    qc_tau: float = 0.005

    # TD discount factor (applied as gamma^chunk_size for chunked transitions)
    qc_discount: float = 0.99

    # Best-of-N sampling count for inference and target Q computation
    qc_best_of_n: int = 32

    # Critic learning rate (separate from policy lr)
    qc_critic_lr: float = 3e-4

    # Ensemble Q-value aggregation: "mean" or "min"
    qc_q_agg: str = "mean"
