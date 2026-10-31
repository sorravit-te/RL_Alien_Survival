"""The copy step fits the written Alien and leaves PPO's own counters untouched."""

import pytest
import torch
from sb3_contrib import MaskablePPO

from rl.env import AlienEnv
from rl.imitation import collect_examples, copy_heuristic, validate_imitation
from rl.pipeline import parameter_hash

SETTINGS = {
    "decisions": 60,
    "random_action_share": 0.2,
    "epochs": 30,
    "batch_size": 32,
    "learning_rate": 0.001,
    "value_weight": 0.05,
}
SCENE = "occluded_loop_random_v1"


class Episodes:
    """Plain seeded resets, standing in for the training manifest."""

    def __init__(self):
        self.env = AlienEnv(scenario=SCENE, bot_kind="evasive_mix")
        self.unwrapped = self.env
        self.seed = 0

    def reset(self):
        self.seed += 1
        return self.env.reset(
            seed=self.seed, options={"scenario_seed": 700 + self.seed, "bot_seed": 900 + self.seed}
        )

    def step(self, action):
        return self.env.step(action)


def new_model():
    return MaskablePPO(
        "MlpPolicy", AlienEnv(scenario=SCENE, bot_kind="evasive_mix"), seed=3, n_steps=64
    )


def test_examples_skip_forced_decisions_and_are_repeatable():
    first = collect_examples(Episodes(), SETTINGS, 1, 0.995)
    again = collect_examples(Episodes(), SETTINGS, 1, 0.995)
    x, m, y, g = first
    assert len(y) >= SETTINGS["decisions"] and (m.sum(dim=1) > 1).all()
    assert m[torch.arange(len(y)), y].all()
    assert torch.equal(x, again[0]) and torch.equal(y, again[2]) and torch.equal(g, again[3])


def agreement(model, examples):
    x, m, y, _ = examples
    distribution = model.policy.get_distribution(x, action_masks=m)
    return (distribution.get_actions(deterministic=True) == y).float().mean().item()


def test_copying_learns_the_written_alien_without_counting_as_training():
    model = new_model()
    examples = collect_examples(Episodes(), SETTINGS, 2, 0.995)
    before, start = agreement(model, examples), parameter_hash(model)
    copied = copy_heuristic(model, Episodes(), SETTINGS, seed=2, gamma=0.995)
    assert copied >= SETTINGS["decisions"] and parameter_hash(model) != start
    assert agreement(model, examples) > before + 0.2
    assert model.num_timesteps == 0 and model._n_updates == 0


def test_copying_is_repeatable():
    one, two = new_model(), new_model()
    copy_heuristic(one, Episodes(), SETTINGS, seed=2, gamma=0.995)
    copy_heuristic(two, Episodes(), SETTINGS, seed=2, gamma=0.995)
    assert parameter_hash(one) == parameter_hash(two)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("decisions", 0),
        ("epochs", 1.5),
        ("batch_size", -1),
        ("learning_rate", 0.0),
        ("value_weight", -0.1),
        ("random_action_share", 1.0),
    ],
)
def test_bad_settings_are_refused(name, value):
    with pytest.raises(ValueError):
        validate_imitation({**SETTINGS, name: value})
