"""The full-deck pilot configs agree with the deck, the contract and each other."""

import json
from pathlib import Path

from game.ship_deck import SHIP_SCENARIO
from rl.contract import SEARCH_CONTROL, VERSION
from rl.evaluation_protocol import evaluation_budget, validate_protocol
from rl.imitation import validate_imitation
from rl.pipeline import ROOT, make_env, validate_environment_config

TRAINING = "configs/ship_deck_training_pilot.json"
SCRATCH = "configs/ship_deck_training_full_scratch.json"


def read(path):
    return json.loads((Path(ROOT) / path).read_text("utf-8"))


def test_pilot_trains_on_the_deck_against_the_runner_mix():
    config = read(TRAINING)
    assert config["scenario"] == SHIP_SCENARIO and config["bot_kind"] == "runner_mix"
    assert config["contract_version"] == VERSION and config["search_control"] == SEARCH_CONTROL
    assert validate_environment_config(config)["name"] == SHIP_SCENARIO


def test_evaluation_protocol_matches_training():
    config = read(TRAINING)
    validate_protocol(config, read(config["evaluation_protocol"]))


def test_pilot_environment_uses_the_room_search():
    env = make_env(read(TRAINING))
    assert env.scenario == SHIP_SCENARIO and env.bot.runner.explorer.walk_steps


def test_pilot_starts_from_a_copy_of_the_written_alien():
    validate_imitation(read(TRAINING)["imitation"])


def test_pilot_budget_stays_bounded():
    budget = evaluation_budget(read(TRAINING))
    assert budget["baseline_validation_episodes"] + budget["checkpoint_validation_episodes"] <= 800

def test_the_from_scratch_run_differs_from_the_full_run_only_by_the_copy():
    """Its PPO starts untrained, so the pair shows what reward teaches without the written Alien."""
    full, scratch = read("configs/ship_deck_training_full.json"), read(SCRATCH)
    assert "imitation" in full and "imitation" not in scratch
    assert {**full, "name": "", "imitation": {}} == {**scratch, "name": "", "imitation": {}}
    validate_protocol(scratch, read(scratch["evaluation_protocol"]))
