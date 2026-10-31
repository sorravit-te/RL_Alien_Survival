"""Player-side sound: it follows the deck from snapshots and can never change a round."""

import os
import wave
from dataclasses import replace

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame
import pytest

from app import audio as sounds
from app.audio import ShipAudio, Speaker, ears
from app.loop import FixedStepLoop
from game.core import GameCore
from game.map import WorldMap
from game.perception import PerceptionSnapshot
from game.state import AlienState, CoreConfig, GameEvent, SurvivorState
from tools.bake_ship_sounds import (
    ALIEN_STEP_OVERTONES,
    HUM_OVERTONES,
    RATE,
    alien_footfall,
    footfalls,
    onsets,
    seamless,
    with_overtones,
)

HALL = (
    "####################",
    "#..................#",
    "#..................#",
    "#..................#",
    "#..................#",
    "####################",
)
SPLIT = (
    "##########",
    "#...#....#",
    "#...#....#",
    "##########",
)
DOOR_HALL = (
    "##########",
    "#........#",
    "##########",
)


class Recorder:
    """Stands in for the mixer and keeps a list of what would have been heard."""

    def __init__(self):
        self.played, self.looped = [], []
        self.holds = self.resumes = self.stops = 0

    def names(self, prefix):
        return [f"{prefix}_{index}" for index in range(3)]

    def play(self, name, left, right):
        self.played.append((name, left, right))

    def loop(self, name, volume):
        self.looped.append((name, volume))

    def hold(self):
        self.holds += 1

    def resume(self):
        self.resumes += 1

    def stop_effects(self):
        self.stops += 1

    def heard(self, prefix):
        return [row for row in self.played if row[0].startswith(prefix)]


def deck(rows=HALL, *, doors=(), alien=None, x=5.5, y=2.5, yaw=0.0):
    world = WorldMap(rows, ((x, y),), doors=doors)
    snapshot = GameCore(world, CoreConfig(alien_enabled=False)).snapshot()
    return replace(snapshot, survivor=SurvivorState(x, y, yaw), alien=alien)


def walked(audio, start, speed, ticks):
    """Feed ticks of the Survivor moving along +x; returns the last snapshot."""
    audio.hear(start)
    now = start
    step = speed / start.config.tick_hz
    for _ in range(ticks):
        survivor = replace(now.survivor, x=now.survivor.x + step)
        now = replace(now, tick=now.tick + 1, survivor=survivor)
        audio.hear(now)
    return now


def fresh():
    speaker = Recorder()
    return speaker, ShipAudio(speaker)


def test_standing_still_is_silent():
    speaker, audio = fresh()
    snapshot = deck()
    audio.hear(snapshot)
    for tick in range(1, 90):
        audio.hear(replace(snapshot, tick=tick))
    assert speaker.played == []


def test_the_first_step_sounds_at_once_then_once_per_stride():
    speaker, audio = fresh()
    start = deck(x=2.5)
    walked(audio, start, start.config.walk_speed, 1)
    assert len(speaker.heard("walk")) == 1
    speaker, audio = fresh()
    walked(audio, start, start.config.walk_speed, 60)  # five cells at 1.3 cells a stride
    assert len(speaker.heard("walk")) == 4


def test_running_uses_the_walking_takes_quicker_and_louder():
    start = deck(x=2.5)
    walker, audio = fresh()
    walked(audio, start, start.config.walk_speed, 60)
    runner, audio = fresh()
    walked(audio, start, start.config.run_speed, 60)
    walking, running = walker.heard("walk"), runner.heard("walk")
    assert len(running) == 7 > 4 == len(walking)  # two seconds of each
    assert running[0][1] == sounds.RUN > sounds.WALK == walking[0][1]


def test_a_run_takes_about_three_steps_a_second():
    config = deck().config
    assert 2.8 <= config.run_speed / sounds.RUN_STRIDE <= 3.5


def test_the_same_take_never_plays_twice_running():
    speaker, audio = fresh()
    start = deck(x=1.5)
    walked(audio, start, start.config.walk_speed, 150)
    names = [row[0] for row in speaker.heard("walk")]
    assert len(names) > 5 and all(a != b for a, b in zip(names, names[1:]))


@pytest.mark.parametrize(
    ("yaw", "place", "louder"),
    [
        (0.0, (5.5, 4.5), "right"),  # facing +x, +y is on the right
        (0.0, (5.5, 1.5), "left"),
        (np.pi, (5.5, 4.5), "left"),
    ],
)
def test_ears_tell_left_from_right(yaw, place, louder):
    left, right = ears(deck(yaw=yaw), place, 16.0)
    assert (right > left) == (louder == "right")


def test_sound_straight_ahead_is_even_and_fades_to_nothing():
    snapshot = deck(x=1.5)
    near, far = ears(snapshot, (3.5, 2.5), 16.0), ears(snapshot, (12.5, 2.5), 16.0)
    assert near[0] == near[1] and far[0] == far[1] and near[0] > far[0] > 0
    assert ears(snapshot, (17.5, 2.5), 16.0) is None


def test_a_sound_on_top_of_the_player_is_full_and_even():
    assert ears(deck(), (5.5, 2.5), 16.0) == (1.0, 1.0)


def test_a_wall_muffles_sound():
    snapshot = deck(SPLIT, x=2.5, y=1.5)
    clear = ears(snapshot, (1.5, 1.5), 16.0)
    behind = ears(snapshot, (6.5, 1.5), 16.0)
    behind_gain = (1 - 4.0 / 16.0) ** sounds.FALL_OFF * sounds.THROUGH_WALL
    assert behind[0] == pytest.approx(behind_gain) and clear[0] > behind[0]


def test_a_door_sounds_when_it_starts_to_move_and_not_while_moving():
    speaker, audio = fresh()
    shut = replace(deck(DOOR_HALL, doors=((5, 1),), x=2.5, y=1.5), door_open=(0.0,))
    audio.hear(shut)
    for tick, amount in enumerate((0.1, 0.5, 1.0, 1.0, 0.9, 0.5, 0.0), 1):
        audio.hear(replace(shut, tick=tick, door_open=(amount,)))
    assert [row[0] for row in speaker.played] == ["door_open", "door_close"]


def test_alien_footfalls_follow_its_walk_and_it_is_quiet_when_it_stands():
    speaker, audio = fresh()
    start = deck(alien=AlienState(10.5, 2.5))
    audio.hear(start)
    now = start
    for tick in range(1, 30):
        now = replace(now, tick=tick, alien=AlienState(10.5 + tick * 0.2, 2.5))
        audio.hear(now)
    steps = speaker.heard("alien_step")
    assert len(steps) == 3  # 5.8 cells at 1.7 cells a stride
    assert steps[0][1] > steps[-1][1]  # it is walking away
    speaker.played.clear()
    for tick in range(30, 90):
        audio.hear(replace(now, tick=tick))
    assert not speaker.heard("alien_step")


def test_being_spotted_roars_but_not_again_right_away():
    speaker, audio = fresh()
    unseen = deck(alien=AlienState(10.5, 2.5))
    seen = replace(unseen, perception=PerceptionSnapshot(visible=True))
    audio.hear(unseen)
    audio.hear(replace(seen, tick=1))
    audio.hear(replace(unseen, tick=2))
    audio.hear(replace(seen, tick=3))
    assert len(speaker.heard("alien_roar")) == 1
    gap = int(sounds.ROAR_GAP_SECONDS * unseen.config.tick_hz)
    audio.hear(replace(unseen, tick=gap + 2))
    audio.hear(replace(seen, tick=gap + 3))
    assert len(speaker.heard("alien_roar")) == 2


def test_growls_come_now_and_then():
    speaker, audio = fresh()
    start = deck(alien=AlienState(9.5, 2.5))
    audio.hear(start)
    for tick in range(1, 60 * start.config.tick_hz):
        audio.hear(replace(start, tick=tick))
    assert 3 <= len(speaker.heard("alien_growl")) <= 8


def test_game_events_have_their_sounds():
    speaker, audio = fresh()
    start = deck()
    audio.hear(start)
    kinds = ("keycard_collected", "exit_offline", "escaped", "captured")
    for tick, kind in enumerate(kinds, 1):
        audio.hear(replace(start, tick=tick, events=(GameEvent(kind, "x", tick),)))
    names = [row[0] for row in speaker.played]
    assert names == ["keycard", "exit_offline", "escaped", "alien_roar"]
    assert speaker.played[-1][1:] == (sounds.ROAR, sounds.ROAR)


def test_a_pause_holds_sound_but_the_end_of_a_round_plays_out():
    speaker, audio = fresh()
    playing = deck()
    audio.follow(True, playing)
    audio.follow(True, playing)
    assert speaker.holds == 1
    audio.follow(False, playing)
    assert speaker.resumes == 1
    audio.follow(True, replace(playing, outcome="captured"))
    assert speaker.holds == 1


def test_a_restart_drops_effects_and_starts_counting_again():
    speaker, audio = fresh()
    start = deck(x=2.5)
    last = walked(audio, start, start.config.walk_speed, 10)
    audio.follow(False, replace(last, tick=0))
    assert speaker.stops == 1 and audio.last is None
    audio.hear(replace(start, tick=0))
    assert audio.last.tick == 0 and audio.player_walked == sounds.WALK_STRIDE


def test_the_hum_starts_on_its_own_loop():
    speaker, audio = fresh()
    audio.start()
    assert speaker.looped == [("ship_hum", sounds.HUM)]


def test_the_loop_reports_every_tick_even_when_a_frame_runs_several():
    core = GameCore(WorldMap(HALL, ((5.5, 2.5),)), CoreConfig(alien_enabled=False))
    loop = FixedStepLoop(core)
    ticks = []
    loop.on_tick = lambda snapshot: ticks.append(snapshot.tick)
    assert loop.advance(0.15) == 4
    assert ticks == [1, 2, 3, 4]


def test_hearing_a_round_leaves_core_untouched():
    def play(listener):
        core = GameCore(WorldMap(HALL, ((5.5, 2.5),)), CoreConfig(alien_enabled=False))
        loop = FixedStepLoop(core)
        loop.on_tick = listener
        loop.inputs.key("w", True)
        for _ in range(30):
            loop.advance(1 / 30)
        return core.snapshot().survivor

    assert play(None) == play(ShipAudio(Recorder()).hear)


def tiny_wav(path):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes(np.zeros(4410, dtype=np.int16).tobytes())


def test_a_speaker_with_no_sounds_stays_silent(tmp_path, capsys):
    speaker = Speaker(tmp_path / "nothing")
    speaker.play("walk_0", 1.0, 1.0)
    speaker.loop("ship_hum", 1.0)
    speaker.hold()
    speaker.resume()
    speaker.stop_effects()
    assert speaker.names("walk") == [] and "bake_ship_sounds" in capsys.readouterr().out
    assert not pygame.mixer.get_init()


def test_a_speaker_with_no_device_stays_silent(tmp_path, monkeypatch, capsys):
    tiny_wav(tmp_path / "walk_0.wav")

    def no_device(*_, **__):
        raise pygame.error("no audio device")

    monkeypatch.setattr(pygame.mixer, "init", no_device)
    speaker = Speaker(tmp_path)
    speaker.play("walk_0", 1.0, 1.0)
    assert speaker.sounds == {} and "no audio device" in capsys.readouterr().out


def test_a_real_speaker_plays_a_sound_and_keeps_the_hum_channel(tmp_path):
    tiny_wav(tmp_path / "walk_0.wav")
    tiny_wav(tmp_path / "ship_hum.wav")
    speaker = Speaker(tmp_path)
    try:
        assert speaker.names("walk") == ["walk_0"]
        speaker.loop("ship_hum", 0.5)
        speaker.play("walk_0", 0.2, 0.8)
        speaker.play("not_baked", 1.0, 1.0)
        hum = pygame.mixer.Channel(sounds.HUM_CHANNEL)
        assert hum.get_busy() and hum.get_volume() == pytest.approx(0.5, abs=0.01)
        speaker.stop_effects()
        assert hum.get_busy() and pygame.mixer.get_busy()
        speaker.hold()
        speaker.resume()
    finally:
        pygame.mixer.quit()


def clicks(times, seconds):
    audio = np.zeros((int(seconds * RATE), 2), dtype=np.float32)
    for time in times:
        start = int(time * RATE)
        burst = np.exp(-np.arange(2000) / 300.0) * 0.8
        audio[start : start + 2000] += burst[:, None]
    return audio


def test_onsets_find_each_footfall():
    found = onsets(clicks((0.5, 1.1, 1.7), 2.4).mean(axis=1))
    assert np.allclose(np.array(found) / RATE, (0.5, 1.1, 1.7), atol=0.02)


def test_footfalls_cut_short_one_shots_that_end_quiet():
    takes = footfalls(clicks((0.5, 1.1, 1.7, 2.3), 3.2), 2, 0.3)
    assert len(takes) == 2
    assert all(len(take) == int(0.304 * RATE) and np.abs(take[-1]).max() < 1e-3 for take in takes)


def share_a_small_speaker_plays(audio):
    power = np.abs(np.fft.rfft(audio[:, 0])) ** 2
    return power[np.fft.rfftfreq(len(audio), 1 / RATE) > 300].sum() / power.sum()


def test_overtones_reach_pitches_a_small_speaker_can_play():
    t = np.arange(RATE) / RATE
    tone = 0.3 * np.sin(2 * np.pi * 110 * t)
    low = np.stack((tone, tone), axis=1)
    richer = with_overtones(low, HUM_OVERTONES)
    assert share_a_small_speaker_plays(low) < 0.001 < 0.05 < share_a_small_speaker_plays(richer)


def test_the_alien_footfall_stays_audible_on_a_small_speaker():
    step = alien_footfall(np.random.default_rng(7))
    richer = with_overtones(step, ALIEN_STEP_OVERTONES)
    assert share_a_small_speaker_plays(richer) > 5 * share_a_small_speaker_plays(step)


def test_overtones_keep_a_loop_seamless():
    t = np.arange(RATE) / RATE
    tone = 0.3 * np.sin(2 * np.pi * 110 * t)
    richer = with_overtones(np.stack((tone, tone), axis=1), HUM_OVERTONES)
    assert abs(richer[0, 0] - richer[-1, 0]) < 0.05


def test_the_hum_loop_joins_without_a_jump():
    t = np.arange(int(5 * RATE)) / RATE
    tone = np.sin(2 * np.pi * 440 * t)
    audio = np.stack((tone, tone), axis=1).astype(np.float32)
    loop = seamless(audio, 4.0, 1.0)
    assert len(loop) == 4 * RATE
    assert abs(loop[0, 0] - audio[4 * RATE, 0]) < 1e-3
