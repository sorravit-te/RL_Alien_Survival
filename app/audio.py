"""What the player hears. Read-only snapshots go in and sound comes out.

The Alien's own hearing lives in game/sound.py and works without a speaker, so nothing here
can change a round. Sounds come from `python -m tools.bake_ship_sounds`; without them, or
without a sound device, the game simply stays silent.
"""

from math import cos, hypot, sin
from pathlib import Path
from random import Random

import pygame

from game.doors import sight_world
from game.interaction import segment_clear

SOUNDS = Path(__file__).resolve().parents[1] / "assets" / "sounds" / "prepared"
CHANNELS = 32
HUM_CHANNEL = 0

# Cells between footfalls; a run takes about three steps a second, as people do.
WALK_STRIDE, RUN_STRIDE, ALIEN_STRIDE = 1.3, 1.2, 1.7
ALIEN_REACH, DOOR_REACH = 16.0, 14.0  # cells a sound carries to the player
FALL_OFF = 1.5  # higher fades a sound faster with distance
THROUGH_WALL = 0.5  # a sound is this much quieter with no clear line to the player
SIDE_WIDTH = 0.8  # 0 keeps both ears equal, 1 mutes the far ear
ROAR_GAP_SECONDS = 10.0
GROWL_GAP_SECONDS = (8.0, 16.0)

HUM, WALK, RUN = 0.5, 0.2, 0.4
ALIEN_STEP, GROWL, ROAR = 1.0, 0.7, 1.0
DOOR, INTERFACE = 0.42, 0.6


class Speaker:
    """pygame.mixer behind a small surface, so the rules can be tested without a device."""

    def __init__(self, folder=SOUNDS):
        self.sounds = {}
        paths = sorted(folder.glob("*.wav")) if folder.is_dir() else ()
        if not paths:
            print("No sound: run python -m tools.bake_ship_sounds")
            return
        try:
            pygame.mixer.init(44100, -16, 2, 512)
            pygame.mixer.set_num_channels(CHANNELS)
            pygame.mixer.set_reserved(1)
            self.sounds = {path.stem: pygame.mixer.Sound(path) for path in paths}
        except pygame.error as exc:
            print(f"No sound: {exc}")

    def names(self, prefix):
        return sorted(name for name in self.sounds if name.startswith(prefix))

    def play(self, name, left, right):
        if name in self.sounds:
            channel = self.sounds[name].play()
            if channel is not None:  # every channel busy: drop it, like a real crowded mix
                channel.set_volume(left, right)

    def loop(self, name, volume):
        if name in self.sounds:
            channel = pygame.mixer.Channel(HUM_CHANNEL)
            channel.play(self.sounds[name], loops=-1)
            channel.set_volume(volume)

    def hold(self):
        if self.sounds:
            pygame.mixer.pause()

    def resume(self):
        if self.sounds:
            pygame.mixer.unpause()

    def stop_effects(self):
        if self.sounds:
            for index in range(HUM_CHANNEL + 1, CHANNELS):
                pygame.mixer.Channel(index).stop()


def ears(snapshot, source, reach):
    """Left and right gain for a sound at source, or None when it does not carry that far."""
    survivor = snapshot.survivor
    dx, dy = source[0] - survivor.x, source[1] - survivor.y
    distance = hypot(dx, dy)
    if distance >= reach:
        return None
    gain = (1 - distance / reach) ** FALL_OFF
    if distance and not segment_clear(sight_world(snapshot), survivor.x, survivor.y, *source):
        gain *= THROUGH_WALL
    # Yaw 0 faces +x and yaw 90 faces +y, so the right ear points along (-sin, cos).
    right_of = (-dx * sin(survivor.yaw) + dy * cos(survivor.yaw)) / distance if distance else 0.0
    return (
        gain * min(1.0, 1 - right_of * SIDE_WIDTH),
        gain * min(1.0, 1 + right_of * SIDE_WIDTH),
    )


class ShipAudio:
    def __init__(self, speaker, seed=0):
        self.speaker = speaker
        self.rng = Random(seed)
        self.groups = {name: speaker.names(name) for name in ("walk", "alien_step")}
        self.groups["growl"] = speaker.names("alien_growl")
        self.heard_last = {}
        self.held = False
        self.forget()

    def forget(self):
        self.last = None
        self.player_walked = WALK_STRIDE
        self.alien_walked = 0.0
        self.next_roar = 0
        self.next_growl = None

    def start(self):
        self.speaker.loop("ship_hum", HUM)

    def follow(self, paused, snapshot):
        """Once per frame: sound waits during a pause and drops on a restart."""
        hold = paused and snapshot.outcome is None
        if hold != self.held:
            self.held = hold
            (self.speaker.hold if hold else self.speaker.resume)()
        if self.last is not None and snapshot.tick < self.last.tick:
            self.speaker.stop_effects()
            self.forget()

    def hear(self, now):
        """One Core tick; called for every tick, so an event between frames is not lost."""
        before, self.last = self.last, now
        if before is None:
            return
        self._player_steps(before, now)
        self._doors(before, now)
        if now.alien is not None and before.alien is not None:
            self._alien(before, now)
        for event in now.events:
            self._event(event, now)

    def _vary(self, group):
        """A random take, never the same one twice running."""
        takes = self.groups[group]
        if not takes:
            return None
        fresh = [take for take in takes if take != self.heard_last.get(group)] or takes
        self.heard_last[group] = self.rng.choice(fresh)
        return self.heard_last[group]

    def _at(self, name, snapshot, source, reach, volume):
        heard = ears(snapshot, source, reach)
        if heard is not None:
            self.speaker.play(name, heard[0] * volume, heard[1] * volume)

    def _player_steps(self, before, now):
        moved = hypot(now.survivor.x - before.survivor.x, now.survivor.y - before.survivor.y)
        if not moved:
            self.player_walked = WALK_STRIDE  # the next step sounds as soon as it is taken
            return
        speed = moved * now.config.tick_hz
        running = speed > (now.config.walk_speed + now.config.run_speed) / 2
        self.player_walked += moved
        if self.player_walked >= (RUN_STRIDE if running else WALK_STRIDE):
            self.player_walked = 0.0
            volume = RUN if running else WALK
            self.speaker.play(self._vary("walk"), volume, volume)

    def _doors(self, before, now):
        for cell, old, new in zip(
            now.world_map.doors, before.door_open, now.door_open, strict=True
        ):
            name = "door_open" if old == 0.0 < new else "door_close" if old == 1.0 > new else None
            if name is not None:
                self._at(name, now, (cell[0] + 0.5, cell[1] + 0.5), DOOR_REACH, DOOR)

    def _alien(self, before, now):
        place = (now.alien.x, now.alien.y)
        self.alien_walked += hypot(place[0] - before.alien.x, place[1] - before.alien.y)
        if self.alien_walked >= ALIEN_STRIDE:
            self.alien_walked = 0.0
            self._at(self._vary("alien_step"), now, place, ALIEN_REACH, ALIEN_STEP)
        spotted = now.perception.visible and not before.perception.visible
        if spotted and now.tick >= self.next_roar:
            self.next_roar = now.tick + ROAR_GAP_SECONDS * now.config.tick_hz
            self._at("alien_roar", now, place, ALIEN_REACH, ROAR)
        if self.next_growl is None or now.tick >= self.next_growl:
            if self.next_growl is not None:
                self._at(self._vary("growl"), now, place, ALIEN_REACH, GROWL)
            gap = self.rng.uniform(*GROWL_GAP_SECONDS)
            self.next_growl = now.tick + gap * now.config.tick_hz

    def _event(self, event, now):
        if event.kind == "captured":
            self.speaker.play("alien_roar", ROAR, ROAR)
        elif event.kind == "keycard_collected":
            self.speaker.play("keycard", INTERFACE, INTERFACE)
        elif event.kind == "exit_offline":
            self.speaker.play("exit_offline", INTERFACE, INTERFACE)
        elif event.kind == "escaped":
            self.speaker.play("escaped", INTERFACE, INTERFACE)
