"""Bake free CC0 recordings into the sounds the ship plays.

Usage: python -m tools.bake_ship_sounds

Recordings download once into assets/sounds/source/. The cut files go to
assets/sounds/prepared/. Both stay local, like the baked textures. Only the Alien's heavy
footfall is made here; it needs no recording.
"""

import io
import os
import wave
import zipfile
from statistics import median
from urllib.request import Request, urlopen

import numpy as np

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from tools.asset_paths import ROOT  # noqa: E402

RATE = 44100
SOURCE = ROOT / "assets" / "sounds" / "source"
PREPARED = ROOT / "assets" / "sounds" / "prepared"
FREESOUND = "https://cdn.freesound.org/previews/{}/{}_{}-hq.mp3"
TERMINALS = "https://opengameart.org/sites/default/files/sci-fi-sfx.zip"
RECORDINGS = {  # name: (Freesound id, uploader id, credit)
    "walk": (398937, 5993580, "Metal Footsteps by SoftDistortionFX"),
    "door": (607265, 11069322, "Sci-Fi Doors-Airlock Sound Effect by RyanKingArt"),
    "roar": (398908, 5411212, "Monster Roar 2 by quadraslayer"),
    "hum": (443499, 4187409, "Spaceship Ambience by PotatokingXII"),
}
TERMINAL_SOUNDS = {
    "keycard": "terminal_01.ogg",
    "exit_offline": "beep_02.ogg",
    "escaped": "terminal_04.ogg",
}
WALK_FOOTFALLS, ALIEN_FOOTFALLS = 10, 4
# Cuts in seconds. Each door hit lands as far in as the door takes to move, 0.35 s or so.
DOOR_OPEN, DOOR_CLOSE = (6.45, 7.35), (4.05, 5.0)
ROAR = (1.5, 4.4)
GROWLS = ((1.7, 2.9, 0.78), (2.6, 3.95, 0.7))  # start, end, pitch
HUM_START, HUM_SECONDS, HUM_OVERLAP = 20.0, 60.0, 4.0
OVERTONE_DRIVE, HUM_OVERTONES, ALIEN_STEP_OVERTONES = 4.0, 0.7, 1.2
OVERTONE_BAND = (180.0, 320.0, 1400.0, 3000.0)  # rise from, full at, fall from, gone at
ALIEN_MODES = (140, 230, 370, 610, 980)  # ringing pitches of a deck plate under a heavy foot


def fetch(url, name):
    path = SOURCE / name
    if not path.exists():
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=60) as response:
            path.write_bytes(response.read())
    return path.read_bytes()


def recording(name):
    number, uploader, _ = RECORDINGS[name]
    url = FREESOUND.format(number // 1000, number, uploader)
    return decode(fetch(url, f"{name}.mp3"))


def decode(data):
    raw = pygame.mixer.Sound(io.BytesIO(data)).get_raw()
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32).reshape(-1, 2) / 32768


def save(name, audio, peak=0.9):
    audio = np.asarray(audio, dtype=np.float32)
    audio = audio * peak / max(float(np.abs(audio).max()), 1e-6)
    with wave.open(str(PREPARED / f"{name}.wav"), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes((audio * 32767).astype(np.int16).tobytes())


def cut(audio, start, end):
    return audio[int(start * RATE) : int(end * RATE)]


def faded(audio, fade_in=0.004, fade_out=0.08):
    audio = audio.copy()
    rise, fall = int(fade_in * RATE), int(fade_out * RATE)
    audio[:rise] *= np.linspace(0, 1, rise)[:, None]
    audio[-fall:] *= np.linspace(1, 0, fall)[:, None]
    return audio


def repitched(audio, factor):
    places = np.arange(0, len(audio) - 1, factor)
    index = np.arange(len(audio))
    return np.stack([np.interp(places, index, channel) for channel in audio.T], axis=1)


def onsets(mono, gap=0.16):
    """Samples where the sound jumps up sharply, at most one per gap seconds."""
    hop = int(0.005 * RATE)
    frames = len(mono) // hop
    energy = np.sqrt((mono[: frames * hop].reshape(frames, hop) ** 2).mean(axis=1))
    smooth = np.convolve(energy, np.ones(4) / 4, mode="same")
    rise = np.maximum(0.0, np.diff(smooth))
    limit = max(float(rise.mean() + 3 * rise.std()), 0.01)
    found, last = [], -gap
    for index in np.flatnonzero(rise > limit):
        if index * hop / RATE - last >= gap:
            last = index * hop / RATE
            found.append(int(index * hop))
    return found


def footfalls(audio, count, seconds):
    """One-shots from a long take: its most ordinary steps, so none stands out."""
    mono = audio.mean(axis=1)
    length, lead = int(seconds * RATE), int(0.004 * RATE)
    starts = [start for start in onsets(mono) if lead <= start < len(mono) - length]
    level = {start: float(np.abs(mono[start : start + RATE // 10]).max()) for start in starts}
    middle = median(level.values())
    chosen = sorted(starts, key=lambda s: abs(level[s] - middle))[:count]
    return [faded(audio[s - lead : s + length]) for s in chosen]


def band(frequencies, centre, width):
    return np.exp(-(((frequencies - centre) / width) ** 2))


def alien_footfall(rng, seconds=0.32):
    """A heavy foot on a plate: a low thud, a short scuff and the plate ringing."""
    t = np.arange(int(seconds * RATE)) / RATE
    thud = 1.6 * np.sin(2 * np.pi * (70 + 60 * np.exp(-t * 60)) * t) * np.exp(-t * 34)
    ring = np.zeros_like(t)
    for index, mode in enumerate(ALIEN_MODES):
        pitch = mode * rng.uniform(0.96, 1.04)
        ring += (
            np.sin(2 * np.pi * pitch * t + rng.uniform(0, 6))
            * np.exp(-t * rng.uniform(26, 48))
            * 0.4 ** (index * 0.5)
            / (index + 1)
        )
    noise = np.fft.rfft(rng.normal(size=len(t)))
    noise = np.fft.irfft(noise * band(np.fft.rfftfreq(len(t), 1 / RATE), 1800, 1500), len(t))
    scuff = noise / np.abs(noise).max() * np.exp(-t * 120)
    return np.repeat((thud + 0.55 * ring + 0.14 * scuff)[:, None], 2, axis=1)


def seamless(audio, seconds, overlap):
    """A loop from a long take: its tail fades across its own start."""
    body, tail = int(seconds * RATE), int(overlap * RATE)
    loop = audio[:body].copy()
    rise = np.sqrt(np.linspace(0, 1, tail))[:, None]
    loop[:tail] = loop[:tail] * rise + audio[body : body + tail] * rise[::-1]
    return loop


def smooth_gate(frequencies, start, end):
    return np.clip((frequencies - start) / (end - start), 0.0, 1.0) ** 2


def with_overtones(audio, amount):
    """Add what small speakers can play to a low sound they cannot.

    Laptop speakers lose nearly everything under 200 Hz, where the hum and the Alien's
    footfall live. Soft clipping makes overtones of the same notes, and the ear fills the
    low notes back in.
    The filter is circular, so a loop stays seamless.
    """
    clipped = np.tanh(OVERTONE_DRIVE * audio / np.abs(audio).max())
    rise_from, full, fall_from, gone = OVERTONE_BAND
    spectrum = np.fft.rfft(clipped, axis=0)
    frequencies = np.fft.rfftfreq(len(clipped), 1 / RATE)
    shape = smooth_gate(frequencies, rise_from, full)
    shape = shape * (1 - smooth_gate(frequencies, fall_from, gone))
    overtones = np.fft.irfft(spectrum * shape[:, None], len(clipped), axis=0)
    return audio + amount * overtones * np.abs(audio).max()


def bake_footfalls():
    for index, step in enumerate(footfalls(recording("walk"), WALK_FOOTFALLS, 0.42)):
        save(f"walk_{index}", step)
    rng = np.random.default_rng(7)
    for index in range(ALIEN_FOOTFALLS):
        save(f"alien_step_{index}", with_overtones(alien_footfall(rng), ALIEN_STEP_OVERTONES))


def bake_doors():
    door = recording("door")
    save("door_open", faded(cut(door, *DOOR_OPEN), fade_out=0.25))
    save("door_close", faded(cut(door, *DOOR_CLOSE), fade_out=0.25))


def bake_alien_voice():
    roar = recording("roar")
    save("alien_roar", faded(cut(roar, *ROAR), fade_out=0.4))
    for index, (start, end, pitch) in enumerate(GROWLS):
        save(f"alien_growl_{index}", faded(repitched(cut(roar, start, end), pitch), 0.1, 0.35))


def bake_ship():
    hum = cut(recording("hum"), HUM_START, HUM_START + HUM_SECONDS + HUM_OVERLAP)
    loop = seamless(hum, HUM_SECONDS, HUM_OVERLAP)
    save("ship_hum", with_overtones(loop, HUM_OVERTONES), peak=0.7)
    with zipfile.ZipFile(io.BytesIO(fetch(TERMINALS, "terminals.zip"))) as archive:
        for name, member in TERMINAL_SOUNDS.items():
            save(name, decode(archive.read(member)))


def credits():
    lines = ["All recordings are CC0; credit is a courtesy, not a requirement.", ""]
    for number, _, credit in RECORDINGS.values():
        lines.append(f"{credit}: https://freesound.org/s/{number}/")
    lines.append(f"50 CC0 Sci-Fi SFX by rubberduck: {TERMINALS}")
    (PREPARED / "CREDITS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    SOURCE.mkdir(parents=True, exist_ok=True)
    PREPARED.mkdir(parents=True, exist_ok=True)
    for old in PREPARED.glob("*.wav"):
        old.unlink()
    pygame.mixer.init(RATE, -16, 2)
    try:
        bake_footfalls()
        bake_doors()
        bake_alien_voice()
        bake_ship()
        credits()
    finally:
        pygame.mixer.quit()
    print(f"Baked {len(list(PREPARED.glob('*.wav')))} sounds into {PREPARED}")


if __name__ == "__main__":
    main()
