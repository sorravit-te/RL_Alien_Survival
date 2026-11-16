# Trained Alien

`ship_deck_full_v3_scratch_s3/run/checkpoints/best.zip` is the Alien people play against: MaskablePPO trained from
scratch for 60,000 steps (training seed 3 of 4). It was chosen by its validation score, on seeds never used in the
reported results.

The run folder holds only what loading the checkpoint needs: the frozen config, round manifest, evaluation
protocol and the record of how `best.zip` was selected. Every file is checked by hash when the game loads it,
so none of them can be edited. Loading also needs the same `game/`, `ai/`, `rl/` and `requirements.lock.txt`
as at training time, and Python 3.12.5.

The play command is in the main [README](../README.md#play).
