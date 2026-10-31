"""Start a policy as a copy of the written Alien, so a run does not depend on early luck.

A run that misses a useful action early never finds it: such a seed still ignored noise
after 300,000 steps (experiments/hearing_analysis_v1). Copying gives every seed the same
sensible start, then PPO trains on reward as usual, so the final policy is still learned.
"""

import numpy as np
import torch

from rl.scripted_controllers import HeuristicSelector


def validate_imitation(settings):
    for name in ("decisions", "epochs", "batch_size"):
        if type(settings[name]) is not int or settings[name] <= 0:
            raise ValueError(f"Imitation {name} must be a positive whole number")
    for name in ("learning_rate", "value_weight"):
        if not settings[name] > 0:
            raise ValueError(f"Imitation {name} must be positive")
    if not 0 <= settings["random_action_share"] < 1:
        raise ValueError("Imitation random_action_share must be in [0, 1)")


def collect_examples(env, settings, seed, gamma):
    """Rollouts of the written Alien; some random actions show it states it would not reach."""
    rng = np.random.default_rng(seed)
    inner = env.unwrapped
    observations, masks, labels, returns = [], [], [], []
    while len(labels) < settings["decisions"]:
        obs, _ = env.reset()
        selector = HeuristicSelector(inner.config.target_arrival_distance)
        first, done = len(labels), False
        while not done:
            mask = inner.action_masks()
            label = selector.choose(inner._allowed_view())
            if mask[label] and mask.sum() > 1:
                observations.append(obs)
                masks.append(mask)
                labels.append(label)
                returns.append(0.0)
            action = label
            if rng.random() < settings["random_action_share"]:
                action = int(rng.choice(np.flatnonzero(mask)))
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if len(labels) > first:
                returns[-1] += reward  # forced decisions count toward the last real choice
        total = 0.0
        for index in range(len(labels) - 1, first - 1, -1):
            total = returns[index] + gamma * total
            returns[index] = total
    return (
        torch.as_tensor(np.array(observations), dtype=torch.float32),
        torch.as_tensor(np.array(masks), dtype=torch.bool),
        torch.as_tensor(np.array(labels), dtype=torch.long),
        torch.as_tensor(np.array(returns), dtype=torch.float32),
    )


def copy_heuristic(model, env, settings, *, seed, gamma):
    """Fit actor and critic to the written Alien; decision and update counters stay at zero."""
    x, m, y, g = collect_examples(env, settings, seed, gamma)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=settings["learning_rate"])
    model.policy.set_training_mode(True)
    for _ in range(settings["epochs"]):
        for batch in torch.randperm(len(y), generator=generator).split(settings["batch_size"]):
            distribution = model.policy.get_distribution(x[batch], action_masks=m[batch])
            values = model.policy.predict_values(x[batch]).flatten()
            value_loss = ((values - g[batch]) ** 2).mean()
            loss = -distribution.log_prob(y[batch]).mean() + settings["value_weight"] * value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.policy.set_training_mode(False)
    return len(y)
