#!/usr/bin/env python3
"""Continue training with reproducible hard-example mining."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from train_unet_path_model import (
    BASE,
    BATCH,
    CASE_ROOT,
    LABEL_ROOT,
    OUTPUT,
    SEED,
    SmallUNet,
    discover_pairs,
    loss_fn,
    prepare_case,
    random_patch,
)


EPOCHS = 24
STEPS = 48
HARD_WEIGHTS = {"TOR10": 2.0, "TOR70": 3.0, "TOR77": 3.0, "TOR112": 4.0, "TOR123": 4.0}


def main() -> None:
    random.seed(SEED + 99)
    np.random.seed(SEED + 99)
    torch.manual_seed(SEED + 99)
    pairs = discover_pairs()
    labeled = sorted(
        [path.parent.name for path in LABEL_ROOT.glob("TOR*/manual_damage_corridor_mask.tif") if path.parent.name in pairs],
        key=lambda value: int(value[3:]),
    )
    cases = {case_id: prepare_case(case_id, pairs[case_id]) for case_id in labeled}
    weights = np.asarray([HARD_WEIGHTS.get(case_id, 1.0) for case_id in labeled], dtype="float64")
    weights /= weights.sum()

    model_dir = OUTPUT / "models" / "final_unet"
    checkpoint_path = model_dir / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = SmallUNet(int(checkpoint["in_channels"]), int(checkpoint.get("base", BASE)))
    model.load_state_dict(checkpoint["state_dict"])
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    rng = np.random.default_rng(SEED + 99)

    history = []
    for epoch in range(EPOCHS):
        running = 0.0
        for _ in range(STEPS):
            indices = rng.choice(len(labeled), size=BATCH, replace=True, p=weights)
            patches = [random_patch(cases[labeled[int(index)]], rng) for index in indices]
            x = torch.from_numpy(np.stack([item[0] for item in patches]))
            y = torch.from_numpy(np.stack([item[1] for item in patches]))[:, None]
            valid = torch.from_numpy(np.stack([item[2] for item in patches]))[:, None]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y, valid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss)
        scheduler.step()
        mean_loss = running / STEPS
        history.append({"epoch": epoch + 1, "loss": mean_loss})
        print(f"hard-mining epoch {epoch + 1}/{EPOCHS} loss={mean_loss:.4f}", flush=True)

    original_backup = model_dir / "model_pre_hard_mining.pt"
    if not original_backup.exists():
        original_backup.write_bytes(checkpoint_path.read_bytes())
    torch.save(
        {"state_dict": model.eval().state_dict(), "in_channels": checkpoint["in_channels"], "base": checkpoint.get("base", BASE)},
        checkpoint_path,
    )
    metadata = {
        "base_checkpoint": str(original_backup),
        "training_cases": labeled,
        "hard_example_weights": HARD_WEIGHTS,
        "epochs": EPOCHS,
        "steps_per_epoch": STEPS,
        "batch_size": BATCH,
        "seed": SEED + 99,
        "history": history,
        "labels_used_only_during_training": True,
    }
    (model_dir / "hard_mining_metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
