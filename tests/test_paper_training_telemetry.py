import json
from types import SimpleNamespace

import torch

from trim.training.telemetry import PaperTrainingTelemetry


def test_telemetry_records_gradient_lr_steps_and_examples(tmp_path):
    model = torch.nn.Linear(3, 2)
    module = SimpleNamespace(device=torch.device("cpu"), parameters=model.parameters)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-5)
    trainer = SimpleNamespace(
        global_step=10,
        current_epoch=2,
        optimizers=[optimizer],
        is_global_zero=True,
    )
    callback = PaperTrainingTelemetry(run_dir=tmp_path)
    callback.on_train_epoch_start(trainer, module)
    model(torch.ones(4, 3)).sum().backward()
    callback.on_after_backward(trainer, module)
    callback.on_train_batch_end(
        trainer,
        module,
        None,
        {
            "pixels": torch.zeros(4, 6, 3, 8, 8),
            "_episode_index": torch.tensor([1, 1, 2, 3]),
            "_clip_start": torch.tensor([0, 1, 0, 5]),
        },
        0,
    )
    trainer.global_step = 11
    callback.on_train_epoch_end(trainer, module)

    payload = json.loads((tmp_path / "training_telemetry.json").read_text())
    record = payload["records"][0]
    assert record["epoch"] == 3
    assert record["global_step"] == 11
    assert record["optimizer_steps"] == 1
    assert record["learning_rates"] == [5.0e-5]
    assert record["gradient_norm_mean"] > 0
    assert record["gradient_norm_max"] == record["gradient_norm_mean"]
    assert len(record["clip_identity_sha256"]) == 64
    assert record["first_batch_clip_identities"] == [
        [1, 0],
        [1, 1],
        [2, 0],
        [3, 5],
    ]
    assert record["examples"] == 4
    assert record["examples_per_second"] > 0


def test_telemetry_state_round_trip(tmp_path):
    source = PaperTrainingTelemetry(run_dir=tmp_path / "source")
    source.records = [{"epoch": 1, "global_step": 3}]
    restored = PaperTrainingTelemetry(run_dir=tmp_path / "restored")
    restored.load_state_dict(source.state_dict())
    assert restored.records == source.records
