from pathlib import Path


def test_external_configs_use_fixed_budget_loss_only_selection():
    root = Path(__file__).resolve().parents[1]
    expected = {
        "paper_gear_acmtm.yaml": ("ac-mtm", 30),
        "paper_gear_smwm.yaml": ("smwm", 30),
        "paper_franka_acmtm.yaml": ("ac-mtm", 20),
        "paper_franka_smwm.yaml": ("smwm", 20),
    }
    for filename, (method, epochs) in expected.items():
        text = (root / "config/train" / filename).read_text(encoding="utf-8")
        assert f"method: {method}" in text
        assert "weight: 0.0" in text
        assert "enabled: true" in text
        assert f"min_epochs: {epochs}" in text
        assert f"max_epochs: {epochs}" in text
