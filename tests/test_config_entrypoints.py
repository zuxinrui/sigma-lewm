"""Compose all release task recipes and resolve their Python targets."""
import importlib
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('arguments', [
    ['train', '--task', 'gear', '--arm', 'c0', '--epochs', '30'],
    ['train', '--task', 'pusht', '--arm', 'A7', '--epochs', '15'],
    ['train', '--task', 'pusht', '--arm', 'lewm', '--epochs', '10'],
    ['eval', '--task', 'cube', '--checkpoint', '/nonexistent/weights.pt'],
])
def test_cli_uses_real_entrypoint_paths(arguments, monkeypatch, capsys):
    import json
    from trim.cli import main
    monkeypatch.setattr('sys.argv', ['reproduce.py'] + arguments)
    main()
    command = json.loads(capsys.readouterr().out)['command']
    assert (ROOT / command[1]).is_file()
    if arguments[0] == 'eval':
        assert 'eval.goal_offset_steps=25' in command
    if 'gear' in arguments:
        assert 'trainer.min_epochs=0' in command
        assert 'early_stopping.enabled=false' in command


@pytest.mark.parametrize('task', ['gear', 'franka'])
@pytest.mark.parametrize('arm', ['l0', 'c0', 'acmtm', 'smwm'])
def test_aligned_recipe_dependency_closure(task, arm):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT/'config/train')):
        cfg = compose(config_name=f'paper_{task}_{arm}')
    config = OmegaConf.to_container(cfg, resolve=False)

    def check(value):
        if isinstance(value, dict):
            target = value.get('_target_')
            if target:
                module, name = target.rsplit('.', 1)
                assert hasattr(importlib.import_module(module), name), target
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)

    check(config)
    assert cfg.data.dataset.transitions == 5
    assert cfg.model._target_.startswith('trim.')
