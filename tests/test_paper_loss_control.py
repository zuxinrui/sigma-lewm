import inspect

from trim.training.stopping import LossCompositeController, LossCompositeEarlyStopping


def _components(value):
    return {
        "teacher": value,
        "free_run": value,
        "terminal": value,
        "action": value,
        "sigreg": value,
    }


def test_plateau_cannot_stop_before_minimum_epoch():
    controller = LossCompositeController(
        min_epochs=5,
        max_epochs=20,
        ema_window=1,
        patience=2,
        slope_window=2,
    )
    decisions = [
        controller.update(
            epoch=epoch,
            validation_components=_components(1.0),
            train_components=_components(1.0),
            embedding_std_mean=1.0,
        )
        for epoch in range(1, 5)
    ]
    assert not any(item.stop for item in decisions)

    stopped = controller.update(
        epoch=5,
        validation_components=_components(1.0),
        train_components=_components(1.0),
        embedding_std_mean=1.0,
    )
    assert stopped.stop
    assert stopped.stop_reason == "validation_plateau"


def test_clear_validation_improvement_resets_patience():
    controller = LossCompositeController(
        min_epochs=3,
        max_epochs=10,
        ema_window=1,
        patience=2,
        slope_window=2,
    )
    values = (1.0, 1.0, 0.8, 0.79)
    decisions = [
        controller.update(
            epoch=epoch,
            validation_components=_components(value),
            train_components=_components(value),
            embedding_std_mean=1.0,
        )
        for epoch, value in enumerate(values, start=1)
    ]
    assert not any(item.stop for item in decisions)
    assert decisions[-1].best_epoch == 4


def test_embedding_collapse_stops_only_after_minimum_and_patience():
    controller = LossCompositeController(
        min_epochs=3,
        max_epochs=10,
        ema_window=1,
        patience=10,
        slope_window=2,
        collapse_std_ratio=0.2,
        collapse_patience=2,
    )
    stds = (1.0, 0.1, 0.1, 0.1)
    decisions = [
        controller.update(
            epoch=epoch,
            validation_components=_components(1.0 - 0.01 * epoch),
            train_components=_components(1.0 - 0.02 * epoch),
            embedding_std_mean=std,
        )
        for epoch, std in enumerate(stds, start=1)
    ]
    assert not decisions[1].stop
    assert not decisions[2].stop
    assert decisions[3].stop_reason == "embedding_collapse"


def test_effective_rank_and_covariance_condition_are_collapse_gates():
    effective_rank = LossCompositeController(
        min_epochs=2,
        max_epochs=10,
        ema_window=1,
        patience=10,
        slope_window=2,
        collapse_patience=2,
    )
    condition = LossCompositeController(
        min_epochs=2,
        max_epochs=10,
        ema_window=1,
        patience=10,
        slope_window=2,
        collapse_patience=2,
    )
    for epoch in range(1, 4):
        rank_decision = effective_rank.update(
            epoch=epoch,
            validation_components=_components(1.0 - 0.01 * epoch),
            train_components=_components(1.0 - 0.02 * epoch),
            embedding_std_mean=1.0,
            embedding_effective_rank=100.0 if epoch == 1 else 5.0,
            embedding_covariance_condition=2.0,
        )
        condition_decision = condition.update(
            epoch=epoch,
            validation_components=_components(1.0 - 0.01 * epoch),
            train_components=_components(1.0 - 0.02 * epoch),
            embedding_std_mean=1.0,
            embedding_effective_rank=100.0,
            embedding_covariance_condition=2.0 if epoch == 1 else 25.0,
        )
    assert rank_decision.stop_reason == "embedding_collapse"
    assert condition_decision.stop_reason == "embedding_collapse"


def test_lightning_adapter_reads_stable_pretraining_fit_metrics(tmp_path):
    callback = LossCompositeEarlyStopping(run_dir=tmp_path)
    metrics = {
        f"fit/{metric}": 1.0
        for metric in callback.component_keys.values()
    }
    assert callback._components(metrics, "fit") == _components(1.0)


def test_early_stopping_source_has_no_simulator_success_dependency():
    source = inspect.getsource(LossCompositeEarlyStopping)
    assert "success_rate" not in source
    assert "per_episode" not in source
    assert "eval.json" not in source
