from slewm.training.sampler import EpochSeededRandomSampler
import pytest
import torch


def test_epoch_order_is_deterministic_and_addressable():
    sampler = EpochSeededRandomSampler(range(20), seed=3072)
    epoch_zero = list(sampler)
    sampler.set_epoch(1)
    epoch_one = list(sampler)
    restored = EpochSeededRandomSampler(range(20), seed=3072)
    restored.set_epoch(1)
    assert list(restored) == epoch_one
    assert epoch_zero != epoch_one
    assert sorted(epoch_zero) == sorted(epoch_one) == list(range(20))


def test_iterator_materializes_after_set_epoch():
    sampler = EpochSeededRandomSampler(range(20), seed=9)
    iterator_constructed_early = iter(sampler)
    sampler.set_epoch(4)
    expected = EpochSeededRandomSampler(range(20), seed=9)
    expected.set_epoch(4)
    assert list(iterator_constructed_early) == list(expected)


def test_initial_epoch_survives_multiworker_dataloader_prefetch():
    sampler = EpochSeededRandomSampler(range(40), seed=9, initial_epoch=4)
    loader = torch.utils.data.DataLoader(
        list(range(40)),
        batch_size=4,
        sampler=sampler,
        num_workers=2,
        prefetch_factor=2,
    )
    iterator_constructed_and_prefetched = iter(loader)
    # This deliberately simulates a too-late framework assignment.  The
    # already-materialized permutation must still be the explicit resume epoch.
    sampler.set_epoch(0)
    actual = torch.cat(list(iterator_constructed_and_prefetched)).tolist()
    expected = list(EpochSeededRandomSampler(range(40), seed=9, initial_epoch=4))
    assert actual == expected


def test_initial_epoch_must_be_nonnegative():
    with pytest.raises(ValueError, match="initial sampler epoch"):
        EpochSeededRandomSampler(range(4), seed=1, initial_epoch=-1)
