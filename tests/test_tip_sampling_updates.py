from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.data.tip_sampling import supervision_groups, TipBalancedBatchSampler, UpdateCounter


def test_partial_review_never_enters_positive_pool():
    data = SimpleNamespace(partial_review_policy='exclude', study_ids=['full', 'partial', 'neg'],
        study_cvc_records={'full': [{'catheter_instance_id': 0}],
                           'partial': [{'catheter_instance_id': 0}, {'catheter_instance_id': 1}]},
        reviewed_tips_map={('full', 0): (1, 1), ('partial', 0): (2, 2)},
        confirmed_negatives={'neg'})
    assert supervision_groups(data) == {'positive': [0], 'negative': [2], 'unknown': [1]}
    data.confirmed_negatives.add('full')
    with pytest.raises(ValueError, match='Contradictory'):
        supervision_groups(data)


def test_sampler_balance_independent_rng_and_state():
    groups = {'positive': [0, 1], 'negative': [2, 3], 'unknown': [4, 5, 6]}
    first = TipBalancedBatchSampler(groups, 20, 42)
    second = TipBalancedBatchSampler(groups, 20, 42)
    batches = list(first)
    np.random.seed(900)
    np.random.rand(200)
    assert batches == list(second)
    for batch in batches:
        assert len(set(batch)) == 4
        assert sum(i in groups['positive'] for i in batch) == 1
        assert sum(i in groups['negative'] for i in batch) == 1
        assert sum(i in groups['unknown'] for i in batch) == 2
    iterator = iter(first)
    next(iterator)
    state = first.state_dict()
    expected = next(iterator)
    restored = TipBalancedBatchSampler(groups, 20, 99)
    restored.load_state_dict(state)
    assert next(iter(restored)) == expected


def test_real_amp_skip_does_not_advance_scheduler():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([p], lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    scaler = torch.amp.GradScaler('cpu', init_scale=128.0)
    counter = UpdateCounter(optimizer)
    start = p.detach().clone()
    attempts = 0
    for bad_gradient in [True, False, False]:
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(p.square().sum()).backward()
        if bad_gradient:
            p.grad.fill_(float('inf'))
        before = counter.successful
        scaler.step(optimizer)
        scaler.update()
        attempts += 1
        if counter.successful > before:
            scheduler.step()
        if bad_gradient:
            assert torch.equal(p.detach(), start)
            assert scheduler.last_epoch == 0
    assert attempts == 3 and counter.successful == 2
    assert scheduler.last_epoch == 2
    assert torch.isfinite(p).all()
