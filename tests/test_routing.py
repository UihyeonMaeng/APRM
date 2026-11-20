"""Scientific contracts for the fixed APRM routing equations and encoder integration."""

import copy

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from aprm.training.trainers.aprm_modules import (
    APRMRoutedStage,
    AnatomyDataLoader,
    NoisyTopkRouter,
    RoutedDatasetNumpy,
    compute_balance_loss,
    prepare_routing_vector,
    routing_to_label_map,
    select_anatomical_gate,
)
from aprm.utilities.label_handling.label_handling import LabelManager


CPU = torch.device("cpu")


def test_equation4_threshold_largest_roi_and_tie():
    occupancy = torch.tensor(
        [
            [0.90, 0.10, 0, 0, 0, 0, 0],
            [0.89, 0.03, 0.08, 0, 0, 0, 0],
            [0.60, 0.20, 0.20, 0, 0, 0, 0],
            [1.0, 0, 0, 0, 0, 0, 0],
            [0.0, 0, 0, 0, 0, 0, 1.0],
        ]
    )
    # OUT need not be smaller than each anatomical region to choose anatomy.
    assert select_anatomical_gate(occupancy).tolist() == [0, 2, 1, 0, 6]


def test_categorical_channels_and_counts_have_identical_patch_occupancy():
    labels = torch.arange(7).repeat(4).reshape(1, 1, 2, 2, 7)
    channels = F.one_hot(labels[:, 0], 7).movedim(-1, 1).float()
    counts = torch.full((1, 7), 4.0)
    expected = torch.full((1, 7), 1 / 7)
    for prior in (labels, channels, counts):
        torch.testing.assert_close(prepare_routing_vector(prior, 1, CPU), expected)


def test_zero_mass_channel_padding_is_outside_not_missing_probability():
    labels = torch.zeros((1, 1, 2, 10, 10))
    labels[0, 0, 0, 0:2, :] = 1  # 20 anatomical / 200 total voxels.
    channels = F.one_hot(labels[:, 0].long(), 7).movedim(-1, 1).float()
    channels[:, :, 1] = 0  # Former inference behavior: all-zero padded channels.
    expected = torch.tensor([[0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]])
    categorical_q = prepare_routing_vector(labels, 1, CPU)
    channel_q = prepare_routing_vector(channels, 1, CPU)
    torch.testing.assert_close(categorical_q, expected)
    torch.testing.assert_close(channel_q, expected)
    assert select_anatomical_gate(channel_q).item() == 0


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        torch.zeros((1, 7)),
        torch.tensor([[float("nan"), 1, 0, 0, 0, 0, 0]]),
        torch.tensor([[-1.0, 2.0, 0, 0, 0, 0, 0]]),
        torch.full((1, 1, 2, 2, 2), 7.0),
        torch.full((1, 1, 2, 2, 2), 0.5),
        torch.zeros((2, 1, 2, 2, 2)),
    ],
)
def test_missing_or_invalid_prior_is_rejected(invalid):
    with pytest.raises(ValueError):
        prepare_routing_vector(invalid, 1, CPU)


def test_numpy_prior_validation_does_not_clip_bad_anatomy_ids():
    for value in (-1, 7, 0.25, np.nan):
        with pytest.raises(ValueError):
            routing_to_label_map(np.full((2, 3, 4), value))
    zeros = np.zeros((7, 2, 3, 4), dtype=np.float32)
    np.testing.assert_array_equal(routing_to_label_map(zeros), np.zeros((1, 2, 3, 4), dtype=np.int16))


def test_noisy_router_train_stochastic_eval_deterministic_and_sparse():
    torch.manual_seed(91)
    router = NoisyTopkRouter(4)
    tokens = torch.zeros((128, 4))
    # Zero logits isolate noise from incidental feature variations.
    with torch.no_grad():
        router.topkroute_linear.weight.zero_()
        router.topkroute_linear.bias.zero_()
        router.noise_linear.weight.zero_()
        router.noise_linear.bias.zero_()
    router.train()
    first = router(tokens)
    second = router(tokens)
    assert not torch.equal(first[0], second[0])
    assert torch.all((first[0] > 0).sum(dim=1) == 2)
    torch.testing.assert_close(first[0].sum(dim=1), torch.ones(128))
    assert torch.all(first[2] > 0)  # Dense probabilities retained before top-K.
    router.eval()
    third, fourth = router(tokens), router(tokens)
    torch.testing.assert_close(third[0], fourth[0], rtol=0, atol=0)
    torch.testing.assert_close(third[2], torch.full((128, 8), 1 / 8))


def test_balance_loss_penalizes_concentration_and_supervises_unselected_logits():
    uniform = torch.full((12, 8), 1 / 8)
    concentrated = F.one_hot(torch.zeros(12, dtype=torch.long), 8).float()
    assert compute_balance_loss(uniform).item() == pytest.approx(1.0)
    assert compute_balance_loss(concentrated).item() == pytest.approx(8.0)
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.2, -0.2, -0.5, -1.0, -2.0]], requires_grad=True)
    compute_balance_loss(logits.softmax(-1)).backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.all(logits.grad[0, 2:].abs() > 0)


def test_sparse_stage_matches_equation6_for_two_anatomical_gates():
    """Analytic oracle checks mixed-batch dispatch, weights, residual, and reshape."""
    stage = APRMRoutedStage(nn.Identity(), out_channels=2).eval()
    logits_a = torch.tensor([4.0, 3.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
    logits_b = torch.tensor([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 3.0, 4.0])
    with torch.no_grad():
        for router in stage.routers:
            router.topkroute_linear.weight.zero_()
            router.topkroute_linear.bias.zero_()
        stage.routers[1].topkroute_linear.bias.copy_(logits_a)
        stage.routers[6].topkroute_linear.bias.copy_(logits_b)
        for index, expert in enumerate(stage.experts):
            expert.net[0].weight.zero_()
            expert.net[0].bias.zero_()
            expert.net[2].weight.zero_()
            expert.net[2].bias.copy_(torch.tensor([index + 1.0, -(index + 1.0)]))
    image = torch.arange(32, dtype=torch.float32).reshape(2, 2, 2, 2, 2) / 10
    prior = F.one_hot(torch.tensor([1, 6]), 7).float()
    output, auxiliary, stats = stage(image, prior)
    high, low = np.exp(1) / (np.exp(1) + 1), 1 / (np.exp(1) + 1)
    expected_delta = torch.tensor([high * 1 + low * 2, low * 7 + high * 8], dtype=torch.float32)
    expected = image + torch.stack([expected_delta, -expected_delta], dim=1)[:, :, None, None, None]
    torch.testing.assert_close(output, expected)
    assert stats["selected_gates"].tolist() == [1, 6]
    assert stats["active_expert_counts"].sum().item() == 2 * 2 * 8
    assert 1 <= auxiliary.item() <= 8


def test_only_selected_experts_receive_segmentation_gradient():
    stage = APRMRoutedStage(nn.Identity(), out_channels=2).eval()
    with torch.no_grad():
        router = stage.routers[1].topkroute_linear
        router.weight.zero_()
        router.bias.copy_(torch.tensor([4.0, 3.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]))
    image = torch.ones((1, 2, 2, 2, 2), requires_grad=True)
    output, auxiliary, _ = stage(image, F.one_hot(torch.tensor([1]), 7).float())
    (output.square().mean() + 0.01 * auxiliary).backward()
    assert stage.experts[0].net[2].bias.grad is not None
    assert stage.experts[1].net[2].bias.grad is not None
    assert all(expert.net[2].bias.grad is None for expert in stage.experts[2:])
    assert stage.routers[1].topkroute_linear.bias.grad[2:].abs().sum() > 0
    assert stage.routers[0].topkroute_linear.weight.grad is None


def test_real_unet_routes_every_encoder_stage_and_backpropagates(tiny_model):
    image = torch.randn((2, 1, 8, 8, 8))
    prior = torch.tensor([[0.95, 0.05, 0, 0, 0, 0, 0], [0.7, 0.1, 0.2, 0, 0, 0, 0]])
    outputs = tiny_model(image, prior)
    assert [tuple(output.shape) for output in outputs] == [(2, 2, 8, 8, 8), (2, 2, 4, 4, 4)]
    assert tiny_model.encoder.routed_stage_indices == (0, 1, 2)
    stats = tiny_model.get_routing_stats()
    assert set(stats) == {"stage_0", "stage_1", "stage_2"}
    for index, stage in enumerate(tiny_model.encoder.stages):
        assert isinstance(stage, APRMRoutedStage)
        assert len(stage.routers) == 7 and len(stage.experts) == 8
        assert stats[f"stage_{index}"]["selected_gates"].tolist() == [0, 2]
    loss = (
        sum(
            F.cross_entropy(output, torch.zeros(output.shape[:1] + output.shape[2:], dtype=torch.long))
            for output in outputs
        )
        + 0.01 * tiny_model.get_aux_loss()
    )
    loss.backward()
    assert torch.isfinite(loss)
    for stage in tiny_model.encoder.stages:
        assert stage.routers[0].topkroute_linear.weight.grad is not None
        assert stage.routers[2].topkroute_linear.weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in tiny_model.parameters() if p.grad is not None)


def test_image_encoder_never_consumes_anatomy_and_input_forms_agree(tiny_model):
    tiny_model.eval()
    tiny_model.decoder.deep_supervision = False
    image = torch.randn((1, 1, 8, 8, 8))
    labels = torch.zeros_like(image)
    labels[:, :, :4] = 1
    labels[:, :, 4:] = 6
    one_hot = F.one_hot(labels[:, 0].long(), 7).movedim(-1, 1).float()
    vector = prepare_routing_vector(labels, 1, CPU)
    seen = []
    handle = tiny_model.encoder.stages[0].stage.register_forward_pre_hook(
        lambda _, args: seen.append(args[0].detach().clone())
    )
    with torch.no_grad():
        expected = tiny_model(image, vector)
        categorical = tiny_model(torch.cat([image, labels], dim=1))
        channels = tiny_model(torch.cat([image, one_hot], dim=1))
    handle.remove()
    for tensor in seen:
        torch.testing.assert_close(tensor, image, rtol=0, atol=0)
    torch.testing.assert_close(categorical, expected, rtol=0, atol=0)
    torch.testing.assert_close(channels, expected, rtol=0, atol=0)
    with pytest.raises(ValueError):
        tiny_model(image)
    with pytest.raises(ValueError):
        tiny_model(image, labels[:, :, :-1])


def test_checkpoint_reload_preserves_predictions(tiny_model, tmp_path):
    tiny_model.eval()
    tiny_model.decoder.deep_supervision = False
    image = torch.randn((1, 1, 8, 8, 8))
    prior = F.one_hot(torch.tensor([3]), 7).float()
    with torch.no_grad():
        expected = tiny_model(image, prior)
    path = tmp_path / "checkpoint.pth"
    torch.save(tiny_model.state_dict(), path)
    restored = copy.deepcopy(tiny_model)
    with torch.no_grad():
        for parameter in restored.parameters():
            parameter.zero_()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(image, prior), expected, rtol=0, atol=0)


def test_augmentation_occupancy_uses_transformed_patch_and_target_excludes_anatomy(tmp_path, monkeypatch):
    anatomy = np.zeros((8, 8, 8), dtype=np.int16)
    anatomy[:2] = 1
    anatomy[2:4] = 6
    anatomy[4:] = 2
    image = anatomy[None].astype(np.float32)
    target = (anatomy[None] > 0).astype(np.int16)
    properties = {"class_locations": {1: np.array([[0, 0, 0, 0]])}}
    RoutedDatasetNumpy.save_case(image, target, properties, str(tmp_path / "subject"))
    np.save(tmp_path / "subject_routing.npy", anatomy)
    dataset = RoutedDatasetNumpy(str(tmp_path), ["subject"])

    def transform(image, segmentation):
        image = image[:, :4, :4, :4].flip(-1)
        segmentation = segmentation[:, :4, :4, :4].flip(-1)
        return {"image": image, "segmentation": [segmentation, segmentation[:, ::2, ::2, ::2]]}

    loader = AnatomyDataLoader(
        dataset, 1, (8, 8, 8), (4, 4, 4), LabelManager({"background": 0, "tumor": 1}, None), transforms=transform
    )
    monkeypatch.setattr(loader, "get_bbox", lambda *args: ([0, 0, 0], [8, 8, 8]))
    batch = loader.generate_train_batch()
    torch.testing.assert_close(batch["routing_vector"], torch.tensor([[0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5]]))
    assert batch["data"].shape == (1, 1, 4, 4, 4)
    assert [target.shape[1] for target in batch["target"]] == [1, 1]
    assert batch["target"][0].max().item() == 1
    assert batch["data"].max().item() == 6
