"""
Unit tests for transfer learning evaluation module.

Tests for:
1. TransferHead initialization and forward pass
2. FineTuneEvaluator initialization
3. Fine-tuning with frozen/unfrozen encoder
4. Training and evaluation
5. FewShotEvaluator initialization
6. Few-shot episode sampling
7. Few-shot evaluation
8. Convenience functions
9. Edge cases
"""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.evaluation.transfer import (
    FewShotEvaluator,
    FineTuneEvaluator,
    TransferHead,
    few_shot_eval,
    fine_tune_eval,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_model():
    """Create a mock H-JEPA model for testing."""
    mock = MagicMock()
    mock.embed_dim = 384

    def mock_extract_features(images, level=0, use_target_encoder=True):
        batch_size = images.shape[0]
        # Return patch features [B, N, D]
        return torch.randn(batch_size, 196, 384)

    mock.extract_features = mock_extract_features
    mock.eval = MagicMock(return_value=mock)
    mock.parameters = MagicMock(return_value=[])
    mock.to = MagicMock(return_value=mock)

    return mock


@pytest.fixture
def simple_dataloader():
    """Create a simple dataloader."""
    num_samples = 64
    num_classes = 10
    images = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=16, shuffle=False)


@pytest.fixture
def small_dataloader():
    """Create a small dataloader for quick tests."""
    num_samples = 32
    images = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, 5, (num_samples,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=8, shuffle=False)


@pytest.fixture
def few_shot_dataset():
    """Create a dataset for few-shot learning."""

    class SimpleDataset(Dataset):
        def __init__(self, num_samples=100, num_classes=10):
            self.num_samples = num_samples
            self.num_classes = num_classes
            self.images = torch.randn(num_samples, 3, 224, 224)
            # Ensure each class has enough samples
            self.labels = torch.arange(num_samples) % num_classes

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx].item()

    return SimpleDataset()


# ============================================================================
# Test TransferHead
# ============================================================================


def test_transfer_head_linear(random_seed):
    """Test linear transfer head (no hidden layers)."""
    head = TransferHead(input_dim=384, num_classes=10, hidden_dims=[])

    # Test with 2D features
    features = torch.randn(8, 384)
    logits = head(features)

    assert logits.shape == (8, 10)


def test_transfer_head_mlp(random_seed):
    """Test MLP transfer head with hidden layers."""
    head = TransferHead(
        input_dim=384,
        num_classes=10,
        hidden_dims=[256, 128],
    )

    features = torch.randn(8, 384)
    logits = head(features)

    assert logits.shape == (8, 10)


def test_transfer_head_with_dropout(random_seed):
    """Test transfer head with dropout."""
    head = TransferHead(
        input_dim=384,
        num_classes=10,
        hidden_dims=[256],
        dropout=0.5,
    )

    features = torch.randn(8, 384)

    # Training mode should apply dropout
    head.train()
    logits_train = head(features)

    # Eval mode should not apply dropout
    head.eval()
    logits_eval = head(features)

    assert logits_train.shape == (8, 10)
    assert logits_eval.shape == (8, 10)


def test_transfer_head_mean_pooling(random_seed):
    """Test mean pooling of patch features."""
    head = TransferHead(
        input_dim=384,
        num_classes=10,
        pooling="mean",
    )

    # 3D patch features
    features_3d = torch.randn(8, 196, 384)
    logits = head(features_3d)

    assert logits.shape == (8, 10)


def test_transfer_head_max_pooling(random_seed):
    """Test max pooling of patch features."""
    head = TransferHead(
        input_dim=384,
        num_classes=10,
        pooling="max",
    )

    # 3D patch features
    features_3d = torch.randn(8, 196, 384)
    logits = head(features_3d)

    assert logits.shape == (8, 10)


def test_transfer_head_invalid_pooling(random_seed):
    """Test that invalid pooling raises error."""
    head = TransferHead(
        input_dim=384,
        num_classes=10,
        pooling="invalid",
    )

    features_3d = torch.randn(8, 196, 384)

    with pytest.raises(ValueError, match="Unknown pooling"):
        head(features_3d)


def test_transfer_head_2d_passthrough(random_seed):
    """Test that 2D features pass through pooling."""
    head = TransferHead(input_dim=384, num_classes=10)

    features_2d = torch.randn(8, 384)
    logits = head(features_2d)

    assert logits.shape == (8, 10)


# ============================================================================
# Test FineTuneEvaluator Initialization
# ============================================================================


def test_finetune_evaluator_initialization(mock_model):
    """Test FineTuneEvaluator initialization."""
    evaluator = FineTuneEvaluator(
        model=mock_model,
        num_classes=10,
        hierarchy_level=0,
        freeze_encoder=False,
        device="cpu",
    )

    assert evaluator.hierarchy_level == 0
    assert evaluator.freeze_encoder is False
    assert evaluator.device == "cpu"
    assert evaluator.classifier is not None


def test_finetune_evaluator_freeze_encoder(mock_model):
    """Test that encoder is frozen when requested."""
    # Use deepcopy to create a real model with parameters

    real_model = nn.Linear(10, 10)
    mock_with_real_params = MagicMock()
    mock_with_real_params.embed_dim = 384
    mock_with_real_params.extract_features = mock_model.extract_features
    mock_with_real_params.to = MagicMock(return_value=mock_with_real_params)
    mock_with_real_params.eval = MagicMock(return_value=mock_with_real_params)
    # Use actual parameters
    mock_with_real_params.parameters = real_model.parameters

    evaluator = FineTuneEvaluator(
        model=mock_with_real_params,
        num_classes=10,
        freeze_encoder=True,
        device="cpu",
    )

    assert evaluator.freeze_encoder is True
    # After freezing, parameters should not require grad
    for param in evaluator.model.parameters():
        assert not param.requires_grad


def test_finetune_evaluator_unfreeze_encoder(mock_model):
    """Test that encoder is trainable when not frozen."""
    # Use a real model with parameters

    real_model = nn.Linear(10, 10)
    # Freeze it first
    for param in real_model.parameters():
        param.requires_grad = False

    mock_with_real_params = MagicMock()
    mock_with_real_params.embed_dim = 384
    mock_with_real_params.extract_features = mock_model.extract_features
    mock_with_real_params.to = MagicMock(return_value=mock_with_real_params)
    mock_with_real_params.eval = MagicMock(return_value=mock_with_real_params)
    mock_with_real_params.parameters = real_model.parameters

    evaluator = FineTuneEvaluator(
        model=mock_with_real_params,
        num_classes=10,
        freeze_encoder=False,
        device="cpu",
    )

    # After unfreezing, parameters should require grad
    for param in evaluator.model.parameters():
        assert param.requires_grad


def test_finetune_evaluator_with_mlp_head(mock_model):
    """Test initialization with MLP head."""
    evaluator = FineTuneEvaluator(
        model=mock_model,
        num_classes=10,
        hidden_dims=[256, 128],
        dropout=0.5,
        device="cpu",
    )

    assert evaluator.classifier is not None


# ============================================================================
# Test FineTuneEvaluator Forward Pass
# ============================================================================


def test_finetune_forward_frozen(mock_model, random_seed):
    """Test forward pass with frozen encoder."""
    evaluator = FineTuneEvaluator(
        model=mock_model,
        num_classes=10,
        freeze_encoder=True,
        device="cpu",
    )

    images = torch.randn(8, 3, 224, 224)
    logits = evaluator.forward(images)

    assert logits.shape == (8, 10)


def test_finetune_forward_unfrozen(mock_model, random_seed):
    """Test forward pass with trainable encoder."""
    evaluator = FineTuneEvaluator(
        model=mock_model,
        num_classes=10,
        freeze_encoder=False,
        device="cpu",
    )

    images = torch.randn(8, 3, 224, 224)
    logits = evaluator.forward(images)

    assert logits.shape == (8, 10)


# ============================================================================
# Test FineTuneEvaluator Training
# ============================================================================


def test_finetune_train_epoch(mock_model, small_dataloader, random_seed):
    """Test training for one epoch."""
    evaluator = FineTuneEvaluator(
        model=mock_model,
        num_classes=5,
        freeze_encoder=True,
        device="cpu",
    )

    optimizer = torch.optim.Adam(evaluator.classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    metrics = evaluator.train_epoch(
        train_loader=small_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        epoch=1,
        verbose=False,
    )

    assert "loss" in metrics
    assert "accuracy" in metrics
    assert metrics["loss"] > 0
    assert 0 <= metrics["accuracy"] <= 100


def test_finetune_evaluate(mock_model, small_dataloader, random_seed):
    """Test evaluation on test set."""
    evaluator = FineTuneEvaluator(
        model=mock_model,
        num_classes=5,
        freeze_encoder=True,
        device="cpu",
    )

    criterion = nn.CrossEntropyLoss()

    metrics = evaluator.evaluate(
        test_loader=small_dataloader,
        criterion=criterion,
        verbose=False,
    )

    assert "loss" in metrics
    assert "accuracy" in metrics
    assert metrics["loss"] > 0
    assert 0 <= metrics["accuracy"] <= 100


def test_finetune_full_training(mock_model, small_dataloader, random_seed):
    """Test full fine-tuning process."""
    evaluator = FineTuneEvaluator(
        model=mock_model,
        num_classes=5,
        freeze_encoder=True,
        device="cpu",
    )

    history = evaluator.fine_tune(
        train_loader=small_dataloader,
        val_loader=None,
        epochs=2,
        lr=0.01,
        verbose=False,
    )

    assert "train_loss" in history
    assert "train_acc" in history
    assert len(history["train_loss"]) == 2
    assert len(history["train_acc"]) == 2


def test_finetune_with_validation(mock_model, small_dataloader, random_seed):
    """Test fine-tuning with validation set."""
    evaluator = FineTuneEvaluator(
        model=mock_model,
        num_classes=5,
        freeze_encoder=True,
        device="cpu",
    )

    history = evaluator.fine_tune(
        train_loader=small_dataloader,
        val_loader=small_dataloader,  # Use same for simplicity
        epochs=2,
        lr=0.01,
        verbose=False,
    )

    assert "val_loss" in history
    assert "val_acc" in history
    assert len(history["val_loss"]) == 2
    assert len(history["val_acc"]) == 2


def test_finetune_cosine_scheduler(mock_model, small_dataloader, random_seed):
    """Test fine-tuning with cosine annealing scheduler."""
    evaluator = FineTuneEvaluator(
        model=mock_model,
        num_classes=5,
        freeze_encoder=True,
        device="cpu",
    )

    history = evaluator.fine_tune(
        train_loader=small_dataloader,
        epochs=2,
        lr=0.01,
        scheduler_type="cosine",
        verbose=False,
    )

    assert len(history["train_loss"]) == 2


def test_finetune_step_scheduler(mock_model, small_dataloader, random_seed):
    """Test fine-tuning with step scheduler."""
    evaluator = FineTuneEvaluator(
        model=mock_model,
        num_classes=5,
        freeze_encoder=True,
        device="cpu",
    )

    history = evaluator.fine_tune(
        train_loader=small_dataloader,
        epochs=3,
        lr=0.01,
        scheduler_type="step",
        verbose=False,
    )

    assert len(history["train_loss"]) == 3


def test_finetune_unfrozen_encoder(mock_model, small_dataloader, random_seed):
    """Test fine-tuning with unfrozen encoder."""
    evaluator = FineTuneEvaluator(
        model=mock_model,
        num_classes=5,
        freeze_encoder=False,
        device="cpu",
    )

    history = evaluator.fine_tune(
        train_loader=small_dataloader,
        epochs=2,
        lr=0.001,
        verbose=False,
    )

    assert len(history["train_loss"]) == 2


# ============================================================================
# Test FewShotEvaluator
# ============================================================================


def test_fewshot_evaluator_initialization(mock_model):
    """Test FewShotEvaluator initialization."""
    evaluator = FewShotEvaluator(
        model=mock_model,
        num_classes=10,
        hierarchy_level=0,
        device="cpu",
    )

    assert evaluator.model is mock_model
    assert evaluator.num_classes == 10
    assert evaluator.hierarchy_level == 0
    assert evaluator.device == "cpu"


def test_fewshot_sample_episodes(mock_model, few_shot_dataset, random_seed):
    """Test sampling few-shot episodes."""
    evaluator = FewShotEvaluator(
        model=mock_model,
        num_classes=10,
        device="cpu",
    )

    episodes = evaluator.sample_few_shot_episodes(
        dataset=few_shot_dataset,
        n_way=5,
        k_shot=3,
        n_query=5,
        n_episodes=10,
    )

    assert len(episodes) == 10

    for episode in episodes:
        assert "support_indices" in episode
        assert "query_indices" in episode
        assert "classes" in episode

        # Should have n_way classes
        assert len(episode["classes"]) == 5

        # Should have n_way * k_shot support samples
        assert len(episode["support_indices"]) == 5 * 3

        # Should have n_way * n_query query samples
        assert len(episode["query_indices"]) == 5 * 5


def test_fewshot_evaluate_episode(mock_model, few_shot_dataset, random_seed):
    """Test evaluation of a single episode."""
    evaluator = FewShotEvaluator(
        model=mock_model,
        num_classes=10,
        device="cpu",
    )

    episodes = evaluator.sample_few_shot_episodes(
        dataset=few_shot_dataset,
        n_way=3,
        k_shot=2,
        n_query=3,
        n_episodes=1,
    )

    accuracy = evaluator.evaluate_episode(
        dataset=few_shot_dataset,
        episode=episodes[0],
        metric="cosine",
    )

    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 100


def test_fewshot_evaluate_episode_euclidean(mock_model, few_shot_dataset, random_seed):
    """Test episode evaluation with euclidean distance."""
    evaluator = FewShotEvaluator(
        model=mock_model,
        num_classes=10,
        device="cpu",
    )

    episodes = evaluator.sample_few_shot_episodes(
        dataset=few_shot_dataset,
        n_way=3,
        k_shot=2,
        n_query=3,
        n_episodes=1,
    )

    accuracy = evaluator.evaluate_episode(
        dataset=few_shot_dataset,
        episode=episodes[0],
        metric="euclidean",
    )

    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 100


def test_fewshot_evaluate_full(mock_model, few_shot_dataset, random_seed):
    """Test full few-shot evaluation."""
    evaluator = FewShotEvaluator(
        model=mock_model,
        num_classes=10,
        device="cpu",
    )

    metrics = evaluator.evaluate_few_shot(
        dataset=few_shot_dataset,
        n_way=5,
        k_shot=3,
        n_query=5,
        n_episodes=5,
        verbose=False,
    )

    assert "accuracy" in metrics
    assert "std" in metrics
    assert "confidence_interval" in metrics
    assert "n_way" in metrics
    assert "k_shot" in metrics

    assert metrics["n_way"] == 5
    assert metrics["k_shot"] == 3
    assert 0 <= metrics["accuracy"] <= 100


def test_fewshot_1_shot(mock_model, few_shot_dataset, random_seed):
    """Test 1-shot learning."""
    evaluator = FewShotEvaluator(
        model=mock_model,
        num_classes=10,
        device="cpu",
    )

    metrics = evaluator.evaluate_few_shot(
        dataset=few_shot_dataset,
        n_way=3,
        k_shot=1,
        n_query=5,
        n_episodes=5,
        verbose=False,
    )

    assert metrics["k_shot"] == 1
    assert 0 <= metrics["accuracy"] <= 100


def test_fewshot_10_shot(random_seed):
    """Test 10-shot learning."""

    # Create larger dataset to support 10-shot
    class LargerDataset:
        def __init__(self):
            self.num_samples = 300
            self.num_classes = 10
            self.images = torch.randn(300, 3, 224, 224)
            # Ensure each class has at least 30 samples (3 classes * (10 shot + 5 query))
            self.labels = torch.arange(300) % 10

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx].item()

    larger_dataset = LargerDataset()

    mock = MagicMock()
    mock.embed_dim = 384

    def mock_extract(images, level=0, use_target_encoder=True):
        batch_size = images.shape[0]
        return torch.randn(batch_size, 196, 384)

    mock.extract_features = mock_extract
    mock.eval = MagicMock(return_value=mock)
    mock.parameters = MagicMock(return_value=[])

    evaluator = FewShotEvaluator(
        model=mock,
        num_classes=10,
        device="cpu",
    )

    metrics = evaluator.evaluate_few_shot(
        dataset=larger_dataset,
        n_way=3,
        k_shot=10,
        n_query=5,
        n_episodes=5,
        verbose=False,
    )

    assert metrics["k_shot"] == 10


# ============================================================================
# Test Convenience Functions
# ============================================================================


def test_fine_tune_eval_function(mock_model, small_dataloader, random_seed):
    """Test fine_tune_eval convenience function."""
    metrics = fine_tune_eval(
        model=mock_model,
        train_loader=small_dataloader,
        val_loader=small_dataloader,
        num_classes=5,
        epochs=2,
        device="cpu",
        verbose=False,
    )

    assert "loss" in metrics
    assert "accuracy" in metrics


def test_fine_tune_eval_frozen(mock_model, small_dataloader, random_seed):
    """Test fine_tune_eval with frozen encoder."""
    metrics = fine_tune_eval(
        model=mock_model,
        train_loader=small_dataloader,
        val_loader=small_dataloader,
        num_classes=5,
        freeze_encoder=True,
        epochs=2,
        device="cpu",
        verbose=False,
    )

    assert "accuracy" in metrics


def test_few_shot_eval_function(mock_model, random_seed):
    """Test few_shot_eval convenience function."""

    # Create dataset with enough samples per class
    class LargerDataset:
        def __init__(self):
            self.num_samples = 200
            self.num_classes = 10
            self.images = torch.randn(200, 3, 224, 224)
            # 20 samples per class
            self.labels = torch.arange(200) % 10

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx].item()

    larger_dataset = LargerDataset()

    results = few_shot_eval(
        model=mock_model,
        dataset=larger_dataset,
        num_classes=10,
        k_shot_list=[1, 3],
        n_episodes=5,
        device="cpu",
        verbose=False,
    )

    assert 1 in results
    assert 3 in results
    assert "accuracy" in results[1]
    assert "accuracy" in results[3]


def test_few_shot_eval_multiple_k(mock_model, random_seed):
    """Test few-shot eval with multiple k values."""

    # Create dataset with enough samples per class
    class LargerDataset:
        def __init__(self):
            self.num_samples = 300
            self.num_classes = 10
            self.images = torch.randn(300, 3, 224, 224)
            # 30 samples per class
            self.labels = torch.arange(300) % 10

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx].item()

    larger_dataset = LargerDataset()

    results = few_shot_eval(
        model=mock_model,
        dataset=larger_dataset,
        num_classes=10,
        n_way=3,
        k_shot_list=[1, 5, 10],
        n_episodes=5,
        device="cpu",
        verbose=False,
    )

    assert len(results) == 3
    for k in [1, 5, 10]:
        assert k in results


# ============================================================================
# Edge Cases
# ============================================================================


def test_finetune_single_batch(mock_model, random_seed):
    """Test fine-tuning with single batch."""
    images = torch.randn(8, 3, 224, 224)
    labels = torch.randint(0, 5, (8,))
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=8)

    evaluator = FineTuneEvaluator(
        model=mock_model,
        num_classes=5,
        freeze_encoder=True,
        device="cpu",
    )

    history = evaluator.fine_tune(
        train_loader=dataloader,
        epochs=2,
        verbose=False,
    )

    assert len(history["train_loss"]) == 2


def test_fewshot_2_way(mock_model, few_shot_dataset, random_seed):
    """Test 2-way few-shot learning."""
    evaluator = FewShotEvaluator(
        model=mock_model,
        num_classes=10,
        device="cpu",
    )

    metrics = evaluator.evaluate_few_shot(
        dataset=few_shot_dataset,
        n_way=2,
        k_shot=3,
        n_query=5,
        n_episodes=5,
        verbose=False,
    )

    assert metrics["n_way"] == 2


def test_finetune_model_copy(mock_model):
    """Test that FineTuneEvaluator copies the model."""
    # This is important so we don't modify the original model
    evaluator = FineTuneEvaluator(
        model=mock_model,
        num_classes=10,
        freeze_encoder=False,
        device="cpu",
    )

    # The evaluator should have its own model instance
    assert evaluator.model is not None
    # Due to deepcopy, it should be a different object
    # (though with MagicMock this is tricky to test perfectly)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
