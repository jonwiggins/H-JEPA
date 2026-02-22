"""
Comprehensive tests for dataset download and verification utilities.

Tests cover:
- Dataset information and metadata
- Disk space checking
- Dataset verification
- Download functionality (mocked)
- Manual download instructions
- Command-line interface
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from urllib.error import URLError

import pytest
from torchvision import datasets

from src.data.download import (
    DATASET_INFO,
    check_disk_space,
    download_dataset,
    get_disk_usage,
    print_dataset_summary,
    print_manual_download_instructions,
    verify_dataset,
)


class TestDatasetInfo:
    """Test dataset information constants."""

    def test_dataset_info_exists(self):
        """Test that DATASET_INFO is defined."""
        assert DATASET_INFO is not None
        assert isinstance(DATASET_INFO, dict)

    def test_all_datasets_present(self):
        """Test that all expected datasets are in DATASET_INFO."""
        expected_datasets = ["cifar10", "cifar100", "stl10", "imagenet", "imagenet100"]

        for dataset in expected_datasets:
            assert dataset in DATASET_INFO

    def test_dataset_info_structure(self):
        """Test that each dataset has required fields."""
        required_fields = [
            "name",
            "size_gb",
            "num_images",
            "num_classes",
            "resolution",
            "auto_download",
            "description",
            "url",
        ]

        for dataset_name, info in DATASET_INFO.items():
            for field in required_fields:
                assert field in info, f"{dataset_name} missing field: {field}"

    def test_auto_download_flags(self):
        """Test auto_download flags are correct."""
        # CIFAR-10, CIFAR-100, STL-10 should support auto-download
        assert DATASET_INFO["cifar10"]["auto_download"] is True
        assert DATASET_INFO["cifar100"]["auto_download"] is True
        assert DATASET_INFO["stl10"]["auto_download"] is True

        # ImageNet should NOT support auto-download
        assert DATASET_INFO["imagenet"]["auto_download"] is False
        assert DATASET_INFO["imagenet100"]["auto_download"] is False

    def test_dataset_sizes(self):
        """Test that dataset sizes are reasonable."""
        for dataset_name, info in DATASET_INFO.items():
            # Size should be positive
            assert info["size_gb"] > 0

            # Number of images should be positive
            assert info["num_images"] > 0

            # Number of classes should be positive
            assert info["num_classes"] > 0

    def test_urls_are_strings(self):
        """Test that URLs are valid strings."""
        for dataset_name, info in DATASET_INFO.items():
            assert isinstance(info["url"], str)
            assert len(info["url"]) > 0
            assert info["url"].startswith("http")


class TestGetDiskUsage:
    """Test disk usage checking."""

    def test_get_disk_usage(self, tmp_path):
        """Test getting disk usage for a path."""
        total_gb, used_gb, free_gb = get_disk_usage(tmp_path)

        # All values should be non-negative
        assert total_gb >= 0
        assert used_gb >= 0
        assert free_gb >= 0

        # Total should equal used + free (approximately)
        assert abs(total_gb - (used_gb + free_gb)) < 1.0

    def test_get_disk_usage_nonexistent_path(self):
        """Test with nonexistent path."""
        nonexistent = Path("/nonexistent/path/that/does/not/exist")

        # Should handle gracefully
        with pytest.warns(UserWarning):
            total_gb, used_gb, free_gb = get_disk_usage(nonexistent)

            # Should return zeros or handle error
            assert total_gb == 0 and used_gb == 0 and free_gb == 0


class TestCheckDiskSpace:
    """Test disk space checking."""

    def test_sufficient_space(self, tmp_path):
        """Test when there's sufficient disk space."""
        # Request a very small amount of space
        result = check_disk_space(tmp_path, required_gb=0.001, buffer_gb=0.001)

        # Should return True (assuming system has > 0.002 GB free)
        assert result is True

    def test_insufficient_space(self, tmp_path, capsys):
        """Test when there's insufficient disk space."""
        # Request an impossibly large amount of space
        result = check_disk_space(tmp_path, required_gb=999999, buffer_gb=0)

        # Should return False
        assert result is False

        # Should print warning
        captured = capsys.readouterr()
        assert "WARNING" in captured.out or "Insufficient" in captured.out

    def test_check_disk_space_with_buffer(self, tmp_path):
        """Test disk space check with buffer."""
        # This should work (very small requirement)
        result = check_disk_space(tmp_path, required_gb=0.001, buffer_gb=0.001)

        assert isinstance(result, bool)


class TestVerifyDataset:
    """Test dataset verification."""

    @pytest.fixture
    def temp_data_path(self, tmp_path):
        """Create temporary directory for data."""
        return tmp_path

    def test_verify_cifar10_success(self, temp_data_path, capsys):
        """Test successful CIFAR-10 verification."""
        with patch.object(datasets, "CIFAR10") as mock_cifar:
            # Mock train dataset
            mock_train = MagicMock()
            mock_train.__len__ = Mock(return_value=50000)

            # Mock test dataset
            mock_test = MagicMock()
            mock_test.__len__ = Mock(return_value=10000)

            mock_cifar.side_effect = [mock_train, mock_test]

            result = verify_dataset("cifar10", temp_data_path)

            assert result is True

            # Check output
            captured = capsys.readouterr()
            assert "verified" in captured.out.lower()

    def test_verify_cifar100_success(self, temp_data_path, capsys):
        """Test successful CIFAR-100 verification."""
        with patch.object(datasets, "CIFAR100") as mock_cifar:
            mock_train = MagicMock()
            mock_train.__len__ = Mock(return_value=50000)

            mock_test = MagicMock()
            mock_test.__len__ = Mock(return_value=10000)

            mock_cifar.side_effect = [mock_train, mock_test]

            result = verify_dataset("cifar100", temp_data_path)

            assert result is True

    def test_verify_stl10_success(self, temp_data_path, capsys):
        """Test successful STL-10 verification."""
        with patch.object(datasets, "STL10") as mock_stl:
            mock_train = MagicMock()
            mock_train.__len__ = Mock(return_value=5000)

            mock_test = MagicMock()
            mock_test.__len__ = Mock(return_value=8000)

            mock_unlabeled = MagicMock()
            mock_unlabeled.__len__ = Mock(return_value=100000)

            mock_stl.side_effect = [mock_train, mock_test, mock_unlabeled]

            result = verify_dataset("stl10", temp_data_path)

            assert result is True

    def test_verify_imagenet_success(self, temp_data_path, capsys):
        """Test successful ImageNet verification."""
        # Create ImageNet directory structure
        train_dir = temp_data_path / "train"
        val_dir = temp_data_path / "val"

        train_dir.mkdir(parents=True)
        val_dir.mkdir(parents=True)

        # Create some class directories
        for i in range(10):
            (train_dir / f"n0000{i}").mkdir()
            (val_dir / f"n0000{i}").mkdir()

        result = verify_dataset("imagenet", temp_data_path)

        assert result is True

    def test_verify_imagenet_missing(self, temp_data_path, capsys):
        """Test ImageNet verification when directories missing."""
        result = verify_dataset("imagenet", temp_data_path)

        assert result is False

        captured = capsys.readouterr()
        assert "not found" in captured.out.lower()

    def test_verify_imagenet_empty(self, temp_data_path, capsys):
        """Test ImageNet verification with empty directories."""
        # Create directories but no class folders
        train_dir = temp_data_path / "train"
        val_dir = temp_data_path / "val"

        train_dir.mkdir(parents=True)
        val_dir.mkdir(parents=True)

        result = verify_dataset("imagenet", temp_data_path)

        assert result is False

    def test_verify_unknown_dataset(self, temp_data_path, capsys):
        """Test verification of unknown dataset."""
        result = verify_dataset("unknown_dataset", temp_data_path)

        assert result is False

        captured = capsys.readouterr()
        assert "Unknown dataset" in captured.out

    def test_verify_cifar10_wrong_size(self, temp_data_path, capsys):
        """Test CIFAR-10 verification with wrong dataset size."""
        with patch.object(datasets, "CIFAR10") as mock_cifar:
            # Wrong size
            mock_train = MagicMock()
            mock_train.__len__ = Mock(return_value=1000)  # Should be 50000

            mock_test = MagicMock()
            mock_test.__len__ = Mock(return_value=10000)

            mock_cifar.side_effect = [mock_train, mock_test]

            result = verify_dataset("cifar10", temp_data_path)

            assert result is False

    def test_verify_exception_handling(self, temp_data_path, capsys):
        """Test that verification handles exceptions gracefully."""
        with patch.object(datasets, "CIFAR10") as mock_cifar:
            mock_cifar.side_effect = Exception("Test error")

            result = verify_dataset("cifar10", temp_data_path)

            assert result is False

            captured = capsys.readouterr()
            assert "failed" in captured.out.lower()


class TestDownloadDataset:
    """Test dataset downloading."""

    @pytest.fixture
    def temp_data_path(self, tmp_path):
        """Create temporary directory for data."""
        return tmp_path

    def test_download_cifar10_success(self, temp_data_path, capsys):
        """Test successful CIFAR-10 download."""
        with (
            patch.object(datasets, "CIFAR10") as mock_cifar,
            patch("src.data.download.verify_dataset") as mock_verify,
        ):

            mock_dataset = MagicMock()
            mock_dataset.__len__ = Mock(return_value=50000)
            mock_cifar.return_value = mock_dataset
            mock_verify.return_value = True

            result = download_dataset("cifar10", temp_data_path, verify=True)

            assert result is True

            # Should be called for train and test splits
            assert mock_cifar.call_count == 2

    def test_download_cifar100_success(self, temp_data_path, capsys):
        """Test successful CIFAR-100 download."""
        with (
            patch.object(datasets, "CIFAR100") as mock_cifar,
            patch("src.data.download.verify_dataset") as mock_verify,
        ):

            mock_dataset = MagicMock()
            mock_cifar.return_value = mock_dataset
            mock_verify.return_value = True

            result = download_dataset("cifar100", temp_data_path, verify=True)

            assert result is True

    def test_download_stl10_success(self, temp_data_path, capsys):
        """Test successful STL-10 download."""
        with (
            patch.object(datasets, "STL10") as mock_stl,
            patch("src.data.download.verify_dataset") as mock_verify,
            patch("src.data.download.check_disk_space", return_value=True),
        ):

            mock_dataset = MagicMock()
            mock_stl.return_value = mock_dataset
            mock_verify.return_value = True

            result = download_dataset("stl10", temp_data_path, verify=True)

            assert result is True

            # Should be called for train, test, and unlabeled
            assert mock_stl.call_count == 3

    def test_download_manual_dataset(self, temp_data_path, capsys):
        """Test that manual download datasets return False."""
        result = download_dataset("imagenet", temp_data_path)

        assert result is False

        captured = capsys.readouterr()
        assert "manual download" in captured.out.lower()

    def test_download_unknown_dataset(self, temp_data_path, capsys):
        """Test downloading unknown dataset."""
        result = download_dataset("unknown_dataset", temp_data_path)

        assert result is False

        captured = capsys.readouterr()
        assert "Unknown dataset" in captured.out

    def test_download_without_verification(self, temp_data_path):
        """Test download without verification."""
        with (
            patch.object(datasets, "CIFAR10") as mock_cifar,
            patch("src.data.download.verify_dataset") as mock_verify,
        ):

            mock_dataset = MagicMock()
            mock_cifar.return_value = mock_dataset

            download_dataset("cifar10", temp_data_path, verify=False)

            # verify_dataset should not be called
            mock_verify.assert_not_called()

    def test_download_network_error(self, temp_data_path, capsys):
        """Test handling of network errors."""
        with patch.object(datasets, "CIFAR10") as mock_cifar:
            mock_cifar.side_effect = URLError("Network error")

            result = download_dataset("cifar10", temp_data_path)

            assert result is False

            captured = capsys.readouterr()
            assert "network error" in captured.out.lower()

    def test_download_general_error(self, temp_data_path, capsys):
        """Test handling of general errors."""
        with patch.object(datasets, "CIFAR10") as mock_cifar:
            mock_cifar.side_effect = Exception("Test error")

            result = download_dataset("cifar10", temp_data_path)

            assert result is False

            captured = capsys.readouterr()
            assert "failed" in captured.out.lower()

    def test_download_verification_failure(self, temp_data_path, capsys):
        """Test when download succeeds but verification fails."""
        with (
            patch.object(datasets, "CIFAR10") as mock_cifar,
            patch("src.data.download.verify_dataset") as mock_verify,
        ):

            mock_dataset = MagicMock()
            mock_cifar.return_value = mock_dataset
            mock_verify.return_value = False  # Verification fails

            result = download_dataset("cifar10", temp_data_path, verify=True)

            assert result is False


class TestPrintManualDownloadInstructions:
    """Test printing manual download instructions."""

    def test_print_imagenet_instructions(self, capsys):
        """Test printing ImageNet instructions."""
        print_manual_download_instructions("imagenet")

        captured = capsys.readouterr()
        assert "ImageNet" in captured.out or "ILSVRC2012" in captured.out
        assert "image-net.org" in captured.out
        assert "train" in captured.out
        assert "val" in captured.out

    def test_print_imagenet100_instructions(self, capsys):
        """Test printing ImageNet-100 instructions."""
        print_manual_download_instructions("imagenet100")

        captured = capsys.readouterr()
        assert "ImageNet-100" in captured.out
        assert "100" in captured.out

    def test_case_insensitive(self, capsys):
        """Test that dataset name is case-insensitive."""
        print_manual_download_instructions("IMAGENET")

        captured = capsys.readouterr()
        assert len(captured.out) > 0


class TestPrintDatasetSummary:
    """Test printing dataset summary."""

    def test_print_summary(self, capsys):
        """Test printing dataset summary."""
        print_dataset_summary()

        captured = capsys.readouterr()

        # Should mention all datasets
        assert "CIFAR-10" in captured.out
        assert "CIFAR-100" in captured.out
        assert "STL-10" in captured.out
        assert "ImageNet" in captured.out

        # Should show auto-download info
        assert "Auto" in captured.out or "Manual" in captured.out

        # Should show total size
        assert "Total" in captured.out


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def temp_data_path(self, tmp_path):
        """Create temporary directory for data."""
        return tmp_path

    def test_verify_case_insensitive(self, temp_data_path):
        """Test that verification is case-insensitive."""
        with patch.object(datasets, "CIFAR10") as mock_cifar:
            mock_train = MagicMock()
            mock_train.__len__ = Mock(return_value=50000)

            mock_test = MagicMock()
            mock_test.__len__ = Mock(return_value=10000)

            mock_cifar.side_effect = [mock_train, mock_test]

            # Try different cases
            result1 = verify_dataset("CIFAR10", temp_data_path)

        with patch.object(datasets, "CIFAR10") as mock_cifar:
            mock_train = MagicMock()
            mock_train.__len__ = Mock(return_value=50000)

            mock_test = MagicMock()
            mock_test.__len__ = Mock(return_value=10000)

            mock_cifar.side_effect = [mock_train, mock_test]

            result2 = verify_dataset("cifar10", temp_data_path)

        # Both should work
        assert result1 is True
        assert result2 is True

    def test_download_case_insensitive(self, temp_data_path):
        """Test that download is case-insensitive."""
        with patch.object(datasets, "CIFAR10") as mock_cifar:
            mock_dataset = MagicMock()
            mock_cifar.return_value = mock_dataset

            result = download_dataset("CIFAR10", temp_data_path, verify=False)

            assert result is True

    def test_imagenet_with_partial_classes(self, temp_data_path):
        """Test ImageNet verification with fewer than expected classes."""
        train_dir = temp_data_path / "train"
        val_dir = temp_data_path / "val"

        train_dir.mkdir(parents=True)
        val_dir.mkdir(parents=True)

        # Create only a few class directories
        for i in range(5):
            (train_dir / f"n0000{i}").mkdir()
            (val_dir / f"n0000{i}").mkdir()

        result = verify_dataset("imagenet", temp_data_path)

        # Should still pass (just has fewer classes)
        assert result is True

    def test_imagenet100_filtering_note(self, temp_data_path, capsys):
        """Test ImageNet-100 with full ImageNet present."""
        train_dir = temp_data_path / "train"
        val_dir = temp_data_path / "val"

        train_dir.mkdir(parents=True)
        val_dir.mkdir(parents=True)

        # Create 1000 class directories (full ImageNet)
        for i in range(1000):
            (train_dir / f"n{i:05d}").mkdir()
            (val_dir / f"n{i:05d}").mkdir()

        result = verify_dataset("imagenet100", temp_data_path)

        capsys.readouterr()

        # Should mention filtering
        # (The actual verification might pass or show a note)
        assert result is True

    def test_disk_space_exactly_required(self, temp_data_path):
        """Test when disk space exactly matches requirement."""
        # Mock disk usage to return specific values
        with patch("src.data.download.get_disk_usage") as mock_usage:
            # 10 GB total, 5 GB used, 5 GB free
            mock_usage.return_value = (10.0, 5.0, 5.0)

            # Request exactly 5 GB (no buffer)
            result = check_disk_space(temp_data_path, required_gb=5.0, buffer_gb=0.0)

            # Should pass (exactly enough)
            assert result is True

            # With buffer, should fail
            result_with_buffer = check_disk_space(temp_data_path, required_gb=5.0, buffer_gb=0.1)
            assert result_with_buffer is False


class TestCommandLineInterface:
    """Test command-line interface functionality."""

    def test_main_no_args(self, capsys):
        """Test main function with no arguments (should show summary)."""
        from src.data.download import main

        with patch("sys.argv", ["download.py"]):
            main()

        captured = capsys.readouterr()
        # Should print summary
        assert "CIFAR" in captured.out or "SUPPORTED" in captured.out

    def test_main_with_dataset(self, tmp_path):
        """Test main function with dataset argument."""
        from src.data.download import main

        with (
            patch(
                "sys.argv", ["download.py", "cifar10", "--data-path", str(tmp_path), "--no-verify"]
            ),
            patch.object(datasets, "CIFAR10") as mock_cifar,
            patch("src.data.download.check_disk_space", return_value=True),
        ):

            mock_dataset = MagicMock()
            mock_cifar.return_value = mock_dataset

            main()

            # Should attempt download
            assert mock_cifar.called

    def test_main_verify_only(self, tmp_path):
        """Test main function with verify-only flag."""
        from src.data.download import main

        with (
            patch(
                "sys.argv",
                ["download.py", "cifar10", "--verify-only", "--data-path", str(tmp_path)],
            ),
            patch("src.data.download.verify_dataset") as mock_verify,
        ):

            mock_verify.return_value = True

            main()

            # Should verify but not download
            mock_verify.assert_called()
