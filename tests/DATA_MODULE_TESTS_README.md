# H-JEPA Data Module Tests

## Overview

This document describes the comprehensive test suite for H-JEPA data modules, specifically designed to achieve 70%+ coverage for modules that previously had 0% coverage.

## Target Modules

The test suite covers four core data modules:

1. **src/data/datasets.py** (159 lines)
   - Dataset implementations (CIFAR-10, CIFAR-100, STL-10, ImageNet, ImageNet-100)
   - JEPA transforms (training and evaluation)
   - Dataset and dataloader factory functions

2. **src/data/transforms.py** (195 lines)
   - RandAugment implementation
   - Mixup and CutMix augmentations
   - Random Erasing
   - DeiT III augmentation pipeline
   - Transform factory functions

3. **src/data/multicrop_dataset.py** (123 lines)
   - MultiCropDataset wrapper
   - MultiCropDatasetRaw for raw image loading
   - Custom collate functions
   - Factory functions for multicrop datasets

4. **src/data/multicrop_transforms.py** (87 lines)
   - MultiCropTransform for training
   - MultiCropEvalTransform for evaluation
   - AdaptiveMultiCropTransform with epoch-based warmup
   - Transform factory functions

## Test File

**Location**: `tests/test_data_modules.py`

**Lines of Code**: ~1,400 lines of comprehensive tests

**Test Classes**: 32 test classes with 150+ individual test methods

## Test Coverage

### datasets.py Tests

#### JEPATransform Tests
- ✅ Default initialization
- ✅ Custom parameters (image size, crop scale, normalization)
- ✅ Output shape validation
- ✅ Color jitter enabled/disabled
- ✅ Horizontal flip enabled/disabled
- ✅ Different input image sizes
- ✅ Custom normalization parameters
- ✅ Tensor input handling

#### JEPAEvalTransform Tests
- ✅ Initialization
- ✅ Output shape validation
- ✅ Deterministic behavior
- ✅ Custom image sizes

#### CIFAR10Dataset Tests
- ✅ Train/val split initialization
- ✅ Dataset length
- ✅ Item retrieval
- ✅ Classes property
- ✅ Mock external downloads

#### CIFAR100Dataset Tests
- ✅ Initialization
- ✅ 100 classes validation
- ✅ Mock external downloads

#### STL10Dataset Tests
- ✅ Train/val/unlabeled splits
- ✅ Mock external downloads

#### ImageNetDataset Tests
- ✅ Missing directory error handling
- ✅ Proper directory structure handling
- ✅ Item retrieval
- ✅ Classes and class_to_idx properties

#### ImageNet100Dataset Tests
- ✅ Class filtering functionality
- ✅ Filtered dataset length
- ✅ Valid indices tracking

#### build_dataset Tests
- ✅ CIFAR-10/100 building
- ✅ STL-10 building
- ✅ Unknown dataset error handling
- ✅ Case-insensitive dataset names

#### build_dataloader Tests
- ✅ Basic dataloader creation
- ✅ Custom parameters (batch size, workers, shuffle, etc.)

### transforms.py Tests

#### RandAugment Tests
- ✅ Initialization with different parameters
- ✅ Image augmentation
- ✅ Different number of operations
- ✅ Different magnitudes
- ✅ Individual operations (identity, auto contrast, equalize, rotate, solarize, color)
- ✅ Magnitude calculation

#### Mixup Tests
- ✅ Initialization
- ✅ Basic mixing operation
- ✅ Alpha parameter variations
- ✅ Probability parameter
- ✅ Target distribution validation

#### CutMix Tests
- ✅ Initialization
- ✅ Basic cut-and-paste operation
- ✅ Random bounding box generation
- ✅ Probability parameter

#### MixupCutmix Tests
- ✅ Initialization with both transforms
- ✅ Random selection between mixup and cutmix

#### RandomErasing Tests
- ✅ Initialization
- ✅ Basic erasing operation
- ✅ Probability control
- ✅ Different value modes (zero, random)
- ✅ Inplace vs copy

#### DeiTIIIAugmentation Tests
- ✅ Initialization
- ✅ Image-level augmentation
- ✅ Batch-level transform retrieval
- ✅ Custom parameters

#### DeiTIIIEvalTransform Tests
- ✅ Initialization
- ✅ Evaluation transform output

#### build_deit3_transform Tests
- ✅ Training transform creation
- ✅ Evaluation transform creation
- ✅ Custom configuration

### multicrop_transforms.py Tests

#### MultiCropTransform Tests
- ✅ Initialization
- ✅ Multiple crop generation
- ✅ Correct crop sizes (global vs local)
- ✅ Different configurations
- ✅ String representation

#### MultiCropEvalTransform Tests
- ✅ Initialization
- ✅ Single crop output

#### AdaptiveMultiCropTransform Tests
- ✅ Initialization with min/max crops
- ✅ Epoch-based crop count adjustment
- ✅ Warmup progression
- ✅ Linear warmup schedule

#### build_multicrop_transform Tests
- ✅ Default transform creation
- ✅ Adaptive transform creation
- ✅ Custom parameters

### multicrop_dataset.py Tests

#### MultiCropDataset Tests
- ✅ Initialization
- ✅ Dataset length
- ✅ Item retrieval with/without labels
- ✅ Epoch setting
- ✅ Properties (num_global_crops, num_local_crops, total_crops, classes)

#### MultiCropDatasetRaw Tests
- ✅ Train/val initialization
- ✅ Transform type selection
- ✅ Properties

#### multicrop_collate_fn Tests
- ✅ Basic collation
- ✅ Empty batch handling
- ✅ Multiple crop types

#### build_multicrop_dataset Tests
- ✅ Default dataset creation
- ✅ Custom parameters

#### build_multicrop_dataloader Tests
- ✅ Dataloader creation with custom collate function

### Integration Tests
- ✅ Full pipeline from dataset to dataloader
- ✅ Complete transform pipeline
- ✅ Multicrop transform pipeline

## Mocking Strategy

To avoid external dependencies and ensure fast, reliable tests:

### Mocked Components
1. **PyTorch Datasets**: CIFAR-10, CIFAR-100, STL-10 downloads
2. **ImageFolder**: For ImageNet dataset structure
3. **External Downloads**: All automatic downloads are mocked

### Mock Fixtures
- `sample_pil_image`: PIL Image for transform tests
- `sample_tensor_image`: Tensor for augmentation tests
- `sample_batch`: Batch of images and targets
- `temp_data_dir`: Temporary directory for test files
- `mock_imagenet_structure`: Mock ImageNet directory structure

## Running the Tests

### Quick Start

```bash
# Run all data module tests
pytest tests/test_data_modules.py -v

# Run with coverage report
pytest tests/test_data_modules.py \
    --cov=src/data/datasets \
    --cov=src/data/transforms \
    --cov=src/data/multicrop_dataset \
    --cov=src/data/multicrop_transforms \
    --cov-report=term-missing

# Use the provided script
./tests/run_data_module_tests.sh
```

### Test Runner Script

The `run_data_module_tests.sh` script provides:
- Colored output for better readability
- Coverage report generation (HTML and terminal)
- Options for verbose/quiet modes
- Automatic verification of test file location

```bash
# Run with default settings
./tests/run_data_module_tests.sh

# Run with verbose output
./tests/run_data_module_tests.sh -v

# Run quietly (only show summary)
./tests/run_data_module_tests.sh -q
```

## Coverage Goals

**Target**: 70%+ coverage for each module

### Expected Coverage by Module

| Module | Lines | Target Coverage | Key Areas Covered |
|--------|-------|----------------|-------------------|
| datasets.py | 159 | 70%+ | Dataset classes, transforms, builders |
| transforms.py | 195 | 70%+ | RandAugment, Mixup, CutMix, DeiT III |
| multicrop_dataset.py | 123 | 70%+ | Dataset wrappers, collate functions |
| multicrop_transforms.py | 87 | 70%+ | Multicrop transforms, adaptive warmup |

### Areas with Full Coverage
- All public class `__init__` methods
- All public class `__call__` methods
- All factory functions (build_*)
- Critical properties and methods
- Error handling paths

### Areas Intentionally Not Covered
- Private utility methods (unless critical)
- Deprecated code paths
- Print statements and logging
- `if __name__ == "__main__"` blocks

## Test Organization

### Test Structure
```
test_data_modules.py
├── Fixtures (setup and mock data)
├── datasets.py Tests
│   ├── TestJEPATransform
│   ├── TestJEPAEvalTransform
│   ├── TestCIFAR10Dataset
│   ├── TestCIFAR100Dataset
│   ├── TestSTL10Dataset
│   ├── TestImageNetDataset
│   ├── TestImageNet100Dataset
│   ├── TestBuildDataset
│   └── TestBuildDataLoader
├── transforms.py Tests
│   ├── TestRandAugment
│   ├── TestMixup
│   ├── TestCutMix
│   ├── TestMixupCutmix
│   ├── TestRandomErasing
│   ├── TestDeiTIIIAugmentation
│   ├── TestDeiTIIIEvalTransform
│   └── TestBuildDeiT3Transform
├── multicrop_transforms.py Tests
│   ├── TestMultiCropTransform
│   ├── TestMultiCropEvalTransform
│   ├── TestAdaptiveMultiCropTransform
│   └── TestBuildMulticropTransform
├── multicrop_dataset.py Tests
│   ├── TestMultiCropDataset
│   ├── TestMultiCropDatasetRaw
│   ├── TestMulticropCollateFunction
│   ├── TestBuildMulticropDataset
│   └── TestBuildMulticropDataloader
└── Integration Tests
    └── TestIntegration
```

## Key Testing Patterns

### 1. Initialization Tests
Every class has initialization tests validating:
- Default parameters
- Custom parameters
- Required attributes exist

### 2. Shape Validation
All transforms and datasets validate:
- Output tensor shapes
- Correct number of crops
- Proper batch dimensions

### 3. Mock Testing
External dependencies are mocked to:
- Avoid downloads during tests
- Ensure consistent test behavior
- Speed up test execution

### 4. Property Tests
All public properties are tested:
- Return correct types
- Return expected values
- Work with mock data

### 5. Error Handling
Critical error paths are tested:
- Missing files/directories
- Invalid parameters
- Unknown dataset names

## Best Practices

### Writing New Tests
1. **Follow Existing Patterns**: Use the same structure as existing test classes
2. **Use Fixtures**: Leverage existing fixtures for common test data
3. **Mock External Dependencies**: Don't rely on real downloads or files
4. **Test Edge Cases**: Include tests for boundary conditions
5. **Validate Shapes**: Always check tensor/image shapes
6. **Test Error Paths**: Include tests for expected errors

### Maintaining Tests
1. **Update When Adding Features**: Add tests for new functionality
2. **Keep Mocks Updated**: Ensure mocks match real interfaces
3. **Monitor Coverage**: Run coverage reports regularly
4. **Remove Obsolete Tests**: Clean up tests for removed features

## Continuous Integration

### Pre-commit Checks
```bash
# Run before committing
pytest tests/test_data_modules.py --tb=short
```

### CI Pipeline Integration
```yaml
# Example GitHub Actions
- name: Test Data Modules
  run: |
    pytest tests/test_data_modules.py \
      --cov=src/data \
      --cov-report=xml \
      --cov-report=term
```

## Troubleshooting

### Common Issues

#### Import Errors
```
ModuleNotFoundError: No module named 'src'
```
**Solution**: Ensure you're running from the project root and `src` is in `PYTHONPATH`

#### Mock Issues
```
AttributeError: Mock object has no attribute 'X'
```
**Solution**: Check mock setup - you may need to add the attribute to the mock

#### Slow Tests
```
Tests taking too long
```
**Solution**: Ensure external downloads are properly mocked

## Performance

### Test Execution Time
- **Target**: < 30 seconds for full suite
- **Typical**: 10-20 seconds with mocking
- **Without Mocking**: 5+ minutes (actual downloads)

### Optimization Tips
1. Use class-scoped fixtures for expensive setup
2. Mock all network operations
3. Use small test images (224x224 or smaller)
4. Minimize actual file I/O

## Future Enhancements

### Planned Additions
- [ ] Performance benchmarking tests
- [ ] Memory usage tests
- [ ] Distributed data loading tests
- [ ] Additional dataset support tests

### Coverage Improvements
- [ ] Increase coverage to 85%+ per module
- [ ] Add mutation testing
- [ ] Add property-based tests with Hypothesis

## References

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Python unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [H-JEPA Documentation](../README.md)

## Contact

For questions about the test suite:
- Review test comments in `test_data_modules.py`
- Check existing test patterns
- Refer to pytest documentation

---

**Last Updated**: 2025-01-21
**Test Suite Version**: 1.0
**Target Coverage**: 70%+ per module
