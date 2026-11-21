from sklearn.preprocessing import StandardScaler

from mini_mlops.preprocessing.strategies import (
    StandardScalerStrategy,
)


def test_standard_scaler_creation():
    strategy = StandardScalerStrategy()

    preprocessor = strategy.create()

    assert isinstance(preprocessor, StandardScaler)


# def test_no_scaling_creation():
#     strategy = NoScalingStrategy()

#     preprocessor = strategy.create()

#     assert isinstance(preprocessor, FunctionTransformer)
