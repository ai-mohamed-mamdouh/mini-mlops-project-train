from unittest.mock import Mock

import numpy as np
import pytest

from mini_mlops.workflows.train_workflow import TrainingWorkflow


def test_training_fits_and_returns_same_model():
    model = Mock(spec=["fit"])
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    assert TrainingWorkflow().run(model, X, y) is model
    model.fit.assert_called_once_with(X, y)


def test_training_propagates_fit_error():
    model = Mock(spec=["fit"])
    model.fit.side_effect = ValueError("fit failed")
    with pytest.raises(ValueError, match="fit failed"):
        TrainingWorkflow().run(model, [[1]], [0])
