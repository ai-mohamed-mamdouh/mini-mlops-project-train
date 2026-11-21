from unittest.mock import Mock

import pytest

from mini_mlops.workflows.train_workflow import TrainingWorkflow


@pytest.fixture
def dependencies():

    return {
        "config": Mock(),
        "data_loader": Mock(),
        "splitter": Mock(),
        "model": Mock(),
        "pre_processor": Mock(),
        "artifact_store": Mock(),
    }


def test_workflow_initialization(dependencies):

    workflow = TrainingWorkflow(**dependencies)

    assert isinstance(workflow, TrainingWorkflow)


def test_dataset_loader_injection(dependencies):

    workflow = TrainingWorkflow(**dependencies)

    assert workflow.data_loader is dependencies["data_loader"]


def test_model_injection(dependencies):

    workflow = TrainingWorkflow(**dependencies)

    assert workflow.model is dependencies["model"]


def test_preprocessor_injection(dependencies):

    workflow = TrainingWorkflow(**dependencies)

    assert workflow.pre_processor is dependencies["pre_processor"]


def test_artifact_store_injection(dependencies):

    workflow = TrainingWorkflow(**dependencies)

    assert workflow.artifact_store is dependencies["artifact_store"]


def test_splitter_injection(dependencies):

    workflow = TrainingWorkflow(**dependencies)

    assert workflow.splitter is dependencies["splitter"]
