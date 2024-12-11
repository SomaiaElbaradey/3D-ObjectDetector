import pytest
import numpy as np
import torch
from modules.utils import (
    BufferList,
    get_individual_labels,
    filter_detections,
    filter_detections_for_tubing,
    filter_detections_for_dumping,
    make_joint_probs_from_marginals,
)

@pytest.fixture
def mock_args():
    """Fixture for mock arguments."""
    class Args:
        CONF_THRESH = 0.5
        NMS_THRESH = 0.4
        TOPK = 10
        GEN_CONF_THRESH = 0.3
        GEN_TOPK = 5
        GEN_NMS = 0.4

    return Args()

@pytest.fixture
def mock_boxes():
    """Fixture for mock bounding boxes."""
    return torch.tensor([[10, 10, 50, 50], [15, 15, 55, 55], [20, 20, 60, 60]], dtype=torch.float32)

@pytest.fixture
def mock_scores():
    """Fixture for mock scores."""
    return torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)

@pytest.fixture
def mock_confidences():
    """Fixture for mock confidences."""
    return torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]], dtype=torch.float32)

def test_filter_detections(mock_args, mock_boxes, mock_scores):
    """Test filter_detections."""
    cls_dets = filter_detections(mock_args, mock_scores, mock_boxes)

    assert cls_dets.shape[1] == 5  # 4 box coordinates + 1 score
    assert cls_dets.shape[0] <= mock_args.TOPK  # Max detections

def test_filter_detections_for_tubing(mock_args, mock_boxes, mock_scores, mock_confidences):
    """Test filter_detections_for_tubing."""
    save_data = filter_detections_for_tubing(mock_args, mock_scores, mock_boxes, mock_confidences)

    assert save_data.shape[1] > 5  # Includes boxes, scores, and confidences
    assert save_data.shape[0] <= mock_args.TOPK * 60

def test_filter_detections_for_dumping(mock_args, mock_boxes, mock_scores, mock_confidences):
    """Test filter_detections_for_dumping."""
    cls_dets, save_data = filter_detections_for_dumping(mock_args, mock_scores, mock_boxes, mock_confidences)

    assert cls_dets.shape[1] == 5  # Includes box coordinates and scores
    assert save_data.shape[1] > 5  # Includes confidences
    assert cls_dets.shape[0] <= mock_args.GEN_TOPK
