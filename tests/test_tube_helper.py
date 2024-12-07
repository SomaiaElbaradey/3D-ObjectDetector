import pytest
import numpy as np
from modules.tube_helper import (
    make_det_tube,
    get_nonnp_det_tube,
    make_gt_tube,
    trim_tubes,
    get_topk_classes,
    dpEMmax,
    bbox_overlaps,
    get_tube_3Diou,
    nms3dt,
)

@pytest.fixture
def mock_boxes():
    """Fixture to provide mock bounding boxes."""
    return np.array([[10, 10, 50, 50], [15, 15, 55, 55], [20, 20, 60, 60]])

@pytest.fixture
def mock_scores():
    """Fixture to provide mock scores."""
    return np.array([0.9, 0.8, 0.85])

@pytest.fixture
def mock_frames():
    """Fixture to provide mock frame indices."""
    return [1, 2, 3]

def test_make_det_tube(mock_scores, mock_boxes, mock_frames):
    """Test the make_det_tube function."""
    label_id = 1
    tube = make_det_tube(mock_scores, mock_boxes, mock_frames, label_id)

    assert tube["label_id"] == label_id
    assert tube["scores"].shape == mock_scores.shape
    assert tube["boxes"].shape == mock_boxes.shape
    assert tube["frames"].shape[0] == len(mock_frames)

def test_get_nonnp_det_tube(mock_scores, mock_boxes):
    """Test the get_nonnp_det_tube function."""
    label_id = 2
    start, end = 0, 3
    tube = get_nonnp_det_tube(mock_scores, mock_boxes, start, end, label_id)

    assert tube["label_id"] == label_id
    assert len(tube["frames"]) == end - start
    assert tube["boxes"].shape == mock_boxes.shape
    assert "score" in tube

def test_make_gt_tube(mock_boxes, mock_frames):
    """Test the make_gt_tube function."""
    label_id = 3
    tube = make_gt_tube(mock_frames, mock_boxes, label_id)

    assert tube["label_id"] == label_id
    assert tube["frames"].shape == (len(mock_frames),)
    assert tube["boxes"].shape == mock_boxes.shape

def test_trim_tubes(mock_boxes, mock_scores):
    """Test the trim_tubes function."""
    paths = [{"allScores": np.tile(mock_scores, (3, 1)), "boxes": mock_boxes, "foundAt": [1, 2, 3]}]
    tubes = trim_tubes(0, 3, paths, [], [3], topk=1, min_len=2, trim_method="none")

    assert len(tubes) > 0
    assert tubes[0]["label_id"] is not None
    assert len(tubes[0]["frames"]) > 0

def test_get_topk_classes(mock_scores):
    """Test the get_topk_classes function."""
    mock_all_scores = np.tile(mock_scores, (3, 1)).T
    topk_classes, topk_scores = get_topk_classes(mock_all_scores, topk=2)

    assert len(topk_classes) == 2
    assert len(topk_scores) == 2
    assert topk_scores[0] >= topk_scores[1]  # Ensure scores are sorted

def test_dpEMmax():
    """Test the dpEMmax function."""
    M = np.random.rand(2, 5)
    alpha = 2
    ps, D = dpEMmax(M, alpha)

    assert ps.shape == (5,)
    assert D.shape == (2, 5)

def test_bbox_overlaps(mock_boxes):
    """Test the bbox_overlaps function."""
    box = mock_boxes[0]
    overlaps = bbox_overlaps(box, mock_boxes)

    assert overlaps.shape == (mock_boxes.shape[0],)
    assert overlaps[0] == 1.0  # Self-overlap should be 1

def test_nms3dt(mock_scores, mock_boxes, mock_frames):
    """Test the nms3dt function."""
    tubes = [make_det_tube(mock_scores, mock_boxes, mock_frames, i) for i in range(3)]
    selected_tubes = nms3dt(tubes, overlap=0.5)

    assert len(selected_tubes) > 0
    assert all(isinstance(tube, dict) for tube in selected_tubes)
