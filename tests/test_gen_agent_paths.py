import pytest
import numpy as np
from modules.gen_agent_paths import (
    update_agent_paths, trim_paths, remove_dead_paths, sort_live_paths, copy_live_to_dead,
    score_of_edge, bbox_overlaps, intersect, fill_gaps
)

@pytest.fixture
def mock_detections():
    """Fixture to create mock detections for testing."""
    dets = {
        "boxes": np.array([[0.1, 0.1, 0.2, 0.2], [0.2, 0.2, 0.3, 0.3], [0.3, 0.3, 0.4, 0.4]]),
        "scores": np.array([0.9, 0.8, 0.7]),
        "allScores": np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
    }
    return dets

@pytest.fixture
def mock_paths():
    """Fixture to create mock live and dead paths."""
    live_paths = [
        {
            "boxes": np.array([[0.1, 0.1, 0.2, 0.2]]),
            "scores": [0.9],
            "allScores": np.array([[0.9, 0.1]]),
            "foundAt": [0],
            "count": 1
        }
    ]
    dead_paths = []
    return live_paths, dead_paths

def test_update_agent_paths(mock_detections, mock_paths):
    """Test the update_agent_paths function."""
    live_paths, dead_paths = mock_paths
    dets = mock_detections
    updated_live_paths, updated_dead_paths = update_agent_paths(
        live_paths, dead_paths, dets, num_classes_to_use=2, time_stamp=1
    )
    assert len(updated_live_paths) >= len(live_paths)
    assert all("boxes" in path for path in updated_live_paths)

def test_trim_paths(mock_paths):
    """Test the trim_paths function."""
    live_paths, _ = mock_paths
    trim_threshold = 1
    keep_num = 1
    trimmed_paths = trim_paths(live_paths, trim_threshold, keep_num)
    for path in trimmed_paths:
        assert len(path["boxes"]) <= keep_num

def test_sort_live_paths(mock_paths):
    """Test the sort_live_paths function."""
    live_paths, dead_paths = mock_paths
    path_order_score = np.array([0.9])
    sorted_paths, sorted_dead_paths = sort_live_paths(live_paths, path_order_score, dead_paths, jumpgap=5, time_stamp=1)
    assert len(sorted_paths) == len(live_paths)

def test_score_of_edge(mock_detections):
    """Test the score_of_edge function."""
    v1 = {
        "boxes": np.array([[0.1, 0.1, 0.2, 0.2]]),
        "scores": [0.9]
    }
    v2 = mock_detections
    as1 = v1["scores"]
    as2 = v2["allScores"]
    scores = score_of_edge(v1, v2, iouth=0.1, costtype="scoreiou", avoid_dets=[], as1=as1, as2=as2, jumpgap=5)
    assert len(scores) == v2["boxes"].shape[0]

def test_bbox_overlaps(mock_detections):
    """Test the bbox_overlaps function."""
    box_a = np.array([0.1, 0.1, 0.2, 0.2])
    box_b = mock_detections["boxes"]
    overlaps = bbox_overlaps(box_a, box_b)
    assert overlaps.shape[0] == box_b.shape[0]
    assert np.all(overlaps >= 0)

def test_intersect(mock_detections):
    """Test the intersect function."""
    box_a = np.array([0.1, 0.1, 0.2, 0.2])
    box_b = mock_detections["boxes"]
    intersections = intersect(box_a, box_b)
    assert intersections.shape[0] == box_b.shape[0]
    assert np.all(intersections >= 0)

def test_fill_gaps(mock_paths):
    """Test the fill_gaps function."""
    live_paths, _ = mock_paths
    min_len_with_gaps = 1
    minscore = 0.3
    filled_paths = fill_gaps(live_paths, min_len_with_gaps, minscore)
    assert len(filled_paths) >= len(live_paths)
