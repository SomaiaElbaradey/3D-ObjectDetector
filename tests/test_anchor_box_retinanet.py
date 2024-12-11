import pytest
import torch
import numpy as np
from modules.utils import BufferList
from modules.anchor_box_retinanet import anchorBox

@pytest.fixture
def anchor_box_instance():
    """Fixture to initialize anchorBox instance."""
    return anchorBox()

def test_initialization(anchor_box_instance):
    """Test initialization of anchorBox class."""
    assert isinstance(anchor_box_instance.sizes, list)
    assert isinstance(anchor_box_instance.ratios, np.ndarray)
    assert isinstance(anchor_box_instance.scales, np.ndarray)
    assert isinstance(anchor_box_instance.strides, list)
    assert anchor_box_instance.ar == len(anchor_box_instance.ratios) * len(anchor_box_instance.ratios)

def test_get_cell_anchors(anchor_box_instance):
    """Test _get_cell_anchors method."""
    cell_anchors = anchor_box_instance._get_cell_anchors()
    assert isinstance(cell_anchors, list)
    assert all(isinstance(anchor, torch.Tensor) for anchor in cell_anchors)
    assert len(cell_anchors) == len(anchor_box_instance.sizes)
    for anchor in cell_anchors:
        assert anchor.shape[1] == 4  # Each anchor should have 4 coordinates

def test_generate_anchors_on_one_level(anchor_box_instance):
    """Test _gen_generate_anchors_on_one_level method."""
    base_size = 32
    anchors = anchor_box_instance._gen_generate_anchors_on_one_level(base_size)
    assert isinstance(anchors, np.ndarray)
    assert anchors.shape[1] == 4  # Each anchor should have 4 coordinates
    assert (anchors[:, 2] > 0).all() and (anchors[:, 3] > 0).all()  # Width and height must be positive

def test_forward(anchor_box_instance):
    """Test forward method."""
    grid_sizes = [(8, 8), (16, 16), (32, 32)]  # Example grid sizes
    anchors = anchor_box_instance(grid_sizes)  # Call forward method
    assert isinstance(anchors, torch.Tensor)
    assert anchors.dim() == 2
    assert anchors.size(1) == 4  # Each anchor has 4 coordinates
    assert (anchors[:, 0] <= anchors[:, 2]).all()  # x_min <= x_max
    assert (anchors[:, 1] <= anchors[:, 3]).all()  # y_min <= y_max

@pytest.mark.parametrize("sizes, ratios, scales, strides", [
    ([32, 64, 128], np.array([0.5, 1.0, 2.0]), np.array([1.0]), [8, 16, 32]),
    ([64, 128, 256], np.array([0.33, 1.0]), np.array([1.0, 1.5]), [16, 32, 64]),
    ([128, 256], np.array([0.5, 1.0, 2.0]), np.array([1.0, 1.2]), [32, 64]),
])
def test_anchorbox_custom_configurations(sizes, ratios, scales, strides):
    """Test anchorBox with custom configurations."""
    anchor_box = anchorBox(sizes=sizes, ratios=ratios, scales=scales, strides=strides)
    assert len(anchor_box.cell_anchors) == len(sizes)
    assert anchor_box.ar == len(ratios) * len(ratios)

