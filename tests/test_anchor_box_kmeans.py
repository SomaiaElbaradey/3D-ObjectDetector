import pytest
import torch
import numpy as np
from modules.utils import BufferList  
from modules.anchor_box_kmeans import anchorBox

@pytest.fixture
def anchor_box_instance():
    """Fixture to initialize anchorBox instance."""
    return anchorBox()

def test_initialization(anchor_box_instance):
    """Test initialization of anchorBox class."""
    assert isinstance(anchor_box_instance.aspect_ratios, list)
    assert isinstance(anchor_box_instance.scale_ratios, list)
    assert isinstance(anchor_box_instance.default_sizes, list)
    assert anchor_box_instance.anchor_boxes == len(anchor_box_instance.aspect_ratios) * len(anchor_box_instance.scale_ratios)
    assert anchor_box_instance.num_anchors == anchor_box_instance.ar

def test_get_cell_anchors(anchor_box_instance):
    """Test _get_cell_anchors method."""
    cell_anchors = anchor_box_instance._get_cell_anchors()
    assert isinstance(cell_anchors, list)
    assert all(isinstance(anchor, torch.Tensor) for anchor in cell_anchors)
    assert len(cell_anchors) == len(anchor_box_instance.default_sizes)
    for anchor in cell_anchors:
        assert anchor.shape[1] == 4  # Each anchor should have 4 coordinates

@pytest.mark.parametrize("aspect_ratios, scale_ratios", [
    ([0.5, 1.0], [1.0]),
    ([0.5, 1.0, 2.0], [1.0, 1.5]),
    ([1.0], [1.0, 2.0, 3.0]),
])
def test_anchorbox_custom_ratios(aspect_ratios, scale_ratios):
    """Test anchorBox with custom aspect and scale ratios."""
    anchor_box = anchorBox(aspect_ratios=aspect_ratios, scale_ratios=scale_ratios)
    assert anchor_box.anchor_boxes == len(aspect_ratios) * len(scale_ratios)
    assert anchor_box.num_anchors == anchor_box.ar

