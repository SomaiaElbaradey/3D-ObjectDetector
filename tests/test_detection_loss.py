import pytest
import torch
from modules import box_utils  # Replace with actual import
from modules.detection_loss import smooth_l1_loss, sigmoid_focal_loss, get_one_hot_labels, FocalLoss  # Replace with actual import

@pytest.fixture
def args_mock():
    """Mock arguments for FocalLoss initialization."""
    class Args:
        POSTIVE_THRESHOLD = 0.5
        NEGTIVE_THRESHOLD = 0.4
        num_classes = 10
        num_label_types = 2
        num_classes_list = [10, 2]
    return Args()

@pytest.fixture
def focal_loss_instance(args_mock):
    """Fixture to create FocalLoss instance."""
    return FocalLoss(args_mock)

def test_smooth_l1_loss():
    """Test smooth_l1_loss function."""
    input = torch.tensor([[0.5, 0.8], [0.4, 0.6]])
    target = torch.tensor([[0.5, 0.6], [0.3, 0.7]])
    loss = smooth_l1_loss(input, target, beta=1. / 9, reduction='mean')
    assert loss.item() > 0

def test_sigmoid_focal_loss():
    """Test sigmoid_focal_loss function."""
    preds = torch.sigmoid(torch.randn(4, 5))
    labels = torch.eye(4, 5)
    num_pos = labels.sum().item()
    loss = sigmoid_focal_loss(preds, labels, num_pos=num_pos, alpha=0.25, gamma=2.0)
    assert loss.item() > 0
