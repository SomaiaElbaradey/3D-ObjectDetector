import pytest
import torch
import numpy as np
from math import log
from modules.box_utils import match_anchors_wIgnore, hard_negative_mining, intersect, nms, encode, decode, jaccard, get_ovlp_cellwise, decode_seq

# Test cases for match_anchors_wIgnore
def test_match_anchors_wIgnore():
    gt_boxes = torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.2, 0.2, 0.4, 0.4]])
    gt_labels = torch.tensor([1, 2])
    anchors = torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.0, 0.0, 0.2, 0.2], [0.25, 0.25, 0.35, 0.35]])
    pos_th = 0.5
    nge_th = 0.3
    seq_len = 1

    conf, loc = match_anchors_wIgnore(gt_boxes, gt_labels, anchors, pos_th, nge_th, seq_len=seq_len)
    assert conf.shape[0] == anchors.shape[0]
    assert loc.shape[1] == 4 * seq_len

# Test cases for hard_negative_mining
def test_hard_negative_mining():
    loss = torch.tensor([[0.5, 0.2, 0.9, 0.1]])
    labels = torch.tensor([[1, 0, 0, 0]])
    neg_pos_ratio = 3
    mask = hard_negative_mining(loss, labels, neg_pos_ratio)
    assert mask.shape == loss.shape
    assert mask.sum() == labels.sum() * (1 + neg_pos_ratio)

# Test cases for jaccard
def test_jaccard():
    box_a = torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.2, 0.2, 0.4, 0.4]])
    box_b = torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.3, 0.3, 0.5, 0.5]])
    overlap = jaccard(box_a, box_b)
    assert overlap.shape == (box_a.shape[0], box_b.shape[0])
    assert (overlap >= 0).all() and (overlap <= 1).all()

# Test cases for intersect
def test_intersect():
    box_a = torch.tensor([[0.1, 0.1, 0.3, 0.3]])
    box_b = torch.tensor([[0.2, 0.2, 0.4, 0.4], [0.0, 0.0, 0.1, 0.1]])
    intersection = intersect(box_a, box_b)
    assert intersection.shape == (box_a.shape[0], box_b.shape[0])
    assert (intersection >= 0).all()

# Test cases for encode
def test_encode():
    matched = torch.tensor([[0.1, 0.1, 0.3, 0.3]])
    anchors = torch.tensor([[0.0, 0.0, 0.2, 0.2]])
    variances = [0.1, 0.2]
    encoded = encode(matched, anchors, variances)
    assert encoded.shape == (anchors.shape[0], 4)

# Test cases for decode
def test_decode():
    loc = torch.tensor([[0.1, 0.1, 0.2, 0.2]])
    anchors = torch.tensor([[0.0, 0.0, 0.2, 0.2]])
    decoded = decode(loc, anchors)
    assert decoded.shape == anchors.shape

# Test cases for nms
def test_nms():
    boxes = torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.15, 0.15, 0.35, 0.35], [0.4, 0.4, 0.6, 0.6]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    overlap = 0.5
    top_k = 3
    keep, count = nms(boxes, scores, overlap, top_k)
    assert len(keep) <= top_k
    assert count <= len(scores)

# Test cases for decode_seq
def test_decode_seq():
    loc = torch.tensor([[0.1, 0.2, 0.1, 0.2, 0.2, 0.3, 0.2, 0.3]])
    anchors = torch.tensor([[0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4]])
    seq_len = 2
    variances = [0.1, 0.2]
    decoded = decode_seq(loc, anchors, variances, seq_len)
    assert decoded.shape[1] == seq_len * 4
