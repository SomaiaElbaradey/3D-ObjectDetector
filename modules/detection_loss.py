import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
from modules import box_utils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def smooth_l1_loss(input, target, beta=1. / 9, reduction='sum'):
    logging.info("Entered smooth_l1_loss function")
    logging.debug(f"Input: {input}, Target: {target}, Beta: {beta}, Reduction: {reduction}")
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if reduction == 'mean':
        result = loss.mean()
    else:
        result = loss.sum()
    logging.debug(f"Smooth L1 Loss Result: {result}")
    return result


def sigmoid_focal_loss(preds, labels, num_pos, alpha, gamma):
    logging.info("Entered sigmoid_focal_loss function")
    logging.debug(f"Preds: {preds}, Labels: {labels}, Num Pos: {num_pos}, Alpha: {alpha}, Gamma: {gamma}")
    loss = F.binary_cross_entropy(preds, labels, reduction='none')
    alpha_factor = alpha * labels + (1.0 - alpha) * (1.0 - labels)
    pt = preds * labels + (1.0 - preds) * (1.0 - labels)
    focal_weight = alpha_factor * ((1-pt) ** gamma)
    loss = (loss * focal_weight).sum() / num_pos
    logging.debug(f"Sigmoid Focal Loss Result: {loss}")
    return loss


def get_one_hot_labels(tgt_labels, numc):
    logging.info("Entered get_one_hot_labels function")
    logging.debug(f"Target Labels: {tgt_labels}, Num Classes: {numc}")
    new_labels = torch.zeros([tgt_labels.shape[0], numc], device=tgt_labels.device)
    new_labels[:, tgt_labels] = 1.0
    logging.debug(f"One-hot Encoded Labels: {new_labels}")
    return new_labels


class FocalLoss(nn.Module):
    def __init__(self, args, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        logging.info("Initializing FocalLoss")
        self.positive_threshold = args.POSTIVE_THRESHOLD
        self.negative_threshold = args.NEGTIVE_THRESHOLD
        self.num_classes = args.num_classes
        self.num_label_types = args.num_label_types
        self.num_classes_list = args.num_classes_list
        self.alpha = 0.25
        self.gamma = 2.0
        logging.debug(f"Initialized with alpha={alpha}, gamma={gamma}")

    def forward(self, confidence, predicted_locations, gt_boxes, gt_labels, counts, anchors, ego_preds, ego_labels):
        logging.info("Entered FocalLoss forward function")
        logging.debug(f"Input confidence shape: {confidence.shape}, predicted_locations shape: {predicted_locations.shape}")
        ego_preds = torch.sigmoid(ego_preds)
        ps = confidence.shape
        preds = torch.sigmoid(confidence)
        logging.debug("Sigmoid applied to confidence and ego_preds")
        # ps = predicted_locations.shape
        # predicted_locations = predicted_locations.view(ps[0],ps[1], -1, [-1])
        ball_labels = []
        bgt_locations = []
        blabels_bin = []
        # mask = torch.zeros([preds.shape[0],preds.shape[1]], dtype=torch.int)

        with torch.no_grad():
            # gt_boxes = gt_boxes.cpu()
            # gt_labels = gt_labels.cpu()
            # anchors = anchors.cpu()
            # device = torch.device("cpu")
            device = preds.device
            zeros_tensor = torch.zeros(1, gt_labels.shape[-1], device=device)
            for b in range(gt_boxes.shape[0]):
                all_labels = []
                gt_locations = []
                labels_bin = []
                for s in range(gt_boxes.shape[1]):
                    gt_boxes_batch = gt_boxes[b, s, :counts[b,s], :]
                    gt_labels_batch = gt_labels[b, s, :counts[b,s], :]
                    if counts[b,s]>0:
                        gt_dumy_labels_batch = torch.LongTensor([i for i in range(counts[b,s])]).to(device)
                        conf, loc = box_utils.match_anchors_wIgnore(gt_boxes_batch, gt_dumy_labels_batch, 
                            anchors, pos_th=self.positive_threshold, nge_th=self.negative_threshold )
                    else:
                        loc = torch.zeros_like(anchors, device=device)
                        conf = ego_labels.new_zeros(anchors.shape[0], device=device) - 1
                    
                    # print(conf.device)
                    # print(loc.device)
                    gt_locations.append(loc)
                    labels_bin.append(conf)

                    dumy_conf = conf.clone()
                    dumy_conf[dumy_conf<0] = 0
                    labels_bs = torch.cat((zeros_tensor, gt_labels_batch),0)
                    batch_labels = labels_bs[dumy_conf,:]
                    all_labels.append(batch_labels)

                all_labels = torch.stack(all_labels, 0).float()
                gt_locations = torch.stack(gt_locations, 0)
                labels_bin = torch.stack(labels_bin, 0).float()
                ball_labels.append(all_labels)
                bgt_locations.append(gt_locations)
                blabels_bin.append(labels_bin)
            
            all_labels = torch.stack(ball_labels, 0)
            gt_locations = torch.stack(bgt_locations, 0)
            labels_bin = torch.stack(blabels_bin, 0)
            # mask = labels_bin > -1
            # device = ego_preds.device
            # all_labels = all_labels.to(device)
            # gt_locations = gt_locations.to(device)
            # labels_bin = labels_bin.to(device)

        # bgt_locations = []
        # blabels_bin = []
        pos_mask = labels_bin > 0
        num_pos = max(1.0, float(pos_mask.sum()))
        logging.debug(f"Positive mask sum: {num_pos}")
        
        gt_locations = gt_locations[pos_mask].reshape(-1, 4)
        predicted_locations = predicted_locations[pos_mask].reshape(-1, 4)
        regression_loss = smooth_l1_loss(predicted_locations, gt_locations)/(num_pos * 4.0)
        logging.debug(f"Regression Loss: {regression_loss}")

        # if regression_loss.item()>40:
        #     pdb.set_trace()
        
        mask = labels_bin > -1 # Get mask to remove ignore examples
        
        masked_labels = all_labels[mask].reshape(-1, self.num_classes) # Remove Ignore labels
        masked_preds = preds[mask].reshape(-1, self.num_classes) # Remove Ignore preds
        cls_loss = sigmoid_focal_loss(masked_preds, masked_labels, num_pos, self.alpha, self.gamma)
        logging.debug(f"Classification Loss: {cls_loss}")

        mask = ego_labels>-1
        numc = ego_preds.shape[-1]
        masked_preds = ego_preds[mask].reshape(-1, numc) # Remove Ignore preds
        masked_labels = ego_labels[mask].reshape(-1) # Remove Ignore labels
        one_hot_labels = get_one_hot_labels(masked_labels, numc)
        ego_loss = 0
        if one_hot_labels.shape[0]>0:
            ego_loss = sigmoid_focal_loss(masked_preds, one_hot_labels, one_hot_labels.shape[0], self.alpha, self.gamma)
        logging.debug(f"Ego Loss: {ego_loss}")
        # print(regression_loss, cls_loss, ego_loss)
        return regression_loss, cls_loss/8.0 + ego_loss/4.0