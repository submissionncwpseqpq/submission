import torch
import torch.nn as nn

class TCLLoss(nn.Module):
    def __init__(self, temperature=0.1, contrast_mode='all', base_temperature=0.07, single_view_mode=False):
        super(TCLLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.single_view_mode = single_view_mode  # New parameter to toggle single-view mode
        
    def forward(self, features, labels=None, mask=None, pair_weights=None):
        device = features.device  # Simplified device assignment

        if self.single_view_mode:
            # Ensure features are 2D (batch_size, feature_dim) for single-view mode
            if len(features.shape) == 3 and features.shape[1] == 1:
                features = features.squeeze(1)
        else:
            # Ensure features are at least 3D (batch_size, n_views, ...) for multi-view mode
            if len(features.shape) < 3:
                raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        elif mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        if self.single_view_mode:
            # Adjust mask for single-view if needed
            mask = mask.float().to(device)
        else:
            # Process features for multi-view mode
            if len(features.shape) > 3:
                features = features.view(features.shape[0], features.shape[1], -1)

        # Applying pair_weights if provided
        if pair_weights is not None:
            if pair_weights.shape != mask.shape:
                raise ValueError('pair_weights must have the same shape as mask')
            mask *= pair_weights.to(device)

        # Compute contrastive logits
        if self.single_view_mode or self.contrast_mode == 'one':
            anchor_feature = features
        else:
            # For multi-view, handle according to the specified contrast_mode
            contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
            anchor_feature = contrast_feature if self.contrast_mode == 'all' else features[:, 0]

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, anchor_feature.T), self.temperature)

        # Numerical stability and mask adjustment
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        if not self.single_view_mode:
            # Adjust mask for multi-view mode
            mask = mask.repeat(features.shape[1], features.shape[1]) if self.contrast_mode == 'all' else mask

        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask *= logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # Compute mean log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

        # Compute loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss