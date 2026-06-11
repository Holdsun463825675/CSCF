import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from FusionLogits import compute_uncertainty, combine_logits_similarity


class MLP(nn.Module):
    def __init__(self, attr_dim, hid_dim, num_class, device='cpu'):
        super(MLP, self).__init__()
        self.device = device
        self.classifier = nn.Sequential(
            nn.Linear(attr_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.ReLU(),
            nn.LayerNorm(hid_dim),
            nn.Linear(hid_dim, num_class),
        ).to(self.device)

    def forward(self, attrs):
        attrs = attrs.to(self.device)
        logits = self.classifier(attrs)
        return logits

class Visual2CommonProj(nn.Module):
    def __init__(self, visual_dim, hid_dim, common_dim, device='cpu'):
        super(Visual2CommonProj, self).__init__()
        self.device = device
        self.v2c_proj = nn.Sequential(
            nn.Linear(visual_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.ReLU(),
            nn.LayerNorm(hid_dim),
            nn.Linear(hid_dim, common_dim),
            nn.ReLU(),
        ).to(self.device)

    def forward(self, visual):
        visual = visual.to(self.device)
        visual_proj = self.v2c_proj(visual)
        return visual_proj


class SingleModeModule(nn.Module):
    def __init__(self, attr_dim: int, hid_dim: int, proto_dim: int, num_class, net_type='semantic', net_num=4, device='cpu'):
        super().__init__()
        self.device = device
        self.net_type = net_type
        self.net_num = net_num
        self.s2c = []
        self.v2c = []
        self.params_to_update = []

        if self.net_type == 'semantic':
            common_dim = attr_dim
            self.v2c = nn.ModuleList([Visual2CommonProj(visual_dim=proto_dim, hid_dim=int(hid_dim * (i + 1) / net_num),
                                          common_dim=common_dim, device=self.device) for i in range(net_num)])
            for i in range(net_num):
                self.params_to_update.extend(list(self.v2c[i].parameters()))
        else:
            raise ValueError('net_type must be "semantic"')


    def get_logits(self, vc, sc, scale=5):
        vc_norm = vc.norm(dim=1, keepdim=True)
        sc_norm = sc.norm(dim=1, keepdim=True)

        vc_ns = torch.where(vc_norm == 0, vc, scale * vc / vc_norm)
        sc_ns = torch.where(sc_norm == 0, sc, scale * sc / sc_norm)

        logits = vc_ns @ sc_ns.t()
        return logits

    def forward(self, x, attrs, train=False):
        v = [x for _ in range(self.net_num)]
        s = [attrs for _ in range(self.net_num)]
        logits = []

        if self.net_type == 'semantic':
            v = [self.v2c[i](x.to(self.device)) for i in range(self.net_num)]
            for i in range(self.net_num):
                logits.append(self.get_logits(v[i], s[i]))


        mean_logit = sum(logits) / self.net_num
        if train:
            return mean_logit, logits, v, s
        else:
            return mean_logit, logits


    def train_loss(self, attrs_seen, feats, targets, mask):
        masked_targets = targets[mask == 1]
        mean_logit, logits, v, s = self.forward(feats, attrs_seen, train=True)
        masked_logit = mean_logit[mask == 1]

        total_loss = 0
        for i in range(self.net_num):
            logits[i] = logits[i][mask == 1]
            loss = F.cross_entropy(logits[i], masked_targets)
            total_loss += loss
        if self.net_num > 1:
            loss_fus = F.cross_entropy(masked_logit, masked_targets)
            total_loss += loss_fus

        return total_loss


class FusionModule(nn.Module):
    def __init__(self, device='cpu', fusion_type='A', topk=0):
        super(FusionModule, self).__init__()
        self.device = device
        self.fusion_type = fusion_type
        self.topk = topk

    def forward(self, logits_multi, sub_logits, mask, similarity_mat, seen_masks=None, balanced_weight=1, train=False):
        logits_multi = torch.stack(logits_multi, dim=0)
        uncertainty = compute_uncertainty(logits_multi, sub_logits)

        if train:
            if self.fusion_type == 'S':
                logits_combine = combine_logits_similarity(logits_multi, uncertainty, similarity_mat, self.topk)
            else:
                raise ValueError(f"Unsupported fusion type: {self.fusion_type}")
            probs = torch.nn.Softmax(dim=-1)(logits_combine)
        else:
            logits_combine = torch.zeros_like(logits_multi[0])

            idx_0 = (mask == 0)
            idx_1 = (mask == 1)
            idx_2 = (mask == 2)

            logits_combine[idx_0] = logits_multi[0, idx_0]
            logits_combine[idx_1] = logits_multi[1, idx_1]

            if idx_2.any():
                uncertainty_idx2 = [tensor[idx_2] for tensor in uncertainty]
                if self.fusion_type == 'S':
                    logits_combine[idx_2] = combine_logits_similarity(logits_multi[:, idx_2], uncertainty_idx2, similarity_mat,
                                                                      self.topk)
                else:
                    raise ValueError(f"Unsupported fusion type: {self.fusion_type}")
            probs = torch.nn.Softmax(dim=-1)(logits_combine)

            idx_0 = (mask == 0).nonzero(as_tuple=True)[0]
            idx_1 = (mask == 1).nonzero(as_tuple=True)[0]
            seen_mask_0 = seen_masks[0].nonzero(as_tuple=True)[0]
            seen_mask_1 = seen_masks[1].nonzero(as_tuple=True)[0]

            if idx_0.numel() > 0 and seen_mask_0.numel() > 0:
                probs.index_put_((idx_0.unsqueeze(1), seen_mask_0.unsqueeze(0)),
                                 probs.index_select(0, idx_0)[:, seen_mask_0] * balanced_weight)
            if idx_1.numel() > 0 and seen_mask_1.numel() > 0:
                probs.index_put_((idx_1.unsqueeze(1), seen_mask_1.unsqueeze(0)),
                                 probs.index_select(0, idx_1)[:, seen_mask_1] * balanced_weight)

        return probs

    def train_loss(self, logits_multi, sub_logits, mask, similarity_mat, targets):
        logits_combine = self.forward(logits_multi, sub_logits, mask, similarity_mat, train=True)
        loss = F.cross_entropy(logits_combine, targets)
        return loss


class CSCF(nn.Module):
    def __init__(self, attr_dim: int, hid_dim: int, proto_dim: int(), modal_num=1, lr=5e-4, weight_decay=0.0001, device='cpu', topk=0, num_class=0):
        super(CSCF, self).__init__()
        self.device = device
        self.modal_num = modal_num
        net_type = 'semantic'
        net_num = 4
        fusion_type = 'S'

        self.single_mode_modules = nn.ModuleList([
            SingleModeModule(attr_dim, hid_dim, proto_dim[i], num_class, net_type, net_num, self.device)
            for i in range(self.modal_num)
        ])

        self.fusion_module = FusionModule(device, fusion_type, topk)

        params_to_update = list()
        for module in self.single_mode_modules:
            params_to_update += module.params_to_update
        self.params_to_update = params_to_update
        self.optim = torch.optim.Adam(self.params_to_update, lr=lr, weight_decay=weight_decay)

    def forward(self, inputs_multi, mask, attrs_multi, similarity_mat, seen_masks, balanced_weight, train=False, sub=False):
        single_mode_outputs = []
        single_mode_sub_logits = []
        for i in range(len(self.single_mode_modules)):
            if train:
                output, sub_logits, _, _ = self.single_mode_modules[i](torch.tensor(inputs_multi[i]).to(self.device), torch.tensor(attrs_multi).to(self.device), train=True)
            else:
                output, sub_logits = self.single_mode_modules[i](torch.tensor(inputs_multi[i]).to(self.device), torch.tensor(attrs_multi).to(self.device), train=False)
            single_mode_outputs.append(output)
            single_mode_sub_logits.append(sub_logits)

        fusion_output = self.fusion_module(single_mode_outputs, single_mode_sub_logits,
                                           torch.tensor(mask).to(self.device),
                                           torch.tensor(similarity_mat).to(self.device),
                                           torch.tensor(np.array(seen_masks)).to(self.device), balanced_weight, train) \
            if self.modal_num > 1 else single_mode_outputs[0]
        fusion_output = fusion_output.cpu()

        if sub:
            return single_mode_sub_logits
        return fusion_output

    def train_loss(self, inputs_multi, attrs_multi, similarity_mat, targets, mask):
        total_loss = torch.tensor(0.0).to(self.device)

        for i in range(len(self.single_mode_modules)):
            mask_single_mode = torch.zeros_like(mask).to(self.device)
            idx = torch.logical_or(mask == i, mask == 2)
            mask_single_mode[idx] = 1
            total_loss += self.single_mode_modules[i].train_loss(
                attrs_multi.to(self.device), inputs_multi[i].to(self.device), targets, mask_single_mode)

        single_mode_outputs = []
        single_mode_sub_logits = []
        for i in range(len(self.single_mode_modules)):
            output, sub_logits, _, _ = self.single_mode_modules[i](inputs_multi[i], attrs_multi, train=True)
            single_mode_outputs.append(output)
            single_mode_sub_logits.append(sub_logits)

        if self.modal_num > 1:
            total_loss += self.fusion_module.train_loss(single_mode_outputs, single_mode_sub_logits, mask, similarity_mat, targets)

        return total_loss

    def train_loop(self, train_dataloader, attrs_multi, similarity_mat, epochs=50):
        for epoch in range(epochs):
            loss_avg = 0
            for i, batch in enumerate(train_dataloader):
                target = torch.tensor(batch[1]).to(self.device)
                input_multi = [torch.tensor(inputs).to(self.device) for inputs in batch[0]]
                attr_multi = torch.tensor(attrs_multi).to(self.device)
                mask = torch.tensor(batch[2]).to(self.device)
                sim_mat = torch.tensor(similarity_mat).to(self.device)
                loss = self.train_loss(input_multi, attr_multi, sim_mat, target, mask)
                loss_avg += loss.item()
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            loss_avg = loss_avg / len(train_dataloader)
            print(f'The loss of epoch {epoch} is {loss_avg}.')
