import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from sskit.camera import image_to_ground
from ..builder import LOSSES

_DIAG_PATH = os.environ.get('LOCSIM_DIAG', '')   # set env var to enable
_DIAG_MAX  = int(os.environ.get('LOCSIM_DIAG_MAX', '5'))  # batches to log


@LOSSES.register_module()
class LocSim3DLoss(nn.Module):

    def __init__(self,
                 keypoint_index=1,
                 image_width=3840,
                 image_height=2160,
                 heatmap_w=48,
                 heatmap_h=64,
                 delta=0.5,
                 temperature=100.0):
        super().__init__()
        self.keypoint_index = keypoint_index
        self.image_width    = image_width
        self.image_height   = image_height
        self.heatmap_w      = heatmap_w
        self.heatmap_h      = heatmap_h
        self.delta          = delta
        self.temperature    = temperature
        self._diag_count    = 0

    def _soft_argmax2d(self, heatmap):
        N, H, W = heatmap.shape
        flat = F.softmax(
            heatmap.view(N, -1) * self.temperature, dim=-1).view(N, H, W)
        xs = torch.arange(W, dtype=flat.dtype, device=flat.device).view(1, 1, W)
        ys = torch.arange(H, dtype=flat.dtype, device=flat.device).view(1, H, 1)
        return (flat * xs).sum(dim=(1, 2)), (flat * ys).sum(dim=(1, 2))

    def _argmax2d(self, heatmap):
        """Hard argmax (non-differentiable) — used only for diagnostics."""
        N, H, W = heatmap.shape
        idx = heatmap.detach().view(N, -1).argmax(dim=-1)
        x = (idx % W).float()
        y = (idx // W).float()
        return x, y

    def _udp_to_full_image(self, hx, hy, centers, scales):
        sc = scales * 200.0
        x_full = hx * sc[:, 0] / (self.heatmap_w - 1) + centers[:, 0] - sc[:, 0] * 0.5
        y_full = hy * sc[:, 1] / (self.heatmap_h - 1) + centers[:, 1] - sc[:, 1] * 0.5
        return x_full, y_full

    def forward(self, output, img_metas):
        N = output.shape[0]

        valid_idx = [
            i for i in range(N)
            if img_metas[i].get('camera_matrix') is not None
            and img_metas[i].get('undist_poly') is not None
            and img_metas[i].get('position_on_pitch') is not None
        ]
        if not valid_idx:
            return output.sum() * 0.0

        heatmap = output[valid_idx, self.keypoint_index]   # [M, H, W]
        hx_soft, hy_soft = self._soft_argmax2d(heatmap)    # [M]
        hx_hard, hy_hard = self._argmax2d(heatmap)         # [M] diagnostics only

        metas   = [img_metas[i] for i in valid_idx]
        # Use orig_center/orig_scale (4K bbox coords) not center/scale —
        # TopDownAffineMosaicAug overwrites the latter with crop dimensions.
        centers = output.new_tensor([m['orig_center'] for m in metas])
        scales  = output.new_tensor([m['orig_scale']  for m in metas])

        x_soft, y_soft = self._udp_to_full_image(hx_soft, hy_soft, centers, scales)
        x_hard, y_hard = self._udp_to_full_image(hx_hard, hy_hard, centers, scales)

        W = self.image_width
        pkt = torch.stack(
            [(x_soft - (W - 1) * 0.5) / W,
             (y_soft - (self.image_height - 1) * 0.5) / W],
            dim=-1)  # [M, 2]

        pred_3d_list = []
        for i, meta in enumerate(metas):
            cam  = output.new_tensor(meta['camera_matrix'])
            if cam.ndim == 2 and cam.shape[0] == 4:
                cam = cam[:3]
            poly = output.new_tensor(meta['undist_poly'])
            pred_3d_list.append(image_to_ground(cam, poly, pkt[i:i + 1])[:, :2])
        pred_3d = torch.cat(pred_3d_list, dim=0)  # [M, 2]

        gt_3d = output.new_tensor([m['position_on_pitch'] for m in metas])
        vis   = output.new_tensor(
            [m['joints_3d_visible'][self.keypoint_index, 0] for m in metas])

        per_sample = F.huber_loss(
            pred_3d, gt_3d, reduction='none', delta=self.delta).sum(-1)
        loss = (per_sample * vis).sum() / vis.sum().clamp(min=1e-6)

        # ── diagnostics ────────────────────────────────────────────────────
        if _DIAG_PATH and self._diag_count < _DIAG_MAX:
            self._save_diag(
                metas, heatmap,
                hx_soft, hy_soft, hx_hard, hy_hard,
                x_soft, y_soft, x_hard, y_hard,
                pkt, pred_3d, gt_3d, vis, per_sample)
            self._diag_count += 1
        # ───────────────────────────────────────────────────────────────────

        return loss

    def _save_diag(self, metas, heatmap,
                   hx_soft, hy_soft, hx_hard, hy_hard,
                   x_soft, y_soft, x_hard, y_hard,
                   pkt, pred_3d, gt_3d, vis, per_sample):
        M = len(metas)
        records = []
        for i in range(M):
            hm = heatmap[i].detach()
            gt_kp = metas[i].get('joints_3d', None)   # GT pixel in 4K space

            records.append({
                # ── step 1: heatmap stats ──────────────────────────────
                'step1_heatmap': {
                    'shape': list(hm.shape),
                    'min':   float(hm.min()),
                    'max':   float(hm.max()),
                    'mean':  float(hm.mean()),
                },
                # ── step 2: soft-argmax vs hard argmax in heatmap space ─
                'step2_heatmap_coords': {
                    'soft_argmax_hx':  float(hx_soft[i]),
                    'soft_argmax_hy':  float(hy_soft[i]),
                    'hard_argmax_hx':  float(hx_hard[i]),
                    'hard_argmax_hy':  float(hy_hard[i]),
                    'diff_hx': float((hx_soft[i] - hx_hard[i]).abs()),
                    'diff_hy': float((hy_soft[i] - hy_hard[i]).abs()),
                },
                # ── step 3: pixel in 4K image space ───────────────────
                'step3_4k_pixel': {
                    'soft_x':  float(x_soft[i]),
                    'soft_y':  float(y_soft[i]),
                    'hard_x':  float(x_hard[i]),
                    'hard_y':  float(y_hard[i]),
                    'gt_kp_x': float(gt_kp[self.keypoint_index, 0]) if gt_kp is not None else None,
                    'gt_kp_y': float(gt_kp[self.keypoint_index, 1]) if gt_kp is not None else None,
                    'soft_vs_gt_x_err': float((x_soft[i] - gt_kp[self.keypoint_index, 0]).abs()) if gt_kp is not None else None,
                    'soft_vs_gt_y_err': float((y_soft[i] - gt_kp[self.keypoint_index, 1]).abs()) if gt_kp is not None else None,
                    'hard_vs_gt_x_err': float((x_hard[i] - gt_kp[self.keypoint_index, 0]).abs()) if gt_kp is not None else None,
                    'hard_vs_gt_y_err': float((y_hard[i] - gt_kp[self.keypoint_index, 1]).abs()) if gt_kp is not None else None,
                    'center':      [float(v) for v in metas[i]['center']],
                    'scale':       [float(v) for v in metas[i]['scale']],
                    'orig_center': [float(v) for v in metas[i]['orig_center']],
                    'orig_scale':  [float(v) for v in metas[i]['orig_scale']],
                },
                # ── step 4: sskit normalised coords ───────────────────
                'step4_normalised_pkt': {
                    'u': float(pkt[i, 0]),
                    'v': float(pkt[i, 1]),
                },
                # ── step 5: image_to_ground output ────────────────────
                'step5_world': {
                    'pred_x': float(pred_3d[i, 0]),
                    'pred_y': float(pred_3d[i, 1]),
                    'gt_x':   float(gt_3d[i, 0]),
                    'gt_y':   float(gt_3d[i, 1]),
                    'error_x_m': float((pred_3d[i, 0] - gt_3d[i, 0]).abs()),
                    'error_y_m': float((pred_3d[i, 1] - gt_3d[i, 1]).abs()),
                    'error_euc_m': float(
                        ((pred_3d[i] - gt_3d[i]) ** 2).sum().sqrt()),
                },
                # ── step 6: Huber contribution ────────────────────────
                'step6_huber': {
                    'per_sample_huber': float(per_sample[i]),
                    'vis':              float(vis[i]),
                    'contributes':      float(vis[i]) > 0,
                },
                # ── meta ──────────────────────────────────────────────
                'meta': {
                    'image_file': metas[i].get('image_file', ''),
                    'vis_from_joints_3d_visible': float(
                        metas[i]['joints_3d_visible'][self.keypoint_index, 0]),
                },
            })

        batch_record = {
            'batch_index': self._diag_count,
            'batch_size':  M,
            'samples':     records,
        }

        # append to JSON file (one record per batch)
        existing = []
        if os.path.exists(_DIAG_PATH):
            with open(_DIAG_PATH) as f:
                try:
                    existing = json.load(f)
                except Exception:
                    existing = []
        existing.append(batch_record)
        with open(_DIAG_PATH, 'w') as f:
            json.dump(existing, f, indent=2)
