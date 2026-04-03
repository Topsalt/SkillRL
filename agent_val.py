import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import cv2
import mmcv
import numpy as np # 仅用于少部分数据读取和scipy交互
from mmengine.config import Config
from mmdet.apis import init_detector
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from mmdet.structures import DetDataSample
import json

# ----------------------------
# 修复 PyTorch 2.6 weights_only 安全加载
# ----------------------------
import mmengine.runner.checkpoint as cp_mod
_original_torch_load = torch.load

def _load_checkpoint_override(filename, map_location=None, logger=None):
    return _original_torch_load(filename, map_location=map_location, weights_only=False)

cp_mod._load_checkpoint = _load_checkpoint_override

# ----------------------------
# GPU 匹配函数 (Code B - 宽松版)
# ----------------------------
def match_predictions_with_gt(all_cls_scores, all_mask_preds, query_feats, gt_masks_np, gt_labels_np, device, num_queries=50, conf_thresh=0.5,
                              cost_class_coeff=1, cost_mask_coeff=0, cost_dice_coeff=0):
    """
    主要逻辑保留在 GPU 上。
    注意：linear_sum_assignment 必须在 CPU 上运行，这是 SciPy 的限制。
    """
    # 取最后一层预测
    logits = all_cls_scores[-1].to(device).squeeze(0)
    mask_preds = all_mask_preds[-1].to(device)
    probs = torch.softmax(logits, dim=-1)[:, :-1]
    Q, num_classes = probs.shape

    # ==========================================
    # [核心修复] 获取特征并提取维度
    # 应对 query_feats = [4, 100, 256] 的情况
    # ==========================================
    if query_feats.dim() == 3:
        # 如果是 3 维，说明第 0 维是层数(Layers)或批次(Batch)
        # 我们直接取 [-1] 代表取最后一层(最成熟)的特征
        q_f = query_feats[-1].to(device)  # 变成 [100, 256]
    else:
        # 如果已经是 [100, 256]，直接用
        q_f = query_feats.to(device) 
        
    feat_dim = q_f.shape[-1]
    # ==========================================


    # 简单的 fallback 尺寸
    H, W = 1024, 1024 
    if (gt_masks_np is not None) and (len(gt_masks_np) != 0):
         # gt_masks_np 依然是 numpy，取 shape 即可
        H, W = int(gt_masks_np.shape[1]), int(gt_masks_np.shape[2])
    
    # Interpolate Masks (GPU)
    if mask_preds.dim() == 3:
        if (mask_preds.shape[1], mask_preds.shape[2]) != (H, W):
            mask_preds = F.interpolate(mask_preds.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
    elif mask_preds.dim() == 4:
        mask_preds = mask_preds.mean(dim=1) if mask_preds.shape[0] != 1 else mask_preds.squeeze(0)
        mask_preds = F.interpolate(mask_preds.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)

    # Filtering & Padding (GPU)
    if Q >= 100:
        conf_scores, _ = probs.max(dim=1)
        topk_scores, topk_indices = torch.topk(conf_scores, k=100, largest=True)
        probs_topk, masks_topk = probs[topk_indices], mask_preds[topk_indices]
        keep1 = topk_scores > conf_thresh
        probs_p, masks_p = probs_topk[keep1], masks_topk[keep1]
        q_k_topk = q_f[topk_indices]
        q_f_p = q_k_topk[keep1]
    else:
        probs_p, masks_p, q_f_p= probs, mask_preds, q_f

    # Pad to num_queries (GPU)
    curr_n = probs_p.shape[0]
    if curr_n < num_queries:
        pad_n = num_queries - curr_n
        probs_p = torch.cat([probs_p, -torch.ones(pad_n, num_classes, device=device)], dim=0)
        masks_p = torch.cat([masks_p, torch.zeros(pad_n, H, W, device=device)], dim=0)
        q_f_p = torch.cat([q_f_p, torch.zeros(pad_n, feat_dim, device=device)], dim=0)
    else:
        probs_p, masks_p, q_f_p = probs_p[:num_queries], masks_p[:num_queries], q_f_p[:num_queries]

    # 如果没有 GT，返回全 -1
    if (gt_labels_np is None) or (len(gt_labels_np) == 0):
        matched = torch.full((num_queries,), -1, dtype=torch.long, device=device)
        return matched, probs_p, q_f_p.detach() # 返回 Tensor

    # Hungarian Calculation
    # 将 GT 转为 Tensor (GPU)
    gt_labels = torch.from_numpy(gt_labels_np).to(device).long()
    
    # Cost Class Calculation (GPU)
    cost_class = -probs_p[:, gt_labels]
    
    gt_masks = torch.from_numpy(gt_masks_np).to(device).float()
    out_flat = masks_p.reshape(num_queries, -1)
    tgt_flat = gt_masks.reshape(gt_masks.shape[0], -1).float()
    
    pos = F.binary_cross_entropy_with_logits(out_flat, torch.ones_like(out_flat), reduction="none")
    neg = F.binary_cross_entropy_with_logits(out_flat, torch.zeros_like(out_flat), reduction="none")
    cost_mask = (pos @ tgt_flat.T + neg @ (1 - tgt_flat).T) / out_flat.shape[1]
    
    out_sigmoid = out_flat.sigmoid()
    cost_dice = 1 - (2 * (out_sigmoid @ tgt_flat.T) + 1) / (out_sigmoid.sum(-1)[:, None] + tgt_flat.sum(-1)[None, :] + 1)
    C = cost_class_coeff * cost_class + cost_mask_coeff * cost_mask + cost_dice_coeff * cost_dice


    # --- CPU Sync for Hungarian ---
    # SciPy 的 linear_sum_assignment 不支持 GPU，必须转 CPU
    #C_cpu = cost_class.detach().cpu().numpy()

    C_cpu = torch.nan_to_num(C, nan=1e10).detach().cpu().numpy()
    idx_q, idx_t = linear_sum_assignment(C_cpu)
    # ------------------------------

    matched = torch.full((num_queries,), -1, dtype=torch.long, device=device)
    # 将匹配结果写回 Tensor
    # idx_q, idx_t 是 numpy array，用来索引 tensor
    matched[idx_q] = gt_labels[idx_t]
        
    return matched, probs_p.detach(), q_f_p.detach() # 返回 Tensor

# ----------------------------
# COCO Dataset (返回 Tensor)
# ----------------------------
class CoCoDataset(Dataset):
    def __init__(self, img_root, ann_file,
                 transforms=None, model=None,
                 device='cpu', num_classes=0,
                 conf_thresh=0.5, max_preds=50):
        self.img_root = img_root
        self.coco = COCO(ann_file)
        self.img_ids = sorted(self.coco.imgs.keys())
        self.cat_ids = sorted(self.coco.cats.keys())
        self.cat_map = {cid: i for i, cid in enumerate(self.cat_ids)}
        self.transforms = transforms
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.max_preds = max_preds
        self.query_features = None

        def hook_fn(module, input, output):
            # mask2former 解码器的输出（喂给cls_embed的特征）
            self.query_features = input[0].detach().clone()
        if hasattr(self.model.panoptic_head, 'cls_embed'):
            self.model.panoptic_head.cls_embed.register_forward_hook(hook_fn)
        else:
                print("⚠️ Warning: cls_embed not found, query features will not be captured.")
                
    def _load_image(self, img_id):
        # 仅用于可视化或传统流程，本RL流程主要用 get_matched_from_mmdet
        info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_root, info['file_name'])
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_target(self, img_id):
        info = self.coco.loadImgs(img_id)[0]
        anns = self.coco.loadAnns(self.coco.getAnnIds(img_id))
        if len(anns) == 0:
            masks = np.zeros((0, info['height'], info['width']), dtype=np.uint8)
            labels = np.array([], dtype=np.int64)
        else:
            masks = np.stack([self.coco.annToMask(a) for a in anns], axis=0)
            labels = np.array([self.cat_map[a['category_id']] for a in anns], dtype=np.int64)
        
        # 这里为了兼容 match_predictions_with_gt，先返回 numpy
        # 后面会转 tensor
        return {'masks': masks, 'labels': labels}

    def get_matched_from_mmdet(self, img_id):
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_root, img_info['file_name'])
        info = self.coco.loadImgs(img_id)[0]
        img = mmcv.imread(img_path)
        img_tensor = torch.from_numpy(img.transpose(2,0,1)).float().unsqueeze(0).to(self.device)

        det_sample = DetDataSample()
        det_sample.set_metainfo({
            'img_id': img_id,
            'img_path': img_path,
            'ori_shape': (info['height'], info['width'], 3),
            'img_shape': (info['height'], info['width'], 3),
            'scale_factor': 1.0
        })

        data = dict(inputs=img_tensor, data_samples=[det_sample])
        data = self.model.data_preprocessor(data, training=False)
        feats = self.model.extract_feat(data['inputs'])
        all_cls_scores, all_mask_preds = self.model.panoptic_head(feats, data['data_samples'])

        q_feats  = self.query_features

        tgt = self._load_target(img_id)
        gt_masks_np = tgt['masks']
        gt_labels_np = tgt['labels']

        # 返回的是 GPU Tensors
        matched, probs, aligned_feats = match_predictions_with_gt(
            all_cls_scores, all_mask_preds, q_feats,
            gt_masks_np, gt_labels_np,
            device=self.device,
            num_queries=self.max_preds,conf_thresh=self.conf_thresh
        )
        return matched, probs, aligned_feats

    def __getitem__(self, idx):
        # 训练循环主要用 get_matched_from_mmdet
        # 这里返回 dummy 以符合 Dataset 接口
        img_id = self.img_ids[idx]
        return None, {'metainfo': {'img_id': img_id}}

    def __len__(self):
        return len(self.img_ids)


# ----------------------------
# 环境类 (全 GPU 实现)
# ----------------------------
class ChromosomeEnv:
    def __init__(self, probs, features, true_labels, device):
        """
        probs: Tensor [Q, C] (GPU)
        features: Tensor [Q, D] (GPU)
        true_labels: Tensor [Q] (GPU)
        """
        self.device = device
        self.probs = probs.clone() # Keep on GPU
        self.num_ch, self.num_cls = probs.shape
        self.features = features.clone() 
        
        # GPU max
        max_vals, max_indices = torch.max(self.probs, dim=1)
        # 判断 padding: 所有类分数都是 -1
        self.orig_padding_mask = (max_vals == -1)
        
        self.hard_lbl = torch.where(self.orig_padding_mask, -1, max_indices).long()
        self.true_lbl = true_labels.clone()
        
        self.correct_modifications = 0
        self.incorrect_modifications = 0
        self.action_count = 0
        
        self._update_state()
        #self._swap_pairs()

    def _update_state(self):
        # Flatten probabilities
        pv = self.probs.reshape(-1)
        h = self.hard_lbl.float()
        fv = self.features.reshape(-1)
        
        # 计算 counts (bincount)
        valid_mask = (self.hard_lbl >= 0)
        valid_lbls = self.hard_lbl[valid_mask]
        
        # bincount on GPU
        if valid_lbls.numel() > 0:
            cnt = torch.bincount(valid_lbls, minlength=self.num_cls).float()
        else:
            cnt = torch.zeros(self.num_cls, device=self.device).float()

        # 计算 vio (Vectorized)
        vio = torch.zeros(self.num_cls, device=self.device, dtype=torch.float32)
        
        # 规则 1: 前 22 对 (0-21) 必须是 2 个
        # 规则 2: 性染色体 (22, 23)
        #   (22=2, 23=0) -> Male (XY? No, usually XX is 2X, XY is 1X1Y. Let's follow your logic)
        #   Your logic: "normal" if (22==2 and 23==0) OR (22==1 and 23==1)
        #   Otherwise vio=1 if count != 0
        
        # 0-21号染色体
        idx_0_21 = torch.arange(22, device=self.device)
        vio[idx_0_21] = (cnt[idx_0_21] != 2).float()
        
        # 性染色体逻辑
        is_normal_sex = ((cnt[22] == 2) & (cnt[23] == 0)) | ((cnt[22] == 1) & (cnt[23] == 1))
        if not is_normal_sex:
            # 如果不正常，且数量不为0，则标记违规
            if cnt[22] != 0: vio[22] = 1.0
            if cnt[23] != 0: vio[23] = 1.0

        self.vio = vio.clone()
        self.cnt = cnt.clone()
        
        # Concat all features [probs, hard_lbl, counts, violations]
        self.state = torch.cat([pv, h, cnt, vio ,fv])
        #print(f"State updated: probs({pv.shape}), hard_lbl({h.shape}), counts({cnt.shape}), vio({vio.shape}), features({fv.shape})")

    def reset(self):
        max_vals, max_indices = torch.max(self.probs, dim=1)
        self.hard_lbl = torch.where(max_vals == -1, -1, max_indices).long()
        
        self.correct_modifications = 0
        self.incorrect_modifications = 0
        self.action_count = 0
        self._update_state()
        #self._swap_pairs()

        return self.state

    def step(self, action):
        typ = action[0]
        reward = 0.0
        done = False
        
        if typ == 'reassign':
            # ... (这部分保持原样，不需要动) ...
            _, idx, new_cls = action
            old = self.hard_lbl[idx].item()
            if old == -1 and new_cls != -1: reward = -5.0
            elif new_cls == -1: self.hard_lbl[idx] = -1
            else: self.hard_lbl[idx] = new_cls
            
            current_val = self.hard_lbl[idx].item()
            true_val = self.true_lbl[idx].item()
            
            if old == current_val: reward = -1.0 
            elif current_val == true_val:
                reward = 10.0 # 提高改对的奖励
                self.correct_modifications += 1
            else:
                reward = -2.0
                self.incorrect_modifications += 1
            self.action_count += 1
            
        elif typ == 'submit':
            # =======================
            # === 核心修改在这里 ===
            # =======================
            
            # 检查结构完整性
            valid_mask = (self.hard_lbl >= 0)
            counts = torch.bincount(self.hard_lbl[valid_mask], minlength=self.num_cls)
            
            # 只检查常染色体 0-21 是否都是 2 个
            autosomes = counts[:22]
            structure_is_perfect = (autosomes == 2).all().item()
            
            if not structure_is_perfect:
                # 结构不对，禁止提交！
                done = False 
                reward = -50.0 # 重罚，告诉它：“没做完不许交卷”
            else:
                # 结构正确，允许提交，结算最终分数
                done = True
                reward = compute_reward_gpu(
                    self.hard_lbl, self.true_lbl,
                    self.action_count, self.correct_modifications, self.device
                )
            
        self._update_state()
        return self.state, reward, done
    #def _swap_pairs(self):
        for i in range(len(self.hard_lbl)):
            lbl = self.hard_lbl[i]
            if lbl == 5: self.hard_lbl[i] = 6
            elif lbl == 6: self.hard_lbl[i] = 5
            elif lbl == 13: self.hard_lbl[i] = 14
            elif lbl == 14: self.hard_lbl[i] = 13

# ----------------------------
# Reward 计算 (全 GPU)
# ----------------------------
def compute_reward_gpu(hard_lbl, true_lbl, action_count, correct_modifications, device):
    """
    修改版奖励函数：
    1. GT 匹配奖励：跟标准答案一致给高分。
    2. 基础结构惩罚：
       - 0-20号：必须为 2 条。
       - 21号：可以是 2 条 或 3 条 (兼容唐氏综合征但不额外奖励)。
    """
    hard_valid = (hard_lbl >= 0)
    true_valid = (true_lbl >= 0)
    
    # -----------------------------------------------------------
    # 1. 基础匹配奖励 (Base Reward) - 依赖 Ground Truth
    # -----------------------------------------------------------
    both_mask = hard_valid & true_valid
    
    if not both_mask.any():
        base_reward = -100.0
    else:
        valid_hard = hard_lbl[both_mask]
        valid_true = true_lbl[both_mask]
        
        errors = (valid_hard != valid_true).sum().item()
        
        if errors == 0:
            base_reward = 100.0 
        else:
            base_reward = -5.0 * errors 

    # -----------------------------------------------------------
    # 2. 结构性惩罚 (Structure Penalty)
    # -----------------------------------------------------------
    # 假设类别 0-23 (共24类)
    counts = torch.bincount(hard_lbl[hard_valid], minlength=24)
    struct_penalty = 0.0
    
    # --- [修改点开始] ---
    
    # A. 检查 0-20 号染色体 (严格 2n)
    # counts[:21] 包含索引 0 到 20
    cnt_0_20 = counts[:21]
    bad_0_20 = (cnt_0_20 != 2).sum().item()
    struct_penalty -= (50.0 * bad_0_20)
    
    # B. 检查 21 号染色体 (2 或 3 均可)
    cnt_21 = counts[21]
    # 逻辑：如果既不是2，也不是3，才算错
    if (cnt_21 != 2) and (cnt_21 != 3):
        struct_penalty -= 50.0
    
    # --- [修改点结束] ---
    
    # C. 缺失严重惩罚 (Count == 0)
    # 检查 0-21 号，如果有任何一个完全缺失(0条)，额外重罚
    # (注：如果21号是0条，上面B步骤罚了一次，这里会再罚一次，这是合理的，因为缺失比数量不对更严重)
    zeros_num = (counts[:22] == 0).sum().item()
    struct_penalty -= (50.0 * zeros_num)

    # -----------------------------------------------------------
    # 3. 动作与修正奖励
    # -----------------------------------------------------------
    action_penalty = -0.5 * action_count 
    bonus = 50.0 * correct_modifications

    total = base_reward + struct_penalty + action_penalty + bonus
    return total

# ----------------------------
# DQN 网络
# ----------------------------
class DQN(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.feature_dim = 12800              
        self.rule_dim = s_dim - self.feature_dim   # 动态算出逻辑规则维度 (1298)

        # =======================================
        # 1. 视觉流 (Feature Stream) - 六层究极平滑漏斗
        # 12800 -> 8192 -> 4096 -> 2048 -> 1024 -> 512 -> 256
        # =======================================
        # 第 1 层：12800 -> 8192 (超大参数层，加强 Dropout)
        self.feat_fc1 = nn.Linear(self.feature_dim, 8192)
        self.feat_ln1 = nn.LayerNorm(8192)
        self.drop_feat1 = nn.Dropout(p=0.4) 
        
        # 第 2 层：8192 -> 4096
        self.feat_fc2 = nn.Linear(8192, 4096)
        self.feat_ln2 = nn.LayerNorm(4096)
        self.drop_feat2 = nn.Dropout(p=0.4)
        
        # 第 3 层：4096 -> 2048
        self.feat_fc3 = nn.Linear(4096, 2048)
        self.feat_ln3 = nn.LayerNorm(2048)
        self.drop_feat3 = nn.Dropout(p=0.3)
        
        # 第 4 层：2048 -> 1024
        self.feat_fc4 = nn.Linear(2048, 1024)
        self.feat_ln4 = nn.LayerNorm(1024)
        self.drop_feat4 = nn.Dropout(p=0.3)
        
        # 第 5 层：1024 -> 512
        self.feat_fc5 = nn.Linear(1024, 512)
        self.feat_ln5 = nn.LayerNorm(512)
        self.drop_feat5 = nn.Dropout(p=0.2)
        
        # 第 6 层：512 -> 256 (提取出终极视觉特征)
        self.feat_fc6 = nn.Linear(512, 256)
        self.feat_ln6 = nn.LayerNorm(256)
        self.drop_feat6 = nn.Dropout(p=0.2)

        # =======================================
        # 2. 逻辑规则解析流 (Rule Stream)
        # =======================================
        self.rule_fc1 = nn.Linear(self.rule_dim, 512)
        self.rule_ln1 = nn.LayerNorm(512)
        self.rule_fc2 = nn.Linear(512, 256)
        self.rule_ln2 = nn.LayerNorm(256)

        # =======================================
        # 3. 决策融合流 (Fusion Stream)
        # 拼接后的输入为: 256(视觉) + 256(逻辑) = 512 维
        # =======================================
        self.fusion_fc1 = nn.Linear(512, 256)
        self.fusion_ln1 = nn.LayerNorm(256)
        self.drop_fusion = nn.Dropout(p=0.2)
        
        self.fc_out = nn.Linear(256, a_dim)

    def forward(self, x):
        # 1. 状态精确切片分流
        rules = x[:, :self.rule_dim]      # 前部：逻辑规则特征
        feats = x[:, self.rule_dim:]      # 后部：Mask2Former 图像特征

        # 2. 视觉流前向传播 (跑完 6 层漏斗)
        f = F.relu(self.feat_ln1(self.feat_fc1(feats)))
        f = self.drop_feat1(f)
        f = F.relu(self.feat_ln2(self.feat_fc2(f)))
        f = self.drop_feat2(f)
        f = F.relu(self.feat_ln3(self.feat_fc3(f)))
        f = self.drop_feat3(f)
        f = F.relu(self.feat_ln4(self.feat_fc4(f)))
        f = self.drop_feat4(f)
        f = F.relu(self.feat_ln5(self.feat_fc5(f)))
        f = self.drop_feat5(f)
        f = F.relu(self.feat_ln6(self.feat_fc6(f)))
        f = self.drop_feat6(f)

        # 3. 逻辑流前向传播
        r = F.relu(self.rule_ln1(self.rule_fc1(rules)))
        r = F.relu(self.rule_ln2(self.rule_fc2(r)))

        # 4. 融合与输出
        out = torch.cat([r, f], dim=1) # 合并两股力量
        out = F.relu(self.fusion_ln1(self.fusion_fc1(out)))
        out = self.drop_fusion(out)
        
        return self.fc_out(out)

# ----------------------------
# 经验回放 (直接存 Tensor)
# ----------------------------
class ReplayMemory:
    def __init__(self, cap):
        self.mem = deque(maxlen=cap)

    def push(self, state, action_global_idx, reward, next_state, done):
        # state: Tensor
        # action: int
        # reward: float
        # next_state: Tensor
        # done: bool
        self.mem.append((state, action_global_idx, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.mem, batch_size)
        return zip(*batch)

    def __len__(self):
        return len(self.mem)

# ----------------------------
# 全局动作构建
# ----------------------------
def build_global_actions(num_chrom, num_cls):
    actions = []
    PAD_IDX = -1 
    for pos in range(num_chrom):
        for cls in range(num_cls):
            actions.append(('reassign', pos, cls))
        actions.append(('reassign', pos, PAD_IDX))
    actions.append(('submit',))
    return actions


def build_action_space_global(env: "ChromosomeEnv", global_actions: list, top_k=None, per_class_k=None):
    """
    策略：违规驱动 + 过剩驱动 (无 Prototype)
    在原有违规类基础上，额外加入过剩类（cnt > 2）的位置，
    支持"先移走过剩染色体再修正其他位置"的间接操作。
    使用 Tensor 操作来判断。
    """
    num_chrom = env.num_ch
    num_cls = env.num_cls
    
    # violated_classes: vio != 0 的类（数量不符合预期，含缺失和过剩）
    violated_classes = set(torch.nonzero(env.vio).view(-1).tolist())

    # surplus_classes: 数量超出正常预期（常染色体标准 2 条）的类
    # 显式加入过剩类，确保 agent 可以把"多余的"染色体先移走
    surplus_classes = set(torch.nonzero(env.cnt > 2).view(-1).tolist())

    # 合并：允许操作违规类或过剩类的位置
    action_classes = violated_classes | surplus_classes

    if len(action_classes) == 0:
        return [len(global_actions) - 1] # submit index

    allowed = []
    
    # 遍历 (可以优化为 Tensor 操作，但为了逻辑清晰保留循环，此处不是瓶颈)
    # env.probs 是 Tensor
    # env.hard_lbl 是 Tensor
    
    # 找到非 padding 的索引
    valid_pos = torch.nonzero(env.probs.max(dim=1).values != -1).squeeze(1)
    
    for pos in valid_pos:
        pos = pos.item()
        current_label = env.hard_lbl[pos].item()
        
        if current_label in action_classes:
            start_idx = pos * (num_cls + 1)
            end_idx = start_idx + (num_cls + 1)
            allowed.extend(range(start_idx, end_idx))

    submit_idx = len(global_actions) - 1
    allowed.append(submit_idx)

    return sorted(allowed)

# ----------------------------
# Pad Helper (Tensor 版本)
# ----------------------------
def pad_state_to_dim(state: torch.Tensor, target_dim: int):
    # state 是一维 Tensor
    curr_len = state.shape[0]
    if curr_len == target_dim:
        return state
    
    if curr_len > target_dim:
        return state[:target_dim]
    
    # Pad with zeros
    padding = torch.zeros(target_dim - curr_len, device=state.device, dtype=state.dtype)
    return torch.cat([state, padding])

# ----------------------------
# 选择动作
# ----------------------------
def select_action_global(state: torch.Tensor,
                         policy_net: torch.nn.Module,
                         allowed_indices: list,
                         global_actions: list,
                         epsilon: float,
                         device: torch.device,
                         target_state_dim: int):
    # state 已经在 GPU 上
    state_p = pad_state_to_dim(state, target_state_dim)
    
    if random.random() < epsilon:
        gidx = random.choice(allowed_indices)
        return gidx, global_actions[gidx]

    # 添加 batch 维度 [1, dim]
    state_t = state_p.unsqueeze(0) 
    
    with torch.no_grad():
        q_vals = policy_net(state_t)[0]
        allowed_tensor = torch.tensor(allowed_indices, dtype=torch.long, device=device)
        q_allowed = q_vals[allowed_tensor]
        best_pos = int(q_allowed.argmax().item())
        gidx = int(allowed_indices[best_pos])
        
    return gidx, global_actions[gidx]

# ----------------------------
# 训练主流程
# ----------------------------
def train_dqn(config_file, checkpoint, data_root, total_episodes, debug=False, pretrained_path=None):
    BATCH_SIZE = 512
    GAMMA = 0.99
    LR = 1e-4
    MEM_CAPACITY = 20000
    EPS_START = 1.0
    EPS_DECAY = 0.99999
    DECAY_INTERVAL = 1 
    MAX_EPISODE_STEPS = 100             #100步
    VALIDATION_FREQ = 500
    TARGET_SYNC_FREQ = 100 
    TOP_K = 3
    PER_CLASS_K = 2
    UPSTEP = 1                          #old 10       之前是每一步更新一次，改为每5步更新一次 再试试其他的
    PRE = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    cfg = Config.fromfile(config_file)
    num_classes = len(cfg.metainfo['classes'])
    detector = init_detector(config_file, checkpoint, device=device)
    detector.eval()

    tr_img_dir = os.path.join(data_root, 'blance')
    tr_ann_file = os.path.join(data_root, 'annotations', 'train_829.json')

    vl_img_dir = os.path.join(data_root, 'val_439')
    vl_ann_file = os.path.join(data_root, 'annotations', 'val_439.json')

    train_dataset = CoCoDataset(
        tr_img_dir, tr_ann_file,
        model=detector, device=device,
        num_classes=num_classes,
        conf_thresh=0.5, max_preds=50
    )
    val_dataset = CoCoDataset(
        vl_img_dir, vl_ann_file,
        model=detector, device=device,
        num_classes=num_classes,
        conf_thresh=0.5, max_preds=50
    )

    memory = ReplayMemory(MEM_CAPACITY)
    policy_net = None
    target_net = None
    optimizer = None
    lr_scheduler = None
    epsilon = EPS_START
    best_val_acc = 0.0
    loss=0

    GLOBAL_STATE_DIM = None
    global_actions = None

    # ==========================================
    # 用来存每一轮的训练 Loss
    train_history = [] 
    # 用来存每 500 轮的验证 Acc
    val_history = []
    # ==========================================


    for ep in range(total_episodes):
        idx = random.randrange(len(train_dataset))
        _, meta = train_dataset[idx] # 直接获取 meta
        img_id = meta['metainfo']['img_id']
        
        # 返回 Tensors (GPU)
        true_labels, probs, feats = train_dataset.get_matched_from_mmdet(img_id)

        # Env 全 GPU
        env = ChromosomeEnv(probs, feats, true_labels, device=device)
        
        if policy_net is None:
            # 第一次初始化
            GLOBAL_STATE_DIM = env.state.shape[0]
            global_actions = build_global_actions(env.num_ch, env.num_cls)
            ACTION_DIM = len(global_actions)
            policy_net = DQN(GLOBAL_STATE_DIM, ACTION_DIM).to(device)
            target_net = DQN(GLOBAL_STATE_DIM, ACTION_DIM).to(device)
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()
            optimizer = optim.Adam(policy_net.parameters(), lr=LR, weight_decay=1e-4)
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                               factor=0.5, patience=5)
            
            if debug:
                print(f"[INIT] GLOBAL_STATE_DIM={GLOBAL_STATE_DIM}, ACTION_DIM={ACTION_DIM}")

        if pretrained_path and os.path.exists(pretrained_path) and PRE:
            print(f"检测到 checkpoint：{pretrained_path}，正在加载...")
            checkpoint_data = torch.load(pretrained_path, map_location=device, weights_only=False)
            policy_net.load_state_dict(checkpoint_data["policy_state_dict"])
            target_net.load_state_dict(checkpoint_data["target_state_dict"])
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            epsilon = checkpoint_data.get("epsilon", EPS_START)
            ep = checkpoint_data.get("epoch", 0) + 1
            best_val_acc = checkpoint_data.get("best_val_acc", 0.0)
            GLOBAL_STATE_DIM = checkpoint_data.get("GLOBAL_STATE_DIM", GLOBAL_STATE_DIM)
            ACTION_DIM = checkpoint_data.get("ACTION_DIM", len(global_actions))
            print(f"checkpoint 加载完成，从第 {ep} 轮继续训练")
            PRE = False
        elif PRE:
            PRE = False
            print("未加载预训练模型从头开始")

        state = env.reset() # Tensor
        state = pad_state_to_dim(state, GLOBAL_STATE_DIM) # Tensor


        current_ep_loss = 0.0  #记录每回合loss
        update_count = 0
        current_ep_reward = 0.0 #记录每回合reward
        
        #total_reward = 0.0
        only_submit_so_far = True

        for step_count in range(MAX_EPISODE_STEPS):
            allowed_idxs = build_action_space_global(
                env, global_actions,
                top_k=TOP_K, per_class_k=PER_CLASS_K
            )

            gidx, action = select_action_global(
                state, policy_net, allowed_idxs, global_actions,
                epsilon, device, target_state_dim=GLOBAL_STATE_DIM
            )

            next_state, reward, done = env.step(action)
            # next_state 是 Tensor, reward 是 float, done 是 bool
            
            next_state_padded = pad_state_to_dim(next_state, GLOBAL_STATE_DIM)

            if action[0] != 'submit':
                only_submit_so_far = False
            if action[0] == 'submit' and only_submit_so_far and reward != 100:
                reward -= 100
                done = False
            
            reward = np.clip( reward / 100.0, -1.0, 1.0) # float

            # ==========================================
            # [新增] 累加奖励
            # ==========================================
            current_ep_reward += reward 
            # ==========================================


            memory.push(state, gidx, reward, next_state_padded, done)

            state = next_state_padded
            
            # 训练步
            if len(memory) >= BATCH_SIZE and ep % UPSTEP ==0:
                states_b, acts_b, rews_b, next_states_b, dones_b = memory.sample(BATCH_SIZE)

                # states_b 是 list of Tensors (GPU), 直接 stack
                S = torch.stack(states_b) 
                NS = torch.stack(next_states_b)
                
                # 其他转 Tensor
                A = torch.tensor(list(acts_b), device=device, dtype=torch.long).unsqueeze(1)
                R = torch.tensor(list(rews_b), device=device, dtype=torch.float).unsqueeze(1)
                D = torch.tensor(list(dones_b), device=device, dtype=torch.float).unsqueeze(1)

                if torch.isnan(S).any() or torch.isinf(S).any():
                    print("⚠️ NaN detected in states!")

                q_vals = policy_net(S).gather(1, A)
                with torch.no_grad():
                    next_actions = policy_net(NS).argmax(1, keepdim=True)
                    next_q = target_net(NS).gather(1, next_actions)
                target_q = R + GAMMA * next_q * (1 - D)

                loss = nn.MSELoss()(q_vals, target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
                optimizer.step()

                # 累加 Loss 以计算本轮平均值 (或者你也可以只记最后一步)
                current_ep_loss += loss.item()
                update_count += 1

            if done:
                break   

        # ==========================================
        # [修改] 记录每一轮的 Loss 和 Reward
        # ==========================================
        # 只有当本轮有训练发生时才记录 Loss，但 Reward 每轮都有
        avg_loss = current_ep_loss / update_count if update_count > 0 else 0.0
        
        train_history.append({
            'episode': ep,
            'loss': avg_loss,
            'reward': current_ep_reward,  # <--- 新增：保存本轮总奖励
            'epsilon': epsilon
        })


        if (ep + 1) % DECAY_INTERVAL == 0: 
            epsilon *= EPS_DECAY
            epsilon = max(epsilon, 0.1)
        if ep % TARGET_SYNC_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # ----------------------------
        # 验证 (全 GPU)
        # ----------------------------
        if ep % VALIDATION_FREQ == 0:
            policy_net.eval()
            correct, total_v = 0, 0
            first_validation = (ep == 0)
            submit_idx = len(global_actions) - 1

            for vid in range(min(500, len(val_dataset))):
                _, v_meta = val_dataset[vid]
                v_true, v_probs ,v_feats = val_dataset.get_matched_from_mmdet(v_meta['metainfo']['img_id'])
                
                v_env = ChromosomeEnv(v_probs, v_feats, v_true, device=device)
                v_state = v_env.reset()
                v_state = pad_state_to_dim(v_state, GLOBAL_STATE_DIM)

                for _ in range(MAX_EPISODE_STEPS):
                    if first_validation:
                        gidx = submit_idx
                        action = global_actions[gidx]
                    else:
                        allowed_idxs = build_action_space_global(
                            v_env, global_actions,
                            top_k=TOP_K, per_class_k=PER_CLASS_K
                        )

                        with torch.no_grad():
                            qv = policy_net(v_state.unsqueeze(0))[0]
                        
                        allowed_tensor = torch.tensor(allowed_idxs, dtype=torch.long, device=device)
                        q_allowed = qv[allowed_tensor]
                        best_pos = int(q_allowed.argmax().item())
                        gidx = int(allowed_idxs[best_pos])
                        action = global_actions[gidx]

                    v_state, vr, vd = v_env.step(action)
                    v_state = pad_state_to_dim(v_state, GLOBAL_STATE_DIM)
                    if vd:
                        break
                
                # Check correctness (Tensor compare)
                if torch.equal(v_env.hard_lbl, v_true):
                    correct += 1
                total_v += 1
                
                print("#############################")
                print(v_env.hard_lbl.tolist())
                print(v_true.tolist())
                print("#############################")
                print("correct",correct)
                print("total_v",total_v)

            val_acc = correct / total_v if total_v > 0 else 0.0
            print(f"Ep {ep}: TrainR={reward:.2f} ValAcc={val_acc:.4f} Eps={epsilon:.4f} Loss={loss:.4f}")
            
            # ==========================================
            # [修改] 3. 记录验证 Acc 到独立列表
            # ==========================================
            val_history.append({
                'episode': ep,
                'val_acc': val_acc
            })


            # ==========================================
            # [修改] 4. 保存到文件 (结构化保存)
            # ==========================================
            log_data = {
                "train_history": train_history, 
                "val_history": val_history      
            }
            
            log_save_path = os.path.join(data_root, '2training_log_reward5update1.json')
            try:
                with open(log_save_path, 'w') as f:
                    json.dump(log_data, f, indent=4)
            except Exception as e:
                print(f"Error saving log: {e}")

            lr_scheduler.step(val_acc)
            best_val_acc = max(best_val_acc, val_acc)
            policy_net.train()
        else:
            print(f"Ep {ep}: TrainR={reward:.2f} Eps={epsilon:.4f} Loss={loss:.4f}")
        """
        # 可保存模型
        if best_val_acc <= val_acc:
            best_val_acc=val_acc
            torch.save(policy_net.state_dict(), os.path.join(data_root, 'best_g2.pth'))
            torch.save({
                "epoch": ep,
                "policy_state_dict": policy_net.state_dict(),
                "target_state_dict": target_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "GLOBAL_STATE_DIM": GLOBAL_STATE_DIM,
                "ACTION_DIM": len(global_actions) if global_actions else 0,
                "best_val_acc": best_val_acc,
                "epsilon": epsilon,
            }, os.path.join(data_root, 'best_checkpoint_g2.pth'))"""
    print(f"Training finished. Best ValAcc={best_val_acc:.4f}")
    """
    torch.save(policy_net.state_dict(), os.path.join(data_root, 'long_g2.pth'))
    torch.save({
        "ACTION_DIM": len(global_actions) if global_actions else 0,
        "best_val_acc": best_val_acc,
        "epsilon": epsilon,
    }, os.path.join(data_root, 'long_checkpoint_g2.pth'))"""
    
# ----------------------------
# 入口
# ----------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='染色体强化学习（全GPU优化版）')
    parser.add_argument('--config', default='/home/cuit/dhy/6k_finetune/cnsn_resnet50_mcls_6k.py')
    parser.add_argument('--checkpoint', default='/home/cuit/dhy/6k_finetune/epoch_50.pth')
    parser.add_argument('--data-root', default='/home/cuit/dhy/6k_finetune/dj_prepare')
    parser.add_argument('--episodes', type=int, default=150000)
    parser.add_argument('--pretrained', type=str, default='none', help='预训练模型路径')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    train_dqn(args.config, args.checkpoint, args.data_root, args.episodes, debug=args.debug,pretrained_path=args.pretrained)