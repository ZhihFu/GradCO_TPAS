import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from models.xvlm import XVLMBase
from models.pose import Block, ConvExpandReduce
from copy import deepcopy
from torch.distributions import Normal, kl_divergence
import math

class GradCO(XVLMBase):
    def __init__(self, config, num_classes=None):
        super().__init__(config, num_classes=num_classes, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=config['mlm']['is_mlm'],
                         use_bbox_loss=False)
        self.init_params = []

        self.be_hard = config.get('be_hard', False)
        self.be_pose_img = config.get('be_pose_img', False)
        self.be_pose_conv = config.get('pose_conv', False)
        if self.be_pose_img:
            self.pose_block = Block()
            self.init_params.extend(['pose_block.' + n for n, _ in self.pose_block.named_parameters()])
            if self.be_pose_conv:
                self.pose_conv = ConvExpandReduce()
                self.init_params.extend(['pose_conv.' + n for n, _ in self.pose_conv.named_parameters()])
        
        vision_model_dim = config['vision_width']
        text_model_dim = config['text_width']
        embed_dim = config['embed_dim'] 
        num_sub_experts = config['num_sub_experts']
        
        self.bmoe_img = BehaviorAwareMoE(
            model_dim=vision_model_dim,
            embed_dim=embed_dim,
            hidden_dim=config.get('moe_hidden_dim', 512),
            num_sub_experts=num_sub_experts,
            modality='vision'
        )
        self.bmoe_txt = BehaviorAwareMoE(
            model_dim=text_model_dim,
            embed_dim=embed_dim,
            hidden_dim=config.get('moe_hidden_dim', 512),
            num_sub_experts=num_sub_experts
        )

        self.col = ConflictOffloadLearning(
            embed_dim=embed_dim,
            config=config, 
            dropout=0.1
        )
       
        self.hia = HierarchicalIntentAlignment(
            embed_dim=embed_dim,
            config=config,
            hidden_dim=config.get('halign_hidden_dim', embed_dim * 2)
        )
        
        self.init_params.extend(['bmoe_img.' + n for n, _ in self.bmoe_img.named_parameters()])
        self.init_params.extend(['bmoe_txt.' + n for n, _ in self.bmoe_txt.named_parameters()])
        self.init_params.extend(['col.' + n for n, _ in self.col.named_parameters()])
        self.init_params.extend(['hia.' + n for n, _ in self.hia.named_parameters()])

    def forward(self, image, text_ids, text_atts, text_ids_masked=None, masked_pos=None, masked_ids=None,
                idx=None, text_ids_eda=None, text_atts_eda=None,
                pose=None, hard_i=None, hard_i_pose=None, hard_text_ids=None, hard_text_atts=None,
                ):

        image_embeds, image_atts = self.get_vision_embeds(image)
        text_embeds = self.get_text_embeds(text_ids, text_atts)

        if self.be_pose_img:
            if self.be_pose_conv:
                pose = self.pose_conv(pose)
            pose, _ = self.get_vision_embeds(pose)
            image_embeds = self.pose_block(image_embeds, pose)

        image_feat = self.get_image_feat(image_embeds)
        text_feat = self.get_text_feat(text_embeds)

        (img_embeds_app, img_embeds_norm, img_embeds_anom), \
        (img_feat_app, img_feat_norm, img_feat_anom), \
        aux_loss_img = self.bmoe_img(image_embeds)
        
        (txt_embeds_app, txt_embeds_norm, txt_embeds_anom), \
        (txt_feat_app, txt_feat_norm, txt_feat_anom), \
        aux_loss_txt = self.bmoe_txt(text_embeds)
        

        B, L_img, D_model = img_embeds_app.shape
        img_embeds_app_flat = img_embeds_app.view(-1, D_model)
        img_embeds_norm_flat = img_embeds_norm.view(-1, D_model)
        img_embeds_anom_flat = img_embeds_anom.view(-1, D_model)
        
        img_ctx_app_flat = self.bmoe_img.proj_vision_app(img_embeds_app_flat)
        img_ctx_norm_flat = self.bmoe_img.proj_vision_norm(img_embeds_norm_flat)
        img_ctx_anom_flat = self.bmoe_img.proj_vision_anom(img_embeds_anom_flat)
        
        img_embeds_app_proj = img_ctx_app_flat.view(B, L_img, -1)
        img_embeds_norm_proj = img_ctx_norm_flat.view(B, L_img, -1)
        img_embeds_anom_proj = img_ctx_anom_flat.view(B, L_img, -1)
        
        img_feat_app_all = img_feat_app
        img_feat_norm_all = img_feat_norm
        img_feat_anom_all = img_feat_anom
        
        txt_feat_app_all = txt_feat_app
        txt_feat_norm_all = txt_feat_norm
        txt_feat_anom_all = txt_feat_anom
        
        img_embeds_app_all = img_embeds_app_proj
        img_embeds_norm_all = img_embeds_norm_proj
        img_embeds_anom_all = img_embeds_anom_proj

        (s_app_all, s_norm_all, s_anom_all) = self.col(
            img_feat_app_all, img_feat_norm_all, img_feat_anom_all,
            txt_feat_app_all, txt_feat_norm_all, txt_feat_anom_all,
            img_embeds_app_all, img_embeds_norm_all, img_embeds_anom_all,
            is_training=self.training 
        )

        loss_con, s_ij_all,\
        text_feat, image_feat, \
        text_embeds_final, image_embeds_final = self.hia(
            s_app_all, s_norm_all, s_anom_all,
            img_feat_app, img_feat_norm, img_feat_anom,    
            txt_feat_app, txt_feat_norm, txt_feat_anom,    
            img_embeds_app, img_embeds_norm, img_embeds_anom, 
            txt_embeds_app, txt_embeds_norm, txt_embeds_anom, 
            idx
        )

        loss_itc = loss_con
        loss_itm = self.get_matching_loss(
            image_embeds, image_atts, image_feat,    
            text_embeds, text_atts, text_feat,      
            idx=idx,
            sim_matrix_l2=s_ij_all
        )
        
        loss_col = self.col.total_reg_loss
        
        text_embeds_eda = self.get_text_embeds(text_ids_eda, text_atts_eda)
        
        (txt_embeds_app_eda, txt_embeds_norm_eda, txt_embeds_anom_eda), \
        (txt_feat_app_eda, txt_feat_norm_eda, txt_feat_anom_eda), \
        _ = self.bmoe_txt(text_embeds_eda)

        (s_app_eda, s_norm_eda, s_anom_eda) = self.col(
            img_feat_app_all, img_feat_norm_all, img_feat_anom_all, 
            txt_feat_app_eda, txt_feat_norm_eda, txt_feat_anom_eda, 
            img_embeds_app_all, img_embeds_norm_all, img_embeds_anom_all,
            is_training=self.training 
        )
        
        loss_col = loss_col + self.col.total_reg_loss 

        txt_feats_concat_eda = torch.cat([txt_feat_app_eda, txt_feat_norm_eda, txt_feat_anom_eda], dim=-1)
        gate_logits_txt_eda = self.hia.gating_net(txt_feats_concat_eda)
        weights_txt_eda = F.softmax(gate_logits_txt_eda, dim=-1)

        w_app_txt_eda = weights_txt_eda[:, 0:1]
        w_norm_txt_eda = weights_txt_eda[:, 1:2]
        w_anom_txt_eda = weights_txt_eda[:, 2:3]
        
        s_ij_all_eda = (w_app_txt_eda * s_app_eda + 
                        w_norm_txt_eda * s_norm_eda + 
                        w_anom_txt_eda * s_anom_eda)

        logits_eda = s_ij_all_eda / self.hia.temp
        
        idx = idx.view(-1, 1)
        idx_all = idx
        pos_idx = torch.eq(idx_all, idx_all.t()).float()
        labels = pos_idx / pos_idx.sum(1, keepdim=True)
        
        if 0 < self.hia.epsilon < 1:
            _, num_classes = logits_eda.shape
            labels = (1 - self.hia.epsilon) * labels + self.hia.epsilon / num_classes
        elif self.hia.epsilon == 1: 
            labels = torch.ones_like(logits_eda) / logits_eda.shape[1]

        loss_i2t_eda = -torch.sum(F.log_softmax(logits_eda, dim=1) * labels, dim=1).mean()
        loss_t2i_eda = -torch.sum(F.log_softmax(logits_eda.t(), dim=1) * labels, dim=1).mean()
        loss_itc_eda = (loss_i2t_eda + loss_t2i_eda) / 2
        
        text_feat_eda_final = (w_app_txt_eda * txt_feat_app_eda + 
                               w_norm_txt_eda * txt_feat_norm_eda + 
                               w_anom_txt_eda * txt_feat_anom_eda)
                       
        loss_itm_eda = self.get_matching_loss(
            image_embeds, image_atts, image_feat, 
            text_embeds_eda, text_atts_eda, text_feat_eda_final, 
            idx=idx, 
        )

        loss_itc = loss_itc + 0.8 * loss_itc_eda
        loss_itm = loss_itm + 0.8 * loss_itm_eda
        
        if self.be_hard:
            image_embeds_hard_orig, image_atts_hard = self.get_vision_embeds(hard_i)
            text_embeds_hard_orig = self.get_text_embeds(hard_text_ids, hard_text_atts)

            if self.be_pose_img:
                if self.be_pose_conv:
                    hard_i_pose = self.pose_conv(hard_i_pose)
                hard_pose, _ = self.get_vision_embeds(hard_i_pose)
                image_embeds_hard_orig = self.pose_block(image_embeds_hard_orig, hard_pose)

            (img_embeds_app_h, img_embeds_norm_h, img_embeds_anom_h), \
            (img_feat_app_h, img_feat_norm_h, img_feat_anom_h), \
            _ = self.bmoe_img(image_embeds_hard_orig)
            
            (txt_embeds_app_h, txt_embeds_norm_h, txt_embeds_anom_h), \
            (txt_feat_app_h, txt_feat_norm_h, txt_feat_anom_h), \
            _ = self.bmoe_txt(text_embeds_hard_orig)

            loss_itm_hard = self.get_matching_loss_hard(
                image_embeds, image_atts, 
                image_embeds_hard_orig, image_atts_hard,
                text_embeds, text_atts, 
                text_embeds_hard_orig, hard_text_atts 
            )
            loss_itm = loss_itm + loss_itm_hard
            
        loss_mlm = self.get_mlm_loss(text_ids_masked, text_atts, image_embeds, image_atts,
                                     masked_pos, masked_ids)
        loss_mlm = loss_mlm + loss_col

        return loss_itc, loss_itm, loss_mlm


class BehaviorAwareExpert(nn.Module):
    def __init__(self, in_features, hidden_dim=512, scale=1.0):
        super().__init__()
        self.conv_A = nn.Conv1d(in_features, hidden_dim, 1, groups=1, bias=True)
        self.conv_B = nn.Conv1d(hidden_dim, in_features, 1, groups=1, bias=True)
        self.dropout = nn.Dropout(0.1)

        nn.init.xavier_uniform_(self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.xavier_uniform_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)
        self.scale = scale

    def forward(self, x):
        x_trans = x.transpose(1, 2) 
        residual = x_trans
        result = self.conv_A(x_trans)
        result = self.dropout(result)
        result = self.conv_B(result)
        result = result * self.scale + residual
        result = result.transpose(1, 2).contiguous()
        return result

    
class SingleStreamMoE(nn.Module):
    def __init__(self, model_dim, hidden_dim, num_sub_experts=4, tau=1.0):
        super().__init__()
        self.model_dim = model_dim
        self.num_sub_experts = num_sub_experts
        self.tau_gate = tau
        self.sub_gate = nn.Linear(self.model_dim, self.num_sub_experts)
        self.sub_experts = nn.ModuleList([
            BehaviorAwareExpert(self.model_dim, hidden_dim) 
            for _ in range(self.num_sub_experts)
        ])

    def _get_load_balancing_loss(self, gate_logits, gate_weights):
        f_i = F.softmax(gate_logits, dim=-1).mean(dim=0) 
        P_i = gate_weights.mean(dim=0)
        loss = 0.01 * self.num_sub_experts * torch.sum(f_i * P_i)
        return loss

    def forward(self, x):
        B, L, D = x.shape
        x_flat = x.view(-1, D)
        sub_gate_logits = self.sub_gate(x_flat)
        
        if self.training:
            sub_gate_weights = F.gumbel_softmax(sub_gate_logits, tau=self.tau_gate, hard=False, dim=-1)
            aux_loss = self._get_load_balancing_loss(sub_gate_logits, sub_gate_weights)
        else:
            indices = torch.argmax(sub_gate_logits, dim=-1)
            sub_gate_weights = F.one_hot(indices, num_classes=self.num_sub_experts).float()
            aux_loss = 0.0

        all_sub_expert_outputs = torch.stack([sub_expert(x) for sub_expert in self.sub_experts], dim=0)
        sub_gate_weights_bcast = sub_gate_weights.view(B, L, self.num_sub_experts).permute(2, 0, 1).unsqueeze(-1)
        stream_output = torch.sum(all_sub_expert_outputs * sub_gate_weights_bcast, dim=0)
        return stream_output, aux_loss


class BehaviorAwareMoE(nn.Module):
    """
    B-MoE: Behavior-Aware Mixture-of-Experts.
    """
    def __init__(self, model_dim, embed_dim, hidden_dim=None, num_sub_experts=4, tau=1.0, modality="text"):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = model_dim * 2 
        self.model_dim = model_dim
        self.modality = modality
        self.embed_dim = embed_dim
        self.num_streams = 3 
        self.tau_gate = tau
        self.aux_loss = 0.0

        self.gate = nn.Linear(self.model_dim, self.num_streams)

        self.expert_app = SingleStreamMoE(model_dim, hidden_dim, num_sub_experts, tau)
        self.expert_norm = SingleStreamMoE(model_dim, hidden_dim, num_sub_experts, tau)
        self.expert_anom = SingleStreamMoE(model_dim, hidden_dim, num_sub_experts, tau)

        self.proj_vision_app = nn.Linear(self.model_dim, self.embed_dim)
        self.proj_vision_norm = nn.Linear(self.model_dim, self.embed_dim)
        self.proj_vision_anom = nn.Linear(self.model_dim, self.embed_dim)
        self.proj_text_app = nn.Linear(self.model_dim, self.embed_dim)
        self.proj_text_norm = nn.Linear(self.model_dim, self.embed_dim)
        self.proj_text_anom = nn.Linear(self.model_dim, self.embed_dim)

    def _get_load_balancing_loss(self, gate_logits, gate_weights):
        f_i = F.softmax(gate_logits, dim=-1).mean(dim=0) 
        P_i = gate_weights.mean(dim=0)
        loss = 0.01 * self.num_streams * torch.sum(f_i * P_i)
        return loss

    def forward(self, x):
        B, L, D = x.shape
        x_flat = x.view(-1, D) 
        top_gate_logits = self.gate(x_flat)
        
        if self.training:
            top_gate_weights = F.gumbel_softmax(top_gate_logits, tau=self.tau_gate, hard=False, dim=-1) 
            self.aux_loss = self._get_load_balancing_loss(top_gate_logits, top_gate_weights)
        else:
            indices = torch.argmax(top_gate_logits, dim=-1)
            top_gate_weights = F.one_hot(indices, num_classes=self.num_streams).float() 
            self.aux_loss = 0.0

        out_app, aux_loss_app = self.expert_app(x)
        out_norm, aux_loss_norm = self.expert_norm(x)
        out_anom, aux_loss_anom = self.expert_anom(x)
        
        self.aux_loss += (aux_loss_app + aux_loss_norm + aux_loss_anom) / 3.0

        top_gate_weights_bcast = top_gate_weights.view(B, L, self.num_streams)
        top_weights_app = top_gate_weights_bcast[..., 0:1]
        top_weights_norm = top_gate_weights_bcast[..., 1:2]
        top_weights_anom = top_gate_weights_bcast[..., 2:3]

        stream_app = out_app * top_weights_app
        stream_norm = out_norm * top_weights_norm
        stream_anom = out_anom * top_weights_anom

        cls_app = stream_app[:, 0, :]
        cls_norm = stream_norm[:, 0, :]
        cls_anom = stream_anom[:, 0, :]

        if self.modality == 'text':
            feat_app = F.normalize(self.proj_text_app(cls_app), dim=-1)
            feat_norm = F.normalize(self.proj_text_norm(cls_norm), dim=-1)
            feat_anom = F.normalize(self.proj_text_anom(cls_anom), dim=-1)
        
        if self.modality == 'vision':
            feat_app = F.normalize(self.proj_vision_app(cls_app), dim=-1)
            feat_norm = F.normalize(self.proj_vision_norm(cls_norm), dim=-1)
            feat_anom = F.normalize(self.proj_vision_anom(cls_anom), dim=-1)

        return (stream_app, stream_norm, stream_anom), \
               (feat_app, feat_norm, feat_anom), \
               self.aux_loss


def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])

class MultiHeadedCrossAttn(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.3):
        super(MultiHeadedCrossAttn, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = dim
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim,dim)
        self.k_proj = nn.Linear(dim,dim)
        self.v_proj = nn.Linear(dim,dim)
        self.out_proj = nn.Linear(dim,dim)

    def forward(self, q_embeds, context_embeds, attn_mask=None):
        t, b_q, a_q, d = q_embeds.shape
        f_kv = context_embeds.shape[2] 
        b_kv = context_embeds.shape[1]
        b = b_q

        q = self.q_proj(q_embeds)
        q = q.reshape(t, b, a_q, self.num_heads, self.head_dim).permute(0,1,3,4,2)
        k = self.k_proj(context_embeds)
        k = k.reshape(t, b, f_kv, self.num_heads, self.head_dim).permute(0,1,3,2,4)
        v = self.v_proj(context_embeds)
        v = v.reshape(t, b, f_kv, self.num_heads, self.head_dim).permute(0,1,3,4,2)

        attention_logits = k @ q
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_mask = attn_mask[None,:,None,:,None] 
            attention_logits = attention_logits.masked_fill(attn_mask == 0, -1e9)
        attention_weights = F.softmax(attention_logits, dim=-2)

        attention = v @ attention_weights
        attention = attention.permute(0,1,4,2,3).reshape(t, b, a_q, self.embed_dim)
        o = self.out_proj(attention)
        return o.permute(0,2,1,3), attention_weights 

class Transformer(nn.Module):
    def __init__(self, embed_dim, attn_heads, dropout=0.3):
        super(Transformer, self).__init__()
        self.embed_dim = embed_dim
        self.cross_attn = MultiHeadedCrossAttn(embed_dim, attn_heads, dropout=dropout)
        self.linear1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, 4*embed_dim), nn.ReLU(inplace=True),
            nn.Linear(4*self.embed_dim, self.embed_dim)
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name: nn.init.eye_(param)
                elif 'bias' in name: param.data.fill_(0.)

    def forward(self, q_embeds, C_embeds):
        q_embeds_norm = self.layer_norm1(q_embeds)
        C_embeds_norm = self.layer_norm1(C_embeds) 
        attn_out,_ = self.cross_attn(q_embeds_norm, C_embeds_norm) 
        q_embeds_permuted = q_embeds.permute(0,2,1,3)
        attn_out = q_embeds_permuted + self.dropout(attn_out)
        attn_out_norm = self.layer_norm2(attn_out)
        linear_out = self.linear1(attn_out_norm)
        out = attn_out_norm + self.dropout(linear_out)
        out = self.layer_norm3(out)
        return out 

class OffloadGenerator(nn.Module):
    def __init__(self, dim, num_mha_heads, num_layers=1, dropout=0.3):
        super(OffloadGenerator, self).__init__()
        cross_layer = Transformer(dim, num_mha_heads, dropout)
        self.num_layers = num_layers
        self.layers = _get_clones(cross_layer, num_layers)
        self.mean_proj = nn.Linear(dim, dim)
        self.log_std_proj = nn.Linear(dim, dim)

    def forward(self, delta, C):
        out = delta.permute(0,2,1,3) 
        for layer in self.layers:
            _out = layer(out, C)
            out = _out.permute(0,2,1,3)
        mean = self.mean_proj(_out)
        log_sigma = self.log_std_proj(_out)
        return mean, log_sigma

class GapPredictor(nn.Module):
    def __init__(self, dim, num_heads=8, num_layers=1, dropout=0.3):
        super(GapPredictor, self).__init__()
        self.psi = OffloadGenerator(dim, num_mha_heads=1, num_layers=num_layers, dropout=dropout)

class OffloadAnchorUnit(nn.Module):
    def __init__(self, embed_dim, config, dropout=0.1):
        super().__init__()
        self.config = config
        self.psi_agent = GapPredictor(dim=embed_dim, num_heads=1, num_layers=1, dropout=dropout)
        self.psi = self.psi_agent.psi
        
        self.beta = self.config.get('col_beta', 0.07) 
        self.lambda_dir = self.config.get('col_lambda_dir', 0.001) 
        self.lambda_epsilon = self.config.get('col_lambda_epsilon', 0.01) 
        self.lambda_lower = self.config.get('col_lambda_bound', 0.5) 
        self.sigma_scale = self.config.get('col_sigma_scale', 2.0) 

    def direction_diversity_loss(self, delta, alpha):
        B_t, B_v, D = delta.shape
        delta_dir = F.normalize(delta, dim=2) 
        loss = 0.0
        for i in range(B_t):
            z = delta_dir[i]
            sim = torch.matmul(z, z.T)
            dist = 1 - sim
            loss += torch.log(torch.exp(-alpha * dist).mean() + 1e-8)
        return loss / B_t

    def norm_based_epsilon_regularization_loss(self, delta, lambda_lower):
        B_t, B_v, D = delta.shape
        dist = delta.norm(p=2, dim=2)
        var = dist.var(dim=1)
        loss = -var.mean()
        return loss  if loss >= -lambda_lower else 0

    def _compute_kl_loss(self, delta_pred):
        mu = delta_pred.mean(dim=0)
        sigma = delta_pred.std(dim=0) + 1e-6
        target_dist = Normal(torch.zeros_like(mu), torch.ones_like(sigma))
        estimated_dist = Normal(mu, sigma)
        kl_div = kl_divergence(estimated_dist, target_dist).mean()
        return self.beta * kl_div

    def forward(self, txt_feat, img_feat_pooled, img_embeds_sequence, is_training=True):
        cls = txt_feat
        v_feat_sequence = img_embeds_sequence
        a, b = cls.size(0), v_feat_sequence.size(0) 

        video_feat_mean = img_feat_pooled.unsqueeze(0) 
        _cls = cls.unsqueeze(1).repeat(1, b, 1) 
        
        semantic_gap = video_feat_mean - _cls 
        gap_psi_in = semantic_gap.unsqueeze(0)
        ctx_psi_in = v_feat_sequence.unsqueeze(0)
        
        delta_pred, _ = self.psi(gap_psi_in, ctx_psi_in) 
        delta_pred = delta_pred.squeeze(0) 

        cls_corrected = _cls + delta_pred 
        _t_feat = cls_corrected / cls_corrected.norm(dim=-1, keepdim=True)
        _v_feat = img_feat_pooled.unsqueeze(0) 
        _v_feat = _v_feat / _v_feat.norm(dim=-1, keepdim=True)

        s_ij = torch.einsum('abd,xbd->ab', [_t_feat, _v_feat])

        reg_loss = torch.tensor(0.0, device=s_ij.device)
        if is_training:
            dir_loss = self.direction_diversity_loss(delta_pred, alpha=self.sigma_scale)
            reg_loss = self.lambda_dir * dir_loss
            
        return s_ij, reg_loss


class ConflictOffloadLearning(nn.Module):
    """
    Module 2: Conflict Offload Learning (COL).
    Resolves gradient conflict by offloading gradients via dynamic anchors.
    """
    def __init__(self, embed_dim, config, dropout=0.1):
        super().__init__()
        self.anchor_unit_app = OffloadAnchorUnit(embed_dim, config, dropout)
        self.anchor_unit_norm = OffloadAnchorUnit(embed_dim, config, dropout)
        self.anchor_unit_anom = OffloadAnchorUnit(embed_dim, config, dropout)
        self.total_reg_loss = 0.0

    def forward(self, 
                img_feat_app, img_feat_norm, img_feat_anom, 
                txt_feat_app, txt_feat_norm, txt_feat_anom,
                img_embeds_app, img_embeds_norm, img_embeds_anom,
                is_training=True):
        
        s_app_ij, reg_loss_app = self.anchor_unit_app(
            txt_feat_app, img_feat_app, img_embeds_app, is_training)
        s_norm_ij, reg_loss_norm = self.anchor_unit_norm(
            txt_feat_norm, img_feat_norm, img_embeds_norm, is_training)
        s_anom_ij, reg_loss_anom = self.anchor_unit_anom(
            txt_feat_anom, img_feat_anom, img_embeds_anom, is_training)

        if is_training:
            self.total_reg_loss = reg_loss_app + reg_loss_norm + reg_loss_anom
        else:
            self.total_reg_loss = 0.0

        return (s_app_ij, s_norm_ij, s_anom_ij)


class HierarchicalIntentAlignment(nn.Module):
    """
    Module 3: Hierarchical Intent Alignment (HIA).
    """
    def __init__(self, embed_dim, config, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim * 2 

        self.gating_net = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3)
        )
        
        self._init_weights()
        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.epsilon = config.get('label_smooth', 0.0) 

    def _init_weights(self):
        for m in self.gating_net.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, 
                s_app_all, s_norm_all, s_anom_all,
                img_feat_app, img_feat_norm, img_feat_anom, 
                txt_feat_app, txt_feat_norm, txt_feat_anom, 
                img_embeds_app, img_embeds_norm, img_embeds_anom, 
                txt_embeds_app, txt_embeds_norm, txt_embeds_anom, 
                idx):
        
        txt_feats_concat = torch.cat([txt_feat_app, txt_feat_norm, txt_feat_anom], dim=-1)
        gate_logits = self.gating_net(txt_feats_concat)
        weights_local = F.softmax(gate_logits, dim=-1)

        w_app_local = weights_local[:, 0:1]
        w_norm_local = weights_local[:, 1:2]
        w_anom_local = weights_local[:, 2:3]
        
        s_ij_all = (w_app_local * s_app_all + 
                    w_norm_local * s_norm_all + 
                    w_anom_local * s_anom_all)

        logits = s_ij_all / self.temp
        idx = idx.view(-1, 1)
        idx_all = idx
        pos_idx = torch.eq(idx_all, idx_all.t()).float()
        labels = pos_idx / pos_idx.sum(1, keepdim=True)

        if 0 < self.epsilon < 1:
            _, num_classes = logits.shape
            labels = (1 - self.epsilon) * labels + self.epsilon / num_classes
        elif self.epsilon == 1: 
            labels = torch.ones_like(logits) / logits.shape[1]

        loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()
        loss_con = (loss_i2t + loss_t2i) / 2


        f_txt_final = (w_app_local * txt_feat_app + 
                       w_norm_local * txt_feat_norm + 
                       w_anom_local * txt_feat_anom)
        
        w_app_bcast = w_app_local.unsqueeze(-1)
        w_norm_bcast = w_norm_local.unsqueeze(-1)
        w_anom_bcast = w_anom_local.unsqueeze(-1)
        
        embeds_txt_final = (w_app_bcast * txt_embeds_app +
                            w_norm_bcast * txt_embeds_norm +
                            w_anom_bcast * txt_embeds_anom)

        img_feats_concat = torch.cat([img_feat_app, img_feat_norm, img_feat_anom], dim=-1)
        gate_logits_img = self.gating_net(img_feats_concat)
        weights_local_img = F.softmax(gate_logits_img, dim=-1)

        w_app_img = weights_local_img[:, 0:1]
        w_norm_img = weights_local_img[:, 1:2]
        w_anom_img = weights_local_img[:, 2:3]
        
        f_img_final = (w_app_img * img_feat_app + 
                       w_norm_img * img_feat_norm + 
                       w_anom_img * img_feat_anom)
                       
        w_app_bcast_img = w_app_img.unsqueeze(-1)
        w_norm_bcast_img = w_norm_img.unsqueeze(-1)
        w_anom_bcast_img = w_anom_img.unsqueeze(-1)
        
        embeds_img_final = (w_app_bcast_img * img_embeds_app +
                            w_norm_bcast_img * img_embeds_norm +
                            w_anom_bcast_img * img_embeds_anom)

        return loss_con, s_ij_all, f_txt_final, f_img_final, embeds_txt_final, embeds_img_final
