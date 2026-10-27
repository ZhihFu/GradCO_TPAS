import os
import json
import numpy as np
import time
import datetime
from prettytable import PrettyTable
import torch
import torch.distributed as dist
import torch.nn.functional as F
import utils

@torch.no_grad()
def evaluation_itc(model, data_loader, tokenizer, device, config):
    model.eval()
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = config['batch_size_test_text']
     
    text_embeds_agg_all = []
    text_embeds_orig_all = []
    text_atts_all = []
     
    text_feats_app_all = []
    text_feats_norm_all = []
    text_feats_anom_all = []
    text_feats_concat_all = []
     
    for i in range(0, num_text, text_bs): 
        text = texts[i: min(num_text, i + text_bs)] 
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                                     return_tensors="pt").to(device)
         
        text_embed_orig = model.get_text_embeds(text_input.input_ids, text_input.attention_mask)
        text_embeds_orig_all.append(text_embed_orig)
        text_feat_orig = model.get_text_feat(text_embed_orig)
         
        (txt_embeds_app, txt_embeds_norm, txt_embeds_anom), \
        (txt_feat_app, txt_feat_norm, txt_feat_anom), \
        _ = model.bmoe_txt(text_embed_orig)
         
        text_atts_all.append(text_input.attention_mask)

        text_feats_app_all.append(txt_feat_app)
        text_feats_norm_all.append(txt_feat_norm)
        text_feats_anom_all.append(txt_feat_anom)
         
        txt_feats_concat = torch.cat([txt_feat_app, txt_feat_norm, txt_feat_anom], dim=-1)
        text_feats_concat_all.append(txt_feats_concat)

        gate_logits = model.hia.gating_net(txt_feats_concat) 
        weights_local = F.softmax(gate_logits, dim=-1)
         
        w_app_bcast = weights_local[:, 0:1].unsqueeze(-1)
        w_norm_bcast = weights_local[:, 1:2].unsqueeze(-1)
        w_anom_bcast = weights_local[:, 2:3].unsqueeze(-1)
         
        text_embed_agg = (w_app_bcast * txt_embeds_app +
                          w_norm_bcast * txt_embeds_norm +
                          w_anom_bcast * txt_embeds_anom)
        text_embeds_agg_all.append(text_embed_agg)

    text_embeds = torch.cat(text_embeds_agg_all, dim=0)
    text_embeds_orig = torch.cat(text_embeds_orig_all, dim=0)
    text_atts = torch.cat(text_atts_all, dim=0)

    txt_feat_app_all = torch.cat(text_feats_app_all, dim=0)
    txt_feat_norm_all = torch.cat(text_feats_norm_all, dim=0)
    txt_feat_anom_all = torch.cat(text_feats_anom_all, dim=0)
    txt_feats_concat_all = torch.cat(text_feats_concat_all, dim=0)

    image_embeds_agg_local = [] 
    image_embeds_orig_local = []
    sims_matrix_t2i_slices = []
 
    for image, pose, img_id in data_loader: 
        image = image.to(device) 
        image_embed_orig, image_atts = model.get_vision_embeds(image)

        if config.get('be_pose_img', False): 
            pose = pose.to(device) 
            if model.be_pose_conv: 
                pose = model.pose_conv(pose) 
            pose_embed, _ = model.get_vision_embeds(pose) 
            image_embed_orig = model.pose_block(image_embed_orig, pose_embed) 
        
        image_embeds_orig_local.append(image_embed_orig)
        imgae_feat_orig = model.get_image_feat(image_embed_orig)

        (img_embeds_app, img_embeds_norm, img_embeds_anom), \
        (img_feat_app, img_feat_norm, img_feat_anom), \
        _ = model.bmoe_img(image_embed_orig) 
         
        img_feats_concat = torch.cat([img_feat_app, img_feat_norm, img_feat_anom], dim=-1)
        gate_logits = model.hia.gating_net(img_feats_concat) 
        weights_local = F.softmax(gate_logits, dim=-1) 

        w_app_bcast = weights_local[:, 0:1].unsqueeze(-1) 
        w_norm_bcast = weights_local[:, 1:2].unsqueeze(-1) 
        w_anom_bcast = weights_local[:, 2:3].unsqueeze(-1) 

        image_embed_agg = (w_app_bcast * img_embeds_app +
                           w_norm_bcast * img_embeds_norm +
                           w_anom_bcast * img_embeds_anom) 
        image_embeds_agg_local.append(image_embed_agg) 

        B, L_img, D_model = img_embeds_app.shape
        img_embeds_app_flat = img_embeds_app.view(-1, D_model)
        img_embeds_norm_flat = img_embeds_norm.view(-1, D_model)
        img_embeds_anom_flat = img_embeds_anom.view(-1, D_model)
         
        img_ctx_app_flat = model.bmoe_img.proj_vision_app(img_embeds_app_flat)
        img_ctx_norm_flat = model.bmoe_img.proj_vision_norm(img_embeds_norm_flat)
        img_ctx_anom_flat = model.bmoe_img.proj_vision_anom(img_embeds_anom_flat)
         
        img_embeds_app_proj = img_ctx_app_flat.view(B, L_img, -1)
        img_embeds_norm_proj = img_ctx_norm_flat.view(B, L_img, -1)
        img_embeds_anom_proj = img_ctx_anom_flat.view(B, L_img, -1)
         
        s_app_ij, _ = model.col.anchor_unit_app(
            txt_feat_app_all, img_feat_app, img_embeds_app_proj, is_training=False
        )
        s_norm_ij, _ = model.col.anchor_unit_norm(
            txt_feat_norm_all, img_feat_norm, img_embeds_norm_proj, is_training=False
        )
        s_anom_ij, _ = model.col.anchor_unit_anom(
            txt_feat_anom_all, img_feat_anom, img_embeds_anom_proj, is_training=False
        )

        gate_logits_txt = model.hia.gating_net(txt_feats_concat_all)
        weights_txt = F.softmax(gate_logits_txt, dim=-1)
         
        w_app_txt = weights_txt[:, 0:1]
        w_norm_txt = weights_txt[:, 1:2]
        w_anom_txt = weights_txt[:, 2:3]
         
        s_ij_agg = (w_app_txt * s_app_ij +
                    w_norm_txt * s_norm_ij +
                    w_anom_txt * s_anom_ij)
        sims_matrix_t2i_slices.append(s_ij_agg)

    image_embeds = torch.cat(image_embeds_agg_local, dim=0)
    image_embeds_orig = torch.cat(image_embeds_orig_local, dim=0)
    sims_matrix_t2i = torch.cat(sims_matrix_t2i_slices, dim=1)
     
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    return sims_matrix_t2i, image_embeds_orig, text_embeds_orig, text_atts

@torch.no_grad()
def evaluation_itm(model, device, config, args, sims_matrix, image_embeds, text_embeds, text_atts):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print('Computing matching score')
    start_time = time.time()

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    score_matrix_t2i = torch.full(sims_matrix.size(), 1000.0).to(device)
    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 500, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_embeds[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.get_cross_embeds(encoder_output, encoder_att,
                                        text_embeds=text_embeds[start + i].repeat(config['k_test'], 1, 1),
                                        text_atts=text_atts[start + i].repeat(config['k_test'], 1),)[:, 0, :]
        score = model.itm_head(output)[:, 1]
        score_matrix_t2i[start + i, topk_idx] = score

    min_values, _ = torch.min(score_matrix_t2i, dim=1)
    replacement_tensor = min_values.view(-1, 1).expand(-1, score_matrix_t2i.size(1))
    for i in range(sims_matrix.size(0)):
        score_matrix_t2i[i][score_matrix_t2i[i] == 1000.0] = replacement_tensor[i][score_matrix_t2i[i] == 1000.0]
    score_matrix_t2i[score_matrix_t2i == 1000.0] = replacement_tensor[score_matrix_t2i == 1000.0]
    score_matrix_t2i = (score_matrix_t2i - score_matrix_t2i.min()) / (score_matrix_t2i.max() - score_matrix_t2i.min())

    score_sim_t2i = sims_matrix.clone()
    score_sim_t2i = (score_sim_t2i - score_sim_t2i.min()) / (score_sim_t2i.max() - score_sim_t2i.min())
    score_matrix_t2i = score_matrix_t2i + 0.002 * score_sim_t2i 

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Computing matching score time {}'.format(total_time_str))
    return score_matrix_t2i.cpu().numpy()


def mAP(scores_t2i, g_pids, q_pids, table=None, settings=None):
    similarity = torch.tensor(scores_t2i)
    indices = torch.argsort(similarity, dim=1, descending=True)
    g_pids = torch.tensor(g_pids)
    q_pids = torch.tensor(q_pids)
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :10].cumsum(1)  
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100

    num_rel = matches.sum(1)  
    tmp_cmc = matches.cumsum(1)  

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  
    mAP = AP.mean() * 100

    t2i_cmc, t2i_mAP, t2i_mINP, _ = all_cmc, mAP, mINP, indices
    t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()

    if settings is not None and table is not None:
        table.add_row([settings, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        print(table)

    eval_result = {'R1': t2i_cmc[0],
                   'R5': t2i_cmc[4],
                   'R10': t2i_cmc[9],
                   'mAP': t2i_mAP,
                   'mINP': t2i_mINP,
                   }
    return eval_result