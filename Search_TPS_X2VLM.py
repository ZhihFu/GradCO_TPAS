import os
from pathlib import Path
import time
import datetime
import argparse
import json
import math
import random
import numpy as np
from ruamel.yaml import YAML
yaml = YAML(typ='safe')
from prettytable import PrettyTable

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.cuda.amp import GradScaler

from transformers import BertTokenizer, XLMRobertaTokenizer

import utils
from models.model_X2VLM import GradCO

from dataset_TPS import create_dataset, create_sampler, create_loader
from dataset import create_dataset as create_dataset_tpas

from dataset.search_dataset import TextMaskingGenerator
from scheduler import create_scheduler
from optim import create_optimizer

from train import train_model, train_model_TPS
from eval_TPS import evaluation_itm, evaluation_itc, mAP

from eval import evaluation_itm as evaluation_itm_tpas
from eval import evaluation_itc as evaluation_itc_tpas


def build_tokenizer(text_encoder: str, dropout=0):
    if ('bert-base-uncased' in text_encoder) or ('bert-large-uncased' in text_encoder):
        tokenizer = BertTokenizer.from_pretrained(text_encoder)
    elif ('xlm-roberta-base' in text_encoder) or ('xlm-roberta-large' in text_encoder):
        tokenizer = XLMRobertaTokenizer.from_pretrained(text_encoder)
    else:
        raise NotImplementedError(f"tokenizer for {text_encoder}")

    tokenizer.add_special_tokens({'bos_token': tokenizer.cls_token})
    tokenizer.add_special_tokens({'eos_token': tokenizer.sep_token})
    return tokenizer


def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    world_size = utils.get_world_size()

    if args.bs > 0: config['batch_size_train'] = args.bs
    if args.epo > 0: config['schedular']['epochs'] = args.epo
    if args.lr > 0:
        config['optimizer']['lr'] = args.lr
        config['schedular']['lr'] = args.lr

    config['num_sub_experts'] = args.num_sub_experts    
    config['col_lambda_dir'] = args.col_lambda_dir

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    print("### output_dir:", args.output_dir)
    
    print("### Creating GradCO model")
    tokenizer = build_tokenizer(config['text_encoder'])
    model = GradCO(config=config) 
    
    if config['load_pretrained']:
        model.load_pretrained(args.checkpoint, config)
        print("Loaded pretrained model from:", args.checkpoint)
    model = model.to(device)
    print("Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    start_time = time.time()

    if args.evaluate:
        # Construct paths relative to data_root
        base_anno = os.path.join(args.data_root, 'annotation')
        test_file_list = {
            "original": os.path.join(base_anno, 'test/attr.json'),
            "ood": os.path.join(base_anno, 'test/ucc.json'),
            "multi_weather_wind": os.path.join(base_anno, 'multi-weather/multi-weather_wind.json'),
            "multi_weather_rain": os.path.join(base_anno, 'multi-weather/multi-weather_rain.json'),
            "multi_weather_snow": os.path.join(base_anno, 'multi-weather/multi-weather_snow.json'),
            "multi_weather_rain_snow": os.path.join(base_anno, 'multi-weather/multi-weather_rain_snow.json'),
            "multi_weather_dark": os.path.join(base_anno, 'multi-weather/multi-weather_dark.json'),
            "multi_weather_dark_wind": os.path.join(base_anno, 'multi-weather/multi-weather_dark_wind.json'),
            "multi_weather_dark_rain": os.path.join(base_anno, 'multi-weather/multi-weather_dark_rain.json'),
            "multi_weather_dark_snow": os.path.join(base_anno, 'multi-weather/multi-weather_dark_snow.json'),
            "multi_weather_light": os.path.join(base_anno, 'multi-weather/multi-weather_light.json'),
        }

        print("### Creating search dataset")
        test_dataset_list = {}
        for key in test_file_list.keys():
            config['test_file'] = test_file_list[key]
            test_dataset_list[key] = create_dataset_tpas(config, args.evaluate)[1]
        tbs = ["task", "R1", "R5", "R10", "mAP", "mINP"]
        table = PrettyTable(tbs)
        for tb in tbs[1:]:
            table.custom_format[tb] = lambda f, v: f"{v:.3f}"

        for key in test_dataset_list.keys():
            print(f"### Start evaluating {key} set ###")
            test_loader = create_loader([test_dataset_list[key]], [None],
                                        batch_size=[config['batch_size_test']],
                                        num_workers=[8],
                                        is_trains=[False],
                                        collate_fns=[None])[0]
            sims_matrix_t2i, image_embeds, text_embeds, text_atts = evaluation_itc(
                model_without_ddp, test_loader, tokenizer, device, config)
            score_test_t2i = evaluation_itm(model_without_ddp, device, config, args,
                                            sims_matrix_t2i, image_embeds, text_embeds, text_atts)
            if utils.is_main_process():
                print('evaluating result:')
                mAP(score_test_t2i, test_loader.dataset.g_pids, test_loader.dataset.q_pids, table=table, settings=key)
            dist.barrier()

    else:
        print("Creating retrieval dataset", flush=True)
        # Dataset creation logic
        if args.task == "tpas":
             train_dataset, test_dataset = create_dataset_tpas(config, args.evaluate)
        elif args.task == "itr_ufine":
            train_dataset, test_dataset = create_dataset('re_ufine', config, args.evaluate)
        elif args.task == "itr_icfg":
            train_dataset, test_dataset = create_dataset('re_icfg', config, args.evaluate)
        elif args.task == "itr_rstp":
            train_dataset, val_dataset, test_dataset = create_dataset('re_rstp', config, args.evaluate)
        elif args.task == "itr_cuhk":
            train_dataset, val_dataset, test_dataset = create_dataset('re_cuhk', config, args.evaluate)
        elif args.task == "itr_pa100k":
            train_dataset, val_dataset, test_dataset = create_dataset('re_pa100k', config, args.evaluate)
        elif args.task == "itr_gene":
            train_dataset, val_dataset, test_dataset = create_dataset('re_gene', config, args.evaluate)
        
        print("### Start training")
        train_dataset_size = len(train_dataset)
        if utils.is_main_process():
            print(f"### data {train_dataset_size}, batch size, {config['batch_size_train']} x {world_size}")
            tbs = ["epoch", "R1", "R5", "R10", "mAP", "mINP"]
            table = PrettyTable(tbs)
            for tb in tbs[1:]:
                table.custom_format[tb] = lambda f, v: f"{v:.3f}"

        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            if args.task in ["itr_icfg", "tpas", "itr_ufine"]:
                samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None]
            else:
                samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
        else:
            samplers = [None, None] if args.task in ["itr_icfg", "tpas", "itr_ufine"] else [None, None, None]

        if args.task in ["itr_icfg", "tpas", "itr_ufine"]:
            train_loader, test_loader = create_loader([train_dataset, test_dataset], samplers,
                                                      batch_size=[config['batch_size_train']] + [
                                                          config['batch_size_test']]* 2,
                                                      num_workers=[8, 8], is_trains=[True, False],
                                                      collate_fns=[None, None])
        else:
            train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                                  batch_size=[config['batch_size_train']] + [
                                                                      config['batch_size_test']] * 2,
                                                                  num_workers=[8, 8, 8], is_trains=[True, False, False],
                                                                  collate_fns=[None, None, None])

        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model_without_ddp)
        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size / (config['batch_size_train'] * world_size))
        lr_scheduler = create_scheduler(arg_sche, optimizer)
        scaler = GradScaler() 

        mask_generator = TextMaskingGenerator(tokenizer, config['mask_prob'], config['max_masks'],
                                              config['skipgram_prb'], config['skipgram_size'],
                                              config['mask_whole_word'])
        best = 0
        best_epoch = 0
        max_epoch = config['schedular']['epochs']
        for epoch in range(0, max_epoch):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            if args.task == "tpas":
                train_stats = train_model(model, train_loader, optimizer, scaler, tokenizer, epoch,
                                        device, lr_scheduler, config, mask_generator)
            else:
                train_stats = train_model_TPS(model, train_loader, optimizer, scaler, tokenizer, epoch,
                                        device, lr_scheduler, config, mask_generator)
            
            if args.task == "tpas":
                sims_matrix_t2i, image_embeds, text_embeds, text_atts = evaluation_itc_tpas(
                model_without_ddp, test_loader, tokenizer, device, config)
                score_test_t2i = evaluation_itm_tpas(model_without_ddp, device, config, args,
                                                sims_matrix_t2i, image_embeds, text_embeds, text_atts,)
            else:
                sims_matrix_t2i, image_embeds, text_embeds, text_atts = evaluation_itc(
                    model_without_ddp, test_loader, tokenizer, device, config)
                score_test_t2i = evaluation_itm(model_without_ddp, device, config, args,
                                                sims_matrix_t2i, image_embeds, text_embeds, text_atts,)
            del sims_matrix_t2i, image_embeds, text_embeds, text_atts

            if utils.is_main_process():
                test_result = mAP(score_test_t2i, test_loader.dataset.g_pids, test_loader.dataset.q_pids, table)
                table.add_row([epoch, test_result['R1'], test_result['R5'], test_result['R10'],
                               test_result['mAP'], test_result['mINP']])
                print(table)

                logs = {'epo': epoch}
                for k, v in test_result.items():
                    logs[k] = np.around(v, 3)
                for k, v in train_stats.items():
                    logs[k] = float(v)
                print('logs', logs)

                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(logs) + "\n")

                result = test_result['R1']
                if result > best:
                    save_obj = {'model': model_without_ddp.state_dict(), 'config': config, }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                    best = result
                    best_epoch = epoch

            dist.barrier()
            torch.cuda.empty_cache()

        if utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write("best epoch: %d" % best_epoch)
            print("### best epoch: %d" % best_epoch)
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('### Time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--data_root', type=str, default='/data/PAB', help="root directory for datasets")
    parser.add_argument('--bs', default=0, type=int, help="mini batch size")
    parser.add_argument('--epo', default=0, type=int, help="epoch")
    parser.add_argument('--lr', default=0.0, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')
    parser.add_argument('--num_sub_experts', default=0, type=int)
    parser.add_argument('--col_lambda_dir', default=0.0, type=float)
    
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config = yaml.load(open(args.config, 'r'))
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
