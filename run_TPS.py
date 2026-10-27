import os
import argparse

# Default distributed settings
NNODES = 1
NODE_RANK = 0
MASTER_ADDR = '127.0.0.1'
MASTER_PORT = 1633
NPROC_PER_NODE = 4 

def get_dist_launch(args):
    return f"CUDA_VISIBLE_DEVICES={args.devices} WORLD_SIZE={args.world_size} python3 -m torch.distributed.run --nproc_per_node={args.nproc} " \
           f"--nnodes={NNODES} --node_rank={NODE_RANK} " \
           f"--master_addr={MASTER_ADDR} --master_port={MASTER_PORT} "

def run(args):
    args.config = 'configs/' + args.task + '_X2VLM.yaml'
    print(f"Task: {args.task}, Config: {args.config}")

    dist_launch = get_dist_launch(args)
    os.system(
        f"{dist_launch} Search_TPS_X2VLM.py --config {args.config} --task {args.task} --output_dir {args.output_dir} "
        f"--checkpoint {args.checkpoint} --bs {args.bs} --epo {args.epo} --lr {args.lr} --seed {args.seed} "
        f"--num_sub_experts {args.num_sub_experts} --col_lambda_dir {args.col_lambda_dir} "
        f"--data_root {args.data_root} "
        f"{'--evaluate' if args.evaluate else ''}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='itr_rstp', type=str)
    parser.add_argument('--devices', default='0,1,2,3', type=str, help="CUDA_VISIBLE_DEVICES ids")
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--nproc', default=4, type=int, help="GPUs per node")
    parser.add_argument('--output_dir', default='out/cmp', type=str)
    parser.add_argument('--checkpoint', default='checkpoint/checkpoint_9.pth', type=str)
    parser.add_argument('--data_root', type=str, default='/data/PAB', help="Path to dataset root")
    parser.add_argument('--bs', default=0, type=int, help="mini batch size")
    parser.add_argument('--epo', default=0, type=int, help="epoch")
    parser.add_argument('--lr', default=0.0, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--evaluate', action='store_true', help="directly evaluation")
    parser.add_argument('--num_sub_experts', default=4, type=int)
    parser.add_argument('--col_lambda_dir', default=0.1, type=float)
    
    args = parser.parse_args()

    run(args)