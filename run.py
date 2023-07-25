# -*- coding: utf-8 -*-
import os
import argparse
import logging
import time

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(filename)s %(lineno)d:\t%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )


if __name__ == "__main__":
    T1 = time.perf_counter()
    logging.info("===================== Begin ======================")
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnodes", type=int)
    parser.add_argument("--nproc_per_node", type=int)
    parser.add_argument("--node_rank", type=int)
    parser.add_argument("--dist-url", type=str)
    parser.add_argument("--train_data_path",
                        type=str, default=r"B224_8fn_20230412_lowiso_v224U_B133_52L_SPLIT0.9")
    parser.add_argument("--valid_data_path",
                        type=str, default=r"B224_8fn_20230412_lowiso_v224U_B133_52L_SPLIT0.1")
    parser.add_argument("--dataset_dir", type=str, default=r"")
    parser.add_argument("--output_dir", type=str, default=r".")
    parser.add_argument("--tensorboard_dir", type=str, default=r"")
    parser.add_argument("--marker_file", type=str, default=r"")
    args = parser.parse_args()
    logging.info(args)

    T2 = time.perf_counter()
    logging.info('Load configs: %ss' % (T2 - T1))

    logging.info("===================== Install Packages and Pretrain_model ======================")
    os.system('python -m pip install --upgrade pip')
    os.system('pip install h5py')
    os.system('pip install contiguous_params')
    os.system('pip install timm')
    os.system('pip install einops')
    os.system('pip install kornia')

    # h5py==3.1.0
    # onnx==1.9.0
    # fvcore==0.1.5.post20221221

    os.makedirs("/root/.cache/torch/hub/checkpoints", exist_ok=True)
    # os.system("cp -r ./vgg19-dcbb9e9d.pth /root/.cache/torch/hub/checkpoints/")  # stage4需要
    T3 = time.perf_counter()
    logging.info('Install Packages: %ss' % (T3 - T2))

    logging.info("===================== Get datasets ======================")
    #################
    # flexml
    #################
    data_path = r'/'  # flexml
    os.makedirs(data_path, exist_ok=True)

    train_data_path = os.path.join(args.dataset_dir, args.train_data_path)
    valid_data_path = os.path.join(args.dataset_dir, args.valid_data_path)
    T4 = time.perf_counter()
    logging.info('Load datasets: %ss' % (T4 - T3))

    logging.info("===================== Start Training ======================")
    # save_path = os.path.join(args.output_dir, "checkpoint")

    # 0628
    if args.marker_file == '20230628_SC1_v5_Universal_const10_addrelu_esr2_exp1_t1_kd1_base':
        my_output_dir = os.path.join(args.output_dir, args.marker_file)
        command = f'python -m torch.distributed.launch --nproc_per_node={args.nproc_per_node} --master_port=12341 train_mef_pytorch_cloud_4_DETAIL_PASY_local_esq.py --net_version SC1_v5_Universal_const10_addrelu --checkpoint_dir_s3 None --checkpoint_dir {my_output_dir} --epoch_init 200 --train_data_dir {train_data_path} --valid_data_dir {valid_data_path} --pca 0 --model-ema --load_tiny_model_file_path 20230609_SC1_v5_Universal_const10_addrelu-ep200/model_best.pth.tar --es_ratio 2 --expanded_loss_weight 1 --tiny_loss_weight 1 --kd_ratio 1 --kd_loss_type base --tensorboard_dir {args.tensorboard_dir}'
    # 0630
    elif args.marker_file == '20230630_SC1_v5_Universal_const10_addrelu_esr3_exp1_t1_kd1_l1_tr':
        my_output_dir = os.path.join(args.output_dir, args.marker_file)
        command = f'python -m torch.distributed.launch --nproc_per_node={args.nproc_per_node} --master_port=12341 train_mef_pytorch_cloud_4_DETAIL_PASY_local_esq.py --net_version SC1_v5_Universal_const10_addrelu --checkpoint_dir_s3 None --checkpoint_dir {my_output_dir} --epoch_init 200 --train_data_dir {train_data_path} --valid_data_dir {valid_data_path} --pca 0 --model-ema --load_tiny_model_file_path 20230609_SC1_v5_Universal_const10_addrelu-ep200/model_best.pth.tar --es_ratio 3 --expanded_loss_weight 1 --tiny_loss_weight 1 --kd_ratio 1 --kd_loss_type l1 --tiny-rescale --tensorboard_dir {args.tensorboard_dir}'

    elif args.marker_file == '20230630_SC1_v5_Universal_const10_addrelu_esr3_exp1_t1_kd2_l1_tr':
        my_output_dir = os.path.join(args.output_dir, args.marker_file)
        command = f'python -m torch.distributed.launch --nproc_per_node={args.nproc_per_node} --master_port=12341 train_mef_pytorch_cloud_4_DETAIL_PASY_local_esq.py --net_version SC1_v5_Universal_const10_addrelu --checkpoint_dir_s3 None --checkpoint_dir {my_output_dir} --epoch_init 200 --train_data_dir {train_data_path} --valid_data_dir {valid_data_path} --pca 0 --model-ema --load_tiny_model_file_path 20230609_SC1_v5_Universal_const10_addrelu-ep200/model_best.pth.tar --es_ratio 3 --expanded_loss_weight 1 --tiny_loss_weight 1 --kd_ratio 2 --kd_loss_type l1 --tiny-rescale --tensorboard_dir {args.tensorboard_dir}'

    elif args.marker_file == '20230630_SC1_v5_Universal_const10_addrelu_esr2_exp1_t1_kd1_l1_tr_scratch':
        my_output_dir = os.path.join(args.output_dir, args.marker_file)
        command = f'python -m torch.distributed.launch --nproc_per_node={args.nproc_per_node} --master_port=12341 train_mef_pytorch_cloud_4_DETAIL_PASY_local_esq.py --net_version SC1_v5_Universal_const10_addrelu --checkpoint_dir_s3 None --checkpoint_dir {my_output_dir} --epoch_init 200 --train_data_dir {train_data_path} --valid_data_dir {valid_data_path} --pca 0 --model-ema --es_ratio 2 --expanded_loss_weight 1 --tiny_loss_weight 1 --kd_ratio 1 --kd_loss_type l1 --tiny-rescale --tensorboard_dir {args.tensorboard_dir}'

    elif args.marker_file == '20230702_SC1_v5_Universal_const10_addrelu_esr2_exp1_t1_kd0_l1_tr_scratch':
        my_output_dir = os.path.join(args.output_dir, args.marker_file)
        command = f'python -m torch.distributed.launch --nproc_per_node={args.nproc_per_node} --master_port=12341 train_mef_pytorch_cloud_4_DETAIL_PASY_local_esq.py --net_version SC1_v5_Universal_const10_addrelu --checkpoint_dir_s3 None --checkpoint_dir {my_output_dir} --epoch_init 200 --train_data_dir {train_data_path} --valid_data_dir {valid_data_path} --pca 0 --model-ema --es_ratio 2 --expanded_loss_weight 1 --tiny_loss_weight 1 --kd_ratio 0 --kd_loss_type l1 --tiny-rescale --tensorboard_dir {args.tensorboard_dir}'

    elif args.marker_file == '20230710_SC1_v5_Universal_const10_addrelu_esr4_exp1_t1_kd1_l1_tr':
        my_output_dir = os.path.join(args.output_dir, args.marker_file)
        command = f'python -m torch.distributed.launch --nproc_per_node={args.nproc_per_node} --master_port=12341 train_mef_pytorch_cloud_4_DETAIL_PASY_local_esq.py --net_version SC1_v5_Universal_const10_addrelu --checkpoint_dir_s3 None --checkpoint_dir {my_output_dir} --epoch_init 200 --train_data_dir {train_data_path} --valid_data_dir {valid_data_path} --pca 0 --model-ema --load_tiny_model_file_path 20230609_SC1_v5_Universal_const10_addrelu-ep200/model_best.pth.tar --es_ratio 4 --expanded_loss_weight 1 --tiny_loss_weight 1 --kd_ratio 1 --kd_loss_type l1 --tiny-rescale --tensorboard_dir {args.tensorboard_dir}'

    logging.info(command)
    os.system(command)
    T5 = time.perf_counter()
    # logging.info('Training: %ss' % (T5 - T4))
    logging.info("===================== Finish! ======================")
