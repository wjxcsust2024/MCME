'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui1109@stud.tjut.edu.cn, liuhui@ieee.org
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import datetime
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import util.misc as utils
from engine import train_one_epoch
from datasets import create_dataset
import cv2
from eval.evaluate import eval
from util.logger import get_logger
from tqdm import tqdm
from semseg.Dice_BCEwithLogits_1 import BCE_Dice as SoftDiceLoss
from mmengine.optim.scheduler.lr_scheduler import PolyLR
from semseg.Net import Net #主网络调用处

def get_args_parser():
    parser = argparse.ArgumentParser('LIDAR FOR MULTI-MODAL CRACK SEGMENTATION', add_help=False)
    # 数据路径参数
    parser.add_argument('--dataset_path', default="../Multimodal_Crack_Dataset/Multimodal_Crack_Dataset/CrackDepth",  #数据集调入路径，不同数据更改，这里需要更改，还需要修改下面的模态输入格式
                        help='path to images')
    parser.add_argument('--scan_list_json_path', default="pretrain", # ./pre_scan/scan_list/CracskDepth_scan_orders_dict_patch8.json
                        help='path to images. If you want to pretrain to generate the mask, change its value to \'pretrain\'')  # 扫描顺序列表路径
    parser.add_argument('--inference_mask', default=True,
                        help='Inference mask. If you want to pretrain to generate the mask, change its value to \'True\'')  # 是否进行掩码推理
    # 训练参数
    parser.add_argument('--batch_size_train', type=int, default=1, help='train input batch size')
    parser.add_argument('--batch_size_test', type=int, default=1, help='test input batch size')
    parser.add_argument('--modals', nargs='+', default=['RGB','dep'], help='modal name for process')  # 多模态输入，更改数据集，这里也需要更改模态格式 #只需要更改后面那个模态格式，红外模态是inf
    # 优化器参数#'RGB','0','45','90','135','dop'
    parser.add_argument('--lr_scheduler', type=str, default='PolyLR')  # 学习率调度器类型
    parser.add_argument('--lr', default=1e-3, type=float)  # 初始学习率
    parser.add_argument('--min_lr', default=1e-6, type=float)  # 最小学习率
    parser.add_argument('--weight_decay', default=0.01, type=float)  # 权重衰减
    # 训练配置
    parser.add_argument('--epochs', default=40, type=int)  # 总训练轮数
    parser.add_argument('--start_epoch', default=0, type=int)  # 起始轮数
    parser.add_argument('--lr_drop', default=30, type=int)  # 学习率下降的epoch数
    # 优化器选择
    parser.add_argument('--sgd', action='store_true')  # 是否使用SGD优化器
    # 输出配置
    parser.add_argument('--output_dir', default='./checkpoints/weights', help='save path for weights') #权重保存路径
    parser.add_argument('--device', default='cuda', help='device to use')
    # 随机种子
    parser.add_argument('--seed', default=42, type=int)
    # 数据集配置
    parser.add_argument('--dataset_mode', type=str, default='crack')
    parser.add_argument('--serial_batches', action='store_true',
                        help='if true, takes images in order to make batches,  takes them randomly') #是否按顺序获取批次
    parser.add_argument('--num_threads', default=1, type=int, help='num_workers') #数据加载线程数
    parser.add_argument('--phase', type=str, default='train', help='train, test, etc') #阶段：train/test等
    # 图像尺寸
    parser.add_argument('--load_width', type=int, default=512, help='load image width') #加载图像宽度
    parser.add_argument('--load_height', type=int, default=512, help='load image height') #加载图像高度

    return parser

def main(args):
    checkpoints_path = "./checkpoints"  # 检查点保存路径
    curTime = time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime(time.time())) # 当前时间戳
    dataset_name = (args.dataset_path).split('/')[-1] # 从路径中提取数据集名称
    modal_num = len(args.modals) # 模态数量
    # 构建模态名称字符串
    modals_name = ''
    for i in range(len(args.modals)):
        modals_name = modals_name + '_' + (args.modals)[i]
    # 创建处理文件夹路径
    process_floder_path = os.path.join(checkpoints_path, curTime + '_Dataset->' + dataset_name + '_modals->' + modals_name)
    # 创建文件夹
    if not os.path.exists(process_floder_path):
        os.makedirs(process_floder_path)
    else:
        print("create process folder error!")
    # 创建日志记录器
    log_train = get_logger(process_floder_path, 'train')
    log_test = get_logger(process_floder_path, 'test')
    log_eval = get_logger(process_floder_path, 'eval')
    # 记录参数信息
    log_train.info("args -> " + str(args))
    log_train.info("args: dataset -> " + str(args.dataset_path))
    log_train.info("processing modal -> " + str(args.modals))
    log_train.info("number of modal -> " + str(modal_num))
    print('processing modal -> ', args.modals)
    print('number of modal -> ', modal_num)

    # 设备设置和随机种子
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 构建模型和损失函数
    criterion = SoftDiceLoss()
    model = Net('CMNeXt-B2',1,args.modals)    #这里的B2确定了解码器每层模块数量，
    model.to(device)
    # 创建训练数据集
    args.batch_size = args.batch_size_train
    train_dataLoader = create_dataset(args)
    dataset_size = len(train_dataLoader)
    print('The number of training images = %d' % dataset_size)
    log_train.info('The number of training images = %d' % dataset_size)
    # 设置优化器参数
    param_dicts = [
        {
            "params":
                [p for n, p in model.named_parameters()], # 所有模型参数
            "lr": args.lr, # 学习率
        },
    ]
    # 选择优化器
    if args.sgd:
        print('use SGD!')
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        print('use AdamW!')
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    # 设置学习率调度器
    lr_scheduler = None
    if args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    elif args.lr_scheduler == 'CosLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-5)
    elif args.lr_scheduler == 'PolyLR':
        lr_scheduler = PolyLR(optimizer, eta_min=args.min_lr, begin=args.start_epoch, end=args.epochs)
    # 创建输出目录
    output_dir = args.output_dir + '/' + 'Dataset->' + dataset_name + '_modals->' + modals_name
    if args.output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir)

    print("Start processing! ")
    log_train.info("Start processing! ")
    start_time = time.time() # 记录开始时间
    # 初始化最佳指标
    max_F1 = 0
    max_Metrics = {'epoch': 0, 'mIoU': 0, 'ODS': 0, 'OIS': 0, 'F1': 0, 'Precision': 0, 'Recall': 0}
    # 训练循环
    for epoch in range(args.start_epoch, args.epochs):
        args.phase = 'train'
        print("---------------------------------------------------------------------------------------")
        print("training epoch start -> ", epoch)
        # 训练一个epoch
        train_one_epoch(
            model, criterion, train_dataLoader, optimizer, epoch, args, log_train,None)
        lr_scheduler.step() # 更新学习率
        # 保存检查点
        if args.output_dir:
            # 保留每一轮的权重
            # checkpoint_paths = [output_dir / 'checkpoint.pth']
            # if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 1 == 0:
            #     checkpoint_paths.append(output_dir / f'checkpoint{epoch}.pth')
            # for checkpoint_path in checkpoint_paths:

            # 保留最后一轮的权重
            checkpoint_path = output_dir / 'checkpoint.pth'
            utils.save_on_master({
                'model': model.state_dict(), # 模型权重
                'optimizer': optimizer.state_dict(), # 优化器状态
                'lr_scheduler': lr_scheduler.state_dict(), # 学习率调度器状态
                'epoch': epoch, # 当前epoch
                'args': args, # 参数
            }, checkpoint_path)
        print("training epoch finish -> ", epoch)
        print("---------------------------------------------------------------------------------------")

        # 测试阶段
        print("testing epoch start -> ", epoch)
        # 设置结果保存路径
        results_path = 'Dataset->' + dataset_name + '_modal->' + modals_name
        save_root = f'./results/{results_path}/results_'   #此处设置不同数据集测试的结果图片集名称
        args.phase = 'test'
        args.batch_size = args.batch_size_test
        # 创建测试数据集
        test_dl = create_dataset(args)
        pbar = tqdm(total=len(test_dl), desc=f"Initial Loss: Pending")
        # 创建保存目录
        if not os.path.isdir(save_root):
            os.makedirs(save_root)
        # 开始测试
        with torch.no_grad(): # 禁用梯度计算
            for batch_idx, (data) in enumerate(test_dl):
                # 准备多模态输入
                modal_imgs = []
                for i in range(modal_num):
                    items = list(data.items())
                    key, value = items[i]
                    modal_imgs.append(value.to(torch.device(args.device)))
                # 获取目标标签
                target = data["label"]
                target = target.to(dtype=torch.int64).cuda()
                # 前向传播
                out = model(modal_imgs) # ← 在这里传递参数
                loss = criterion(out, target.float()) # 计算损失
                # 转换为numpy数组
                target = target[0, 0, ...].cpu().numpy()
                out = out[0, 0, ...].cpu().numpy()
                root_name = data["image_path"][0].split("/")[-1][0:-4]  # 获取文件名
                # 归一化到0-255范围
                target = 255 * (target / np.max(target))
                out = 255 * (out / np.max(out))
                # 记录日志
                log_test.info('----------------------------------------------------------------------------------------------')
                log_test.info("loss -> " + str(loss))
                log_test.info(str(os.path.join(save_root, "{}_lab.png".format(root_name))))
                log_test.info(str(os.path.join(save_root, "{}_pre.png".format(root_name))))
                log_test.info('----------------------------------------------------------------------------------------------')
                # 保存结果图像
                cv2.imwrite(os.path.join(save_root, "{}_lab.png".format(root_name)), target)
                cv2.imwrite(os.path.join(save_root, "{}_pre.png".format(root_name)), out)
                # 更新进度条
                pbar.set_description(f"Loss: {loss.item():.4f}")
                pbar.update(1)

        pbar.close()
        log_test.info("model -> " + str(epoch) + " test finish!")
        log_test.info('----------------------------------------------------------------------------------------------')
        print("testing epoch finish -> ", epoch)
        print("---------------------------------------------------------------------------------------")

        # 评估阶段
        print("evaluating epoch start -> ", epoch)
        # 评估模型性能
        metrics = eval(log_eval, save_root, epoch)
        # 打印评估指标
        for key, value in metrics.items():
            print(str(key) + ' -> ' + str(value))
        # 更新最佳模型
        if(max_F1 < metrics['F1']):
            max_Metrics = metrics
            max_F1 = metrics['F1']
            checkpoint_paths = [output_dir / f'checkpoint_best.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
            log_train.info("\nupdate and save best model -> " + str(epoch))
            print("\nupdate and save best model -> ", epoch)

        print("evaluating epoch finish -> ", epoch)
        print('\nmax_F1 -> ' + str(max_Metrics['F1']) + '\nmax Epoch -> ' + str(max_Metrics['epoch']))
        print("---------------------------------------------------------------------------------------")
        # 记录评估日志
        log_eval.info("evaluating epoch finish -> " + str(epoch))
        log_eval.info('\nmax_F1 -> ' + str(max_Metrics['F1']) + '\nmax Epoch -> ' + str(max_Metrics['epoch']))
        log_eval.info("---------------------------------------------------------------------------------------")
    # 记录最终的最佳指标
    for key, value in max_Metrics.items():
        log_eval.info(str(key) + ' -> ' + str(value))
    log_eval.info('\nmax_F1 -> ' + str(max_Metrics['F1']) + '\nmax Epoch -> ' + str(max_Metrics['epoch']))
    # 计算总训练时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Process time {}'.format(total_time_str))
    log_train.info('Process time {}'.format(total_time_str))

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser('LIDAR FOR CRACK', parents=[get_args_parser()])
    args = parser.parse_args()
    # 运行主函数
    main(args)
