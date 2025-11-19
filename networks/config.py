import argparse
import os


def train_para_set():
    home_dir = r"/".join(os.getcwd().split("\\"))
    parent_dir = (os.path.dirname(home_dir) +'/ours') # 获取当前目录的上一级目录路径

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    parser = argparse.ArgumentParser()

    # change dataset need rewrite
    ######################################################################################
    parser.add_argument('--parent_dir', type=str, default=parent_dir + '/data/X/', help='dataset use to apply path')
    parser.add_argument('--snapshot_path', type=str, default=parent_dir + "/result/",
                        help='save model_name')
    parser.add_argument('--num_classes', type=str, default=5, help='use to prediction')

    parser.add_argument('--epoch_num', type=str, default=0, help='use to record eppoch nums')
    parser.add_argument('--continue_train', type=str, default=0, help='use to load checkpoint')
    parser.add_argument('--continue_train_iter', type=str, default=7500, help='use to load checkpoint model pth')
    parser.add_argument('--slice_num', type=int, default=32, help='number of labeled volume')
    parser.add_argument('--num', type=int, default=40, help='number of labeled volume')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    #######################################################################################

    parser.add_argument('--adaptive_ratio', type=float, default=0.5, help='自适应补丁数量的比例')
    parser.add_argument('--swap_probability', type=float, default=0.5, help='补丁置换的概率')
    parser.add_argument('--throughput', action='store_true',
                        help='Test throughput only')


    # patch size
    parser.add_argument('--patch_size', type=int, default=56, help='patch_size')
    parser.add_argument('--h_size', type=int, default=4, help='h_size')
    parser.add_argument('--w_size', type=int, default=4, help='w_size')
    # top num

    parser.add_argument('--max_iterations', type=int, default=20000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
    # parser.add_argument('--labeled_bs', type=int, default=1, help='labeled_batch_size per gpu')
    parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--sliceseed', type=int, default=0, help='random seed')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--split', type=str, default='train', help='datalist to use')


    ### costs
    parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
    parser.add_argument('--slice_strategy', type=int, default=2, help='ema_decay') #20/
    parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
    parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')

    return parser.parse_args()


def test_para_set():
    home_dir = r"/".join(os.getcwd().split("\\"))
    parent_dir = (os.path.dirname(home_dir) + '/ours')  # 获取当前目录的上一级目录路径
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    parser = argparse.ArgumentParser()
    # change dataset need rewrite
    ######################################################################################
    parser.add_argument('--snapshot_path', type=str, default=parent_dir + "/result/X/",
                        help='Name of model save')
    parser.add_argument('--test_save_path', type=str,
                        default=parent_dir + "/result/X_post/",
                        help='Name of inference save')
    parser.add_argument('--parent_dir', type=str, default=parent_dir + '/data/X/', help='dataset use to apply path')
    parser.add_argument('--num_classes', type=str, default=5, help='use to prediction')
    ######################################################################################
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')

    # parser.add_argument('--num', type=int, default=10, help='number of labeled volume')

    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--sliceseed', type=int, default=0, help='random seed')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--modeleffe', type=int, default=1, help='model to use')
    # parser.add_argument('--mid_iterations', type=int, default=4000)
    parser.add_argument('--max_iteration', type=int, default=20000)
    parser.add_argument('--iteration_step', type=int, default=500)
    parser.add_argument('--split', type=str, default='test', help='testlist to use')
    parser.add_argument('--min_iteration', type=int, default=1000)

    return parser.parse_args()
