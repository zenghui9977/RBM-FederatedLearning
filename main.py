import numpy as np
import torch

from rbm import *
from load_foursquare_data import *
from federatedlearning import *
from utils import *
from divide_user_fs_5 import divide_user

# 设置训练设备
CUDA = torch.cuda.is_available()
CUDA_DEVICE = 0
if CUDA:
    torch.cuda.set_device(CUDA_DEVICE)

# 定义文件名变量
DATA_FOLDER = 'data/'
FOURSQUARE_FILE = 'sub_users5_items5.txt'
RESULT_FOLDER = 'result/fs_fl_rbm/'


dd = 100 / 1000.0
UD = 20

[(user_num, item_num), pois_cordis, (tra_buys, tes_buys), (tra_dist, tes_dist)] = \
    load_data(os.path.join(DATA_FOLDER, FOURSQUARE_FILE), 'test', -1,dd, int(UD/dd))


train_set = build_interact_matrix(user_num, item_num, tra_buys)

EPOCH = 5 # Federated learning parameter E
BACTH_SIZE = 1 # Federated learning parameter B
CR = 1 # Communication round - CR

VISIBLE_UNIT = item_num
HIDDEN_UNIT = 128
CD_K = 4

Top_K = [5, 10, 15, 20]

params = Params(b=BACTH_SIZE, c=0.2, client_num=user_num, CUDA=CUDA,
                e=EPOCH, max_communication_round=CR,
                rbm_visible_unit=VISIBLE_UNIT, rbm_hidden_unit= HIDDEN_UNIT, rbm_k=CD_K)

area_best, globalbest = FederatedLearning(params, train_set, tes_buys, tra_buys, divide_user, Top_K)


# 全局最优结果输出
globalbest.print_best_metrics(EPOCH)
# 分区最优结果输出
for i in range(len(divide_user)):
    area_best[i].print_best_metrics(i)

save_best_area(RESULT_FOLDER, 'fs_rbm_result.txt', [globalbest], EPOCH)
save_best_area(RESULT_FOLDER, 'fs_5area_result.txt', area_best, EPOCH)

