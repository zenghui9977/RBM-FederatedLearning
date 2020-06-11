from load_foursquare_data import *
from utils import *
from rbm import *
from federatedlearning import *
from divide_user_fs_5 import divide_user
import time


# 设置训练设备
CUDA = torch.cuda.is_available()
CUDA_DEVICE = 0
if CUDA:
    torch.cuda.set_device(CUDA_DEVICE)

# 定义文件名变量
DATA_FOLDER = 'data/'
FOURSQUARE_FILE = 'sub_users5_items5.txt'
RESULT_FOLDER = 'result/fs_rbm/'

# user_num, item_num, \
# item_aliases_dict, user_aliases_dict, \
# train_set, test_set, train_set_item = load_foursquare_data(DATA_FOLDER + FOURSQUARE_FILE)

dd = 100 / 1000.0
UD = 20

[(user_num, item_num), pois_cordis, (tra_buys, tes_buys), (tra_dist, tes_dist)] = \
    load_data(os.path.join(DATA_FOLDER, FOURSQUARE_FILE), 'test', -1, dd, int(UD/dd))

train_set = build_interact_matrix(user_num, item_num, tra_buys)
# test_set = build_interact_matrix(user_num, item_num, tes_buys)


EPOCH = 5
BACTH_SIZE = 64
VISIBLE_UNIT = item_num
HIDDEN_UNIT = 128
CD_K = 2


Top_K = [5, 10, 15, 20]

weights, visible_bias, hidden_bias = init_global_model((VISIBLE_UNIT, HIDDEN_UNIT))
rbm = RBM(VISIBLE_UNIT, HIDDEN_UNIT, CD_K, weights, visible_bias, hidden_bias, use_cuda=CUDA)

train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=BACTH_SIZE)

precision_list, recall_list, f_measure_list, ndcg_list, hit_num_list = [], [], [], [], []

area_num = len(divide_user)
area_best = [GlobalBest(Top_K) for i in range(area_num)]
globalbest = GlobalBest(Top_K)

epoch_train_time = []

for epoch in range(EPOCH):
    epoch_error = 0.0
    t0 = time.time() # 开始训练时间
    print('==========================')
    print('training......')
    for _, batch in enumerate(train_data_loader):
        if CUDA:
            batch = batch.cuda()
        batch_error = rbm.contrastive_divergence(batch)
        epoch_error += batch_error
    print('Epoch %d' % epoch)
    print('\t Epoch Error: %.4f' % epoch_error)
    t1 = time.time() # 训练结束时间
    epoch_train_time.append(t1 - t0)
    print('train time %0.4f' % (t1 - t0))
    print('==========================')
    print('predict......')
    rec_top_k_list = predict_model(rbm.weights, rbm.visible_bias, rbm.hidden_bias,
                                   user_num, item_num, train_set, tra_buys, CUDA, BACTH_SIZE)
    pre, rec, f_mea, ndcg, hit_num, _map = compute_metrice_according_to_rec_list(rec_top_k_list, tes_buys, item_num,
                                                                                 Top_K)

    globalbest.update_best_metrics(Top_K, pre, rec, f_mea, ndcg, hit_num, _map, epoch)

    for i in range(area_num):
        area_divide_user = divide_user[i]
        pre, rec, f_mea, ndcg, hit_num, _map = eval_area_metrics(rec_top_k_list, area_divide_user, item_num, tes_buys, Top_K)
        area_best[i].update_best_metrics(Top_K, pre, rec, f_mea, ndcg, hit_num, _map, epoch)

    # 每一轮的全局最优结果输出
    globalbest.print_best_metrics(epoch)
    # 每一轮的分区最优结果输出
    for i in range(area_num):
        area_best[i].print_best_metrics(i)

globalbest.print_best_metrics(EPOCH)
for i in range(area_num):
    area_best[i].print_best_metrics(i)
save_best_area(RESULT_FOLDER, 'gw_rbm_result.txt', [globalbest], EPOCH)
save_best_area(RESULT_FOLDER, 'gw_5area_result.txt', area_best, EPOCH)

print('total train time is %0.4f' % sum(epoch_train_time))
