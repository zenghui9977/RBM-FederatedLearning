import torch
import numpy as np
from rbm import RBM, RBM_EVAL
from metrics import *
from utils import *
import time


def init_global_model(weight_shape):
    weights = torch.randn(weight_shape) * 0.1
    visible_bias = torch.ones(weight_shape[0]) * 0.5
    hidden_bias = torch.zeros(weight_shape[1])
    return weights, visible_bias, hidden_bias


def evaluate_global_model(weights, visible_bias, hidden_bias,
                          user_num, item_num, train_data, test_set, train_set_item,
                          CUDA, rec_batch):
    rbm_eval = RBM_EVAL(weights, visible_bias, hidden_bias)
    rec = torch.zeros([user_num, item_num])

    for i, batch in enumerate(train_data):
        if CUDA:
            batch = batch.cuda()
        pro = rbm_eval.model_output(batch)

        rec[i * rec_batch: i * rec_batch + len(batch)] = pro

    rec_pro, rec_index = rec.sort(1, descending=True)

    top_k = 100
    rec_top_k_list = []

    for u in range(user_num):
        u_train_set = set(train_set_item[u])
        re_sort_index = rec_index[u].cpu().numpy()
        temp = []
        for i in range(item_num):
            if re_sort_index[i] not in u_train_set:
                temp.append(re_sort_index[i])
            if len(temp) == top_k:
                break
        rec_top_k_list.append(temp)

    # evaluate the metrics
    Top_K = [5, 10, 15, 20]

    precision_mean, recall_mean, f_measure_mean, ndcg_mean, hit_num, map_mean = [], [], [], [], [], []
    for top_k in Top_K:
        pre_list, rec_list, f_me_list, ndcg_list, map_list = [], [], [], [], []
        hit_num_list = 0
        for u in range(user_num):
            pred = rec_top_k_list[u][:top_k]
            gt = test_set[u]
            pre = precision(gt, pred)
            rec = recall(gt, pred)
            f_me = f_measure(gt, pred)
            ndcg = getNDCG(gt, pred)
            _map = mapk(gt, pred)

            pre_list.append(pre)
            rec_list.append(rec)
            f_me_list.append(f_me)
            ndcg_list.append(ndcg)
            map_list.append(_map)

            hit_num_list += hit_num_k(gt, pred)

        hit_num_list = hit_num_list / user_num
        # print('top_k is %d' % top_k)
        # print('[precision, recall, f_measure, NDCG, hit_num] --> [%f, %f, %f, %f, %f]' % (np.mean(pre_list), np.mean(rec_list), np.mean(f_me_list), np.mean(ndcg_list), hit_num_list))


        precision_mean.append(np.mean(pre_list))
        recall_mean.append(np.mean(rec_list))
        f_measure_mean.append(np.mean(f_me_list))
        ndcg_mean.append(np.mean(ndcg_list))
        map_mean.append(np.mean(map_list))
        hit_num.append(hit_num_list)

    return precision_mean, recall_mean, f_measure_mean, ndcg_mean, hit_num, map_mean


def FederatedLearning(params, train_data, test_set, train_set_item, divide_user, Top_K):
    print('Federated Learning with local RBM')
    print('\t Initialize global model ......')
    global_model_weights, global_model_visible_bias, global_model_hidden_bias = \
        init_global_model((params.rbm_visible_unit, params.rbm_hidden_unit))

    # load the train data
    rec_batch = 64
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=rec_batch)

    precision_each_round, recall_each_round, f_measure_each_round, ndcg_each_round, hit_num_each_round = [], [], [], [], []

    area_num = len(divide_user)
    area_best = [GlobalBest(Top_K) for i in range(area_num)]
    globalbest = GlobalBest(Top_K)

    round_train_time = []
    print('\t Start Federated Learning communication round ......')
    for r in range(params.max_communication_round):
        t0 = time.time() # communicatoin round r training start
        print('\t \t Communication round %d training:' % r)
        # 用户随机采样，生成乱序的的用户排序，取前c * N个
        perm = np.random.permutation(params.client_num)
        sample_user_num = int(params.c * params.client_num)
        sample_user = perm[:sample_user_num].tolist()

        weight_accountant, visible_bias_accountant, hidden_bias_accountant = [], [], []
        error_accountant = []
        # 遍历采样得到的每一个节点（用户），每一个节点的数据训练得到一个RBM模型
        for u in sample_user:
            u_train_data = torch.zeros([params.b, params.rbm_visible_unit])
            u_train_data[0] = train_data[u]
            # download global model and initialize local RBM
            u_rbm = RBM(params.rbm_visible_unit, params.rbm_hidden_unit, params.rbm_k,
                        global_model_weights, global_model_visible_bias, global_model_hidden_bias,
                        use_cuda=params.CUDA)
            for epoch in range(params.e):
                epoch_error = 0.0
                if params.CUDA:
                    u_train_data = u_train_data.cuda()
                epoch_error = u_rbm.contrastive_divergence(u_train_data)
            error_accountant.append(epoch_error)
            weight_accountant.append(u_rbm.weights)
            visible_bias_accountant.append(u_rbm.visible_bias)
            hidden_bias_accountant.append(u_rbm.hidden_bias)

        # 模型聚合
        print('\t \t update the global model')
        global_model_weights = 1 / sample_user_num * sum(weight_accountant)
        global_model_visible_bias = 1 / sample_user_num * sum(visible_bias_accountant)
        global_model_hidden_bias = 1 / sample_user_num * sum(hidden_bias_accountant)
        t1 = time.time()
        round_train_time.append(t1 - t0)
        print('\t \t train time is %0.4f' % (t1 - t0))
        print('\t \t evaluate the global model')
        print('********************')
        print('the total error is %0.4f' % sum(error_accountant))

        rec_top_k_list = predict_model(global_model_weights,global_model_visible_bias, global_model_hidden_bias,
                                       params.client_num, params.rbm_visible_unit,
                                       train_data, train_set_item, params.CUDA, rec_batch)

        pre, rec, f_mea, ndcg, hit_num, _map = compute_metrice_according_to_rec_list(rec_top_k_list, test_set, params.rbm_visible_unit,
                                                                                     Top_K)
        globalbest.update_best_metrics(Top_K, pre, rec, f_mea, ndcg, hit_num, _map, r)
        for i in range(area_num):
            area_divide_user = divide_user[i]
            pre, rec, f_mea, ndcg, hit_num, _map = eval_area_metrics(rec_top_k_list, area_divide_user, params.rbm_visible_unit, test_set,
                                                               Top_K)
            area_best[i].update_best_metrics(Top_K, pre, rec, f_mea, ndcg, hit_num, _map, r)
        print('********************')

    print('total train time is %0.4f' % sum(round_train_time))
    return area_best, globalbest
