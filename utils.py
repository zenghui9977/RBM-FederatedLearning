import numpy as np
import os
import pickle
import torch
from rbm import RBM_EVAL
from metrics import *
import datetime


def save_as_csv(data_dir, file_name, data):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    np.savetxt(data_dir+file_name, data, delimiter=',')

def save_as_npy(data_dir, file_name, data):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    np.save(data_dir+file_name, data)

def save_as_pkl(data_dir, filename, data):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    filehandler = open(data_dir + "/" + filename + ".pkl", "wb")
    pickle.dump(data, filehandler)
    filehandler.close()

def save_as_pt(data_dir, filename, data):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    torch.save(data, data_dir + filename)

def read_from_npy(data_dir, file_name):
    return np.load(data_dir + file_name).item()

def read_from_pkl(save_dir, filename):
    return pickle.load(open(save_dir + filename , 'rb'))

def read_from_pt(save_dir, filename):
    return torch.load(save_dir + filename)


def build_interact_matrix(user_num, item_num, values):
    interact_matrix = torch.zeros(user_num, item_num)
    for u in range(user_num):
        u_interact_vector = values[u]
        for i in u_interact_vector:
            interact_matrix[u][i] += 1

    return interact_matrix

def predict_model(weights, visible_bias, hidden_bias, user_num, item_num, train_set, train_set_item, CUDA, rec_batch):
    rbm_eval = RBM_EVAL(weights, visible_bias, hidden_bias)
    rec = torch.zeros([user_num, item_num])
    for i, batch in enumerate(train_set):
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

    return rec_top_k_list


def compute_metrice_according_to_rec_list(rec_top_k_list, test_set, item_num, Top_K):

    user_num = len(rec_top_k_list)
    # evaluate the metrics
    # print(user_num, len(test_set))

    precision_mean, recall_mean, f_measure_mean, ndcg_mean, hit_num, map_mean = [], [], [], [], [], []
    for top_k in Top_K:
        pre_list, rec_list, f_me_list, ndcg_list, map_list = [], [], [], [],[]
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


def eval_area_metrics(rec_top_k_list, area_user_id, item_num, test_set, Top_K):
    area_rec_top_k_list = [rec_top_k_list[u] for u in area_user_id]

    test_data = [test_set[u] for u in area_user_id]

    precision_mean, recall_mean, f_measure_mean, ndcg_mean, hit_num, map_mean = compute_metrice_according_to_rec_list(area_rec_top_k_list, test_data, item_num, Top_K)
    return precision_mean, recall_mean, f_measure_mean, ndcg_mean, hit_num, map_mean

class Params:
    def __init__(self, b, c, client_num, CUDA, e, max_communication_round, rbm_visible_unit, rbm_hidden_unit, rbm_k):
        self.b = b
        self.c = c
        self.client_num = client_num
        self.CUDA = CUDA
        self.e = e
        self.max_communication_round = max_communication_round

        # RBM parameters
        self.rbm_visible_unit = rbm_visible_unit
        self.rbm_hidden_unit = rbm_hidden_unit
        self.rbm_k = rbm_k


class GlobalBest():
    def __init__(self, at_nums):
        # at_nums = [5, 10, 15, 20]
        ranges = np.arange(len(at_nums))
        val_float = np.array([0.0 for _ in ranges])
        epoch_int = np.array([0 for _ in ranges])

        self.best_precision = val_float.copy()
        self.best_recall = val_float.copy()
        self.best_f_measure = val_float.copy()
        self.best_ndcg = val_float.copy()
        self.best_hit_ratio = val_float.copy()
        self.best_map = val_float.copy()

        self.best_precision_epoch = epoch_int.copy()
        self.best_recall_epoch = epoch_int.copy()
        self.best_f_measure_epoch = epoch_int.copy()
        self.best_ndcg_epoch = epoch_int.copy()
        self.best_hit_ratio_epoch = epoch_int.copy()
        self.best_map_epoch = epoch_int.copy()

    def update_best_metrics(self, at_nums, precision_list, recall_list, f_measure_list, ndcg_list, hit_ratio_list, map_list,epoch):
        # at_nums = len([5,10,15,20])
        for k in range(len(at_nums)):
            if precision_list[k] > self.best_precision[k]:
                self.best_precision[k] = precision_list[k]
                self.best_precision_epoch[k] = epoch
            if recall_list[k] > self.best_recall[k]:
                self.best_recall[k] = recall_list[k]
                self.best_recall_epoch[k] = epoch
            if f_measure_list[k] > self.best_f_measure[k]:
                self.best_f_measure[k] = f_measure_list[k]
                self.best_f_measure_epoch[k] = epoch
            if ndcg_list[k] > self.best_ndcg[k]:
                self.best_ndcg[k] = ndcg_list[k]
                self.best_ndcg_epoch[k] = epoch
            if hit_ratio_list[k] > self.best_hit_ratio[k]:
                self.best_hit_ratio[k] = hit_ratio_list[k]
                self.best_hit_ratio_epoch[k] = epoch
            if map_list[k] > self.best_map[k]:
                self.best_map[k] = map_list[k]
                self.best_map_epoch[k] = epoch

    def fun_obtain_best(self, epoch):
        """
        :param epoch:
        :return: 由最优值组成的字符串
        """
        # def truncate4(x):
        #     """
        #     把输入截断为六位小数
        #     :param x:
        #     :return: 返回一个字符串而不是list里单项是字符串，
        #     这样打印的时候就严格是小数位4维，并且没有字符串的起末标识''
        #     """
        #     return ', '.join(['%0.4f' % k for k in x])
        amp = 100
        one = '\t'
        two = one * 2
        a = one + '-----------------------------------------------------------------'
        # 指标值、最佳的epoch
        b = one + 'All values is the "best * {v1}" on area {v2}: | {v3}'\
            .format(v1=amp, v2=epoch, v3=datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S"))
        c = two + 'Precision       = [{val}], '.format(val=(self.best_precision * amp)) + \
            two + '{val}'.format(val=self.best_precision_epoch)
        d = two + 'Recall    = [{val}], '.format(val=(self.best_recall * amp)) + \
            two + '{val}'.format(val=self.best_recall_epoch)
        # e = two + 'Precision = [{val}], '.format(val=(self.best_precis * amp)) + \
        #     two + '{val}'.format(val=self.best_epoch_precis)    # 不输出precision
        f = two + 'F1-score  = [{val}], '.format(val=(self.best_f_measure * amp)) + \
            two + '{val}'.format(val=self.best_f_measure_epoch)
        g = two + 'MAP       = [{val}], '.format(val=(self.best_map * amp)) + \
            two + '{val}'.format(val=self.best_map_epoch)
        h = two + 'NDCG      = [{val}], '.format(val=(self.best_ndcg * amp)) + \
            two + '{val}'.format(val=self.best_ndcg_epoch)

        i = two + 'Hit_Ratio      = [{val}], '.format(val=(self.best_hit_ratio * amp)) + \
            two + '{val}'.format(val=self.best_hit_ratio_epoch)

        return '\n'.join([a, b, c, d, f, g, h, i])

    def print_best_metrics(self, epoch):

        print(self.fun_obtain_best(epoch))


def save_best_area(path, file_name, area_best, epoch):
    # 建立目录、文件名
    if os.path.exists(path):
        print('\t\tdir exists: {v1}'.format(v1=path))
    else:
        os.makedirs(path)
        print('\t\tdir is made: {v1}'.format(v1=path))

    file_path = os.path.join(path, file_name)
    f = open(file_path, 'a')

    for i in range(len(area_best)):
        best_info = area_best[i].fun_obtain_best(epoch)
        f.write(best_info)

    f.close()