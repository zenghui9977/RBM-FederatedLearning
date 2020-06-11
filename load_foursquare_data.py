import numpy as np
import pandas as pd
import random
import torch
from collections import Counter
from math import sin, cos, sqrt, asin


def load_foursquare_data(dataset_dir):
    print('Loading original data ......')
    pois = pd.read_csv(dataset_dir, sep=' ')

    all_user_pois = [[i for i in upois.split('/')] for upois in pois['u_pois']]
    all_users = [i for i in pois['u_id']]
    all_train_data = [item for upois in all_user_pois for item in upois]
    all_items = set(all_train_data)

    user_num, item_num = len(all_users), len(set(all_train_data))
    print('\tusers, items:  = {v1}, {v2}'.format(v1=user_num, v2=item_num))

    # 0-n的重新映射
    item_aliases_dict = dict(zip(all_items, range(item_num)))
    user_aliases_dict = dict(zip(all_users, range(user_num)))

    # 映射为Index
    item_aliases_list = [[item_aliases_dict[i] for i in item] for item in all_user_pois]
    user_aliases_list = [user_aliases_dict[i] for i in all_users]

    print('\tsplit train data set and test data set ...')
    train_set = []
    test_set = []
    for u in user_aliases_list:
        the_user_item_list = item_aliases_list[u]
        the_user_train = the_user_item_list[:-1]
        the_user_test = [the_user_item_list[-1]]

        train_set.append(the_user_train)
        test_set.append(the_user_test)

    print('\tnegativa sample')

    train_set_with_mask, train_set_mask = full_data_buys_mask(train_set, tail=[item_num])
    test_set_with_mask, test_set_mask = full_data_buys_mask(test_set, tail=[item_num])

    train_set_neg_sample = fun_random_neg_masks_tra(item_num, train_set_with_mask)
    test_set_neg_sample = fun_random_neg_masks_tes(item_num, train_set_with_mask, test_set_with_mask)

    combined_train_set = combine_2_array(train_set_with_mask, train_set_neg_sample)
    test_set = combine_2_array(test_set_with_mask, test_set_neg_sample)

    user_num = len(combined_train_set)

    train_set = torch.zeros([user_num, item_num])
    train_set_item = dict()

    for u in range(user_num):
        the_user_item_list = combined_train_set[u]

        train_set_item[u] = set(the_user_item_list)
        if item_num in train_set_item[u]:
            train_set_item[u].remove(item_num)

        for l in the_user_item_list:
            if l != item_num:
                train_set[u][l] += 1


    return user_num, item_num, \
            item_aliases_dict, user_aliases_dict, \
            train_set, test_set, train_set_item



def full_data_buys_mask(all_user_buys, tail):
    # 将train/test中序列补全为最大长度，补的idx值=item_num. 为了能在theano里对其进行shared。
    # tail, 添加的。商品索引是0~item_num-1，所以该值[item_num]是没有对应商品实物的。
    us_lens = [len(ubuys) for ubuys in all_user_buys]
    len_max = max(us_lens)
    us_buys = [ubuys + tail * (len_max - le) for ubuys, le in zip(all_user_buys, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_buys, us_msks


def fun_random_neg_masks_tra(item_num, tras_mask):

    # 从num件商品里随机抽取与每个用户的购买序列等长且不在已购买商品里的标号。后边补全的负样本用虚拟商品[item_num]

    us_negs = []
    for utra in tras_mask:     # 每条用户序列
        unegs = []
        for i, e in enumerate(utra):
            if item_num == e:                    # 表示该购买以及之后的，都是用虚拟商品[item_num]来补全的
                unegs += [item_num] * (len(utra) - i)   # 购买序列里对应补全商品的负样本也用补全商品表示
                break
            j = random.randint(0, item_num - 1)  # 负样本在商品矩阵里的标号
            while j in utra:                     # 抽到的不是用户训练集里的。
                j = random.randint(0, item_num - 1)
            unegs += [j]
        us_negs.append(unegs)
    return us_negs


def fun_random_neg_masks_tes(item_num, tras_mask, tess_mask):

    # 从num件商品里随机抽取与测试序列等长且不在训练序列、也不再测试序列里的标号

    us_negs = []
    for utra, utes in zip(tras_mask, tess_mask):
        unegs = []
        for i, e in enumerate(utes):
            if item_num == e:                   # 尾部补全mask
                unegs += [item_num] * (len(utes) - i)
                break
            j = random.randint(0, item_num - 1)
            while j in utra or j in utes:         # 不在训练序列，也不在预测序列里。
                j = random.randint(0, item_num - 1)
            unegs += [j]
        us_negs.append(unegs)
    return us_negs

def combine_2_array(a, b):
    return np.append(a, b, axis=0)


def unique_the_list(list_with_some_same_ele):
    return np.array(list(set(tuple(t) for t in list_with_some_same_ele)))


def cal_dis(lat1, lon1, lat2, lon2, dd, dist_num):
    """
    Haversine公式: 计算两个latitude-longitude点之间的距离. [ http://www.cnblogs.com/softidea/p/6925673.html ]
    二倍角公式：cos(2a) = 1 - 2sin(a)sin(a)，即sin(a/2)*sin(a/2) = (1 - cos(a))/2
    dd = 25m，距离间隔。
    """
    d = 12742                           # 地球的平均直径。
    p = 0.017453292519943295            # math.pi / 180, x*p由度转换为弧度。
    a = (lat1 - lat2) * p
    b = (lon1 - lon2) * p
    # c = pow(sin(a / 2), 2) + cos(lat1 * p) * cos(lat2 * p) * pow(sin(b / 2), 2)     # a/b别混了。
    c = (1.0 - cos(a)) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1.0 - cos(b)) / 2     # 二者等价，但这个更快。
    dist = d * asin(sqrt(c))            # 球面上的弧线距离(km)

    interval = int(dist / dd)    # 该距离落在哪个距离间隔区间里。
    interval = min(interval, dist_num)
    # 间隔区间范围是[0, 379+1]。即额外添加一个idx=380, 表示两点间距>=38km。
    # 对应的生成分析计算出来的各区间概率，也添加一个位置来表示1520的概率，就是0.
    return interval


def load_data(dataset, mode, split, dd, dist_num):
    """
    加载购买记录文件，生成数据。
    dd = 25m, dt = 60min,
    """
    # 用户购买历史记录，原纪录. 嵌套列表, 元素为一个用户的购买记录(小列表)
    print('Original data ...')
    pois = pd.read_csv(dataset, sep=' ')
    all_user_pois = [[i for i in upois.split('/')] for upois in pois['u_pois']]
    all_user_cods = [[i.split(',') for i in ucods.split('/')] for ucods in pois['u_coordinates']]  # string
    # 删除Gowalla里的空值：'LOC_null'和['null', 'null']
    if 'sub_users5_items5(1)' in dataset:
        tmp_pois, tmp_cods = [], []
        for upois in all_user_pois:
            if 'LOC_null' in upois:
                tmp = [poi for poi in upois if poi != 'LOC_null']
                tmp_pois.append(tmp)
            else:
                tmp_pois.append(upois)
        for ucods in all_user_cods:
            if ['null', 'null'] in ucods:
                tmp = [cod for cod in ucods if cod != ['null', 'null']]
                tmp_cods.append(tmp)
            else:
                tmp_cods.append(ucods)
        all_user_pois, all_user_cods = tmp_pois, tmp_cods
    all_user_cods = [[[float(ucod[0]), float(ucod[1])] for ucod in ucods] for ucods in all_user_cods]  # float
    all_trans = [item for upois in all_user_pois for item in upois]
    all_cordi = [ucod for ucods in all_user_cods for ucod in ucods]
    poi_cordi = dict(zip(all_trans, all_cordi))  # 每个poi都有一个对应的的坐标。
    tran_num, user_num, item_num = len(all_trans), len(all_user_pois), len(set(all_trans))
    print('\tusers, items, trans:  = {v1}, {v2}, {v3}'.format(v1=user_num, v2=item_num, v3=tran_num))
    print('\tavg. user check:      = {val}'.format(val=1.0 * tran_num / user_num))
    print('\tavg. poi checked:     = {val}'.format(val=1.0 * tran_num / item_num))
    print('\tdistance interval     = [0, {val}]'.format(val=dist_num))

    # 选取训练集、验证集(测试集)，并对test去重。不管是valid还是test模式，统一用train，test表示。
    print('Split the training set, test set: mode = {val} ...'.format(val=mode))
    tra_pois, tes_pois = [], []
    tra_dist, tes_dist = [], []  # idx0 = max_dist, idx1 = 0/1的间距并划分到距离区间里。
    for upois, ucods in zip(all_user_pois, all_user_cods):

        left, right = upois[:split], [upois[split]]  # 预测最后一个。此时right只有一个idx，加[]变成list。

        # 两个POI之间距离间隔落在哪个区间。
        dist = []
        for i, cord in enumerate(ucods[1:]):  # 从idx=1和idx=0的距离间隔开始算。
            pre = ucods[i]
            dist.append(cal_dis(cord[0], cord[1], pre[0], pre[1], dd, dist_num))
        dist = [dist_num] + dist  # idx=0的距离间隔，就用最大的。
        dist_lf, dist_rt = dist[:split], [dist[split]]

        # 保存
        tra_pois.append(left)
        tes_pois.append(right)
        tra_dist.append(dist_lf)
        tes_dist.append(dist_rt)

    # 建立商品别名字典。更新购买记录，替换为0~len(se)-1的别名。
    print('Use aliases to represent pois ...')
    all_items = set(all_trans)
    aliases_dict = dict(zip(all_items, range(item_num)))  # 将poi转换为[0, n)标号。
    tra_pois = [[aliases_dict[i] for i in utra] for utra in tra_pois]
    tes_pois = [[aliases_dict[i] for i in utes] for utes in tes_pois]
    # 根据别名对应关系，更新poi-坐标的表示，以list表示，并且坐标idx就是poi别名替换后的idx。
    cordi_new = dict()
    for poi in poi_cordi.keys():
        cordi_new[aliases_dict[poi]] = poi_cordi[poi]  # 将poi和坐标转换为：poi的[0, n)标号、坐标。
    pois_cordis = [cordi_new[k] for k in sorted(cordi_new.keys())]

    return [(user_num, item_num), pois_cordis, (tra_pois, tes_pois), (tra_dist, tes_dist)]
