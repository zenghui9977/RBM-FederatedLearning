from metrics import *
from utils import *

RESULT_FOLDER = 'result/'

ground_truth = read_from_npy(RESULT_FOLDER, 'ground_truth.npy')

prediction_list = read_from_pkl(RESULT_FOLDER, 'top@k_recommedation.pkl')

precision_list = []
recall_list = []
fmeasure_list = []
NDCG_list = []

top_k = 10
hit_num = 0

for u in range(len(prediction_list)):
    pred = prediction_list[u][:top_k]
    gt = [ground_truth[u]]

    pre = precision(gt, pred)
    rec = recall(gt, pred)
    f_me = f_measure(gt, pred)
    ndcg = getNDCG(gt, pred)


    if pre > 0:
        print('the precision@%d is %s' % (top_k, pre))
        hit_num += 1
    if rec > 0:
        print('the recall@%d is %s' % (top_k, rec))

    if f_me > 0:
        print('the f_measure@%d is %s' % (top_k, f_me))

    if ndcg > 0:
        print('the NDCG@%d is %s' % (top_k, ndcg))

    precision_list.append(pre)
    recall_list.append(rec)
    fmeasure_list.append(f_me)
    NDCG_list.append(ndcg)

print('all users(total %d) statistical information' % len(prediction_list))

print('the mean of all precision@%d is %f' % (top_k, np.mean(precision_list)))
print('the mean of all recall@%d is %f' % (top_k, np.mean(recall_list)))
print('the mean of all f_measure@%d is %f' % (top_k, np.mean(fmeasure_list)))
print('the mean of all NDCG@%d is %f' % (top_k, np.mean(NDCG_list)))
print('%d/%d(target/all users)' % (hit_num, len(prediction_list)))
