from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import pandas as pd
from datetime import datetime
import pytz
import math


def get_sent_ids(features_df, is_test=False):
    if is_test:
        sent_ids = list(set(features_df['sent_n'].tolist()))
    else:
        sent_ids = []
        for l in features_df[['dscode', 'sent_n']].values.tolist():
            if l not in sent_ids:
                sent_ids.append(l)  # list of [dscode, sent_n]

    return sent_ids


def get_sentence_ids(features_df):
    sentence_ids = features_df['sentence_id'].drop_duplicates().values

    return sentence_ids


def get_ds_lens_statistics(features_df, dsname):
    one_ds = features_df.groupby('dsname').get_group(dsname)
    one_ds_lens = one_ds[['sentence_id', 'sent_len']].drop_duplicates()['sent_len']
    one_ds_lens_dict = one_ds_lens.describe().to_dict()
    meanl, meanh, minLen, medianLen, maxLen = int(one_ds_lens_dict['mean']), math.ceil(one_ds_lens_dict['mean']), one_ds_lens_dict['min'], one_ds_lens_dict['50%'], one_ds_lens_dict['max']
    print(one_ds_lens_dict)
    
    return meanl, meanh, minLen, medianLen, maxLen


def calculations(y_true, y_pred):  # sentence-level acc
    MAE = mean_absolute_error(y_true, y_pred)
    R2 = r2_score(y_true, y_pred)
    MSE = mean_squared_error(y_true, y_pred)
    # if sum(y_true) != 0 and sum(y_pred) != 0:
    try:
        Pearson = pearsonr(y_true, y_pred)[0]
        Spearman = spearmanr(y_true, y_pred, nan_policy='omit').correlation
    except: 
        Pearson = 0
        Spearman = 0

    return MAE, R2, MSE, Pearson, Spearman


class PredictionAnalysis():
    def by_dsname(prediction_folder):
        ffd_pred_df = pd.read_csv(prediction_folder + 'FFD.csv')
        trt_pred_df = pd.read_csv(prediction_folder + 'TRT.csv')
        preds_df = pd.merge(ffd_pred_df, trt_pred_df)
        preds_df['dsname'] = preds_df['sentence_id'].str.split('-').apply(lambda x: x[0])

        cols = ['FFDAvg', 'FFDStd', 'TRTAvg', 'TRTStd']
        for ds, ds_df in preds_df.groupby('dsname'):
            print(ds)
            print(calculations(ds_df['TRTAvg'], ds_df['pred_TRTAvg']))
            # print(ds_df.describe())

        return preds_df

def one_ds_test(input_folder, fname):
    lab_f = input_folder + fname
    with open(lab_f, 'r') as fi:
        sentences = eval(fi.read())
    # print(sentences)

    for k,v in sentences.items():
        print('-----', k, len(v))
        # for sk, sv in v.items():
        #     print(sk, len(sv.replace(' ', '')))


def BSC_test_sent0():
    sent_i = 0


ip_folder = '/home/shukai/CMCL2022_GPT3/data/output/'

# one_ds_test(ip_folder, 'sentences_test_subtask1_truth.csv')
# one_ds_test(ip_folder, 'sentences_train.csv')

# ##### prediction data #####
# p_folder = '/home/shukai/CMCL2022_GPT3/prediction/' + 'OS_F3_P2/'
# merged_results_df = PredictionAnalysis.by_dsname(p_folder)

sgtime_str = str(int(datetime.now(pytz.timezone('Asia/Singapore')).strftime('%m%d%H%M'))-800)
