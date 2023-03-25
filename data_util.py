import pandas as pd
import numpy as np
import textstat, string
import os
import string
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
import torch
import math
from sklearn import preprocessing
from wordfreq import word_frequency
import pickle
from indicnlp.syllable import syllabifier
from RU.ru_transformers.yt_encoder import YTEncoder
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModel, AutoTokenizer, BertTokenizer, AutoModelWithLMHead


# modelEN = GPT2LMHeadModel.from_pretrained("gpt2")
# tokenizerEN = GPT2Tokenizer.from_pretrained("gpt2")

# modelRU = GPT2LMHeadModel.from_pretrained("/home/shukai/CMCL2022_GPT3/data/RU/gpt2/m_checkpoint-3364613")
# tokenizerRU = YTEncoder.from_pretrained("/home/shukai/CMCL2022_GPT3/data/RU/gpt2/m_checkpoint-3364613")

# modelHI = GPT2LMHeadModel.from_pretrained("surajp/gpt2-hindi")
# tokenizerHI = AutoTokenizer.from_pretrained("surajp/gpt2-hindi")

# modelZH = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
# tokenizerZH = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

# modelNL = GPT2LMHeadModel.from_pretrained("GroNLP/gpt2-small-dutch")
# tokenizerNL = AutoTokenizer.from_pretrained("GroNLP/gpt2-small-dutch")

# modelDE = AutoModelWithLMHead.from_pretrained("dbmdz/german-gpt2")
# tokenizerDE = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")


class DataLoader():
    def load_data(dsname):
        """
        sents_df: add columns=['sent_n', 'word_n', 'sent_len']
        lang_lab: {'lang':
            {'sent_n': sentence tokens (string)}
            }
        """
        df = pd.read_csv(dsname)
        df = df.astype({'word_id': int})
        df.loc[:, ['word']] = df.loc[:, ['word']].replace(" ","")

        sent_df_list = []
        sent_n = 0
        for id, sent_df in df.groupby('sentence_id'):
            # add sent_len col
            sent_len = sent_df.shape[0]
            sent_df['sent_len'] = sent_len

            # add sent_n col
            sent_df['sent_n'] = sent_n
            sent_n += 1

            # add word_n col
            word_n = [n for n in range(sent_len)]
            sent_df['word_n'] = word_n

            sent_df_list.append(sent_df)
        sents_df = pd.concat(sent_df_list, ignore_index=True)
        sents_df['dsname'] = sents_df['sentence_id'].str.split('-').apply(lambda x: x[0])
        # print(sents_df)
        # print(sents_df.columns)

        ds_lab = {}
        for lang, lang_df in sents_df.groupby('dsname'):
            ds_lab[lang] = {}
            for i, sent_df in lang_df.groupby('sent_n'):
                ds_lab[lang][i] = ' '.join(sent_df['word'].tolist())
        # print(ds_lab)

        return sents_df, ds_lab


class FeatureExtraction():
    def get_lang_ids(lang):
        lang_ids = {}
        for i, l in enumerate(['zh', 'hi', 'ru', 'en', 'nl', 'de', 'dk']):
            lang_ids[l] = i + 1
        
        return lang_ids[lang]

    def is_two_columns_equal(df, col1, col2):
        # df['diff_col'] = 0
        df['diff_col'] = np.where(df[col1] == df[col2], 0, 1)
        diff_col = set(df['diff_col'].tolist())
        diff_num = sum(list(diff_col))
        if diff_num != 0:
            print('comparing (%s) (%s), diff_num=%s'%(col1, col2, diff_num))
            print(df.loc[(df['diff_col'] == 1)][['sent_n', col1, col2, 'diff_col', 'language', 'word']])

        return diff_num

    def get_syllable_count(wordlanguage, wordtext, wordlen):
        if wordlanguage not in ["hi", "zh"]:
            textstat.set_lang(wordlanguage)
            syllab = textstat.syllable_count(wordtext)
        elif wordlanguage == "zh":
            syllab = wordlen
        # elif wordlanguage == "hi":
        #     syllab=len(syllabifier.orthographic_syllabify(wordtext,'hi'))
        else:
            syllab = 0
        
        return syllab
        

    def word_feature_extract(data_df):
        # cur_word_pos scaled by sentlen
        data_df['word_pos_norm'] = data_df.word_n / data_df.sent_len

        # cur_word_len
        data_df['cur_word_len'] = data_df.apply(lambda x: len(x['word']), axis=1)
        # df[feat_out_name] =  df.apply(lambda x: x[feat_in_name] / x['sent_len'] if x['sent_len']!=0 else 0, axis=1)
        
        # prev_word_len
        tmp = []
        for i, row in data_df.iterrows():
            if row['word_n'] == 0:
                tmp.append(0)
            else:
                tmp.append(data_df.loc[i-1, 'cur_word_len'])
        data_df['prev_word_len'] = tmp

        # cur_word_logfreq
        data_df['cur_word_freq'] = data_df.apply(lambda x: word_frequency(x['word'].lower(), x['language']), axis=1)
        data_df['cur_word_logfreq'] = data_df.apply(lambda x: -np.log(x['cur_word_freq']) if x['cur_word_freq'] != 0 else 0, axis=1)
        
        # prev_word_logfreq
        tmp = []
        for i, row in data_df.iterrows():
            if row['word_n'] == 0:
                tmp.append(0)
            else:
                tmp.append(data_df.loc[i-1, 'cur_word_logfreq'])
        data_df['prev_word_logfreq'] = tmp

        # is_upper
        data_df['is_upper'] = data_df.apply(lambda x: 1 if x['word'].upper() == x['word'] else 0, axis=1)

        # is_capital_start
        data_df['is_capital_start'] = data_df.apply(lambda x: 1 if x['word'][0] == x['word'][0].upper() else 0, axis=1)
        
        # map language(string) to lang_id
        data_df['lang_id'] = data_df.apply(lambda x: FeatureExtraction.get_lang_ids(x['language']), axis=1)

        # syllable_count
        data_df['syllable_count'] = data_df.apply(lambda x: FeatureExtraction.get_syllable_count(x['language'], x['word'], x['cur_word_len']), axis=1)

        # TODO
        # map wordtext tp word_embedding

        # TODO
        # delete punctuation/non-character for each token;
        # is_digit
        # contains_punctuation

        # # syllable_count
        # if wordlanguage not in ["hi", "zh"]:
        #     textstat.set_lang(wordlanguage)
        #     syllab = textstat.syllable_count(wordtext)
        # if wordlanguage == "zh":
        #     syllab = wordlen
        # if wordlanguage == "hi":
        #     syllab=len(syllabifier.orthographic_syllabify(wordtext,'hi'))
        # line.append(syllab)		
        #print(wordtext, textstat.syllable_count(wordtext))

        # splittedSent = sents[sent_id].split(' ')
        # index_token = splittedSent.index(wordtext)

        # # load gpt2 models for surprisal 
        # if wordlanguage == "en":
        #     tokenizer = tokenizerEN
        #     model = modelEN
        # if wordlanguage == "zh":
        #     tokenizer = tokenizerZH
        #     model = modelZH
        # if wordlanguage == "hi":
        #     tokenizer = tokenizerHI
        #     model = modelHI
        # if wordlanguage == "ru":
        #     tokenizer = tokenizerRU
        #     model = modelRU
        # if wordlanguage == "nl":
        #     tokenizer = tokenizerNL
        #     model = modelNL
        # if wordlanguage == "de":
        #     tokenizer = tokenizerDE
        #     model = modelDE

        # tok_tens = torch.tensor(tokenizer.encode(sents[sent_id]))  # token embed? embed_dim=32, all the same if not trained
        # loss = model(tok_tens, labels=tok_tens)
        # loss_v = loss[0].item()
        # surprisal = -1 * math.log(loss_v)
        # surprisal = -1 * math.log(-1 * loss_v)
        # surprisal = 0.01
        # line.append(surprisal)
        # if wordlanguage=='zh' and sentlen==10:
        #     print(sent_id, wordseq_id, wordtext, tok_tens, surprisal)

        print('word-level', data_df.columns)

        return data_df
    

    def sent_feature_extract(wordfeature_df):
        """get sent-level features from wordfeat_df

        Args:
            wordfeat_df (_type_): _description_

        Returns:
            _type_: _description_
        """
        group_dict = {
            'sent_n': 'max',  # 句子唯一编号
            'lang_id': 'max',  # 句子语言编号
            'sent_len': 'max',  # 句子token总数
            'FFDAvg': 'mean',  # 句子平均FFDAvg
            'FFDStd': 'mean',  # 句子平均FFDStd
            'TRTAvg': 'mean',  # 句子平均TRTAvg
            'TRTStd': 'mean',  # 句子平均TRTStd
            'cur_word_len': 'mean',  # 平均单词长度
            'cur_word_logfreq': 'mean',  # 平均单词频率log
            'is_upper': 'mean',  # 单句中全部大写的单词总数
            'is_capital_start': 'mean',  # 单句中首字母大写的单词总数
        }   
        sent_grouped = wordfeature_df.groupby('sent_n')
        sentfeat_df = sent_grouped.agg(group_dict)
        sentfeat_df.columns = [k + '_' + v.upper() if k != 'sent_n' else 'sent_n' for k,v in group_dict.items()]
        sentfeat_df.reset_index(inplace=True, drop=True)  # 把sent_n的index删掉
        print('sent-level:', sentfeat_df.columns)

        return sentfeat_df
    

    def save_features(input_folder, dsname, output_folder):  # merge word-level and sent-level features for given dsname
        dscode = {
            'train.csv': 0,
            'dev.csv': 1, 
            'test_subtask1_truth.csv': 2
        }    
        sents_df, ds_lab = DataLoader.load_data(input_folder + dsname)
        wf_df = FeatureExtraction.word_feature_extract(sents_df)
        sf_df = FeatureExtraction.sent_feature_extract(wf_df)
        features_df = pd.merge(wf_df, sf_df, on='sent_n', suffixes=['_W', '_S'])
        features_df.insert(0, 'dscode', dscode[dsname])  # 在第一列i添加dscode区分train/dev/test
        FeatureExtraction.is_two_columns_equal(features_df, 'sent_len_MAX', 'sent_len')
        FeatureExtraction.is_two_columns_equal(features_df, 'lang_id_MAX', 'lang_id')

        # save features_df
        merge_f = output_folder + 'features_' + dsname
        print(features_df.columns)
        if not os.path.exists(merge_f):
            features_df.to_csv(merge_f, index=False)
            print('saving merged of word-level and sent-level features to %s'%(merge_f))
            print(features_df)
        else:
            features_df = pd.read_csv(merge_f)
            print('loading %s:%s'%(merge_f, features_df.shape))
        
        # save sentences_dict
        ds_lab_f = output_folder + 'sentences_' + dsname
        if not os.path.exists(ds_lab_f):
            with open(ds_lab_f, 'w') as fi:
                fi.write(str(ds_lab))
        else:
            with open(ds_lab_f, 'r') as fo:
                ds_lab = eval(fo.read())
        # print(ds_lab)

        return features_df, ds_lab


ip_folder = '/home/shukai/CMCL2022_GPT3/data/raw_data/'
op_folder = '/home/shukai/CMCL2022_GPT3/data/output/'


##### get features_df and sentences_dict for each dsname #####
dev_feats, dev_lab = FeatureExtraction.save_features(ip_folder, 'dev.csv', op_folder)
train_feats, train_lab = FeatureExtraction.save_features(ip_folder, 'train.csv', op_folder)
test_feats, test_lab = FeatureExtraction.save_features(ip_folder, 'test_subtask1_truth.csv', op_folder)


##### get train and dev features #####
mergedfeats_f = op_folder + 'features_nontest.csv'
if not os.path.exists(mergedfeats_f):
    mergedfeats_df = pd.concat([dev_feats, train_feats], ignore_index=True)
    print(mergedfeats_df)
    mergedfeats_df.to_csv(mergedfeats_f, index=False)
else:
    mergedfeats_df = pd.read_csv(mergedfeats_f)
    print('loading %s:%s'%(mergedfeats_f, mergedfeats_df.shape))