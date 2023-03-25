import os
import openai
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
import re
import math
from random import sample
from log_tool import logger_config
from sklearn.cluster import KMeans
import csv
import prediction_util as pu
# pd.set_option('mode.chained_assignment', None)


api_key = {
    # '1': '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',
    '2': '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',
    '3': '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',
    '4': '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',
    '5': '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',
    '6': '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',
    '7': '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',
    # '8': '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'
}
openai.api_key = api_key['6']
para={
'model': 'text-davinci-003',  # text-ada-001, text-babbage-001, text-curie-001, text-davinci-002
'max_tokens': 2000,
}
directions = {
    'FFDAvg': ['the first fixation duration', 'the duration of the first fixation on the prevailing word'],
    'FFDStd': ['the standard deviation of the first fixation duration', 'the standard deviation of FFD across readers'],
    'TRTAvg': ['the total fixation duration (including regressions)', 'the sum of all fixation durations on the current word (including regressions)'],
    'TRTStd': ['the standard deviation of the total fixation duration', 'the standard deviation of TRT across readers']
}
data_folder = '/home/shukai/CMCL2022_GPT3/data/output/'
predictions_folder = '/home/shukai/CMCL2022_GPT3/prediction/predictions/'


class ExamplesFilter():
    def sent_len_filter(examples_df, test_len, t, dsname, learning_mode):
        # get len statitics for each dsname
        dslen_dict = examples_df.groupby('dsname').get_group(dsname)['sent_len'].describe().to_dict()
        mean1_l, mean2_l, min_l, median_l, max_l = int(dslen_dict['mean']), math.ceil(dslen_dict['mean']), dslen_dict['min'], dslen_dict['50%'], dslen_dict['max']
        
        # 排除和test一样长的句
        examples_df_tmp = examples_df.loc[~(examples_df['sent_len']==test_len)]
        if learning_mode == 'oneshot':
            # examples_df_tmp = examples_df_tmp.loc[(examples_df_tmp['sent_len'] > test_len) | (examples_df_tmp['sent_len'] == max_l)]
            examples_df_tmp = examples_df_tmp.loc[(examples_df_tmp['sent_len'] > test_len)]
        elif learning_mode == 'fewshot':
            examples_df_tmp = examples_df_tmp.loc[(examples_df_tmp['sent_len'] > test_len) | (examples_df_tmp['sent_len'].isin([mean1_l, mean2_l, median_l, max_l]))]
        else:
            examples_df_tmp = examples_df
        # sent_lenMax, sent_lenMin = test_len + t, test_len - t
        # examples_df_tmp = examples_df_tmp.loc[(examples_df_tmp['sent_len'] <= sent_lenMax) & (examples_df_tmp['sent_len'] >= sent_lenMin)]
        # examples_df_tmp = examples_df_tmp.loc[(examples_df_tmp['sent_len'] > test_len)]
        if len(pu.get_sentence_ids(examples_df_tmp)) == 0:
            examples_df_tmp = examples_df_tmp.loc[(examples_df_tmp['sent_len'] == max_l)]
        #     t -= 1
        #     print('t=%s, len=%s, perv_num=%s'%(t, test_len, len(pu.get_sentence_ids(examples_df_tmp))))
        #     examples_df_tmp = ExamplesFilter.sent_len_filter(examples_df, test_len, t, dsname, learning_mode)
        # examples_df_tmp = examples_df_tmp.loc[(examples_df_tmp['sent_len'].isin([mean1_l, mean2_l, median_l, max_l]))]

        return examples_df_tmp


    def meanWrodLen_filter(examples_df, test_word_len_MEAN, t):
        meanWrodLenMax, meanWrodLenMin = test_word_len_MEAN + t, test_word_len_MEAN - t
        examples_df_tmp = examples_df.loc[(examples_df['cur_word_len_MEAN'] <= meanWrodLenMax) & (examples_df['sent_len'] >= meanWrodLenMin)]
        if len(pu.get_sentence_ids(examples_df_tmp)) == 0:
            t += 0.1
            # print('t=%s, test_word_len_MEAN=%s, perv_num=%s'%(t, test_word_len_MEAN, len(pu.get_sentence_ids(examples_df_tmp))))
            examples_df_tmp = ExamplesFilter.meanWrodLen_filter(examples_df, test_word_len_MEAN, t)

        return examples_df_tmp


    def wordlogfreq_filter(examples_df, test_word_logfreq_MEAN, t):
        wordlogfreqMax, wordlogfreqMin = test_word_logfreq_MEAN + t, test_word_logfreq_MEAN - t
        examples_df_tmp = examples_df.loc[(examples_df['cur_word_logfreq_MEAN'] <= wordlogfreqMax) & (examples_df['sent_len'] >= wordlogfreqMin)]
        if len(pu.get_sentence_ids(examples_df_tmp)) == 0:
            t += 0.1
            # print('t=%s, test_word_len_MEAN=%s, perv_num=%s'%(t, test_word_logfreq_MEAN, len(pu.get_sentence_ids(examples_df_tmp))))
            examples_df_tmp = ExamplesFilter.wordlogfreq_filter(examples_df, test_word_logfreq_MEAN, t)

        return examples_df_tmp


    def get_examples(sent_i_df, examples_df, learning_mode):
        test_i_df = sent_i_df.reset_index()
        
        test_dsname = test_i_df.loc[0, 'dsname']
        test_lang = test_i_df.loc[0, 'language']
        test_len = test_i_df.loc[0, 'sent_len']
        test_word_len_MEAN = test_i_df.loc[0, 'cur_word_len_MEAN']
        test_word_logfreq_MEAN = test_i_df.loc[0, 'cur_word_logfreq_MEAN']

        ##### filter by dataset #####
        examples_df2 = examples_df.loc[lambda x: x['sentence_id'].str.startswith(test_dsname)]
        # print('filter by dataset:', len(pu.get_sentence_ids(examples_df2)))

        ##### filter by language #####
        examples_df2 = examples_df2.loc[examples_df2['language'] == test_lang]
        # print('filter by language:', len(pu.get_sentence_ids(examples_df2)))

        ##### filter by sent_len #####
        len_delta = 0
        examples_df2 = ExamplesFilter.sent_len_filter(examples_df2, test_len, len_delta, test_dsname, learning_mode)
        print('filter by sent_len:', len(pu.get_sentence_ids(examples_df2)))

        # ##### filter by meanWordLen #####
        # wordlen_delta = 0.1
        # examples_df2 = ExamplesFilter.meanWrodLen_filter(examples_df2, test_word_len_MEAN, wordlen_delta)
        # print('filter by meanWordLen:', len(pu.get_sentence_ids(examples_df2)))

        # ##### filter by word_logfreq #####
        # wordlogfreq_delta = 0.1
        # examples_df2 = ExamplesFilter.wordlogfreq_filter(examples_df2, test_word_logfreq_MEAN, wordlogfreq_delta)
        # print('filter by word_logfreq:', len(pu.get_sentence_ids(examples_df2)))   

        return test_lang, test_len, examples_df2


class PromptGeneration():
    def prompt1(test_i_df, targets, example_df):
        f1, f2 = targets[0], targets[1]

        # example part
        example_df = example_df.round(3)
        if example_df is None or len(example_df) == 0:
            example_prompt = ''
        else:
            example_sent_len = str(len(example_df['word'].tolist()))
            observations = ''
            for i, row in example_df.iterrows():
                observations += row['word'] + '\t' + str(row[f1]) + '\t' + str(row[f2]) + '\n'
            example_l1 = 'given a sentence (%s words):\n%s\n'%(example_sent_len, '\n'.join(example_df['word'].tolist()))
            example_l2 = '%s and %s on each word:\n%s'%(directions[f1][0], directions[f2][0], observations)
            example_prompt = example_l1 + example_l2

        # test part
        test_sent_len = str(len(test_i_df['word'].tolist()))
        test_l1 = 'given a sentence (%s words):\n%s\n'%(test_sent_len, '\n'.join(test_i_df['word'].tolist()))
        test_l2 = '%s and %s on each word:\n'%(directions[f1][0], directions[f2][0])
        test_prompt = test_l1 + test_l2
    
        return example_prompt, test_prompt

    def prompt2(test_i_df, targets, example_df):
        f1, f2 = targets[0], targets[1]

        # example part
        example_df = example_df.round(3)
        if example_df is None or len(example_df) == 0:
            example_prompt = ''
        else:
            example_sent_len = str(len(example_df['word'].tolist()))
            observations = ''
            for i, row in example_df.iterrows():
                observations += row['word'] + '\t' + str(row[f1]) + '\t' + str(row[f2]) + '\n'
            example_l1 = 'example sentence: %s\n'%(' '.join(example_df['word'].tolist()))
            example_l2 = 'question: what are %s and %s on each word in the example sentence (%s words)?\n'%(directions[f1][0], directions[f2][0], example_sent_len)
            example_l3 = 'answer:\n%s'%(observations)
            example_prompt = example_l1 + example_l2 + example_l3

        # test part
        test_sent_len = str(len(test_i_df['word'].tolist()))
        test_l1 = 'test sentence: %s\n'%(' '.join(test_i_df['word'].tolist()))
        test_l2 = 'question: what are %s and %s on each word in the test sentence (%s words)?\n'%(directions[f1][0], directions[f2][0], test_sent_len)
        test_l3 = 'answer:\n'
        test_prompt = test_l1 + test_l2 + test_l3
    
        return example_prompt, test_prompt        

    def prompt3(test_i_df, targets, example_df):
        f1, f2 = targets[0], targets[1]

        # example part
        example_df = example_df.round(3)
        if example_df is None or len(example_df) == 0:
            example_prompt = ''
        else:
            example_sent_len = str(len(example_df['word'].tolist()))
            observations = ''
            for i, row in example_df.iterrows():
                observations += row['word'] + '\t' + str(row[f1]) + '\t' + str(row[f2]) + '\n'
            example_l1 = 'descriptions: %s is %s, %s is %s.\n'%(directions[f1][0], directions[f1][1], directions[f2][0], directions[f2][1])
            example_l2 = 'example sentence: %s\n'%(' '.join(example_df['word'].tolist()))
            example_l3 = 'question: what are %s and %s on each word in the example sentence (%s words)?\n'%(directions[f1][0], directions[f2][0], example_sent_len)
            example_l4 = 'answer:\n%s'%(observations)
            example_prompt = example_l1 + example_l2 + example_l3 + example_l4

        # test part
        test_sent_len = str(len(test_i_df['word'].tolist()))
        test_l1 = 'test sentence: %s\n'%(' '.join(test_i_df['word'].tolist()))
        test_l2 = 'question: what are %s and %s on each word in the test sentence (%s words)?\n'%(directions[f1][0], directions[f2][0], test_sent_len)
        test_l3 = 'answer:\n'
        test_prompt = test_l1 + test_l2 + test_l3
    
        return example_prompt, test_prompt
    
    def prompt4(test_i_df, targets, example_df):
        f1, f2 = targets[0], targets[1]

        # example part


def submit_request(args, input_prompt):
  try:
    response = openai.Completion.create(
      model=args['model'],
      prompt=input_prompt,
      temperature=0,
      max_tokens=args['max_tokens'],
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    
    return response
  except Exception as e:
    logger.error(e)
    logger.error('Fails to submit_request for input_prompt::\t%s'%input_prompt)

    return None


def decode_response(response, input_x=[]):
  output_x, y1_pred, y2_pred = [], [], []
  response_text = response.choices[0]['text'].strip()  # tokens and features in prediction, strip special tokens of the left and right sides
  try: 
    for line in re.split('\n|\n\n', response_text):
      if '\t' in line:
        token = line.split('\t')[0]
        y1 = line.split('\t')[1]
        y2 = line.split('\t')[2]
      else:
        token = line.split(':')[0]
        y1 = line.split(',')[1].replace('ms','')
        y2 = line.split(',')[2].replace('ms','') 
      output_x.append(token)
      y1_pred.append(float(y1))
      y2_pred.append(float(y2))
    assert len(input_x) == len(output_x) == len(y1_pred) == len(y2_pred)
  except:
    logger.error('输入输出长度不等')
    logger.error('Assign zeros to output, response_text error in line:{%s}'%line)
    logger.error('input_x(%s):\n%s'%(str(len(input_x)), input_x))
    logger.error('output_x(%s):\n%s'%(str(len(output_x)), output_x))
    logger.error('y1_pred=%s,y2_pred=%s'%(str(len(y1_pred)), str(len(y2_pred))))
    logger.error('response_text:\n%s'%response_text)
    logger.error('response:\n%s'%response)   
    output_x, y1_pred, y2_pred = np.zeros(len(input_x)), np.zeros(len(input_x)), np.zeros(len(input_x))

  return output_x, y1_pred, y2_pred


def run_one_test(test_sent_i_df, targets, prompting_df, output_fp=''):
    t1, t2 = targets[0], targets[1]
    fail_df, pred_df = [], []
    
    sentence_id = test_sent_i_df['sentence_id'].values[0]
    sent_lang = test_sent_i_df['language'].values[0]
    sent_len = test_sent_i_df['sent_len'].values[0]
    x_in = test_sent_i_df['word'].tolist()
    y1_true = test_sent_i_df[t1].tolist()
    y2_true = test_sent_i_df[t2].tolist()
    print('test_sentence: length=%s, lang=%s'%(sent_len, sent_lang))

    # get promopt for test_sent_i
    prompt_for_i = ''
    prompting_ids = pu.get_sentence_ids(prompting_df)
    for i, id in enumerate(prompting_ids):
        p_df = prompting_df[prompting_df['sentence_id']==id]
        print('example_sentence: length=%s, sentence_id=%s'%(p_df.shape[0], id))
        p1, p2 = PromptGeneration.prompt2(test_sent_i_df, targets, p_df)
        prompt_for_i += p1
    prompt_for_i = prompt_for_i + p2
    print(prompt_for_i)
    
    # # get response for test_sent_i
    # response_for_i = submit_request(args=para, input_prompt=prompt_for_i)
    # if response_for_i is None:
    #     fail_df = [sentence_id, sent_lang, prompting_ids, 'response_None']
    
    # # get output_x, pred_y1, pred_y2 for test_i
    # x2, y1_pred, y2_pred = decode_response(response=response_for_i, input_x=x_in)
    # if sum(y1_pred) == 0 or sum(y2_pred) == 0:
    #     fail_df = [sentence_id, sent_lang, prompting_ids, 'response_length_error']

    # # get results for test_sent_i
    # y1_results = list(pu.calculations(y1_true, y1_pred))
    # y2_results = list(pu.calculations(y2_true, y2_pred))
    # y1_mae, y2_mae = y1_results[0].round(3), y2_results[0].round(3)
    # print('%s_MAE:%s'%(t1, y1_mae))
    # print('%s_MAE:%s'%(t2, y2_mae))

    # # get pred_df for test_i including truth and prediction
    # pred_df = test_sent_i_df.round(3)[['dscode', 'language', 'sentence_id', 'word_n', 'sent_n', 'word'] + targets]
    # pred_df['pred_' + t1] = y1_pred
    # pred_df['pred_' + t2] = y2_pred
    # for i, id in enumerate(prompting_ids):
    #     pred_df['example_' + str(i+1)] = id
    # pred_df[t1 + '_MAE'] = y1_mae
    # pred_df[t2 + '_MAE'] = y2_mae

    # if output_fp != '':
    #     if len(fail_df) != 0:
    #         fail_df = pd.DataFrame(fail_df, columns=['test_i', 'test_lang', 'prompting_ids', 'reason'])
    #         fail_df.to_csv(output_fp + '_fail.csv', index=False)
    #     if pred_df is not None:
    #         pred_df.to_csv(output_fp + '_pred.csv', index=False)            
    # print(pred_df)
    # print(fail_df)


    # return fail_df, pred_df, y1_results, y2_results


#### test set predictions #####
def run(test_df, targets, examples_df, learning_mode):
    preds, fails = [], []
    for test_sent_i, test_sent_df in test_df.groupby('sent_n'):  # 遍历test_df中每个句子test_sent_i
        if 0 <= test_sent_i <= 10:  # 324
            print('----------sent_%s----------'%test_sent_i)
            """
            对于每一个测试句子id，
            给定：测试句子特征test_i_featrues_df, 指标targets，学习样例prompts_df
            返回：指标预测值，及其准确率
            """
            
            # get input_x, y1_true, y2_true for test_sent_i
            x_in = test_sent_df['word'].tolist()
            y1_true, y2_true = test_sent_df[targets[0]].tolist(), test_sent_df[targets[1]].tolist()

            # filter examples for test_sent_i
            sent_lang, sent_len, filtered_examples_df = ExamplesFilter.get_examples(test_sent_df, examples_df, learning_mode)
            filtered_examples_ids = pu.get_sentence_ids(filtered_examples_df)  # final example(s) for test_sent_i
            print('final filtered examples_num:', len(filtered_examples_ids))
            
            # promopt for test_sent_i
            example_ids_for_i, prompt_for_i = PromptGeneration.gen_prompt2(test_sent_df, targets, learning_mode, candidates_df=filtered_examples_df)

            # response for test_sent_i
            response_for_i = submit_request(args=para, input_prompt=prompt_for_i)

            # get output_x, pred_y1, pred_y2 for test_i
            x2, y1_pred, y2_pred = decode_response(response=response_for_i, input_x=x_in)

            # print test_sent_i results
            y1_r = pu.calculations(y1_true, y1_pred)
            y2_r = pu.calculations(y2_true, y2_pred)
            print('y1_MAE:', list(y1_r)[0].round(3))
            print('y2_MAE:', list(y2_r)[0].round(3))

            # generate new df for test_i including truth and predictions
            test_sent_tmp_df = test_sent_df.round(3)[['dscode', 'language', 'sentence_id', 'word_n', 'sent_n', 'word'] + targets]
            test_sent_tmp_df['pred_' + targets[0]] = y1_pred
            test_sent_tmp_df['pred_' + targets[1]] = y2_pred
            preds.append(test_sent_tmp_df)

            # write results
            if response_for_i == None:
                error = [test_sent_i, sent_lang, example_ids_for_i, 'response_None']
                if write_to_files:
                    fails_writer.writerow(error)
                fails.append(error)
                continue

            if sum(y1_pred) == 0 or sum(y2_pred) == 0:
                error = [test_sent_i, sent_lang, example_ids_for_i, 'response_length_error']
                if write_to_files:
                    fails_writer.writerow(error)
                fails.append(error)
                continue
            if write_to_files:
                test_sent_tmp_df.to_csv(predictions_f, mode='a', index=False, header=False)

    success_df = pd.concat(preds, ignore_index=True)
    print(success_df)
    print(success_df.describe())
    fails_df = pd.DataFrame(fails)
    print(fails_df)

    cals = []
    for sent_i, sent_df in success_df.groupby('sent_n'):
        sent_lang = sent_df['language'].values[0]
        y1_true, y1_pred= sent_df[targets[0]].tolist(), sent_df['pred_' + targets[0]].tolist()
        y1_r = pu.calculations(y1_true, y1_pred)

        y2_true, y2_pred = sent_df[targets[1]].tolist(), sent_df['pred_' + targets[1]].tolist()
        y2_r = pu.calculations(y2_true, y2_pred)

        cals.append([sent_i, sent_lang] + list(y1_r + y2_r))


    results_sum_f = predict_folder + targets[0][:3] + '_sum.csv'
    cals_df = pd.DataFrame(cals, columns=['sent_n', 'sent_lang', 'y1_MAE', 'y1_R2', 'y1_MSE', 'y1_Pearson', 'y1_Spearman', 'y2_MAE', 'y2_R2', 'y2_MSE', 'y2_Pearson', 'y2_Spearman'])
    print(cals_df)
    print(cals_df.describe())
    if write_to_files:
        csvfile.close()
        cals_df.describe().to_csv(results_sum_f)


##### get train&dev features #####
merge_f = data_folder + 'features_nontest.csv'
candidates_df = pd.read_csv(merge_f)
print('Loading %s: %s'%(merge_f, candidates_df.shape))
# print(candidates_df.columns)

##### get test features #####
test_merge_f = data_folder + 'features_test_subtask1_truth.csv'
test_df = pd.read_csv(test_merge_f)
print('Loading %s: %s'%(test_merge_f, test_df.shape))

##### exp settings #####
targets = ['TRTAvg', 'TRTStd']   # ['FFDAvg', 'FFDStd'] / ['TRTAvg', 'TRTStd'] / ['FFDAvg', 'TRTAvg']
# learning_mode='fewshot'  # fewshot/oneshot

##### BSC sent_0 test #####
test_sent_i = 0
test_sent_i_df = test_df[test_df['sent_n'] == test_sent_i]

test_i_folder = predictions_folder + str(test_sent_i) + '/'
os.mkdir(test_i_folder) if not os.path.exists(test_i_folder) else print('Folder already exists: %s'%(test_i_folder))
test_i_subfolder = test_i_folder + '2S_P2_dvc003/'
os.mkdir(test_i_subfolder) if not os.path.exists(test_i_subfolder) else print('Folder already exists: %s'%(test_i_subfolder))

##### get one_ds_candidates_df #####
ds_df = candidates_df[candidates_df['dsname'] == 'BSC']  # for one dsname

# candidates of maxLen_sentences
meanl, meanh, minLen, medianLen, maxLen = pu.get_ds_lens_statistics(candidates_df, dsname='BSC')  # get len statitics for dsname
ds_maxLen_dfs = ds_df[(ds_df['sent_len'] == maxLen)]  
maxLen_ids = pu.get_sentence_ids(ds_maxLen_dfs)
# print(maxLen_ids)

##### further filter candidates #####
sel_dfs = ds_df[~ds_df['sentence_id'].isin(maxLen_ids)]  # candidates except maxLen_sentences
# sel_dfs = sel_dfs[(sel_dfs['sent_len'] > 10)]
sel_ids = pu.get_sentence_ids(sel_dfs)
print(len(sel_ids))

cands_ids = sel_ids
y1_mae_tmp, y2_mae_tmp = 1000, 1000
for i, id in enumerate(cands_ids):
    if i<5:
        ids = [id] + ['BSC-65']
        exams_df = ds_df[ds_df['sentence_id'].isin(ids)]
        exams_ids = pu.get_sentence_ids(exams_df)
        exams_i_output_f = test_i_subfolder + '__'.join(exams_ids).replace('-', '_')
        run_one_test(test_sent_i_df, targets, exams_df, exams_i_output_f)
        # if os.path.exists(exams_i_output_f + '_pred.csv') or os.path.exists(exams_i_output_f + '_fail.csv'):
        #     print('File already exists: %s'%(exams_i_output_f))
        #     continue
        # else:
        #     _, _, min1, min2 = run_one_test(test_sent_i_df, targets, exams_df, exams_i_output_f)
#         if min1 < y1_mae_tmp:
#             y1_mae_tmp = min1
#             y1_min_sent_ids = exams_ids
#         if min2 < y2_mae_tmp: 
#             y2_mae_tmp = min2
#             y2_min_sent_ids = exams_ids
# print(y1_min_sent_ids, y2_min_sent_ids)
# ##### parameters #####
# settings = learning_mode + '_diffleMeanMaxMedian_samedsname_samelang_prompt2_'  # output folder name
# write_to_files = False

# ##### check output file paths #####
# sgtime_str = str(int(datetime.now(pytz.timezone('Asia/Singapore')).strftime('%m%d%H%M'))-800)

# if write_to_files:
#     log_f = output_folder + targets[0][:3] + '.log'
#     predictions_f = output_folder + targets[0][:3] + '_preds.csv'
#     fails_f = output_folder + targets[0][:3] + '_fails.csv'
#     if os.path.exists(log_f) or os.path.exists(predictions_f) or os.path.exists(fails_f):
#         print('File path exists!!!')
#         exit()

#     # init log_f
#     logger = logger_config(log_name = settings, log_path = log_f)  

#     # init predictions_f
#     preds_cols = ['dscode', 'language', 'sentence_id', 'word_n', 'sent_n', 'word'] + targets + ['pred_' + targets[0], 'pred_' + targets[1]]
#     pd.DataFrame([preds_cols]).to_csv(predictions_f, index=False, header=False)

#     # init fails file
#     csvfile = open(fails_f, 'a')
#     csvfile.truncate()  # 清空原内容
#     fails_writer = csv.writer(csvfile)  
#     fails_writer.writerow(['test_i', 'language', 'dscode_sentid', 'reason'])


# # # nohup python -u /home/shukai/CMCL2022_GPT3/prediction.py > run.log 2>&1 &
