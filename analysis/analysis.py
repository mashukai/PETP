from time import time
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import math
import numpy as np
import scipy.stats as stats
from statsmodels.stats.diagnostic import lilliefors
import scipy
from datetime import datetime
import pytz
from sklearn.cluster import KMeans



class Visual():
    def plot_sentfeat_corr():
        ########## train ##########
        plt.subplots(figsize=(9,9),dpi=1080,facecolor='w')# 设置画布大小，分辨率，和底色
        fig=sns.heatmap(trainsentfeat_df.corr(),annot=True, vmax=1, square=True, cmap="Blues", fmt='.2g')
        #annot为热力图上显示数据；fmt='.2g'为数据保留两位有效数字,square呈现正方形，vmax最大值为1
        fig.get_figure().savefig(op_folder + 'train_sent_corr.png',bbox_inches='tight',transparent=True)#保存图片
        print(trainsentfeat_df.corr())

        ########## dev ##########
        print(devsentfeat_df.corr())
        plt.subplots(figsize=(9,9),dpi=1080,facecolor='w')# 设置画布大小，分辨率，和底色
        fig=sns.heatmap(devsentfeat_df.corr(),annot=True, vmax=1, square=True, cmap="Blues", fmt='.2g')
        fig.get_figure().savefig(op_folder + 'dev_sent_corr.png',bbox_inches='tight',transparent=True)#保存图片

        ########## test ##########
        print(testsentfeat_df.corr())
        plt.subplots(figsize=(9,9),dpi=1080,facecolor='w')# 设置画布大小，分辨率，和底色
        fig=sns.heatmap(testsentfeat_df.corr(),annot=True, vmax=1, square=True, cmap="Blues", fmt='.2g')
        fig.get_figure().savefig(op_folder + 'test_sent_corr.png',bbox_inches='tight',transparent=True)#保存图片


    def plot_his(df, feature_name, bin_n, fig_name):
        print(df[feature_name].describe())
        matplotlib.rcParams['axes.unicode_minus']=False
        plt.hist(x=df[feature_name], bins=bin_n, color="steelblue", edgecolor="black")
        min_x = int(df.describe()[feature_name]['min'])
        max_x = math.ceil(df.describe()[feature_name]['max'])
        # max_x = 40
        plt.xlim((min_x, max_x))

        #添加x轴和y轴标签
        plt.xlabel(feature_name)
        # plt.ylabel("sentence_num")
        #添加标题
        plt.title(feature_name)
        plt.savefig(fig_name)


    def plot_two_feats(df, col1, col2, kind='hex', fig_name=''):
        # plot joinplot for two features
        sns.jointplot(data=df, x=col1, y=col2, kind=kind)  # kind=hex/kde/hist/reg
        plt.savefig(fig_name)


class Quantify():
    def scale_feature(df, feature_name, scale_method='sigmoid', fig_f='_dev.png'):
        t = pd.DataFrame()
        if scale_method == 'sigmoid':
            t['TRTAvg'] = df[feature_name].apply(lambda x :sigmoid(x))
        elif scale_method == 'softmax':
            t['TRTAvg'] = df[feature_name].apply(lambda x :softmax(x))
        print(t.describe())
        plot_his(t, feature_name, bin_n=20, fig_name = op_folder + feature_name + scale_method + fig_f)
        return t


    def get_skew_kurt():
        f = 'TRTStd'  # TRTStd,TRTAvg
        s = pd.Series(mergedfeats_df[f])
        print('merge,%s,%s,%s'%(f, s.skew(), s.kurt()))


    def check_dist(df, feature_name, dist_type='norm'):
        # https://blog.csdn.net/pearl8899/article/details/103034857
        data = df[feature_name]
        print(feature_name, dist_type)
        print(scipy.stats.normaltest(data, axis=0, nan_policy='omit'))
        #输出（统计量D的值,P值）=(0.058248638723832402, 0.88658843653019392)
        #统计量D的值越接近0就越表明数据和标准正态分布拟合得越好，
        # P值>指定水平,不拒绝原假设，可以认为样本数据服从正态分布。
        # 科尔莫戈罗夫检验(Kolmogorov-Smirnov test)
        # print(stats.kstest(data, dist_type))

        # 用Anderson-Darling检验生成的数组是否服从正态分布
        # '''输出AndersonResult(statistic=0.18097695613924714, 
        #                       critical_values=array([ 0.555,  0.632,  0.759,  0.885,  1.053]), 
        #                       significance_level=array([ 15. ,  10. ,   5. ,   2.5,   1. ]))
        # 如果输出的统计量值statistic < critical_values,则表示在相应的significance_level下,接受原假设,认为样本数据来自给定的正态分布。'''    
        # print(stats.anderson(data, dist=dist_type))  # 'norm', 'expon', 'gumbel', 'extreme1', 'logistic'

        #输出(统计量的值,P值)=(0.019125294462402076, 0.48168672320192013)，P值>指定水平0.05,接受原假设，可以认为样本数据服从正态分布
        # print(lilliefors(data))

        # t='norm'
        # check_dist(dev_wordfeats_df, 'TRTAvg')
        # check_distribution(train_wordfeats_df, 'TRTAvg', dist_type=t)
        # check_distribution(dev_wordfeats_df, 'TRTStd', dist_type=t)
        # check_distribution(train_wordfeats_df, 'TRTStd', dist_type=t)
        return None



class LoadData():
    def load_features(input_filepath, cols=[]):
        subsets = {}
        features_df = pd.read_csv(input_filepath)
        for dsname, subds_df in features_df.groupby('dsname'):
            # print(dsname)
            # print(subds_df[cols].describe())
            if cols == []: 
                subsets[dsname] = subds_df
            else:
                subsets[dsname] = subds_df[cols]
        # print(subsets)
        return subsets

    
    def load_sentences():
        pass


def xx():
    train_d = np.array(train_df['FFDAvg'].tolist()).reshape(-1,1)
    dev_d = np.array(dev_df['FFDAvg'].tolist()).reshape(-1,1)
    # x = np.random.random(1000).reshape(-1,1)
    # km = KMeans(n_clusters=5,max_iter=1000).fit(x)
    # print(km.cluster_centers_)
    # y = KMeans(n_clusters=5,max_iter=1000).fit_predict(x) # 预测每个sample属于哪一类
    # print(y)

    kms = []
    for id, l in enumerate(km.labels_.tolist()):
        kms.append(km.cluster_centers_[l][0])
    print('pred::',kms[:20])
    print('true::',dev_d[:20].flatten().tolist())
    print(km.labels_[:20])
    print(km.inertia_)

    kms = []
    for id, l in enumerate(km.labels_.tolist()):
        kms.append(km.cluster_centers_[l][0])


class KM():
    def ClusterNum_analysis(input_filepath, output_folder='', cols=['TRTAvg'], max_cluster=20):
        distortions_dfs = []
        features_df = pd.read_csv(input_filepath)
        for dsname, subds_df in features_df.groupby('dsname'):
            distortions = []
            X = np.array(subds_df[cols].values.tolist()).reshape(-1, 1)
            clusters_num = range(1, max_cluster + 1)
            for i in clusters_num:
                kmeans = KMeans(n_clusters = i)
                kmeans.fit(X)
                distortions.append(kmeans.inertia_)
            distortions_df = pd.DataFrame(distortions, columns=[dsname])
            distortions_dfs.append(distortions_df)
            plt.plot(clusters_num, distortions, label=dsname)
            plt.legend()
        df = pd.concat(distortions_dfs, axis=1)
        plt.xlabel('Number of Clusters')
        plt.ylabel('distortions')
        if output_folder != '':
            plt.savefig(output_folder + 'cluster_num_analysis.png')

        return df
    
    def clusters_analysis(input_filepath, output_folder='', cols=['TRTAvg', 'TRTStd'], cluster_num=10):
        features_df = pd.read_csv(input_filepath)
        for dsname, subds_df in features_df.groupby('dsname'):
            X = np.array(subds_df[cols[0]].values.tolist()).reshape(-1, 1)
            # X2 = np.array(subds_df[cols[1]].values.tolist()).reshape(-1, 1)
            # X2 = np.array(np.ones((len(X)))).reshape(-1, 1)
            X2 = np.array([x[0] + 10 for x in X]).reshape(-1, 1)
            
            km = KMeans(n_clusters = cluster_num)
            km.fit(X)
            cluster_preds = km.fit_predict(X)  # 预测每个sample属于哪个类
            cluster_labels = km.labels_ # 聚类结果标签
            centers = km.cluster_centers_ # 聚类中心
            centers2 = np.array([x[0] + 10 for x in centers]).reshape(-1, 1)
            plt.cla()
            plt.scatter(X, X2, c=cluster_preds)
            plt.scatter(centers, centers2, c = 'yellow', label = 'Centroids')
            plt.legend()

            print(dsname)
            # print('X:', X)
            # print('X2: ', X2)
            print('聚类后类标：', cluster_labels)
            print('聚类质心：', centers)
            # for i in range(len(X)):
            #     cluster_id = cluster_labels[i]
            #     print(X[i][0], centers[cluster_id][0], cluster_id)
            if output_folder != '':
                plt.savefig(output_folder + dsname + '_3.png')


##### features data #####
ip_folder = '/home/shukai/CMCL2022_GPT3/data/output/'

# output
# timeSG = datetime.now(pytz.timezone('Asia/Singapore')).strftime("%m%d%H%M")
op_folder = '/home/shukai/CMCL2022_GPT3/analysis/122019/'
# os.mkdir(op_folder) if not os.path.exists(op_folder) else print('Folder exists!!', op_folder)


##### generate figs #####
KM.ClusterNum_analysis(ip_folder + 'features_nontest.csv')
KM.clusters_analysis(ip_folder + 'features_nontest.csv', output_folder=op_folder)
# nontest_features_dfs = LoadData.load_features(ip_folder + 'features_nontest.csv', cols=['TRTAvg'])


# # plot_his(mergedfeats_df, 'TRTStd', bin_n=100, fig_name = op_folder + 'TRTStd' + '_merge.png')
# feats = ['FFDAvg','FFDStd','TRTAvg','TRTStd']
# feats = ['cur_word_len', 'cur_word_logfreq', 'upper_rate', 'cap_rate', 'is_upper_W', 'is_capital_start_W']
# print(mergedfeats_df[feats].describe())
# # print(train_wordfeats_df[feats].describe())
# # print(dev_wordfeats_df[feats].describe())
# plot_his(mergedfeats_df, feature_name='is_capital_start_W', bin_n=50, fig_name=op_folder+ 'is_capital_start_W' +'_merge.png')








