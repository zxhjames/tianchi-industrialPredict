'''
Author: your name
Date: 2020-11-06 18:11:38
LastEditTime: 2020-11-12 14:53:38
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PyCode/project_demo/TIanchi/p1/test.py
'''

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


train_data_file = "/Users/mac/GitCode/PyCode/project_demo/TIanchi/p1/zhengqi_train.txt"
test_data_file = "/Users/mac/GitCode/PyCode/project_demo/TIanchi/p1/zhengqi_test.txt"

train_data = pd.read_csv(train_data_file,sep='\t',encoding='utf-8')
test_data = pd.read_csv(test_data_file,sep='\t',encoding='utf-8')
train_data.info()
flg = plt.figure(figsize=(4,6))
sns.boxplot(train_data['V0'],orient="v",width=0.5)
plt.show()


# 绘制多个象限图
# column = train_data.columns.tolist()[:39]
# fig = plt.figure(figsize=(80,60),dpi=75)
# for i in range (38):
#     plt.subplot(7,8,i+1)
#     sns.boxplot(train_data[column[i]],orient="v",width=0.5)
#     plt.ylabel(column[i],fontsize=36)
# plt.show()




# 岭回归
# 此方法是采用模型预测的形式，找出异常值
def find_outliers(model,X,y,sigma = 3):
    # 使用模型预测y值
    try:
        y_pred = pd.Series(model.predict(X),index=y.index)
    # 如果预测失败就适应这个模型
    except:
        model.fit(X,y)
        y_pred=pd.Series(model.predict(X),index=y.index)

    # 计算模型预测值与真实值之间的残差
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()

    # 计算统计，定义异常分类
    z = (resid - mean_resid) / std_resid
    outliers = z[abs(z) > sigma].index

    # 打印图和结果
    print('R2=',model.score(X,y))
    print("mse=",mean_squared_error(y,y_pred))
    print('-----------------------------')
    
# 测试数据集合
X_train = train_data.iloc[:,0:-1]
y_train = train_data.iloc[:,-1]
outliers = find_outliers(Ridge(),X_train,y_train)