# GA_for_Feature_Selection
使用遗传算法结合决策树做特征选择
Using genetic algorithm for feature selection with decision tree
* 原始遗传算法参考https://blog.csdn.net/czrzchao/article/details/52314455
```jupyter notebook
import numpy as np
import pandas as pd
import random
```

```jupyter notebook
data_train = pd.read_csv('\data_train.csv')
data_test = pd.read_csv('\data_test.csv')
```
```jupyter notebook
#合并训练集测试集
data = data_train.append(data_test).drop(['id'], axis=1)
feature_names = data.columns
```

```jupyter notebook
pop_size = 20      # 种群数量  
max_value = 10      # 基因中允许出现的最大值  
chrom_length = 42       # 染色体长度  
pc = 0.6            # 交配概率  
pm = 0.01           # 变异概率  
results = [[]]      # 存储每一代的最优解，N个二元组  
fit_value = []      # 个体适应度  
fit_mean = []       # 平均适应度  

#初始化种群
def geneEncoding(pop_size, chrom_length):  
    pop = [[]]  
    for i in range(pop_size):  
        temp = []  
        for j in range(chrom_length):  
            temp.append(random.randint(0, 1))  
        pop.append(temp)  
  
    return pop[1:] 
pop = geneEncoding(pop_size, chrom_length)
```

```jupyter notebook
from sklearn import tree
from sklearn.model_selection import cross_val_score

#提取数据与标签，根据自己的数据集作相应处理
data_X = data.drop(['label'], axis=1)
data_y = data.label

#计算适应度，使用决策树的准确率作为适应度
def calFitness(pop, chrom_length, max_value, data_X, data_y, feature_names):  
    
    obj_value = []     
    for i in range(len(pop)): 
        data_X_test = data_X
        for j in range(len(pop[i])):
            if pop[i][j] == 0:
                data_X_test = data_X_test.drop([feature_names[j]], axis=1)
        clf = tree.DecisionTreeClassifier()
        score = cross_val_score(clf, data_X_test, data_y, cv=5, scoring='f1').mean()
        obj_value.append(score)
    return obj_value
```

```jupyter notebook
#种群选择
def sum(fit_value):  
    total = 0  
    for i in range(len(fit_value)):  
        total += fit_value[i]  
    return total  
  
  
def cumsum(fit_value):  
    for i in range(len(fit_value)-2, -1, -1):  
        t = 0  
        j = 0  
        while(j <= i):  
            t += fit_value[j]  
            j += 1  
        fit_value[i] = t  
        fit_value[len(fit_value)-1] = 1  
  
  
def selection(pop, fit_value):  
    newfit_value = []  
    # 适应度总和  
    total_fit = sum(fit_value)  
    for i in range(len(fit_value)):  
        newfit_value.append(fit_value[i] / total_fit)  
    # 计算累计概率  
    cumsum(newfit_value)  
    ms = []  
    pop_len = len(pop)  
    for i in range(pop_len):  
        ms.append(random.random())  
    ms.sort()  
    fitin = 0  
    newin = 0  
    newpop = pop  
    # 转轮盘选择法  
    while newin < pop_len:  
        if(ms[newin] < newfit_value[fitin]):  
            newpop[newin] = pop[fitin]  
            newin = newin + 1  
        else:  
            fitin = fitin + 1  
    pop = newpop
```

```jupyter notebook
#交配
def crossover(pop, pc):  
    pop_len = len(pop)  
    for i in range(pop_len - 1):  
        if(random.random() < pc):  
            cpoint = random.randint(0,len(pop[0]))  
            temp1 = []  
            temp2 = []  
            temp1.extend(pop[i][0:cpoint])  
            temp1.extend(pop[i+1][cpoint:len(pop[i])])  
            temp2.extend(pop[i+1][0:cpoint])  
            temp2.extend(pop[i][cpoint:len(pop[i])])  
            pop[i] = temp1  
            pop[i+1] = temp2
```

```jupyter notebook
#变异
def mutation(pop, pm):  
    px = len(pop)  
    py = len(pop[0])  
      
    for i in range(px):  
        if(random.random() < pm):  
            mpoint = random.randint(0, py-1)  
            if(pop[i][mpoint] == 1):  
                pop[i][mpoint] = 0  
            else:  
                pop[i][mpoint] = 1
```

```jupyter notebook
#最优解
def best(pop, fit_value):
    px = len(pop)
    best_individual = []
    best_fit = fit_value[0]
    for i in range(1, px):
        if(fit_value[i] > best_fit):
            best_fit = fit_value[i]
            best_individual = pop[i]
    return [best_individual, best_fit]
```

```jupyter notebook
def mean(obj_value):
    return sum(obj_value)/len(obj_value)
for i in range(50):  
    obj_value = calFitness(pop, chrom_length, max_value, data_X, data_y, feature_names)        # 个体评价  
     
    best_individual, best_fit = best(pop, obj_value)        # 第一个存储最优的解, 第二个存储最优基因  
    results.append([best_individual,best_fit,mean(obj_value)])  
    
    selection(pop, obj_value)       # 新种群复制  
    crossover(pop, pc)      # 交配  
    mutation(pop, pm)       # 变异  
  
results = results[1:]  
results.sort()
```
