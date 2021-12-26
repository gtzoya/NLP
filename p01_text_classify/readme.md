
### 简介
功能：中文文本分类模型的训练和测试，主要原理是朴素贝叶斯模型   
数据预处理：使用jieba分词、去除停用词后，建立词袋，借助TF-IDF将文本转成向量
模型训练：朴素贝叶斯

### 运行环境(Environment of software)
```
python 3.7.4
pandas 0.25.1
sklearn 0.21.3
jieba  0.42.1  
```

### 数据集来源(Data resource)
清华NLP组提供的THUCNews新闻文本分类数据集的一个子集  
数据集协议：CC0 公共领域共享 https://creativecommons.org/publicdomain/zero/1.0/deed.zh  

### 文件目录(file directions)
```
└─p01_text_classify
    ├─data
    │  └─cnews     # 训练和测试数据路径
    ├─resource     # 停用词文件路径 
    └─src          # python 脚本路径
```

### 运行脚本(run scripts)

```
cd ./p01_test_classify/src
python text_classify.py
```

### 测试的日志(log of test)
```
2021-12-26 16:45:43:  stop words has been loaded

 - - - - - - - processing training data- - - - - -
2021-12-26 16:45:44:  ..\data\cnews\cnews.train.txt has been loaded
2021-12-26 16:49:02:  texts have been converted to words
2021-12-26 16:49:11:  train data have been converted to be vec

 - - - - - - - processing testing data - - - - - -
2021-12-26 16:49:11:  ..\data\cnews\cnews.test.txt has been loaded
2021-12-26 16:49:54:  texts have been converted to words
2021-12-26 16:49:56:  test data has been converted to be vec

 - - - - - - - training - - - - - -

 - - - - - - - testing - - - - - -

 - - - - - - - report - - - - - - - - 

 confusion matix:
[[998   1   0   0   0   0   1   0   0   0]
 [  1 989   0   0   1   2   2   3   2   0]
 [  0  22 369 530  24   9  20   8  10   8]
 [  1   6   9 922  12   2  18   1   1  28]
 [  1  11   2   9 945   1   1   3  23   4]
 [  0  15   2   1   2 974   1   1   4   0]
 [  0   3   0  44  35   0 907   1   4   6]
 [  0  13   0   3   3   3   1 971   6   0]
 [  0   3   0   4   5   0   0   2 986   0]
 [  0   0   0   6   0   0   3   0   0 991]]

 precision & recall:
              precision    recall  f1-score   support

          体育       1.00      1.00      1.00      1000
          娱乐       0.93      0.99      0.96      1000
          家居       0.97      0.37      0.53      1000
          房产       0.61      0.92      0.73      1000
          教育       0.92      0.94      0.93      1000
          时尚       0.98      0.97      0.98      1000
          时政       0.95      0.91      0.93      1000
          游戏       0.98      0.97      0.98      1000
          科技       0.95      0.99      0.97      1000
          财经       0.96      0.99      0.97      1000

    accuracy                           0.91     10000
   macro avrg       0.92      0.91      0.90     10000
weighted avrg       0.92      0.91      0.90     10000

Total elapse: 253.36621832847595
The End ... ...
```

### 引用（citation）
```
Author:gtzoya
url: 
email: gtzoya@163.com
```