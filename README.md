# mkl_tools
多核学习工具箱
## 主要是使用合成核方法进行多核学习
## 使用说明
该软件包依赖 python2.7,sklearn,numpy,cvxopt 

将特征文件放入feature文件夹（没有数量限制）

填写gamma.csv配置文件

gamma.csv中共两行，第一行填写特征文件的文件名，第二行填写相应特征对应的rbf核的gamma参数

## 运行：

## 平均权重合成核

  python averageMKL_svm.py [-cv] [-l]
  
  两个参数都是可选参数
  
  -cv 交叉验证的折数，默认为5
  
  -l 如果为平衡数据集，且前一半为正例，后一半为反例，不需要此参数。
  
  非平衡数据集需要自己准备一个一维的标签文件

## EasyMKL权重合成核：
  
  python easyMKL_svm.py [-cv] [-l]
  
  用法同上


程序运行完会生成一个csv的结果文件，第一列为参数c的值，第二列为acc 第三列为auc_roc
