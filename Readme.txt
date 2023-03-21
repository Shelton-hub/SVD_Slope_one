环境：python3.7 使用科学计算库numpy 实验数据集movielens,一个电影评分的数据集,在ml-100k文件夹中

本次SVD_slopeone项目有两个版本:
1.正常的slopeone,在文件SVD_Slopeone.py
2.乘法版本的slopeone,在文件SVD_Slopeone_mulversion.py

主要流程：
1.使用biasSVD将一半的训练集（选训练集的一半用户，一半电影）训练,biasSVD用随机梯度下降法训练
2.将训练集中未使用的评分补上，将SVD预测的评分补上
3.运行slopeonne算法,补全剩下的评分

注意：
实验运行过程由于slopeone要计算项目之间流行度差异，频数，考虑到实际计算过程很久，故将此表信息存储在表ml-100k/devi.txt和ml-100k/devimul.txt中，
分别对应原本slopeone版本和乘法版本的slopeone中;这样下次访问的时候只需要访问此文件即可

最终的代码在ml-100k/u1.base作为训练集，ml-100k/u1.test作为测试集实验
这是以原数据ml-100k/u.data八二比例随机分开得到的