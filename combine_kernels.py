#coding=utf-8
import numpy as np
'''
知识点：对于三维矩阵A[n,row,col] n表示第几个二维矩阵；row为二维矩阵的行数；col为二维矩阵的列数
'''
def combine_kernels(weights, kernels):  # weights是一个一维的矩阵，kernels则是三维的矩阵，它是由二维核矩阵拼接成的。
    '''
    :param weights: 
    :param kernels: 
    :return: 
    '''
    result = np.zeros(kernels[1, :, :].shape)
    n = len(weights)
    for i in range(n):
        result = result + weights[i] * kernels[i, :, :]
    return result

