'''
Created on 04.10.2016

@author: Daniel Stromer
@modified by Charly, Christian, Max (23.12.2020)
'''
import numpy as np
import matplotlib.pyplot as plt
# do not import more modules!


def calculate_R_Distance(Rx, Ry):
    '''
    calculate similarities of Ring features
    :param Rx: Ring features of Person X
    :param Ry: Ring features of Person Y
    :return: Similiarity index of the two feature vectors
    '''

    return np.mean(np.abs(Rx - Ry))

def calculate_Theta_Distance(Thetax, Thetay):
    '''
    calculate similarities of Fan features
    :param Thetax: Fan features of Person X
    :param Thetay: Fan features of Person Y
    :return: Similiarity index of the two feature vectors
    '''

    n = len(Thetax)

    # Calculate the means of Thetax and Thetay
    mean_Thetax = np.sum(Thetax) / n
    mean_Thetay = np.sum(Thetay) / n

    lxx = np.sum((Thetax - mean_Thetax) ** 2)
    lyy = np.sum((Thetay - mean_Thetay) ** 2)
    lxy = np.sum((Thetax - mean_Thetax) * (Thetay - mean_Thetay))
    D_theta_xy = (1 - ((lxy * lxy) / (lxx * lyy))) * 100

    return D_theta_xy