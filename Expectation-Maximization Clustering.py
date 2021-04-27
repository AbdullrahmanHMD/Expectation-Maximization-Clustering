import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import math
import pandas as pd
from random import uniform
from scipy.stats import multivariate_normal
import matplotlib.patches as mp

np.random.seed(152)
#-------------------------------------------------------------------------------
#--------------------------------Initializing Data------------------------------
N = [50, 50, 50, 50, 100]
N = np.array(N)

sample_means = [[2.5, 2.5],
                [-2.5, 2.5],
                [-2.5, -2.5],
                [2.5, -2.5],
                [0.0, 0.0]]

cov_matrices = [[[0.8, -0.6],[-0.6, 0.8]],
                [[0.8, 0.6],[0.6, 0.8]],
                [[0.8, -0.6],[-0.6, 0.8]],
                [[0.8, 0.6],[0.6, 0.8]],
                [[1.6, 0.0],[0.0, 1.6]]]

data_points = [np.random.multivariate_normal(a, b, c) for a, b, c in zip(sample_means, cov_matrices, N)]
data_points = np.concatenate((data_points[0], data_points[1], data_points[2], data_points[3], data_points[4]))

K = 5

#-------------------------------------------------------------------------------
#--------------------------------Initializing Centoids--------------------------
minimum_point = (min(data_points[:,0]), min(data_points[:,1])) # Not a real point, used to find the minimum x and y to initialize the centroids.
maximum_point = (max(data_points[:,0]), max(data_points[:,1])) # Not a real point, used to find the maximum x and y to initialize the centroids.

def initializeCentroids():
    centroids = []
    for i in range(K):
        centroids.append((uniform(minimum_point[0],maximum_point[0]), uniform(minimum_point[1],maximum_point[1])))
    return np.array(centroids)

centroids = initializeCentroids()

#-------------------------------------------------------------------------------
#--------------------------------Helper Functions-------------------------------
def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def distance_matrix(centroids, points):
    return np.array([[distance(a, point) for a in centroids] for point in points])

def estimated_sample_mean(Z, data_points, N): # TODO: Refactor.
    return np.array([(lambda a, x: a / x if x != 0 else [0.001, 0.001])(a, b) for a, b in zip(np.dot(Z.T, data_points), N)])

def estimated_covariance_matrices(Z, data_points, sample_mean, N):
    mat = []
    row = []

    for i in range(len(sample_mean)):
        for j in range(len(data_points)):
            a = np.array((data_points[j] - sample_mean[i]).reshape((2,1)))
            row.append(Z[:,i][j] * np.dot(a, a.T))
        mat.append(sum(row) / N[i])
        row = []
    return np.array(mat)

def number_of_points_in_clusters(z_values):
    return np.dot(np.ones(shape = (1, len(data_points))), z_values)[0]

#-------------------------------------------------------------------------------
#--------------------------------Cluster memberships Functions------------------

def Z_values(distances_matrix): # returns one-hot encoded Z matrix
    z_values = np.zeros((len(distances_matrix), len(distances_matrix[0])))
    for i in range(len(distances_matrix)):
        z_values[i][np.argmin(distances_matrix[i])] = 1

    return np.array(z_values)

def EM_cluster_memberships(h_ik, data_points):
    mat = []
    row = []
    for i in range(K):
        for j in range(len(data_points)):
            if np.argmax(h_ik[j]) == i:
                row.append(data_points[j])
        mat.append(np.array(row))
        row = []
    return np.array(mat)

#----------------------------------------------------------------------------------------
#--------------------------------Running k-means for 2 iterations------------------------
distances_matrix = distance_matrix(centroids, data_points)

Z = Z_values(distances_matrix)

N_c = number_of_points_in_clusters(Z)

k_means_sample = estimated_sample_mean(Z, data_points, N_c)

def k_means(iterations, data_points):

    centroids = initializeCentroids()

    distances_matrix = distance_matrix(centroids, data_points)
    Z = Z_values(distances_matrix)
    N_c = number_of_points_in_clusters(Z)
    
    k_means_sample = estimated_sample_mean(Z, data_points, N_c)

    for i in range(iterations):
        distances_matrix = distance_matrix(centroids, data_points)
        Z = Z_values(distances_matrix)
        N_c = number_of_points_in_clusters(Z)
        k_means_sample = estimated_sample_mean(Z, data_points, N_c)
        centroids = k_means_sample
        
    return centroids, distances_matrix, N_c

centroids, distances_matrix, N_c = k_means(2, data_points)

#-------------------------------------------------------------------------------
#--------------------------------EM Alogorithm----------------------------------
def multivariate_normals(points, sample_mean, cov_matrix):
    return np.array([multivariate_normal.pdf(point, mean=sample_mean, cov=cov_matrix, allow_singular=True) for point in points])

def multivariate_normals_matrix(data_points, centroids, cov_matrices):
    return np.array([multivariate_normals(data_points, mean, cov) for mean, cov in zip(centroids, cov_matrices)])

def priors(points, N):
    return np.array([p / N for p in points])

def point_per_cluster(hik):
    return np.array([sum(c) for c in hik.T])

estimated_priors = priors(N_c, len(data_points))
estimated_cov_matrices = estimated_covariance_matrices(Z, data_points, centroids, N_c)
multi_mat = multivariate_normals_matrix(data_points, centroids, estimated_cov_matrices)

def hik_matrix(priors, multi_mat):
    mat = [a * b for a, b in zip(multi_mat, priors)]
    return np.divide(mat, np.dot(priors, multi_mat))

def EM(data_points, h_ik, iterations):
    e_mean, distances_matrix, n_c = k_means(2, data_points)
    hik = h_ik
    for i in range(iterations):

        e_mean = estimated_sample_mean(hik, data_points, n_c)
        e_priors = priors(n_c, len(data_points))
        e_cov_mats = estimated_covariance_matrices(hik, data_points, e_mean, n_c)
        m_mat = multivariate_normals_matrix(data_points, e_mean, e_cov_mats)
        hik = hik_matrix(e_priors, m_mat).T
        n_c = point_per_cluster(hik)

        p = EM_cluster_memberships(hik, data_points)
        
    return p , e_mean, e_cov_mats

hik = hik_matrix(estimated_priors, multi_mat).T
p, centroids, estimated_cov_matrices = EM(data_points, hik, 50)

print(pd.DataFrame(centroids))

#-------------------------------------------------------------------------------
#--------------------------------Plotting Confidence Ellipses-------------------
fig = plt.figure(figsize=(8,8))

plt.xlabel("x1")
plt.ylabel("x2")

figure_frame = fig.gca()

def confidance_ellipse(s):
    u, s, dummy = np.linalg.svd(s)
    an = np.degrees(np.arctan2(u[1, 0], u[0, 0]))
    w, h = 2 * np.sqrt(s)
    return an, w, h

for cov, mean in zip(cov_matrices, sample_means):
    _angle, _width, _height = confidance_ellipse(cov)
    con_ellipse = mp.Ellipse(xy=mean, width=_width, height=_height, angle=_angle,edgecolor='black', facecolor='none', linestyle='solid')
    figure_frame.add_artist(con_ellipse)

for cov, mean in zip(estimated_cov_matrices, centroids):
    _angle, _width, _height = confidance_ellipse(cov)
    con_ellipse = mp.Ellipse(xy=mean, width=_width, height=_height, angle=_angle, edgecolor='black', facecolor='none', linestyle='dashed')
    figure_frame.add_artist(con_ellipse)

#-------------------------------------------------------------------------------
#--------------------------------Plotting Data----------------------------------

plt.plot(data_points[:,0], data_points[:,1], 'k.')
colors = ['bo', 'ro','go', 'mo','yo']

for i in range(K):
    plt.plot(p[i][:,0], p[i][:,1], colors[i])

plt.plot(centroids[:,0], centroids[:,1], 'ko')
plt.show()
