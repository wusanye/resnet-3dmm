#############################################
# Copyright (c) 2018-present
# written by Kai Wu on 2018-08-15
#############################################

"""This is implementation of test for 3dmm"""

import os
import sys
import h5py
import ntpath
import numpy as np
import open3d
from utils import read_txt
from resnet_variant import ResNet


# project coefficients to 3dmm model
def project2bfm09(coefs, use_std=True):

    bfm09h5 = h5py.File("./bfm09/bfm09withexp.h5", 'r')

    shape_mu = bfm09h5['shape/mean'].value
    exp_mu = bfm09h5['expression/mean'].value

    shape_basis = bfm09h5['shape/basis'].value[:, :100]
    exp_basis = bfm09h5['expression/basis'].value

    shape_std = bfm09h5['shape/std'].value[:100]
    exp_std = bfm09h5['expression/std'].value

    # shape recovery
    alpha_id = coefs[:100]
    if use_std:
        alpha_id = np.multiply(shape_std, alpha_id)
    shape = np.matmul(shape_basis, alpha_id)

    shape = shape + shape_mu

    num_vertex = int(shape.shape[0] / 3)
    shape = np.reshape(shape, (num_vertex, 3))

    # expression recovery
    alpha_exp = coefs[100:179]
    if use_std:
        alpha_exp = np.multiply(exp_std, alpha_exp)
    exp = np.matmul(exp_basis, alpha_exp)

    exp = exp + exp_mu

    exp = np.reshape(exp, (num_vertex, 3))

    bfm09h5.close()

    return shape, exp


if __name__ == '__main__':

    # if len(sys.argv) < 3 or len(sys.argv) > 4:
    #    print("Usage: python test_3dmm.py <modelPath> <inputList> <outputDir>")

    model_path = "./experiment/resnet-variant/model_epoch{}.ckpt".format(100)  # sys.argv[1]
    file_list = './list/performance.list'  # sys.argv[2]
    out_dir = "out/"  # sys.argv[3]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(file_list, 'r') as ins:

        for img_path in ins:

            if len(img_path) < 6:
                print("Skipping" + img_path + ", file path too short")
                continue

            img_path = img_path[:-1]

            predict = ResNet.inference(model_path, img_path)

            predict = predict.reshape(185)

            im_name = ntpath.basename(img_path)
            im_name = im_name.split(im_name.split('.')[-1])[0][0:-1]

            out_file = out_dir + '/' + im_name

            np.savetxt(out_file + '_est.txt', predict, fmt="%f")

            est_shape, est_exp = project2bfm09(predict)

            est_geom = est_shape + est_exp

            est = open3d.PointCloud()
            est.points = open3d.Vector3dVector(est_geom)

            label_path = img_path[:-3] + 'txt'
            label = np.asarray(read_txt(label_path))
            truth_shape, truth_exp = project2bfm09(label, use_std=False)

            truth_geom = truth_shape + truth_exp

            truth = open3d.PointCloud()
            truth.points = open3d.Vector3dVector(truth_geom + np.array([200000, 0, 0]))

            open3d.draw_geometries([est, truth])















