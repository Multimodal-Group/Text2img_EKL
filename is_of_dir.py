# import myutils
import tensorflow as tf
import numpy as np
import inception_score  ## will occupy huge memory
# import fid
import scipy
import os
import random


# dataset= 'fashion'
# dataset= 'cifar10'

# if dataset=='fashion':
#   path= './fid_stats_fashion_train.npz'
# else:
#   path= './fid_stats_cifar10_train.npz'
# f = np.load(path)
# m1, s1 = f['mu'][:], f['sigma'][:]
# f.close()
# inception_path = './tmp/imagenet'
# inception_path = fid.check_or_download_inception(inception_path)

def load_data(fullpath):
    print('Eval data: '+ fullpath)
    images = []
    for path, subdirs, files in os.walk(fullpath):
        for name in files:
            if name.rfind('jpg') != -1 or name.rfind('png') != -1:
                filename = os.path.join(path, name)
                # print('filename', filename)
                # print('path', path, '\nname', name)
                # print('filename', filename)
                if os.path.isfile(filename):
                    img = scipy.misc.imread(filename)
                    images.append(img)
    print('Load images ok, images', len(images), images[0].shape)
    random.shuffle(images)  ## random images

    return images

# img_root= './eval_epoch250_birds_Twostages-x2_2019_02_28_23_31_54/single_samples'
# img_root= './eval_epoch206_birds_Twostages-x2_2019_02_28_23_31_54/single_samples'
# img_root= './eval_epoch150_birds_Twostages-x2_2019_02_28_23_31_54/single_samples'

# img_root= './eval_epoch291_flowers_Flowers-twostages-x2_2019_03_02_13_46_10/single_samples'
# img_root= './eval_epoch253_flowers_Flowers-twostages-x2_2019_03_02_13_46_10/single_samples'
# img_root= './eval_epoch298_flowers_Flowers-twostages-x2_2019_03_02_13_46_10/single_samples'
# img_root= './eval_epoch232_flowers_Flowers-twostages-x2_2019_03_02_13_46_10/single_samples'

## two-path eval
img_root= './eval_epoch211_flowers_Flowers-twostages-x2_2019_03_07_20_46_44/single_samples'
img_root= './eval_epoch211_flowers_Flowers-twostages-x2_2019_03_07_20_46_44_EVAL_MODE/single_samples'

img_root= './eval_images_epoch299_birds_Twopath-twostages_2019_02_20_21_59_26/single_samples'
# img_root= './eval_varnoise_epoch249_birds_Twopath-twostages_2019_02_20_21_59_26/single_samples'
# img_root= './eval_varnoise_epoch299_birds_Twopath-twostages_2019_02_20_21_59_26/single_samples'

img_root= './eval_varnoise_epoch211_flowers_Flowers-twostages-x2_2019_03_07_20_46_44_test/single_samples'
img_root= './Testset_trainmode_varnoise_epoch211_flowers_Flowers-twostages-x2_2019_03_07_20_46_44/single_samples'

root= True
# n=50000
if root:
  # imgs= myutils.imgs2ndarray(img_root, save=False)
  imgs= load_data(img_root)
  # assert len(imgs)==n  

  ## cal is by root dir
  print('>>>>>>>>>>get IS...')      
  print('IS of %s:'%img_root, inception_score.get_inception_score(imgs))

  ## cal fid
  # imgs= np.stack(imgs, axis= 0)  ## 3d list -> 4d
  # assert imgs.shape[0]==n
  # print('>>>>>>>>>>get FID...')
  # fid.create_inception_graph(inception_path)  # load the graph into the current TF graph    
  # with tf.Session() as sess:
  #   sess.run(tf.global_variables_initializer())
  #   single= True
  #   # single=False
  #   if single:
  #     m2, s2 = fid.calculate_activation_statistics(imgs[: 10000], sess, batch_size=100, verbose= True)
  #     fid_value = fid.calculate_frechet_distance(m1, s1, m2, s2)
  #     print('Gen %d, model %s, FID: %.3f\n'%(10000, img_root, fid_value))
  #   else:
  #     m= 10000
  #     for i in range(n//m):
  #       m2, s2 = fid.calculate_activation_statistics(imgs[i*m: (i+1)*m], sess, batch_size=100, verbose= True)
  #       fid_value = fid.calculate_frechet_distance(m1, s1, m2, s2)
  #       print('Gen %d, model %s, FID: %.3f\n'%(10000, img_root, fid_value))
else:
  ## cal is realtime
  print('>>>>>>>>>>get IS realtime...')
