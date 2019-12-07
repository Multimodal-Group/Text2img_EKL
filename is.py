import myutils
import tensorflow as tf
import numpy as np
import inception_score  ## will occupy huge memory
import fid


dataset= 'fashion'
dataset= 'cifar10'

if dataset=='fashion':
  path= './fid_stats_fashion_train.npz'
else:
  path= './fid_stats_cifar10_train.npz'
f = np.load(path)
m1, s1 = f['mu'][:], f['sigma'][:]
f.close()
inception_path = './tmp/imagenet'
inception_path = fid.check_or_download_inception(inception_path)

## evaluate
img_root= 'cifar10_cbg_att_circle3_ublr_epoch295_gen_tn2'
img_root= 'cifar10_cbg_att_circle3_ublr_epoch295_gen_tn1.5'
img_root= 'cifar10_cbg_att_circle3_ublr_epoch295_gen_tn1.0'
img_root= 'cifar10_cbg_att_circle3_ublr_epoch295_gen_tn0.5'
img_root= 'cifar10_cbg_att_circle3_bs500_epoch214_gen'
img_root= 'cifar10_cbg_att_circle3_bs500_epoch211_gen'
img_root= 'cifar10_cbg_att_circle3_ublr2_bs500_epoch418_gen'
img_root= 'cifar10_cbg_att_circle3_bs500_epoch211_gen_tn2.0'

## cave
img_root= './cptbrg-att-tsf/cifar10_cvae_sum2_epoch162_gen'
img_root= './cptbrg-att-tsf/cifar10_cvae_sum2_epoch162_gen_tn2.0'
img_root= './cptbrg-att-tsf/cifar10_cvae_sum2_epoch162_gen_tn1.5'
img_root= './cptbrg-att-tsf/cifar10_cvae_sum2_epoch162_gen_tn1.0'
img_root= './cptbrg-att-tsf/cifar10_cvae_sum2_epoch162_gen_tn0.5'

## fashion
img_root= './cptbrg-att-tsf/fashion_cbg_circle_epoch499_gen'
img_root= './cptbrg-att-tsf/fashion_cbg_circle_epoch499_gen_tn1.5'
img_root= './cptbrg-att-tsf/fashion_cbg_circle_blr_tw_epoch138_gen'
img_root= './cptbrg-att-tsf/fashion_cbg_circle_macf_epoch127_gen' # is 4.87
img_root= './cptbrg-att-tsf/fashion_cbg_circle_macf_epoch82_gen' # fid 22.1

## cifar with best vcgan
# img_root= './best-vcgan/cifar10_vcgan_bnenc_epoch352_gen' ## fid 33.7
# img_root= './best-vcgan/cifar10_vcgan_bnenc_epoch352_gen_tn2.0'
# img_root= './best-vcgan/cifar10_vcgan_bnenc_epoch352_gen_tn1.5'
img_root= './best-vcgan/cifar10_vcgan_bnenc_epoch352_gen_tn1.0'
img_root= './best-vcgan/cifar10_vcgan_bnenc_epoch352_gen_tn0.5'
img_root= './best-vcgan/cifar10_vcgan_bnenc_epoch292_gen' ## is 8.7

## sentence level , cub_catcls
img_root= './Testset_evalmode_varnoise_epoch187_birds_birds-twostages-x2-cat_2019_06_01_15_06_48/single_samples'

root= True
fid= False
n=50000

if root:
  imgs= myutils.imgs2ndarray(img_root, save=False)
  # assert len(imgs)==n
  print('load imgs ok')
  print('sizeof imgs: %d'%len(imgs))

  ## cal is by root dir
  print('>>>>>>>>>>get IS...')      
  print('IS of %s:'%img_root, inception_score.get_inception_score(imgs))

  ## cal fid
  if fid:
      imgs= np.stack(imgs, axis= 0)  ## 3d list -> 4d
      assert imgs.shape[0]==n
      print('>>>>>>>>>>get FID...')
      fid.create_inception_graph(inception_path)  # load the graph into the current TF graph    
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        single= True
        # single=False
        if single:
          m2, s2 = fid.calculate_activation_statistics(imgs[: 10000], sess, batch_size=100, verbose= True)
          fid_value = fid.calculate_frechet_distance(m1, s1, m2, s2)
          print('Gen %d, model %s, FID: %.3f\n'%(10000, img_root, fid_value))
        else:
          m= 10000
          for i in range(n//m):
            m2, s2 = fid.calculate_activation_statistics(imgs[i*m: (i+1)*m], sess, batch_size=100, verbose= True)
            fid_value = fid.calculate_frechet_distance(m1, s1, m2, s2)
            print('Gen %d, model %s, FID: %.3f\n'%(10000, img_root, fid_value))
else:
  ## cal is realtime
  print('>>>>>>>>>>get IS realtime...')
