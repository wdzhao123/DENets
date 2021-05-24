import numpy as np
import tensorflow as tf
from Model import Net_test
from Dataloader import Dataloader, Dataloader_test
from PIL import Image
import numpy as np
import time

proto = tf.ConfigProto()
proto.gpu_options.per_process_gpu_memory_fraction = 0.5


config = {
'batch_num':1,
'num_classes':1,
'max_size':(320,320),
'weight_decay': 0.005,
'base_lr': 0.0001,
'momentum': 0.9
}

if __name__ == '__main__':
    split = 'test'
    model = Net_test(config)
    data_loader = Dataloader_test(split, config)
    saver = tf.train.Saver()
    with tf.Session(config=proto) as session:
        saver.restore(session, '../model/final.ckpt')
        print ('Model restored.')
        for i in range(data_loader.num_images):
            minibatch = data_loader.get_minibatch_at(i)
            feed_dict = {model.img: minibatch[0]}
            pred  = session.run(model.get_output('final'), feed_dict=feed_dict)
            pred[0] = pred[0] * 255
            img0 = pred[0].reshape(320, 320)
            im0 = Image.fromarray(img0.astype(np.uint8))
            print (str(i+1) + '/' + str(data_loader.num_images) + ' done!')
            im0.save('../result/result_xu/'+str(i+1)+'.bmp')
