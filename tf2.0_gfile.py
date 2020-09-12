import tensorflow as tf
import os
import time
'''tensoflowb版本2.x和1.x差别比较大'''

dirname = 'local_run'
print(tf.__version__,type(tf.__version__))
start_time = time.time()
if tf.__version__ < '2.0':
    tf.logging.set_verbosity(tf.logging.INFO)
    if tf.gfile.IsDirectory(dirname):
        filenames = tf.gfile.ListDirectory(dirname)
        tf.logging.info('文件数量：%d' % len(filenames))
        if len(filenames) != 0:
            for filename in filenames:
                filename_path = os.path.join(dirname, filename)
                tf.logging.info('删除文件：%s' % filename_path)
                tf.gfile.Remove(filename_path)
            tf.logging.info('删除文件完毕！！')
        else:
            tf.logging.info('当前目录下没有文件！')
else:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    if tf.io.gfile.isdir(dirname):
        filenames = tf.io.gfile.listdir(dirname)
        tf.compat.v1.logging.info('文件数量：%d' % len(filenames))
        if len(filenames) != 0:
            for filename in filenames:
                filename_path = os.path.join(dirname, filename)
                tf.compat.v1.logging.info('删除文件：%s' % filename_path)
                tf.io.gfile.remove(filename_path)
            tf.compat.v1.logging.info('删除文件完毕！！')
        else:
            tf.compat.v1.logging.info('当前目录下没有文件！')
print(time.time() - start_time)

