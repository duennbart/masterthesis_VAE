from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
from functools import partial
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from general.utilTF1.utils import session
from general.kneeOsteoarthritisDataset.KneeOsteoarthritsDataset import KneeOsteoarthritsDataset

_N_CPU = multiprocessing.cpu_count()


def batch_dataset(dataset, batch_size, prefetch_batch=_N_CPU + 1, drop_remainder=True, filter=None,
                  map_func=None, num_threads=_N_CPU, shuffle=True, buffer_size=4096, repeat=-1):
    if filter:
        dataset = dataset.filter(filter)

    if map_func:
        dataset = dataset.map(map_func, num_parallel_calls=num_threads)

    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    if drop_remainder:
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    else:
        dataset = dataset.batch(batch_size)

    dataset = dataset.repeat(repeat).prefetch(prefetch_batch)

    return dataset


class Dataset(object):

    def __init__(self):
        self._dataset = None
        self._iterator = None
        self._batch_op = None
        self._sess = None

        self._is_eager = tf.executing_eagerly()
        self._eager_iterator = None

    def __del__(self):
        if self._sess:
            self._sess.close()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            b = self.get_next()
        except:
            raise StopIteration
        else:
            return b

    next = __next__

    def get_next(self):
        if self._is_eager:
            return self._eager_iterator.get_next()
        else:
            return self._sess.run(self._batch_op)

    def reset(self, feed_dict={}):
        if self._is_eager:
            self._eager_iterator = tfe.Iterator(self._dataset)
        else:
            self._sess.run(self._iterator.initializer, feed_dict=feed_dict)

    def _bulid(self, dataset, sess=None):
        self._dataset = dataset

        if self._is_eager:
            self._eager_iterator = tfe.Iterator(dataset)
        else:
            self._iterator = dataset.make_initializable_iterator()
            self._batch_op = self._iterator.get_next()
            if sess:
                self._sess = sess
            else:
                self._sess = session()

        try:
            self.reset()
        except:
            pass

    @property
    def dataset(self):
        return self._dataset

    @property
    def iterator(self):
        return self._iterator

    @property
    def batch_op(self):
        return self._batch_op

def get_dataset(data_set_path,shuffle=True):
    # shape
    img_shape = [256,256, 1]
    # dataset
    def _map_func(img,label):
        img = tf.image.resize_images(img, [img_shape[0], img_shape[1]], method=tf.image.ResizeMethod.BICUBIC)
        img = tf.clip_by_value(tf.cast(img, tf.float32), 0, 255) / 255 # / 127.5 - 1

        return img,label

    # get image pathes
    #
    kneeosteo_train = KneeOsteoarthritsDataset(data_path=data_set_path)

    labels = list(kneeosteo_train.dict_url_class.values())
    paths = list(kneeosteo_train.dict_url_class.keys())
    assert (len(paths) == len(labels))
    print('The dataset %s has %f elements. ' % (data_set_path, len(labels)))

    Dataset = partial(DiskImageData, img_paths=paths,labels=labels, repeat=1, map_func=_map_func,shuffle=shuffle)



    # index func
    def get_imgs(batch):
        return batch

    return Dataset, img_shape, get_imgs

def disk_image_batch_dataset(img_paths, batch_size, labels=None, prefetch_batch=_N_CPU + 1, drop_remainder=True, filter=None,
                             map_func=None, num_threads=_N_CPU, shuffle=True, buffer_size=4096, repeat=-1):
    """Disk image batch dataset.

    This function is suitable for jpg and png files

    Arguments:
        img_paths : String list or 1-D tensor, each of which is an iamge path
        labels    : Label list/tuple_of_list or tensor/tuple_of_tensor, each of which is a corresponding label
    """
    if labels is None:
        dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    elif isinstance(labels, tuple):
        dataset = tf.data.Dataset.from_tensor_slices((img_paths,) + tuple(labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))

    def parse_func(path, *label):
        img = tf.read_file(path)
        img = tf.image.decode_png(img, 1)
        return (img,) + label

    if map_func:
        def map_func_(*args):
            return map_func(*parse_func(*args))
    else:
        map_func_ = parse_func

    # dataset = dataset.map(parse_func, num_parallel_calls=num_threads) is slower

    dataset = batch_dataset(dataset, batch_size, prefetch_batch, drop_remainder, filter,
                            map_func_, num_threads, shuffle, buffer_size, repeat)

    return dataset


class DiskImageData(Dataset):
    """DiskImageData.

    This class is suitable for jpg and png files

    Arguments:
        img_paths : String list or 1-D tensor, each of which is an iamge path
        labels    : Label list or tensor, each of which is a corresponding label
    """

    def __init__(self, img_paths, batch_size, labels=None, prefetch_batch=_N_CPU + 1, drop_remainder=True, filter=None,
                 map_func=None, num_threads=_N_CPU, shuffle=True, buffer_size=4096, repeat=-1, sess=None):
        super(DiskImageData, self).__init__()
        dataset = disk_image_batch_dataset(img_paths, batch_size, labels, prefetch_batch, drop_remainder, filter,
                                           map_func, num_threads, shuffle, buffer_size, repeat)
        self._bulid(dataset, sess)
        self._n_data = len(img_paths)

    def __len__(self):
        return self._n_data
