import h5py
import numpy as np
import os
import sys
import tarfile
from six.moves import cPickle as pickle
def get_attr(c, i, attr):
    d = c[c['digitStruct']['bbox'][i][0]][attr].value.squeeze()
    if d.dtype == 'float64':
        return d.reshape(-1)
    return np.array([c[x].value for x in d]).squeeze()

def get_label(c, i):
    d = c[c['digitStruct']['name'][i][0]].value.tostring()
    return d.replace('\x00', '')
def load_data(path):
    c = h5py.File(path, 'r')
    images = a = np.ndarray(shape=(c['digitStruct']['name'].shape[0], ), dtype='|S15')
    labels = np.zeros((len(c['digitStruct']['bbox']), 6), dtype=float)
    labels.fill(10)
    tops = np.zeros((len(c['digitStruct']['bbox']), 6), dtype=float)
    heights = np.zeros((len(c['digitStruct']['bbox']), 6), dtype=float)
    widths = np.zeros((len(c['digitStruct']['bbox']), 6), dtype=float)
    lefts = np.zeros((len(c['digitStruct']['bbox']), 6), dtype=float)
    for i in xrange(c['digitStruct']['name'].shape[0]):
        images[i] = get_label(c, i)
        l = get_attr(c, i, 'label')
        t = get_attr(c, i, 'top')
        h = get_attr(c, i, 'height')
        w = get_attr(c, i, 'width')
        le = get_attr(c, i, 'left')
        
        labels[i, :l.shape[0]] = l
        tops[i, :t.shape[0]] = t
        heights[i, :h.shape[0]] = h
        widths[i, :w.shape[0]] = w
        lefts[i, :le.shape[0]] = le
    
        if (i % 5000 == 0):
            print(i, "elapsed")
    
    return labels, images, tops, heights, widths, lefts
train__tuple = load_data('train/digitStruct.mat')
test__tuple = load_data('test/digitStruct.mat')
extra__tuple = load_data('extra/digitStruct.mat')
def maybe_pickle(struct, force=False):
    if os.path.exists(struct + '.pickle') and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % struct)
    else:
      print('Pickling %s.' % struct + '.pickle')
      permutation = np.random.permutation(extra__tuple[1].shape[0])[:2000]
      dataset = {
            'train': {
                'labels': train__tuple[0],
                'images': train__tuple[1],  
                'tops': train__tuple[2],
                'heights': train__tuple[3],
                'widths': train__tuple[4],
                'lefts': train__tuple[5],
                
                
            }, 
            'test': {
                'labels': test__tuple[0],
                'images': test__tuple[1],  
                'tops': test__tuple[2],
                'heights': test__tuple[3],
                'widths': test__tuple[4],
                'lefts': test__tuple[5],
            },
            'extra': {
                'labels': extra__tuple[0],
                'images': extra__tuple[1],  
                'tops': extra__tuple[2],
                'heights': extra__tuple[3],
                'widths': extra__tuple[4],
                'lefts': extra__tuple[5],
            },
            'valid': {
                'labels': extra__tuple[0][permutation],
                'images': extra__tuple[1][permutation],  
                'tops': extra__tuple[2][permutation],
                'heights': extra__tuple[3][permutation],
                'widths': extra__tuple[4][permutation],
                'lefts': extra__tuple[5][permutation],
            }
      }
      try:
        with open( struct + '.pickle', 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to',  struct + '.pickle', ':', e)
  
    return  struct + '.pickle'
maybe_pickle('svhn')
