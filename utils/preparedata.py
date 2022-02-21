import numpy as np


def start_points(size: int, split_size: int, overlap: float = None) -> list:
    points = [0]
    if overlap is None:
        overlap = (split_size - size / (size // split_size + 1)) / split_size
    stride = int(split_size * (1 - overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points


def restore_from_patches(patches: np.ndarray, img_shape: (int, int)) -> np.ndarray:
    n_tot, h_patch, w_patch = patches.shape
    h, w = img_shape
    starts_x = start_points(w, w_patch, overlap=None)
    starts_y = start_points(h, h_patch, overlap=None)
    n_patches = len(starts_y) * len(starts_x)
    assert n_tot % n_patches == 0, 'First dimension of input array must be the product between the number of images ' \
                                   'and the number of patches in each image but it seems not to be the case.'
    n_imgs = n_tot // n_patches
    img_restored = np.zeros((n_imgs, h, w), dtype=patches.dtype)
    count = 0
    for k in range(n_imgs):
        for y in starts_y:
            for x in starts_x:
                img_restored[k, y: y+h_patch, x: x+w_patch] = patches[count]
                count += 1
    return img_restored


def normalize(func: np.ndarray, minimum: float = 0., maximum: float = 1.,
              func_min: float = None, func_max: float = None) -> np.ndarray:
    func_min = func.min() if not func_min else func_min
    func_max = func.max() if not func_max else func_max
    func_norm = (func - func_min) / (func_max - func_min) * (maximum - minimum) + minimum
    return func_norm


def to_uint8(dat: np.ndarray, max_val: float = None) -> np.ndarray:
    dat_norm = normalize(dat, maximum=255., minimum=0., func_max=max_val, func_min=0.)
    return np.uint8(np.round(dat_norm))


def label_map(labels: np.ndarray) -> np.ndarray:
    label_map = np.zeros([*labels.shape, 2])
    label_map[labels == 0, 0] = 1
    label_map[labels == 1, 1] = 1
    return label_map


def prepare_samples4prediction(n: int, img_h: int, img_w: int, max_val: float = None,
                               verbose: bool = False) -> np.ndarray:
    file_names = [str(k) if len(str(k)) > 1 else '0' + str(k) for k in range(n)]
    
    x_starts = start_points(2000, img_w, overlap=None)
    y_starts = start_points(2000, img_h, overlap=None)
    N = len(x_starts) * len(y_starts)
    
    X = np.zeros((n*N, img_h, img_w, 1), dtype=np.uint8)
    
    count = 0
    for k, name in enumerate(file_names):
        img = np.load('./data/test/' + name + '.npz')['arr_0']
        for i in y_starts:
            for j in x_starts:
                X[count] = to_uint8(img[i:i + img_h, j:j + img_w, None], max_val=max_val)
                count += 1

    if verbose:
        print('All training data has been loaded from ./data/train')
        print('\tshapes: {}'.format(X.shape))
        print('\ttypes:  {}'.format(X.dtype))
        print('\tmemory: {} MB'.format(X.nbytes / 1048576))

    return X


def prepare_dataset(n: int, img_h: int, img_w: int, max_val: float = None, max_weight: float = None,
                    verbose: bool = False) -> (np.ndarray, np.ndarray, np.ndarray):
    file_names = [str(k) if len(str(k)) > 1 else '0' + str(k) for k in range(n)]
    
    x_starts = start_points(2000, img_w, overlap=None)
    y_starts = start_points(2000, img_h, overlap=None)
    N = len(x_starts) * len(y_starts)
    
    X = np.zeros((n*N, img_h, img_w, 1), dtype=np.uint8)  # dtype=np.float64
    y = np.zeros((n*N, img_h, img_w, 2), dtype=bool)
    w = np.zeros((n*N, img_h, img_w, 1), dtype=np.uint8)  # dtype=np.float32
    
    count = 0
    for k, name in enumerate(file_names):
        img = np.load('./data/train/' + name + '.npz')['arr_0']
        msk = np.load('./data/train/' + name + '-gt.npz')['arr_0'][..., 0]
        wht = np.load('./data/train/' + name + '-weights.npz')['arr_0'][..., 0]

        for i in y_starts:
            for j in x_starts:
                X[count] = to_uint8(img[i:i + img_h, j:j + img_w, None], max_val=max_val)     # img[i:i + img_h, j:j + img_w, None]
                y[count] = label_map(msk[i:i + img_h, j:j + img_w])
                w[count] = to_uint8(wht[i:i + img_h, j:j + img_w, None], max_val=max_weight)  # wht[i:i + img_h, j:j + img_w, None]
                count += 1
    
    if verbose:
        print('All training data has been loaded from ./data/train')
        print('\tshapes: {}, {}, {}'.format(X.shape, y.shape, w.shape))
        print('\ttypes:  {}, {}, {}'.format(X.dtype, y.dtype, w.dtype))
        print('\tmemory: {}, {}, {} MB'.format(X.nbytes / 1048576, y.nbytes / 1048576, w.nbytes / 1048576))
        print('\ttot memory:  {} GB'.format((X.nbytes + y.nbytes + w.nbytes) / 1073741824))

    return X, y, w


def find_max(data_path: str, n: int) -> float:
    file_names = [str(k) if len(str(k)) > 1 else '0' + str(k) for k in range(n)]
    max_vals = np.zeros(n, dtype=np.float64)
    count = 0
    for k, name in enumerate(file_names):
        img = np.load(data_path + name + '.npz')['arr_0']
        max_vals[k] = np.max(img)
    return np.max(max_vals)


def find_max_weight(data_path: str, n: int) -> float:
    file_names = [str(k) if len(str(k)) > 1 else '0' + str(k) for k in range(n)]
    max_vals = np.zeros(n, dtype=np.float64)
    count = 0
    for k, name in enumerate(file_names):
        img = np.load(data_path + name + '-weights.npz')['arr_0']
        max_vals[k] = np.max(img)
    return np.max(max_vals)


def find_bad_patches(n, img_h, img_w, th_val=100., return_max=True):
    h, w = 2000, 2000
    x_starts = start_points(w, img_w, overlap=None)
    y_starts = start_points(h, img_h, overlap=None)
    file_names = [str(k) if len(str(k)) > 1 else '0' + str(k) for k in range(n)]
    bad, max_vals = [], []
    nans = 0
    count = 0
    for k, name in enumerate(file_names):
        img = np.load('./data/train/' + name + '.npz')['arr_0']
        
        if np.any(np.isnan(img)):
            nans += 1
        
        for i in y_starts:
            for j in x_starts:
                mval = np.max(img[i:i + img_h, j:j + img_w])
                if mval < th_val:
                    max_vals.append(mval)
                else:
                    bad.append(count)
                count += 1
    if return_max:
        return bad, np.max(max_vals)
    
    print('There are', nans, 'images with nan values')
    return bad


def find_nans(n):
    file_names = [str(k) if len(str(k)) > 1 else '0' + str(k) for k in range(n)]
    nans = 0
    for k, name in enumerate(file_names):
        img = np.load('./data/train/' + name + '.npz')['arr_0']
#         img = np.load('./data/train/' + name + '-weights.npz')['arr_0'][..., 0]
        if np.any(np.isnan(img)):
            nans += 1
    print('There are', nans, 'images with nan values')
    return nans
