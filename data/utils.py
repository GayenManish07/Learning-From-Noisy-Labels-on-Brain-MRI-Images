from __future__ import print_function
import os
import os.path
import hashlib
import errno
import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.utils import assert_all_finite


def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)


def list_dir(root, prefix=False):
    """List all directories at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files

'''
# basic function
def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    print(np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]  # P should be a square matrix
    assert np.max(y) < P.shape[0]    # Labels should be within the range of classes

    # Ensure P is a row stochastic matrix (rows sum to 1)
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()  # All probabilities should be non-negative

    m = y.shape[0]
    print(m)
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # Draw a new label from the probability distribution P[i, :]
        flipped = flipper.multinomial(1, P[i, :])  # Removed the unnecessary [0] and second argument
        new_y[idx] = np.where(flipped == 1)[0]  # Find the index of the flipped label

    return new_y


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    print(P)

    return y_train, actual_noise

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    print(P)

    return y_train, actual_noise

def noisify(dataset='mnist', nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=0):
    if noise_type == 'pairflip':
        train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    if noise_type == 'symmetric' or noise_type == 'sn':
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    return train_noisy_labels, actual_noise_rate

'''
from sklearn.utils import assert_all_finite

def multiclass_noisify(y, P, random_state=0):
    """
    Flip classes according to the transition probability matrix P.
    Args:
        y: Ground truth labels (1D array of integers).
        P: Transition probability matrix (square matrix, rows sum to 1).
        random_state: Random seed for reproducibility.
    Returns:
        new_y: Noisy labels.
    """
    # Validate inputs
    assert P.shape[0] == P.shape[1], "P must be a square matrix."
    assert np.max(y) < P.shape[0], "Labels must be in [0, nb_classes - 1]."
    assert_all_finite(P), "P contains NaN or infinite values."
    assert np.allclose(P.sum(axis=1), 1), "Rows of P must sum to 1."
    assert np.all(P >= 0), "P must contain non-negative probabilities."

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # Draw a new label from the probability distribution P[i, :]
        flipped = flipper.multinomial(1, P[i, :])
        new_label = np.where(flipped == 1)[0]

        # Ensure the new label is valid
        if new_label < 0 or new_label >= P.shape[0]:
            raise ValueError(f"Invalid label {new_label} generated during noise injection!")

        new_y[idx] = new_label

    return new_y


def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """
    Introduce pairflip noise into the labels.
    Args:
        y_train: Ground truth labels (1D array of integers).
        noise: Noise rate (fraction of labels to corrupt).
        random_state: Random seed for reproducibility.
        nb_classes: Number of classes.
    Returns:
        y_train_noisy: Noisy labels.
        actual_noise: Actual fraction of labels corrupted.
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # Define the pairflip transition matrix
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes - 1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes - 1, nb_classes - 1], P[nb_classes - 1, 0] = 1. - n, n

        # Inject noise
        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()

        # Validate noisy labels
        if np.any(y_train_noisy < 0) or np.any(y_train_noisy >= nb_classes):
            raise ValueError("Invalid labels detected after noise injection!")

        print(f"Actual noise rate: {actual_noise:.2f}")
        return y_train_noisy, actual_noise

    return y_train, 0.0


def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """
    Introduce symmetric noise into the labels.
    Args:
        y_train: Ground truth labels (1D array of integers).
        noise: Noise rate (fraction of labels to corrupt).
        random_state: Random seed for reproducibility.
        nb_classes: Number of classes.
    Returns:
        y_train_noisy: Noisy labels.
        actual_noise: Actual fraction of labels corrupted.
    """
    P = np.ones((nb_classes, nb_classes)) * (noise / (nb_classes - 1))
    np.fill_diagonal(P, 1. - noise)

    if noise > 0.0:
        # Inject noise
        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()

        # Validate noisy labels
        if np.any(y_train_noisy < 0) or np.any(y_train_noisy >= nb_classes):
            raise ValueError("Invalid labels detected after noise injection!")

        print(f"Actual noise rate: {actual_noise:.2f}")
        return y_train_noisy, actual_noise

    return y_train, 0.0


def noisify(dataset='mnist', nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=0):
    """
    Add noise to the labels based on the specified noise type.
    Args:
        dataset: Name of the dataset (unused in this function but kept for compatibility).
        nb_classes: Number of classes.
        train_labels: Ground truth labels (1D array of integers).
        noise_type: Type of noise ('pairflip' or 'symmetric').
        noise_rate: Fraction of labels to corrupt.
        random_state: Random seed for reproducibility.
    Returns:
        train_noisy_labels: Noisy labels.
        actual_noise_rate: Actual fraction of labels corrupted.
    """
    # Validate input labels
    if np.any(train_labels < 0) or np.any(train_labels >= nb_classes):
        raise ValueError(f"Invalid labels detected in train_labels. Labels must be in [0, {nb_classes - 1}].")

    if noise_type == 'pairflip':
        train_noisy_labels, actual_noise_rate = noisify_pairflip(
            train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes
        )
    elif noise_type == 'symmetric' or noise_type == 'sn':
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(
            train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes
        )
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    # Validate noisy labels
    if np.any(train_noisy_labels < 0) or np.any(train_noisy_labels >= nb_classes):
        raise ValueError(f"Invalid labels detected after noise injection. Labels must be in [0, {nb_classes - 1}].")

    return train_noisy_labels, actual_noise_rate
