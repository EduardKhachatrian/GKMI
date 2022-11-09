# Parallel classification for an output of GKMI attribute selection algorithm.

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# (1) PARALLEL CLASSIFICATION ------------------------------------------------------------------------------- #
def perform_classification(
    dataset, ground_truth, pixel_idx, idx_GKMI, test_size=0.8, classifier="SVM"
):
    """Classification for each superpixel.

    Parameters
    ----------
    dataset            : array-like, shape (rows, columns, n_attributes)
        Set of different (multisensor/multiband/multifrequency) images.

    ground_truth       : array-like, shape (rows, columns) or (n_samples,)
        Ground truth or labels used for training and test of the algorithm.

    pixels_idx         : array-like
        Pixels indices for each superpixel.

    idx_GKMI           : array-like
        Indexes of relevant attributes selected by GKMI for each superpixel.

    test_size          : int, optional
        Test size that will be used for classification. By default test size
        is equal to 0.8, therefore train size is equal to 0.2 (1-0.8).

    classifier         : str, optional
        Classification algorithm to be used, either 'RF' or 'SVM'.
        By default, the algorithm is using 'SVM'.

    Returns
    -------
    OA                 : int
        Overall accuracy.

    Kappa              : int
        Cohen's kappa coefficient.

    AA                 : int
        Average accuracy.

    cm                 : array-like
        Confusion matrix.

    classified_map     : array-like
        Classified dataset.
    """
    if len(dataset.shape) == 2:
        array_reshaped = dataset
    if len(dataset.shape) == 3:
        rows, cols, attributes = dataset.shape
        array_reshaped = np.reshape(dataset, (np.multiply(rows, cols), attributes))

    if len(ground_truth.shape) == 1:
        ground_truth = ground_truth
    if len(ground_truth.shape) == 2:
        ground_truth = ground_truth.flatten()

    classified_map = np.zeros_like(ground_truth).astype("int")

    # Create training samples
    Xtrain, ytrain, _, _ = preparation_for_classification(
        array_reshaped, ground_truth, test_size
    )

    counter = 1
    # Classify the pixels in each superpixels.
    print("Start Classification : ")
    logger.info("Start Parallel Classification")
    logger.info("Applying " + str(classifier) + " Classifier")
    logger.info("%s : %s" % ("Test Size", test_size))

    for i in range(idx_GKMI.shape[0]):
        if classifier == "SVM":
            classified_map[pixel_idx[i]] = svm(
                array_reshaped[pixel_idx[i], :][:, np.sort(idx_GKMI[i])],
                Xtrain[:, np.sort(idx_GKMI[i])],
                ytrain,
            )
        elif classifier == "RF":
            classified_map[pixel_idx[i]] = rf(
                array_reshaped[pixel_idx[i], :][:, np.sort(idx_GKMI[i])],
                Xtrain[:, np.sort(idx_GKMI[i])],
                ytrain,
            )

        print("%s / %s" % (counter, idx_GKMI.shape[0]))
        counter += 1

    _, _, y_pred, y_test = preparation_for_classification(
        classified_map, ground_truth, test_size
    )
    # Evaluate the performance
    OA, Kappa, AA, cm = accuracy_evaluation(y_test, y_pred)
    logger.info("Parallel Classification Finished")
    return OA, Kappa, AA, cm, classified_map


# (2) GENERATING TEST AND TRAIN SAMPLES --------------------------------------------------------------------- #
def preparation_for_classification(dataset, ground_truth, test_size):
    """Split the data to train and test samples."""
    # Find the indexes that are not zero for all the classes
    idx_known = ground_truth != 0
    # Keep track on all the indexes
    yknown = ground_truth[idx_known]
    # Keep the data not equal to zero
    Xknown = dataset[idx_known]
    # Make the array with number from 0 to len of each class but in mixed order
    idx = np.random.permutation(len(Xknown))
    # Mix the data and indexes in each class
    Xknown, yknown = Xknown[idx], yknown[idx]
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        Xknown, yknown, test_size=test_size
    )
    return X_train, y_train, X_test, y_test


# (3) PERFORMANCE EVALUATION -------------------------------------------------------------------------------- #
def accuracy_evaluation(y_test, y_pred):
    """Evaluate thealgorithm performance using several criterias."""
    # Overall accuracy and Kappa
    OA = np.round(accuracy_score(y_test, y_pred) * 100, 1)
    Kappa = np.round(cohen_kappa_score(y_test, y_pred) * 100, 1)
    # Average accuracy
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)
    AA = np.round(np.sum(precision) / len(precision) * 100, 1)
    # Confusion matrix
    c = confusion_matrix(y_test, y_pred)
    cm = np.round(c / c.astype(np.float).sum(axis=0), 3)
    return OA, Kappa, AA, cm


# (4) ClASSIFIERS ------------------------------------------------------------------------------------------- #
def svm(X, Xtrain, ytrain, C=10, gamma=10):
    """Support Vector Machine algorithm with fixed parameters."""
    clf = SVC(C=C, kernel="rbf", gamma=gamma)
    clf.fit(Xtrain, ytrain)
    y_pred = clf.predict(X)
    return y_pred


def rf(X, Xtrain, ytrain):
    """Random Forest algorithm with fixed parameters."""
    clf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        max_depth=80,
        min_samples_split=5,
        bootstrap=True,
        max_features="sqrt",
        n_jobs=-1,
    )
    clf.fit(Xtrain, ytrain)
    y_pred = clf.predict(X)
    return y_pred
