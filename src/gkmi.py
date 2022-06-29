# Unsupervised attribute selection method - GKMI.
# Authors :  Eduard Khachatrian    <eduard.khachatrian@uit.no>
#            Saloua Chlaily        <saloua.chlaily@uit.no> 
#            Andrea Marinoni       <andrea.marinoni@uit.no>

import logging
import itertools
import numpy as np
import numpy.matlib as matlib
from scipy.linalg import eigh
from scipy.sparse import csgraph
from skimage.filters import sobel
from sklearn.cluster import KMeans
from scipy.spatial import distance
from skimage import exposure, img_as_float
from skimage.segmentation import watershed, slic
from sklearn.metrics import pairwise_distances_argmin_min, normalized_mutual_info_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# (1) GKMI ATTRIBUTE SELECTION ---------------------------------------------------------------------------- #
def GKMI(dataset, n_clusters, attribute_n=0, n_superpixels=100, segmentation_algorithm='watershed'):
    """ GKMI is a flexible, accurate, efficient, and interpretable attribute selection method 
    that allows determining the most informative and relevant attributes in heterogeneous datasets. 

    Parameters
    ----------
    dataset                  : array-like, shape (rows, columns, n_attributes)
        Set of different (multisensor/multiband/multifrequency) images.

    n_clusters               : int
        Number of clusters that will be used in k-means. Note that this parameter is 
        equal to the final number of attributes selected. If 'auto', then the number
        of clusters will be selected automatically for each superpixel.

    attribute_n              : int, optional
        Attribute number/index that will be used for creating the superpixels regions.
        By default, the algorithm is using the first attribute. 

    n_superpixels            : int, optional
        Number of superpixels (homogeneous areas) to be generated.
        By default, the algorithm is using 100 superpixels.

    segmentation_algorithm   : str, optional
        Superpixels segmentation algorithm to be used, either 'slic' or 'watershed'.
        By default, the algorithm is using watershed segmentation.

    Returns
    -------
    pixels_idx                : array-like
        Pixels indices for each superpixel.

    attributes_selected       : array-like
        Indexes of relevant attributes selected by GKMI for each superpixel.
    """
    rows, cols, attributes = dataset.shape
    dataset_reshaped = np.reshape(dataset, (np.multiply(rows, cols), attributes))
    
    attributes_selected = []  
    # Generating Supepixels
    image = dataset[:,:,attribute_n]
    pixels_idx, superpixels = superpixels_generation(image, n_superpixels, segmentation_algorithm)
        
    counter = 1

    # Calculating Mutual Information, global image-wise similarity measure 
    print ('Start Processing : Mutual Information')
    logger.info('Start Processing : Mutual Information')

    mi_laplacian_mx =  normalize(mutual_information(dataset_reshaped))
    print ('Finished Processing : Mutual Information')
    logger.info('Finished Processing : Mutual Information')

    print ('Start Processing : Gaussian Kernel')
    logger.info('Start Processing : Gaussian Kernel')
    for i in range(pixels_idx.shape[0]):
        print ('%s : %s' % ('Size of superpixel', pixels_idx[i].shape))
        logger.info('%s : %s' % ('Size of superpixel', pixels_idx[i].shape))
        
        # Calculating Gaussian Kernel, local superpixel-wise similarity measure
        gk_laplacian_mx_ij = normalize(gaussian_kernel(dataset_reshaped[pixels_idx[i][:]]))

        # Performing joint diaganilization of stacked Laplacians
        joint_eigenvalues, joint_eigenvectors = eigh(gk_laplacian_mx_ij, mi_laplacian_mx)

        # Searching for an optimal number of clusters
        if n_clusters == 'auto':
            clusters = optimal_cluster_number(joint_eigenvalues)
            print ('%s : %s' % ('Optimal number of clusters', clusters))
            logger.info('%s : %s' % ('Optimal number of clusters', clusters))
        else:
            clusters = n_clusters

        if joint_eigenvectors.flags['C_CONTIGUOUS']==False:
            joint_eigenvectors = np.ascontiguousarray(joint_eigenvectors)

        # Applying k-means clustering
        try:
            closest_idx = kmeans(joint_eigenvalues, joint_eigenvectors, clusters)
        except ValueError: 
            pass

        attributes_selected.append(np.sort(closest_idx))
        logger.info('%s : %s' % ('Indexes of attributes selected', np.sort(closest_idx)))

        print('%s / %s' % (counter, pixels_idx.shape[0]))
        counter += 1
    
    print ('Finished Processing : Gaussian Kernel')
    logger.info('Finished Processing : Gaussian Kernel')

    attributes_selected = np.array(attributes_selected)
    return pixels_idx, attributes_selected

# (2) SEGMENTATION FUNCTION ------------------------------------------------------------------------------- #
def superpixels_generation(image, n_superpixels, segmentation_algorithm):
    """ Superpixel segmentation divides the image into homogeneous areas that share common characteristics.  

    Parameters
    ----------
    image           : array-like, shape (rows, columns)

    Returns
    -------
    superpixels_idx : array_like
        Indexes of all points that fall into the superpixels area.

    superpixels     : array_like
        Segmented image.
    """
    # Streching the image histogram (contrast streching)
    equalized_img = exposure.equalize_hist(image)
    # Performing  superpixel segmentation
    if segmentation_algorithm == 'watershed':
        gradient = sobel(equalized_img)
        superpixels = watershed(gradient, markers=n_superpixels, compactness=0.00001)
    elif segmentation_algorithm == 'slic':
        superpixels = slic(equalized_img, n_segments=n_superpixels, compactness=0.1, sigma=1)
    else:
        raise ValueError('GKMI is called with wrong parameters, use either slic or watershed') 
        logger.info('GKMI is called with wrong parameters, use either slic or watershed')

    # Grouping pixels indices that correspond to each superpixel
    superpixels_idx = []
    for i in np.unique(superpixels):
        superpixels_idx.append(np.where(superpixels.flatten() == i)[0])
    superpixels_idx = np.array(superpixels_idx, dtype=object)
    return superpixels_idx, superpixels

# (3) GRAPH BUILDING -------------------------------------------------------------------------------------- #
def gaussian_kernel(array, sigma=1):
    """ Building the graph with Gaussian Kernel.

    Parameters
    ----------
    array        : array-like, shape (n_samples, n_attributes)

    Returns
    -------
    laplacian    : array_like or sparse matrix, 2 dimensions
        Normalized Symmetric Laplacian. 
    """
    # Building the graph
    gk_dist = distance.cdist(array.T, array.T, 'sqeuclidean')
    adjacency_mx_ij = np.exp(-(gk_dist) / (2*(sigma**2)))
    gk_laplacian = csgraph.laplacian(adjacency_mx_ij, normed=True)
    return gk_laplacian
def mutual_information(array):
    """ Building the graph with Mutual Information. 

    Parameters
    ----------
    array        : array-like, shape (n_samples, n_attributes)

    Returns
    -------
    laplacian    : array_like or sparse matrix, 2 dimensions
        Normalized Symmetric Laplacian. 
    """
    if array.shape[0] < 1e4:
        array = array[::10,:]
    elif array.shape[0] > 1e6:
        array = array[::1000,:]
    else:
        array = array[::100,:]

    # Building the graph
    mi_matrix = np.reshape([normalized_mutual_info_score(array[:,ii].ravel(),array[:,jj].ravel()) 
                for ii,jj in itertools.product(range(array.shape[1]), range(array.shape[1]))], 
                (array.shape[1], array.shape[1]))         
    mi_laplacian = csgraph.laplacian(mi_matrix, normed=True)    
    return mi_laplacian

# (4) GRAPH CLUSTERING ------------------------------------------------------------------------------------ # 
def kmeans(eigenvalues, eigenvectors, n_clusters):
    """ Performing unsupervised k-means clustering.

    Parameters
    ----------
    eigenvalues     : array-like
    eigenvectors    : array-like

    Returns
    -------
    closest         : array_like 
        Indexes of closest points to the centroids.  
    """
    # Skip the first eigenvalue and corresponding eigenvector column if it is zero or negative
    if eigenvalues[0]<=0:
        eigenvectors = eigenvectors[:, 1:n_clusters + 1]
    else:
        eigenvectors = eigenvectors[:, 0:n_clusters] 
    # Norm of eigenvectors
    eigenvectors = eigenvectors / np.reshape(np.linalg.norm(eigenvectors, axis=1), 
                                            (eigenvectors.shape[0], 1))
    # Perform the k-means clustering
    kmeans = KMeans(n_clusters)
    kmeans.fit(eigenvectors)
    centroids = kmeans.cluster_centers_
    # Choose the closest point to the centroids
    closest_idx, _ = pairwise_distances_argmin_min(centroids, eigenvectors)
    return closest_idx

# (5) OPTIMAL CLUSTER NUMBER SELECTION -------------------------------------------------------------------- #
def optimal_cluster_number(eigenvalues):
    """ Select the optimal number of clusters by detecting a knee points.
    
    Parameters
    ----------
    eigenvalues       : array-like

    Returns
    -------
    best_point_idx    : int 
        Optimal number of clusters.  
    """
    # Discarding the first and last eigenvalue 
    curve = eigenvalues[1:-1]
    # Get coordinates for each point
    coordinates = np.vstack((range(len(curve)), curve)).T
    first_point = coordinates[0]
    line_vector = coordinates[-1] - coordinates[0]
    line_vector_norm = line_vector / np.sqrt(np.sum(line_vector**2))
    # Calculate the distance from poins to line: vector between all points and first point
    vector_to_first = coordinates - first_point
    scalar_product = np.sum(vector_to_first * matlib.repmat(line_vector_norm, len(curve), 1), axis=1)
    # Calculate the distance to the line, by applying parrarel and perpendicular to the line components
    parralel = np.outer(scalar_product, line_vector_norm)
    perpendicular = vector_to_first - parralel
    distance = np.sqrt(np.sum(perpendicular**2, axis=1))
    best_point_idx = np.argmax(distance) + 1
    
    min_limit = 3
    if best_point_idx  < min_limit:
        return min_limit
    else:
        return best_point_idx

# (6) ADDITIONAL FUNCTIONS -------------------------------------------------------------------------------- #
def normalize(array):
    """ Changing the original numerical values to fit within a certain range."""
    array = img_as_float(array)
    return np.array((array - np.min(array)) / (np.max(array) - np.min(array)))
    



