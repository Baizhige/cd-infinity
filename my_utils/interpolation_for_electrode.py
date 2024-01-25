import numpy as np
import os

def gaussian_kernel(Adj: np.ndarray, sigma: float = 0.3) -> np.ndarray:
    """
    Convert a distance matrix to a weight matrix using a Gaussian kernel.

    Parameters:
    - Adj (np.ndarray): The adjacency matrix representing distances between electrodes.
                        It should be a square matrix with zero diagonal.
    - sigma (float): The standard deviation for the Gaussian kernel.

    Returns:
    - np.ndarray: The weight matrix after applying the Gaussian kernel.
    """
    if not isinstance(Adj, np.ndarray):
        raise TypeError("Adj must be a NumPy array")
    if Adj.shape[0] != Adj.shape[1]:
        raise ValueError("Adj must be a square matrix")
    if not isinstance(sigma, float):
        raise TypeError("sigma must be a float")

    return np.exp(-Adj ** 2 / (2 * sigma ** 2))


def get_interpolation_matrix(Adj: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    Compute the interpolation matrix based on the adjacency matrix and the selection vector.

    Parameters:
    - Adj (np.ndarray): The adjacency matrix where the i-th row and j-th column
                        represent the distance from node i to j.
    - S (np.ndarray): A diagonal matrix where each diagonal element is 1 if data
                      for that electrode is available and 0 otherwise.

    Returns:
    - np.ndarray: The interpolation matrix to interpolate the missing data.
    """
    if not (isinstance(Adj, np.ndarray) and isinstance(S, np.ndarray)):
        raise TypeError("Adj and S must be NumPy arrays")
    if Adj.shape[0] != Adj.shape[1]:
        raise ValueError("Adj must be a square matrix")
    if S.shape[0] != Adj.shape[0] or S.shape[1] != Adj.shape[1]:
        raise ValueError("S must have the same dimensions as Adj")

    n_channels = Adj.shape[0]

    # Convert distances to weights using a Gaussian kernel
    W = gaussian_kernel(Adj=Adj)

    # Set weights to zero for channels with missing data (empty electrodes)
    W_modified = W @ S

    # Normalize weights
    W_norm = W_modified / np.sum(W_modified, axis=1, keepdims=True)

    # Construct interpolation matrix
    T_int = S + (np.eye(n_channels) - S) @ W_norm

    return T_int


def find_empty_rows(matrix: np.ndarray) -> np.ndarray:
    """
    Identify the empty rows in a matrix.

    Parameters:
    - matrix (np.ndarray): The input matrix.

    Returns:
    - np.ndarray: A vector where 0 indicates an empty row and 1 indicates a non-empty row.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("The input must be a NumPy array")

    # Check if each row sums to 0 (which would indicate it's an empty row)
    is_row_empty = np.sum(matrix, axis=1) == 0

    # Convert boolean array to int, and then invert it (since we want 1 for non-empty rows)
    row_indicator = (~is_row_empty).astype(int)

    return row_indicator

orinal_dataset = "PhysioNetMI"
transform_dataset = "MengExp12"
shape = "(62_64)"
T_replace  = np.load(os.path.join("..","config",f"transformation_matrix_{orinal_dataset}_{transform_dataset}_{shape}_null.npy"))
print(np.sum(T_replace, axis=1))
Adj = np.load(os.path.join("..","config",f"{transform_dataset}_Geo_Adj.npy"))
S = np.diag(find_empty_rows(T_replace))
T_int = get_interpolation_matrix(Adj=Adj,S=S)
T_final = T_int @ T_replace
np.save(os.path.join("..","config",f"transformation_matrix_{orinal_dataset}_{transform_dataset}_{shape}_new.npy"), T_final)
print(np.sum(T_final, axis=1))
print(T_final)