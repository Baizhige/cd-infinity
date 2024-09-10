import torch
import numpy as np
def compute_average_cov_matrix(X):
    """
    Compute the average covariance matrix for the given data.

    Parameters:
    X (torch.Tensor): A tensor of shape (n_samples, c, T), where
                      n_samples is the number of samples,
                      c is the number of channels (features),
                      T is the number of time points.

    Returns:
    torch.Tensor: The average covariance matrix of shape (c, c).
    """
    n_samples, c, T = X.shape

    # Initialize the covariance matrix
    cov_matrix = torch.zeros((c, c), dtype=X.dtype, device=X.device)

    # Calculate the covariance matrix for each sample and accumulate
    for i in range(n_samples):
        # Center the data (subtract the mean)
        centered_data = X[i] - X[i].mean(dim=1, keepdim=True)

        # Compute the covariance matrix for this sample
        cov_sample = centered_data @ centered_data.T / (T - 1)

        # Accumulate
        cov_matrix += cov_sample

    # Take the average
    cov_matrix /= n_samples

    return cov_matrix

def solve_T(S_a, S_b):
    """
    Attempts to solve for the transformation matrix T given covariance matrices S_a and S_b.

    Parameters:
    S_a (torch.Tensor): Covariance matrix for X_a.
    S_b (torch.Tensor): Covariance matrix for X_b.

    Returns:
    torch.Tensor or None: The transformation matrix T, if a solution exists, otherwise None.
    """
    c_a = S_a.size(0)
    c_b = S_b.size(0)

    # Check if S_a is invertible (non-singular)
    if torch.det(S_a) != 0:
        # Eigenvalue decomposition of S_a
        eigenvalues_a, eigenvectors_a = torch.linalg.eigh(S_a)

        # Check if all eigenvalues are non-zero (non-singular)
        if torch.all(eigenvalues_a != 0):
            # Attempt to solve for T
            try:
                D_inv_sqrt = torch.diag(1.0 / torch.sqrt(eigenvalues_a))
                Q = eigenvectors_a

                # Check if S_b is non-negative definite
                if torch.all(torch.linalg.eigvalsh(S_b) >= 0):
                    S_b_sqrt = torch.linalg.cholesky(S_b)

                    # Compute T
                    T = S_b_sqrt @ Q @ D_inv_sqrt @ Q.T
                    return T
                else:
                    print("S_b is not non-negative definite.")
            except RuntimeError as e:
                print(f"Error while solving for T: {e}")
        else:
            print("S_a has zero eigenvalues, it's singular.")
    else:
        print("S_a is singular (not invertible).")

    # If all methods fail, return None
    return None


def solve_T_numpy(S_a, S_b):
    """
    Attempts to solve for the transformation matrix T given covariance matrices S_a and S_b using numpy.

    Parameters:
    S_a (np.ndarray): Covariance matrix for X_a.
    S_b (np.ndarray): Covariance matrix for X_b.

    Returns:
    np.ndarray or None: The transformation matrix T, if a solution exists, otherwise None.
    """
    c_a = S_a.shape[0]
    c_b = S_b.shape[0]

    # Check if S_a is invertible (non-singular)
    if np.linalg.det(S_a) != 0:
        # Eigenvalue decomposition of S_a
        eigenvalues_a, eigenvectors_a = np.linalg.eigh(S_a)

        # Check if all eigenvalues are non-zero (non-singular)
        if np.all(eigenvalues_a != 0):
            # Attempt to solve for T
            try:
                D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues_a))
                Q = eigenvectors_a

                # Check if S_b is non-negative definite
                if np.all(np.linalg.eigvalsh(S_b) >= 0):
                    S_b_sqrt = np.linalg.cholesky(S_b)

                    # Compute T
                    T = S_b_sqrt @ Q @ D_inv_sqrt @ Q.T
                    return T
                else:
                    print("S_b is not non-negative definite.")
            except np.linalg.LinAlgError as e:
                print(f"Error while solving for T: {e}")
        else:
            print("S_a has zero eigenvalues, it's singular.")
    else:
        print("S_a is singular (not invertible).")

    # If all methods fail, return None
    return None


