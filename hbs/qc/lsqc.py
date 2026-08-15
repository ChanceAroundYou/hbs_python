import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

from hbs.mesh import Mesh


def lsqc_solver(
    mu: np.ndarray[np.complexfloating],
    landmark: np.ndarray[np.integer],
    target: np.ndarray[np.floating],
    mesh: Mesh,
    eps: float = 0.0,
):
    """
    Solves the Beltrami equation
    :param mu: n x 1 complex array, Beltrami coefficients defined on triangles
    :param landmark: m x 1 array, indices of landmark vertices
    :param target: m x 2 array, target positions of constrained vertices
    :param mesh: Mesh object
    :return: vertex coordinates of the solved mesh
    """
    assert isinstance(mesh, Mesh), "mesh must be Mesh object"

    assert isinstance(mu, np.ndarray) and np.issubdtype(mu.dtype, np.complexfloating), (
        "mu must be complex array"
    )
    assert mu.ndim == 1 and mu.shape[0] == mesh.face_num, (
        "mu must be n x 1 array and n is the number of faces"
    )

    assert isinstance(landmark, np.ndarray) and np.issubdtype(
        landmark.dtype, np.integer
    ), "landmark must be integer array"
    assert landmark.ndim == 1, "landmark must be 1D array"

    assert isinstance(target, np.ndarray) and np.issubdtype(
        target.dtype, np.floating
    ), "target must be float array"
    assert (
        target.ndim == 2
        and target.shape == (landmark.shape[0], 2)
    ), "target must be m x 2 array and m is the number of landmarks"

    landmark = np.concatenate([landmark, landmark + mesh.vert_num])
    target = target.T.reshape(-1)
    L = div_PtP_grad(mu, mesh, eps)
    A = unsigned_area_matrix(mu, mesh)
    M = lil_matrix(L - 2 * A)
    b = -M[:, landmark] @ target
    M[landmark, :] = 0
    M[:, landmark] = 0
    M[landmark, landmark] = 1
    M = M.tocsr()
    u = spsolve(M, b)
    u[landmark] = target
    u = u.reshape(2, -1).T
    return u


def unsigned_area_matrix(mu, mesh):
    idx = np.abs(mu) > 1
    A = np.ones(len(mu))
    A[idx] = -A[idx]

    def compute_q(ei, ej):
        yi, xi = ei.T
        yj, xj = ej.T
        return A * (yi * xj - xi * yj) / (4 * mesh.area)

    x_order = [0, 1, 2, 0, 1, 0, 2, 1, 2]
    y_order = [0, 1, 2, 1, 0, 2, 0, 2, 1]

    x_pos = np.concatenate([mesh.face[:, k] for k in x_order])
    y_pos = np.concatenate([mesh.face[:, k] for k in y_order])
    value = np.concatenate(
        [compute_q(mesh.edges[i], mesh.edges[j]) for i, j in zip(x_order, y_order)]
    )

    x_pos = np.concatenate([x_pos, x_pos + mesh.vert_num])
    y_pos = np.concatenate([y_pos + mesh.vert_num, y_pos])
    value = np.concatenate([value, -value]) / 2
    matrix = csr_matrix((value, (x_pos, y_pos)))
    return matrix


def div_PtP_grad(mu, mesh, eps=0.0):
    abs_mu = np.abs(mu)
    # eps：尖角处 |mu|→1⁻ 使正分母 1-|mu|² 接近 0（病态），抬到 eps 稳定。
    # 默认 0 保持原行为；>0 牺牲极小精度换取尖角重建稳定。
    # |mu|>1 的分母为负（unsigned 处理），不受影响。
    denom = 1.0 - abs_mu**2
    denom = np.where((denom > 0) & (denom < eps), eps, denom)
    A = -(1 - 2 * np.real(mu) + abs_mu**2) / denom
    B = 2 * np.imag(mu) / denom
    C = -(1 + 2 * np.real(mu) + abs_mu**2) / denom

    idx = abs_mu > 1
    A[idx] = -A[idx]
    B[idx] = -B[idx]
    C[idx] = -C[idx]

    def compute_v(ei, ej):
        yi, xi = ei.T
        yj, xj = ej.T
        return (A * (xi * xj) - B * (xi * yj + xj * yi) + C * (yi * yj)) / mesh.area

    x_order = [0, 1, 2, 0, 1, 1, 2, 2, 0]
    y_order = [0, 1, 2, 1, 0, 2, 1, 0, 2]

    x_pos = np.concatenate([mesh.face[:, k] for k in x_order])
    y_pos = np.concatenate([mesh.face[:, k] for k in y_order])
    value = np.concatenate(
        [compute_v(mesh.edges[i], mesh.edges[j]) for i, j in zip(x_order, y_order)]
    )

    x_pos = np.concatenate([x_pos, x_pos + mesh.vert_num])
    y_pos = np.concatenate([y_pos, y_pos + mesh.vert_num])
    value = -np.concatenate([value, value]) / 4
    matrix = csr_matrix((value, (x_pos, y_pos)))
    return matrix
