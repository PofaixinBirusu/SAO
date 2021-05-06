import numpy as np
import open3d as o3d


class LRF:
    def __init__(self, pcd, pcd_tree, lrf_kernel):

        self.pcd = pcd
        self.pcd_tree = pcd_tree
        self.patch_kernel = lrf_kernel

    def get(self, pt):

        _, patch_idx, _ = self.pcd_tree.search_radius_vector_3d(pt, self.patch_kernel)
        ptall = np.asarray(self.pcd.points)[patch_idx, :].T

        lRg = self.get_lrf(pt)

        # rotate w.r.t local frame and centre in zero using the chosen point
        ptall = (lRg.T @ (ptall - pt[:, np.newaxis])).T

        # this is our normalisation
        ptall /= self.patch_kernel

        T = np.zeros((4, 4))
        T[-1, -1] = 1
        T[:3, :3] = lRg
        T[:3, -1] = pt

        return ptall

    def get_lrf(self, pt):
        _, patch_idx, _ = self.pcd_tree.search_radius_vector_3d(pt, self.patch_kernel)

        ptnn = np.asarray(self.pcd.points)[patch_idx[1:], :].T

        # eq. 3
        ptnn_cov = 1 / len(ptnn) * np.dot((ptnn - pt[:, np.newaxis]), (ptnn - pt[:, np.newaxis]).T)

        if len(patch_idx) < self.patch_kernel / 2:
            _, patch_idx, _ = self.pcd_tree.search_knn_vector_3d(pt, self.patch_kernel)

        # The normalized (unit “length”) eigenvectors, s.t. the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
        a, v = np.linalg.eig(ptnn_cov)
        smallest_eigevalue_idx = np.argmin(a)
        np_hat = v[:, smallest_eigevalue_idx]

        # eq. 4
        zp = np_hat if np.sum(np.dot(np_hat, pt[:, np.newaxis] - ptnn)) > 0 else - np_hat

        v = (ptnn - pt[:, np.newaxis]) - (np.dot((ptnn - pt[:, np.newaxis]).T, zp[:, np.newaxis]) * zp).T
        alpha = (self.patch_kernel - np.linalg.norm(pt[:, np.newaxis] - ptnn, axis=0)) ** 2
        beta = np.dot((ptnn - pt[:, np.newaxis]).T, zp[:, np.newaxis]).squeeze() ** 2

        # e.q. 5
        xp = 1 / (np.linalg.norm(np.dot(v, (alpha * beta)[:, np.newaxis])) + 1e-32) * np.dot(v, (alpha * beta)[:, np.newaxis])
        xp = xp.squeeze()

        yp = np.cross(xp, zp)

        lRg = np.asarray([xp, yp, zp]).T
        return lRg