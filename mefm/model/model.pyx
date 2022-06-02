import multiprocessing
from collections import Counter, OrderedDict
from typing import Tuple

cimport cython
from cython.parallel import prange, parallel
from cython cimport floating, integral
from libc.math cimport sqrt

import numpy as np
cimport numpy as np
import scipy.sparse as sp
from scipy.linalg.cython_blas cimport sdot, ddot

from cornac.models.recommender import Recommender
from cornac.exception import ScoreException
from cornac.utils.common import intersects
from cornac.utils import get_rng
from cornac.utils.init_utils import uniform
from cornac.data import Dataset


cdef floating _dot(int n, floating *x, int incx,
                   floating *y, int incy) nogil:
    if floating is float:
        return sdot(&n, x, &incx, y, &incy)
    else:
        return ddot(&n, x, &incx, y, &incy)


class ModifiedEFM(Recommender):
    def __init__(
            self,
            num_explicit_factors: int,
            num_latent_factors: int,
            num_most_cared_features: int,
            rating_scale: float,
            alpha: float,
            lambda_x: float,
            lambda_y: float,
            lambda_u: float,
            lambda_h: float,
            lambda_v: float,
            max_iter: int,
            user_matrix_mode: str="count",
            item_matrix_mode: str="indicator",
            sigmoid_b: float=1.0,
            name: str="EFM",
            num_threads: int=0,
            trainable: bool=True,
            verbose: bool=False,
            init_params: dict=None,
            seed: int=None
    ):
        """
        Initialize Modified EFM.

        :param num_explicit_factors: The dimension of the explicit factors.
        :param num_latent_factors: The dimension of the latent factors.
        :param num_most_cared_features: The number of most cared features for each user.
        :param rating_scale: The maximum rating score of the dataset.
        :param alpha: Trace off factor for constructing ranking score.
        :param lambda_x: The regularization parameter for user feature attentions.
        :param lambda_y: The regularization parameter for item feature qualities.
        :param lambda_u: The regularization parameter for user and item explicit factors.
        :param lambda_h: The regularization parameter for user and item latent factors.
        :param lambda_v: The regularization parameter for V.
        :param max_iter: Maximum number of iterations or the number of epochs.
        :param user_matrix_mode: Modification of user-feature matrix construction.
        :param item_matrix_mode: Modification of item-quality matrix construction.
        :param sigmoid_b: The growth rate parameter of sigmoid function.
        :param name: The name of the recommender model.
        :param num_threads: Number of parallel threads for training. If 0, all CPU cores will be utilized.
        :param trainable: When False, the model is not trained, and it is assumed that the model already pre-trained (U1, U2, V, H1, and H2 are not None).
        :param verbose: When True, running logs are displayed.
        :param init_params: List of initial parameters, e.g., init_params = {‘U1’:U1, ‘U2’:U2, ‘V’:V, ‘H1’:H1, ‘H2’:H2}
        :param seed: Random seed for weight initialization.
        """

        self._user_matrix_modes=["count", "frequency", "indicator", "tf-idf"]
        if user_matrix_mode not in self._user_matrix_modes:
            raise AssertionError(f"User matrix mode should be one of " + str(self._user_matrix_modes))
        self._item_matrix_modes=["rating", "indicator", "tf-idf"]
        if item_matrix_mode not in self._item_matrix_modes:
            raise AssertionError(f"Item matrix mode should be one of " + str(self._item_matrix_modes))
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.num_explicit_factors = num_explicit_factors
        self.num_latent_factors = num_latent_factors
        self.num_most_cared_features = num_most_cared_features
        self.rating_scale = rating_scale
        self.alpha = alpha
        self.lambda_x = lambda_x
        self.lambda_y = lambda_y
        self.lambda_u = lambda_u
        self.lambda_h = lambda_h
        self.lambda_v = lambda_v
        self.max_iter = max_iter
        self.user_matrix_mode = user_matrix_mode
        self.item_matrix_mode = item_matrix_mode
        self.sigmoid_b = sigmoid_b
        self.seed = seed
        
        if self.user_matrix_mode=="indicator":
            self._user_func=np.max
        elif self.user_matrix_mode=="frequency":
            self._user_func=np.mean
        else:
            self._user_func=np.sum

        if seed is not None:
            self.num_threads = 1
        elif num_threads > 0 and num_threads < multiprocessing.cpu_count():
            self.num_threads = num_threads
        else:
            self.num_threads = multiprocessing.cpu_count()

        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.U1 = self.init_params.get('U1', None)
        self.U2 = self.init_params.get('U2', None)
        self.V = self.init_params.get('V', None)
        self.H1 = self.init_params.get('H1', None)
        self.H2 = self.init_params.get('H2', None)

    def _init(self):
        """
        Initialize Modified EFM.
        """
        rng = get_rng(self.seed)
        n_users, n_items = self.train_set.num_users, self.train_set.num_items
        n_features = self.train_set.item_feature.feature_dim
        n_efactors = self.num_explicit_factors
        n_lfactors = self.num_latent_factors
        n_factors = n_efactors + n_lfactors
        high = np.sqrt(self.rating_scale / n_factors)
        
        if self.U1 is None:
            self.U1 = uniform((n_users, n_efactors), high=high, random_state=rng)
        if self.U2 is None:
            self.U2 = uniform((n_items, n_efactors), high=high, random_state=rng)
        if self.V is None:
            self.V = uniform((n_features, n_efactors), high=high, random_state=rng)
        if self.H1 is None:
            self.H1 = uniform((n_users, n_lfactors), high=high, random_state=rng)
        if self.H2 is None:
            self.H2 = uniform((n_items, n_lfactors), high=high, random_state=rng)

    def fit(self, train_set: Dataset, val_set: Dataset=None) -> Recommender:
        """
        Fit the model to observations.

        :param train_set: User-Item preference data as well as additional modalities.
        :param val_set: User-Item preference data for model selection purposes (e.g., early stopping).
        :return: self
        """
        Recommender.fit(self, train_set, val_set)

        self._init()

        if self.trainable:
            A, X, Y = self._build_matrices(self.train_set)
            A_user_counts = np.ediff1d(A.indptr)
            A_item_counts = np.ediff1d(A.tocsc().indptr)
            A_uids = np.repeat(np.arange(self.train_set.num_users), A_user_counts).astype(A.indices.dtype)
            X_user_counts = np.ediff1d(X.indptr)
            X_feature_counts = np.ediff1d(X.tocsc().indptr)
            X_uids = np.repeat(np.arange(self.train_set.num_users), X_user_counts).astype(X.indices.dtype)
            Y_item_counts = np.ediff1d(Y.indptr)
            Y_feature_counts = np.ediff1d(Y.tocsc().indptr)
            Y_iids = np.repeat(np.arange(self.train_set.num_items), Y_item_counts).astype(Y.indices.dtype)
            
            self._fit_efm(
                self.num_threads,
                A.data.astype(np.float32), A_uids, A.indices, A_user_counts, A_item_counts,
                X.data.astype(np.float32), X_uids, X.indices, X_user_counts, X_feature_counts,
                Y.data.astype(np.float32), Y_iids, Y.indices, Y_item_counts, Y_feature_counts,
                self.U1, self.U2, self.V, self.H1, self.H2
            )

        return self        


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit_efm(self, int num_threads,
                 floating[:] A, integral[:] A_uids, integral[:] A_iids, integral[:] A_user_counts, integral[:] A_item_counts,
                 floating[:] X, integral[:] X_uids, integral[:] X_fids, integral[:] X_user_counts, integral[:] X_feature_counts,
                 floating[:] Y, integral[:] Y_iids, integral[:] Y_qids, integral[:] Y_item_counts, integral[:] Y_feature_counts,
                 floating[:, :] U1, floating[:, :] U2, floating[:, :] V, floating[:, :] H1, floating[:, :] H2):
        """
        Fit the model parameters (U1, U2, V, H1, H2)

        :param num_threads: Number of parallel threads for training. If 0, all CPU cores will be utilized.
        :param A: User-item rating matrix.
        :param A_uids: User-item rating matrix users' indices (row indices).
        :param A_iids: User-item rating matrix items' indices (column indices).
        :param A_user_counts: Counts of interactions by user.
        :param A_item_counts: Counts of interactions by item.
        :param X: User-feature matrix.
        :param X_uids: User-feature matrix users' indices (row indices).
        :param X_fids: User-feature matrix features' indices (column indices).
        :param X_user_counts: Counts of existing user-feature pairs by user.
        :param X_feature_counts: Counts of existing user-feature pairs by feature.
        :param Y: Item-quality matrix.
        :param Y_iids: Item-feature matrix items' indices (row indices).
        :param Y_qids: Item-feature matrix qualities' indices (column indices).
        :param Y_item_counts: Counts of existing item-quality pairs by item.
        :param Y_feature_counts: Counts of existing item-feature pairs by quality.
        :param U1: The user explicit factors.
        :param U2: The item explicit factors.
        :param V: The feature factors.
        :param H1: The user latent factors.
        :param H2: The item latent factors.
        """
        cdef:
            long num_users = self.train_set.num_users
            long num_items = self.train_set.num_items
            long num_features = self.train_set.item_feature.feature_dim
            int num_explicit_factors = self.num_explicit_factors
            int num_latent_factors = self.num_latent_factors

            floating lambda_x = self.lambda_x
            floating lambda_y = self.lambda_y
            floating lambda_u = self.lambda_u
            floating lambda_h = self.lambda_h
            floating lambda_v = self.lambda_v

            floating prediction, score, loss

            np.ndarray[np.float32_t, ndim=2] U1_numerator = np.empty((num_users, num_explicit_factors), dtype=np.float32)
            np.ndarray[np.float32_t, ndim=2] U1_denominator = np.empty((num_users, num_explicit_factors), dtype=np.float32)
            np.ndarray[np.float32_t, ndim=2] U2_numerator = np.empty((num_items, num_explicit_factors), dtype=np.float32)
            np.ndarray[np.float32_t, ndim=2] U2_denominator = np.empty((num_items, num_explicit_factors), dtype=np.float32)
            np.ndarray[np.float32_t, ndim=2] V_numerator = np.empty((num_features, num_explicit_factors), dtype=np.float32)
            np.ndarray[np.float32_t, ndim=2] V_denominator = np.empty((num_features, num_explicit_factors), dtype=np.float32)
            np.ndarray[np.float32_t, ndim=2] H1_numerator = np.empty((num_users, num_latent_factors), dtype=np.float32)
            np.ndarray[np.float32_t, ndim=2] H1_denominator = np.empty((num_users, num_latent_factors), dtype=np.float32)
            np.ndarray[np.float32_t, ndim=2] H2_numerator = np.empty((num_items, num_latent_factors), dtype=np.float32)
            np.ndarray[np.float32_t, ndim=2] H2_denominator = np.empty((num_items, num_latent_factors), dtype=np.float32)
            int i, j, k, idx
            long n_ratings

            floating eps = 1e-9

        for t in range(1, self.max_iter + 1):
            loss = 0.
            U1_numerator.fill(0)
            U1_denominator.fill(0)
            U2_numerator.fill(0)
            U2_denominator.fill(0)
            V_numerator.fill(0)
            V_denominator.fill(0)
            H1_numerator.fill(0)
            H1_denominator.fill(0)
            H2_numerator.fill(0)
            H2_denominator.fill(0)

            with nogil, parallel(num_threads=num_threads):
                # compute numerators and denominators for all factors
                for idx in prange(A.shape[0]):
                    i = A_uids[idx]
                    j = A_iids[idx]
                    prediction = _dot(num_explicit_factors, &U1[i, 0], 1, &U2[j, 0], 1) \
                                 + _dot(num_latent_factors, &H1[i, 0], 1, &H2[j, 0], 1)
                    score = A[idx]
                    loss += (prediction - score) * (prediction - score)
                    for k in range(num_explicit_factors):
                        U1_numerator[i, k] += score * U2[j, k]
                        U1_denominator[i, k] += prediction * U2[j, k]
                        U2_numerator[j, k] += score * U1[i, k]
                        U2_denominator[j, k] += prediction * U1[i, k]

                    for k in range(num_latent_factors):
                        H1_numerator[i, k] += score * H2[j, k]
                        H1_denominator[i, k] += prediction * H2[j, k]
                        H2_numerator[j, k] += score * H1[i, k]
                        H2_denominator[j, k] += prediction * H1[i, k]

                for idx in prange(X.shape[0]):
                    i = X_uids[idx]
                    j = X_fids[idx]
                    prediction = _dot(num_explicit_factors, &U1[i, 0], 1, &V[j, 0], 1)
                    score = X[idx]
                    loss += (prediction - score) * (prediction - score)
                    for k in range(num_explicit_factors):
                        V_numerator[j, k] += lambda_x * score * U1[i, k]
                        V_denominator[j, k] += lambda_x * prediction * U1[i, k]
                        U1_numerator[i, k] += lambda_x * score * V[j, k]
                        U1_denominator[i, k] += lambda_x * prediction * V[j, k]

                for idx in prange(Y.shape[0]):
                    i = Y_iids[idx]
                    j = Y_qids[idx]
                    prediction = _dot(num_explicit_factors, &U2[i, 0], 1, &V[j, 0], 1)
                    score = Y[idx]
                    loss += (prediction - score) * (prediction - score)
                    for k in range(num_explicit_factors):
                        V_numerator[j, k] += lambda_y * score * U2[i, k]
                        V_denominator[j, k] += lambda_y * prediction * U2[i, k]
                        U2_numerator[i, k] += lambda_y * score * V[j, k]
                        U2_denominator[i, k] += lambda_y * prediction * V[j, k]

                # update V
                for i in prange(num_features):
                    for j in range(num_explicit_factors):
                        loss += lambda_v * V[i, j] * V[i, j]
                        V_denominator[i, j] += (X_feature_counts[i] + Y_feature_counts[i]) * lambda_v * V[i, j] + eps
                        V[i, j] *= sqrt(V_numerator[i, j] / V_denominator[i, j])

                # update U1, H1
                for i in prange(num_users):
                    for j in range(num_explicit_factors):
                        loss += lambda_u * U1[i, j] * U1[i, j]
                        U1_denominator[i, j] += (A_user_counts[i] + X_user_counts[i])* lambda_u * U1[i, j] + eps
                        U1[i, j] *= sqrt(U1_numerator[i, j] / U1_denominator[i, j])
                    for j in range(num_latent_factors):
                        loss += lambda_h * H1[i, j] * H1[i, j]
                        H1_denominator[i, j] += A_user_counts[i] * lambda_h * H1[i, j] + eps
                        H1[i, j] *= sqrt(H1_numerator[i, j] / H1_denominator[i, j])

                # update U2, H2
                for i in prange(num_items):
                    for j in range(num_explicit_factors):
                        loss += lambda_u * U2[i, j] * U2[i, j]
                        U2_denominator[i, j] += (A_item_counts[i] + Y_item_counts[i]) * lambda_u * U2[i, j] + eps
                        U2[i, j] *= sqrt(U2_numerator[i, j] / U2_denominator[i, j])
                    for j in range(num_latent_factors):
                        loss += lambda_h * H2[i, j] * H2[i, j]
                        H2_denominator[i, j] += A_item_counts[i] * lambda_h * H2[i, j] + eps
                        H2[i, j] *= sqrt(H2_numerator[i, j] / H2_denominator[i, j])

            if self.verbose:
                print('iter: %d, loss: %f' % (t, loss))

        if self.verbose:
            print('Optimization finished!')

    def _build_matrices(self, data_set: Dataset) -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
        """
        Build user-item rating matrix, user-feature matrix and item-quality matrix.

        :param data_set: User-Item preference data as well as additional modalities.
        :return: User-item rating matrix, user-feature matrix and item-quality matrix
        """
        features = self.train_set.item_feature
        ratings = []
        map_uid = []
        map_iid = []

        for uid, iid, rating in data_set.uir_iter():
            if self.train_set.is_unk_user(uid) or self.train_set.is_unk_item(iid):
                continue
            ratings.append(rating)
            map_uid.append(uid)
            map_iid.append(iid)

        ratings = np.asarray(ratings, dtype=np.float64).flatten()
        map_uid = np.asarray(map_uid, dtype=np.int64).flatten()
        map_iid = np.asarray(map_iid, dtype=np.int64).flatten()
        A = sp.csr_matrix((ratings, (map_uid, map_iid)), shape=(self.train_set.num_users, self.train_set.num_items))
        self.A=A
        
        attention_scores=[]
        map_uid=[]
        map_feature_id=[]
        
        if self.user_matrix_mode=="tf-idf":
            idf=np.zeros(shape=features.feature_dim, dtype=np.int64)
        
        for uid in range(self.train_set.num_users):
            if self.train_set.is_unk_user(uid):
                continue
            user_feature_count=self._user_func(features.features[A.getrow(uid).indices], axis=0)
            if self.user_matrix_mode=="tf-idf":
                user_feature_count=user_feature_count/user_feature_count.sum()
                idf+=(user_feature_count>0)
            for fid, count in enumerate(user_feature_count):
                attention_scores.append(count)
                map_uid.append(uid)
                map_feature_id.append(fid)

        if self.user_matrix_mode=="tf-idf":
            idf=np.log(self.train_set.num_users / idf)
        attention_scores=np.asarray(attention_scores, dtype=np.float64)
        if self.user_matrix_mode=="tf-idf":
            attention_scores=attention_scores*np.tile(idf, self.train_set.num_users)

        attention_scores = self._compute_attention_score(attention_scores.flatten())
        map_uid = np.asarray(map_uid, dtype=np.int64).flatten()
        map_feature_id = np.asarray(map_feature_id, dtype=np.int64).flatten()
        X = sp.csr_matrix((attention_scores, (map_uid, map_feature_id)), shape=(self.train_set.num_users, features.feature_dim))
        self.X=X

        quality_scores=features.features.copy()
        if self.item_matrix_mode=="rating":
            item_rating_means=A.sum(axis=0) / (A>0).sum(axis=0)
            quality_scores=quality_scores*item_rating_means.A[0].reshape(-1,1)
        elif self.item_matrix_mode=="tf-idf":
            idf=np.log(self.train_set.num_items/quality_scores.sum(axis=0))
            quality_scores=quality_scores/quality_scores.sum(axis=1).reshape(-1,1)
            quality_scores=quality_scores*idf
        
        quality_scores = self._compute_quality_score(quality_scores.flatten())
        map_iid = np.arange(self.train_set.num_items).repeat(features.feature_dim)
        map_feature_id = np.tile(np.arange(features.feature_dim), self.train_set.num_items)
        Y = sp.csr_matrix((quality_scores, (map_iid, map_feature_id)), shape=(self.train_set.num_items, features.feature_dim))
        self.Y=Y
        
        if self.verbose:
            print('Building matrices completed!')
            
        return A, X, Y
    
    def _compute_attention_score(self, count):
        """
        Transform user-feature raw count to matrix value.

        :param count: User-feature raw count/counts.
        :return: User-feature matrix value/values
        """
        if self.user_matrix_mode=="indicator":
            return count * self.rating_scale
        else:
            return np.where(count==0, 0, 1 + (self.rating_scale - 1) * (2 / (1 + np.exp(-self.sigmoid_b * count)) - 1))
    
    def _compute_quality_score(self, count):
        """
        Transform item-quality raw count to matrix value.

        :param count: Item-quality raw count/counts.
        :return: Item-quality matrix value/values
        """
        if self.item_matrix_mode=="rating":
            return count
        elif self.item_matrix_mode=="indicator":
            return count * self.rating_scale
        else:
            return np.where(count==0, 0, 1 + (self.rating_scale - 1) * (2 / (1 + np.exp(-self.sigmoid_b * count)) - 1))

    def score(self, user_idx: int, item_idx: int=None) -> np.array:
        """
        Predict the scores/ratings of a user for an item.

        :param user_idx: The index of the user for whom to perform score prediction.
        :param item_idx: The index of the item for which to perform score prediction. If None, scores for all known items will be returned.
        :return: Relative scores that the user gives to the item or to all known items
        """
        if item_idx is None:
            if self.train_set.is_unk_user(user_idx):
                raise ScoreException("Can't make score prediction for (user_id=%d" & user_idx)
            item_scores = self.U2.dot(self.U1[user_idx, :]) + self.H2.dot(self.H1[user_idx, :])
            return item_scores
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(item_idx):
                raise ScoreException("Can't make score prediction for (user_id=%d, item_id=%d)" % (user_idx, item_idx))
            item_score = self.U2[item_idx, :].dot(self.U1[user_idx, :]) + self.H2[item_idx, :].dot(self.H1[user_idx, :])
            return item_score

    def rank(self, user_idx: int, item_indices=None) -> Tuple[np.array, np.array]:
        """
        Rank all test items for a given user.

        :param user_idx: The index of the user for whom to perform item raking.
        :param item_indices:
        :return: Tuple of item_rank, and item_scores. The order of values in item_scores are corresponding to the order of their ids in item_indices
        """
        X_ = self.U1[user_idx, :].dot(self.V.T)
        most_cared_features_indices = (-X_).argsort()[:self.num_most_cared_features]
        most_cared_X_ = X_[most_cared_features_indices]
        most_cared_Y_ = self.U2.dot(self.V[most_cared_features_indices, :].T)
        explicit_scores = most_cared_X_.dot(most_cared_Y_.T) / (self.num_most_cared_features * self.rating_scale)
        item_scores = self.alpha * explicit_scores + (1 - self.alpha) * self.score(user_idx)

        if item_indices is None:
            item_scores = item_scores
            item_rank = item_scores.argsort()[::-1]
        else:
            num_items = max(self.train_set.num_items, max(item_indices) + 1)
            item_scores = np.ones(num_items) * np.min(item_scores)
            item_scores[:self.train_set.num_items] = item_scores
            item_rank = item_scores.argsort()[::-1]
            item_rank = intersects(item_rank, item_indices, assume_unique=True)
            item_scores = item_scores[item_indices]
        return item_rank, item_scores