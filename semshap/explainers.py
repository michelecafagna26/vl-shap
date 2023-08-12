import math
from copy import copy
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sentence_transformers import SentenceTransformer, util

from semshap.masking import apply_mask
from semshap.core import powerset, shapley_kernel


class BaseExplainer:
    def __init__(self, model, device="cpu", **kwargs):

        self.ref_emb = None
        self.ref_caption = None
        self.M = None
        self.feature_masks = None
        self.device = device
        self.ref_val = kwargs.get("ref_val", None)
        self.model = model
        self.logger = kwargs.get("logger", None)

        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)

    def _f(self, features_masks, **kwargs):

        if self.ref_emb is None:
            raise ValueError(
                f"not valid reference embedding, make sure to pass a valid reference caption = {self.ref_emb}")

        cosine_scores = []
        for masked_img in tqdm(features_masks):

            caption = self.model(masked_img, **kwargs)
            emb = self.text_encoder.encode(caption, convert_to_tensor=True).to(self.device)
            cosine_score = util.cos_sim(self.ref_emb, emb)
            cosine_scores.append(cosine_score.item())

            if self.logger:
                self.logger.info({"gen": caption, "cos_sim": cosine_score.item()})

        return np.array(cosine_scores)

    def _init_v(self, k):
        return [self.ref_val] * k

    def _generate_row(self, s, x):

        if len(s) == 0:
            # empty set return the original image
            return copy(x)
        else:

            if self.logger:
                self.logger.info({
                    "subset": s
                })

            subset_masks = []
            for f_idx in s:
                subset_masks.append(copy(self.feature_masks[f_idx]))

            # apply_mask return the masked image
            return apply_mask(x, subset_masks)

    def explain(self, image, feature_masks, k=500, **kwargs):

        self.M = len(feature_masks)
        self.feature_masks = copy(feature_masks)
        self.ref_caption = self.model(image)
        self.ref_emb = self.text_encoder.encode(self.ref_caption, convert_to_tensor=True).to(self.device)

        return self.solve(image, k, **kwargs)

    def solve(self, x, k=500, **kwargs):

        sampling = kwargs.pop("sampling", "optimal")
        approx_method = kwargs.pop("approx_method", "matrix")  # matrix or lreg

        if k < 0:
            k = int(math.pow(2, self.M))
            if self.logger is not None:
                self.logger.info("Running BRUTE FORCE SHAP")

        if k > math.pow(2, self.M):
            k = int(math.pow(2, self.M))
            if self.logger is not None:
                self.logger.warning(" k > 2^M -->  set k = 2^M")
                self.logger.info("Running BRUTE FORCE SHAP")

        if k < self.M:
            raise ValueError(f"Number of samples 'k' cannot be lower then the number of features 'M': {k} < {self.M}")

        if k < math.pow(2, self.M):
            if self.logger is not None:
                self.logger.info(f"sampling {k / math.pow(2, self.M) * 100:.2f} % of the sampling space")

        X = np.zeros((k, self.M + 1))
        X[:, -1] = 1
        weights = np.zeros(k)
        V = self._init_v(k)

        sample = powerset(range(self.M), sampling=sampling)

        for i, s in enumerate(sample):

            # sampling budget limit
            if i == k:
                break

            s = list(s)
            V[i] = self._generate_row(s, x)
            X[i, s] = 1
            weights[i] = shapley_kernel(self.M, len(s))

        # f is the black box model computing the value function
        # prune the arguments and pass kwargs
        y = self._f(V, **kwargs)

        if approx_method == "matrix":
            # matrix formulation
            tmp = np.linalg.inv(np.dot(np.dot(X.T, np.diag(weights)), X))
            phi = np.dot(tmp, np.dot(np.dot(X.T, np.diag(weights)), y))

            res = (phi[:-1], phi[-1])

        elif approx_method == "lreg":
            # linear regression
            reg = LinearRegression().fit(X, y, sample_weight=weights)
            res = (reg.coef_[:-1], reg.intercept_)

        else:
            raise ValueError(f"{approx_method} is not recognized. Allowed modes are ['matrix', 'lreg']")

        return res
