from ._univariate import (dist_plot,
                          count_plot)
from ._bivariate import (joint_plot,
                         corrmatrix,
                         compute_kl_by_prob,
                         compute_kl)


__all__ = [
    "dist_plot",
    "count_plot",
    "joint_plot",
    "corrmatrix",
    "compute_kl_by_prob",
    "compute_kl",
]
