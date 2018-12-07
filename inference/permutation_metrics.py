import numpy as np

# p-val <= 0.05: strong evidence against null hypothesis

# Kendall Rank Correlation Coefficient
# https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
# Between -1 and 1. close to 1: strong agreement, close to -1: strong disagreement (in rank)
# Based off Kendall Tau distance: metric that counts the number of pairwise disagreements between two ranking lists
# Implementation: tau-b, accounts for ties
# Kendall Tau seems more appropiate for our application
from scipy.stats import kendalltau as kt

# Spearman's rank correlation coefficient
# https://en.wikipedia.org/wiki/Spearman's_rank_correlation_coefficient
# It assesses how well the relationship between two variables can be described using a monotonic function.
# Coefficient: -1, 1
from scipy.stats import spearmanr as sp


def get_ranking(M, weights=None):
    """
    :param M: n x X binary matrix of student responses or skills
    :param weights: np array (len M.shape[1] (X)). optional, pass if we want to do a weighted average of the X responses/skills
    :return: ranking of students (student represented by index) based off number of correct responses.
    The first element in return is the index of the "best" student
    """
    if weights is None:
        X = M.shape[1]
        weights = np.ones(X) / X

    weighted_skills = M.dot(weights)
    return np.argsort(weighted_skills)[::-1]


def rank_similarities(A_true, R, A_pred, skill_weights=None, question_weights=None):
    """
    RELEVANT METHOD FOR THIS MODULE
    Simply call this method, and we will return a dictionary with all the relevant results.

    We use as goal the (true) rankings implied by A_true. We compare the implied rankings from R and A_pred to
    those implied by A_true, and return the results. Ideally, the correlation between A_true and A_pred is higher than
    that of A_true and R (the baseline).

    :param R: n x q matrix (dataset, student responses)
    :param A_true: n x m (true latent skills [from simulation])
    :param A_pred: n x m matrix (predicted latent skills)
    :param skill_weights: see get_ranking_from_A
    :return:
    Dictionary with correlation metrics based off true predictions and latent predictions
    dict {
        'summary': <string with summary of results>,
        'metric': <see below>
        'baseline': {
            'kendall_tau': { 'correlation': <float [-1,1]>, 'pvalue': <float> },
            'spearman': { 'correlation': <float [-1,1]>, 'pvalue': <float> },
        },
        'pred': {
            'kendall_tau': { 'correlation': <float [-1,1]>, 'pvalue': <float> }
            'spearman': { 'correlation': <float [-1,1]>, 'pvalue': <float> }
        }
    }

    metric: absolute difference between the average coefficient of the prediction and the baseline.
    """
    rank_true = get_ranking(A_true, weights=skill_weights)
    rank_baseline = get_ranking(R, weights=question_weights)
    rank_pred = get_ranking(A_pred, weights=skill_weights)

    kt_baseline = np.around(kt(rank_true, rank_baseline), decimals=3)
    sp_baseline = np.around(sp(rank_true, rank_baseline), decimals=3)
    kt_pred = np.around(kt(rank_true, rank_pred), decimals=3)
    sp_pred = np.around(sp(rank_true, rank_pred), decimals=3)


    metric = np.average([kt_pred[0], sp_pred[0]]) - np.average([kt_baseline[0], sp_baseline[0]])
    string = \
        """
        Summary of Ranking Evaluation: 
        The correlations are with the true rankings derived from A_true.
        For the baseline, we get a Kendall Rank correlation of {}, with p-value of {}, and a Spearman correlation of {}, with p-value of {}. 
        For the prediction, we get a Kendall Rank correlation of {}, with p-value of {}, and a Spearman correlation of {}, with p-value of {}.  
        Which gives us an average difference of {} versus the baseline. 
        """.format(kt_baseline[0], kt_baseline[1], sp_baseline[0], sp_baseline[1], kt_pred[0], kt_pred[1],sp_pred[0] ,sp_pred[1], metric)

    d = {
        'summary': string,
        'metric': metric,
        'baseline': {
            'kendall_tau': { 'correlation': kt_baseline[0], 'pvalue': kt_baseline[1] },
            'spearman': { 'correlation': sp_baseline[0], 'pvalue': sp_baseline[1]},
        },
        'pred': {
            'kendall_tau': { 'correlation': kt_pred[0], 'pvalue': kt_pred[1] },
            'spearman': { 'correlation': sp_pred[0], 'pvalue': sp_pred[1]},
        }
    }

    return d


if __name__ == '__main__':
    print("Kendall Tau")
    print(kt([1,2,3], [1,2,3]))
    print(kt([1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]))
    print(kt([1,2,3,4], [2,1,4,3]))
    print(kt([1,2,3,4], [4,3,2,1]))
    print()

    print("Spearman")
    print(sp([1,2,3], [1,2,3]))
    print(sp([1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]))
    print(sp([1,2,3,4], [2,1,4,3]))
    print(sp([1,2,3,4], [4,3,2,1]))
    print()

    A_true = np.array([
        [.2, .3, .4],
        [.3, .4, .5],
        [.6, .4, .5],
    ])

    A_pred = np.array([
        [.3, .4, .5],
        [.2, .3, .4],
        [.6, .4, .5],
    ])

    R = np.array([
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [0, 0, 0, 0],
    ])

    print(rank_similarities(A_true, R, A_pred)['summary'])






