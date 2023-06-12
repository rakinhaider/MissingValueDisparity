import numpy as np


def conditional_ndcg(y_true, y_score, prot_attr='sex', target_attr='label',
                     privilege=None, target=None):
    ndcgs = []
    for true, score in zip(y_true, y_score):
        index = true.index
        true = true.values
        score = score.values
        dg = dgs(true, score, reduce=False)
        idg = dgs(true, true, reduce=False)
        selection = index.to_frame()
        selection['dg'] = dg
        if privilege is not None:
            selection = selection[selection[prot_attr] == privilege]
        if target is not None:
            selection = selection[selection[target_attr] == target]
        ndcg = selection.dg.sum() / idg.sum()
        ndcgs.append(ndcg)

    return np.mean(ndcgs)


def dgs(true, score, reduce=True):
    ranks = np.argsort(score)[::-1]
    rank_pos = np.array([np.where(ranks == i)[0][0] for i in range(len(ranks))])
    discounts = 1 / (np.log2(rank_pos + 2))
    dg = true * discounts
    if reduce:
        return dg.sum()
    else:
        return dg


def ndd(ranking, orig_ranking, reduce=True):
    dd = discounted_difference(ranking)
    inverse_ranking = get_priv_inverse_ranking(orig_ranking)
    max_dd = discounted_difference(inverse_ranking)
    if reduce:
        return sum(dd) / sum(max_dd)
    else:
        return np.array(dd) / sum(max_dd)


def get_priv_inverse_ranking(orig_ranking):
    inverse_ranking = orig_ranking.copy()
    n_upriv = len(inverse_ranking.loc[:, 0])
    cur_upriv_rank, cur_priv_rank = 0, 0
    for (index, s), val in inverse_ranking.items():
        if s == 1:
            inverse_ranking.at[index, s] = n_upriv + cur_priv_rank
            cur_priv_rank += 1
        else:
            inverse_ranking.at[index, s] = cur_upriv_rank
            cur_upriv_rank += 1
    return inverse_ranking


def discounted_difference(ranking):
    n = len(ranking)
    df = ranking.index.to_frame()
    df['ranks'] = ranking.values
    s_plus = df[df['sex'] == 1]

    diffs = []
    for i in np.round(np.arange(0.1, 1, 0.1) * n):
        top_i = df[df['ranks'] <= i]
        s_plus_top_i = top_i[top_i['sex'] == 1]
        diff = np.abs(len(s_plus_top_i) / (i+1) - len(s_plus) / n)
        diffs.append(diff)
    return diffs
