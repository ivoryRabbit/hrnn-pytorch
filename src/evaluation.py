import torch


def get_nDCG(indices, output, k):
    """
    Args:
    Returns:
    """
    outputs = output.view(-1, 1).expand_as(indices)
    hits = (outputs == indices).float()

    dcg_weight = torch.reciprocal(torch.log2(torch.arange(2, k+2)))
    dcg = torch.sum(hits * dcg_weight.expand_as(hits), dim=1)
    idcg = 1.0
    ndcg = dcg / idcg
    return torch.mean(ndcg).item()


def get_recall(indices, output):
    """
    Args:
    Returns:
    """
    outputs = output.view(-1, 1).expand_as(indices)
    n_hits = (outputs == indices).nonzero()[:, :-1].size(0)
    recall = float(n_hits) / outputs.size(0)
    return recall


def get_precision(indices, output, k):
    """
    Args:
    Returns:
    """
    outputs = output.view(-1, 1).expand_as(indices)
    hits = (outputs == indices)
    precision = torch.sum(hits, axis=1) / k
    return torch.mean(precision).item()


def get_mrr(indices, output):
    """
    Args:
    Returns:
    """
    outputs = output.view(-1, 1).expand_as(indices)
    hits = (outputs == indices).nonzero()
    ranks = hits[:, -1] + 1
    rr = torch.reciprocal(ranks)
    return torch.sum(rr).item() / output.size(0)


def get_coverage(indices, n_items):
    """
    Args:
    Returns:
    """
    rec_items = indices.view(-1).unique()
    return rec_items.size(0) / n_items


def get_topk(score, k):
    _, indices = torch.topk(score, k, -1)
    return indices


def evaluate(score, output, k=20):
    indices = get_topk(score, k)

    ndcg = get_nDCG(indices, output, k)
    precision = get_precision(indices, output, k)
    recall = get_recall(indices, output)
    mrr = get_mrr(indices, output)

    return {
        "ndcg": ndcg,
        "precision": precision,
        "recall": recall,
        "mrr": mrr,
    }