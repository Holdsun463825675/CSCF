import torch
import torch.nn.functional as F

def compute_entropy(logits):
    prob = F.softmax(logits, dim=-1)
    entropy = -torch.sum(prob * torch.log(prob), dim=-1)
    return entropy

def compute_e_uncertainty(logits_multi):
    entropies_multi = [compute_entropy(logits) for logits in logits_multi]
    entropies_tensor = torch.stack(entropies_multi, dim=0)
    column_sums = entropies_tensor.sum(dim=0)
    column_sums[column_sums == 0] = 1
    uncertainty = entropies_tensor / column_sums
    return uncertainty

def compute_entropy_inconsistency(sub_logits):
    num_networks = len(sub_logits)

    if num_networks == 1:
        batch_size = sub_logits[0].shape[0]
        return torch.zeros(batch_size, device=sub_logits[0].device)  # [B]

    probs_tensor = torch.stack([F.softmax(logits, dim=-1) for logits in sub_logits], dim=0)  # [K, B, C]
    entropy_tensor = -(probs_tensor * probs_tensor.log()).sum(dim=-1)  # [K, B]
    entropy_inconsistency = entropy_tensor.std(dim=0)  # [B]

    return entropy_inconsistency  # [B]

def inconsistency_trans(inconsistency):
    denominator = torch.sum(inconsistency, dim=0)
    denominator[denominator == 0] = 1
    inconsistency = inconsistency / denominator
    return inconsistency

def process_similarity_mat(similarity_mat, topk, type=0):
    num_classes = similarity_mat.size(0)
    half_classes = num_classes // 2

    if topk == 0:
        similarity_mat[:, :] = 0
        return similarity_mat

    if type == 0:
        similarity_mat[:half_classes, :] = 0
        similarity_mat[:, half_classes:] = 0

        for j in range(half_classes):
            col = similarity_mat[:, j]
            topk_values, _ = torch.topk(col, topk)
            threshold = topk_values[-1]
            col[col < threshold] = 0
    elif type == 1:
        similarity_mat[:, :half_classes] = 0
        similarity_mat[half_classes:, :] = 0

        for j in range(half_classes, num_classes):
            col = similarity_mat[:, j]
            topk_values, _ = torch.topk(col, topk)
            threshold = topk_values[-1]
            col[col < threshold] = 0

    return similarity_mat


def compute_uncertainty(logits_multi, sub_logits):
    Inconsistency = compute_e_uncertainty(logits_multi)
    Inconsistency.reshape(Inconsistency.shape[1], -1)
    difference = torch.stack([compute_entropy_inconsistency(sub_logits[i]) for i in range(len(sub_logits))], dim=0)
    difference = inconsistency_trans(difference)
    uncertainty = Inconsistency + difference

    return uncertainty


def combine_logits_similarity(logits_multi, uncertainty, similarity_mat, top_k):
    logits_A, logits_B = logits_multi

    similarity_matB = process_similarity_mat(similarity_mat.clone(), top_k, type=0)
    similar_A = torch.matmul(logits_B, similarity_matB)
    similarity_matA = process_similarity_mat(similarity_mat.clone(), top_k, type=1)
    similar_B = torch.matmul(logits_A, similarity_matA)

    logits = logits_multi[0].clone()
    mask = uncertainty[0] < uncertainty[1]
    logits[mask] = logits_A[mask] + similar_A[mask]
    logits[~mask] = logits_B[~mask] + similar_B[~mask]

    return logits
