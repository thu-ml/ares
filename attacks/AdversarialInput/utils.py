import torch


def cosine_similarity(x: list):
    '''
    input a list of tensor with same shape. return the mean cosine_similarity
    '''
    x = torch.stack(x, dim=0)
    N = x.shape[0]
    x = x.reshape(N, -1)

    norm = torch.norm(x, p=2, dim=1)
    x /= norm.reshape(-1, 1)  # N, D
    similarity = x @ x.T  # N, N
    # matrix_heatmap(similarity.cpu().numpy())
    mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=0).to(torch.bool)
    similarity = similarity[mask]
    return torch.mean(similarity).item()


def clamp(x: torch.tensor, min_value: float = 0, max_value: float = 1):
    return torch.clamp(x, min=min_value, max=max_value)
