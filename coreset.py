import torch

def execute_coreset_selection(feature_dict_path, quality_dict_path, alpha, size, device=None):
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    feature_dict = torch.load(feature_dict_path, map_location="cpu", weights_only=False)
    files = list(feature_dict.keys())
    feature = torch.stack([feature_dict[f] for f in files]).float().to(device)
    assert 0 < size <= len(files)

    if alpha != 0:
        quality_dict = torch.load(quality_dict_path, map_location="cpu", weights_only=False)
        quality = torch.tensor([quality_dict[f] for f in files]).float().to(device)

    selected = [0]
    score = torch.zeros(len(files), dtype=torch.float32, device=device)
    while len(selected) < size:
        last = selected[-1]
        score[last] = -float("inf")

        distance = ((feature - feature[last]) ** 2).sum(dim=1)

        if alpha != 0:
            weight = (quality[last] * quality) ** alpha
            score += weight * distance
        else:
            score += distance

        selected.append(int(torch.argmax(score).item()))

    assert len(set(selected)) == size

    return [files[i] for i in selected]
