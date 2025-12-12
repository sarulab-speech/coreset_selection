import torch

def execute_coreset_selection(feature_dict_path, quality_dict_path, size, device=None):
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    feature_dict = torch.load(feature_dict_path, map_location="cpu", weights_only=False)
    files = list(feature_dict.keys())
    feature = torch.stack([feature_dict[f] for f in files]).float().to(device)
    assert 0 < size <= len(files)

    if quality_dict_path is None:
        quality = torch.tensor([1.0 for f in files]).float().to(device)
    else:
        quality_dict = torch.load(quality_dict_path, map_location="cpu", weights_only=False)
        quality = torch.tensor([quality_dict[f] for f in files]).float().to(device)

    init = torch.argmax(quality)
    
    selected = [init]
    score = torch.zeros(len(files), dtype=torch.float32, device=device)
    while len(selected) < size:
        last = selected[-1]
        score[last] = -float("inf")

        distance = ((feature - feature[last]) ** 2).sum(dim=1)

        weight = quality[last] * quality
        score += weight * distance

        selected.append(int(torch.argmax(score).item()))

    assert len(set(selected)) == size

    return [files[i] for i in selected]
