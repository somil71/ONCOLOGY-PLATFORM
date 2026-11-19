import json


def feature_shapes(model):
    slices = model.rad_encoder.pos_embedding.shape[1]
    return {
        "wsi": (slices, 768),
        "rad": (slices, 512),
        "rna": (model.rna_encoder.encoder[0].in_features,),
        "cli": (model.cli_encoder.net[0].in_features,),
    }


def demo_features(shapes):
    import torch

    generator = torch.Generator().manual_seed(42)
    return {key: torch.randn(shape, generator=generator) for key, shape in shapes.items()}


def read_case_features(payload, shapes):
    import torch

    if len(payload) > 2_000_000:
        raise ValueError("Use a preprocessed JSON file smaller than 2 MB.")
    document = json.loads(payload)
    if not isinstance(document, dict) or not document or document.keys() - shapes.keys():
        raise ValueError("Provide an object with one or more of these keys: wsi, rad, rna, cli.")
    features = {}
    for key, value in document.items():
        try:
            tensor = torch.tensor(value)
        except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
            raise ValueError(f"{key}: provide a rectangular numeric array.") from exc
        if tensor.dtype == torch.bool or tensor.is_complex():
            raise ValueError(f"{key}: provide real numeric features.")
        tensor = tensor.to(dtype=torch.float32)
        if tuple(tensor.shape) != shapes[key]:
            raise ValueError(f"{key}: expected shape {shapes[key]}, got {tuple(tensor.shape)}.")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{key}: all values must be finite.")
        features[key] = tensor
    return features


def run_case(model, shapes, features, selected):
    import torch

    if not selected or set(selected) - features.keys():
        raise ValueError("Select at least one supplied modality.")
    if features.keys() - shapes.keys():
        raise ValueError("Unrecognized modality input.")
    for key, value in features.items():
        if tuple(value.shape) != shapes[key] or not torch.isfinite(value).all():
            raise ValueError(f"Invalid {key} feature shape or values.")
    mask = torch.tensor(
        [[key in features for key in shapes], [key in selected for key in shapes]],
        dtype=torch.float32,
    )
    inputs = []
    for key, shape in shapes.items():
        blank = torch.zeros(shape)
        supplied = features.get(key, blank)
        inputs.append(torch.stack((supplied, supplied if key in selected else blank)))
    with torch.inference_mode():
        risk, uncertainty, weights, _, _ = model(*inputs, mask)
    if not all(torch.isfinite(output).all() for output in (risk, uncertainty, weights)):
        raise ValueError("The model produced non-finite outputs. This run was not saved.")
    return risk.tolist(), uncertainty.tolist(), weights.tolist()
