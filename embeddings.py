# embeddings.py
from __future__ import annotations
import numpy as np

def encode_patches(
    patches: np.ndarray,
    *,
    model_name: str = "resnet18",
    batch_size: int = 128,
    device: str | None = None,
    normalize_input: bool = True,
) -> np.ndarray:
    import torch
    import torch.nn as nn
    import torchvision.models as tvm
    """
    Encode RGB patches into CNN embeddings using a pretrained torchvision model.

    Inputs:
      patches: (N, P, P, 3) numpy array, dtype can be float or uint.
              Expected range:
                - if normalize_input=True: values will be scaled to [0,1] if needed
      model_name: "resnet18" or "resnet50"
      batch_size: encode in batches to avoid OOM
      device: "cuda" or "cpu". If None, auto-detect.
      normalize_input: apply ImageNet normalization (recommended)

    Returns:
      embeddings: (N, d) float32, where d=512 for resnet18, d=2048 for resnet50
    """
    if patches.ndim != 4 or patches.shape[-1] != 3:
        raise ValueError(f"patches must be (N,P,P,3). Got {patches.shape}")

    N = patches.shape[0]

    # ----------------------------
    # Device
    # ----------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------
    # Model (remove classifier head)
    # ----------------------------
    model_name = model_name.lower()
    if model_name == "resnet18":
        net = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
        out_dim = 512
    elif model_name == "resnet50":
        net = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
        out_dim = 2048
    else:
        raise ValueError("model_name must be 'resnet18' or 'resnet50'")

    # Replace fc with identity so output is the penultimate feature vector
    net.fc = nn.Identity()
    net.eval()
    net.to(device)

    # ----------------------------
    # Prepare input
    # ----------------------------
    x = patches.astype(np.float32)

    # If the GeoTIFF values are like 0..3000, bring to 0..1
    # Heuristic: if max > 1.5, assume it's not normalized yet.
    if x.max() > 1.5:
        # robust scaling: divide by a high percentile to avoid a few bright pixels
        scale = np.percentile(x, 99.5)
        scale = max(scale, 1.0)
        x = np.clip(x / scale, 0.0, 1.0)

    # Convert NHWC -> NCHW for torch
    x = np.transpose(x, (0, 3, 1, 2))  # (N,3,P,P)

    # ResNet expects 224x224. We'll resize patches in torch.
    # (Works fine for embeddings. Later you can use a satellite encoder for better.)
    x_t = torch.from_numpy(x)

    # ImageNet normalization
    # mean/std for RGB
    if normalize_input:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        mean = mean.to(device)
        std = std.to(device)

    # ----------------------------
    # Batch encoding
    # ----------------------------
    embeddings = np.zeros((N, out_dim), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            xb = x_t[start:end].to(device)  # (B,3,P,P)

            # Resize to 224x224 (ResNet default)
            xb = torch.nn.functional.interpolate(xb, size=(224, 224), mode="bilinear", align_corners=False)

            if normalize_input:
                xb = (xb - mean) / std

            feat = net(xb)  # (B, out_dim)
            embeddings[start:end] = feat.detach().cpu().numpy().astype(np.float32)

    return embeddings

# embeddings_torchgeo
def encode_patches_torchgeo(
    patches: np.ndarray,
    *,
    batch_size: int = 128,
    device: str | None = None,
) -> np.ndarray:
    import torch
    import torch.nn as nn
    import torchvision.models as tvm
    from torchgeo.models import get_weight

    if patches.ndim != 4 or patches.shape[-1] != 3:
        raise ValueError(f"Expected patches shape (N,P,P,3), got {patches.shape}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1) Build plain torchvision resnet50 WITHOUT torchvision weights
        net = tvm.resnet50(weights=None)
        net.fc = nn.Identity()

        # 2) Load TorchGeo pretrained weights into the model
        weights = get_weight("ResNet50_Weights.FMOW_RGB_GASSL")
        state_dict = weights.get_state_dict(progress=True)
        net.load_state_dict(state_dict, strict=False)

        net.eval().to(device)

        x = patches.astype(np.float32)

        # Scale Sentinel-like reflectance to [0,1]
        if x.max() > 1.5:
            scale = np.percentile(x, 99.5)
            scale = max(scale, 1.0)
            x = np.clip(x / scale, 0.0, 1.0)

        x = np.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
        x = torch.from_numpy(x)

        preprocess = weights.transforms

        feats = []
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                xb = x[i:i + batch_size].to(device)
                xb = torch.nn.functional.interpolate(
                    xb, size=(224, 224), mode="bilinear", align_corners=False
                )
                xb = preprocess(xb)
                fb = net(xb)
                feats.append(fb.cpu())

        return torch.cat(feats, dim=0).numpy().astype(np.float32)
def encode_image_to_patch_embeddings_torchgeo(
    img: np.ndarray,
    *,
    patch_size: int,
    device: str | None = None,
) -> np.ndarray:
    """
    Whole-image encoder using TorchGeo-pretrained ResNet50.

    Steps:
    1. Run CNN once on the whole image
    2. Take the convolutional feature map
    3. Average-pool the feature map over each patch region

    Args:
        img: (H, W, 3) numpy array
        patch_size: patch size in original image pixels
        device: "cuda" or "cpu"

    Returns:
        F: (N, C) patch embeddings in row-major patch order
    """
    import torch
    import torch.nn as nn
    import torchvision.models as tvm
    from torchgeo.models import get_weight

    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected img shape (H,W,3), got {img.shape}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Build plain torchvision ResNet50 backbone
    net = tvm.resnet50(weights=None)

    # 2) Load TorchGeo pretrained weights
    weights = get_weight("ResNet50_Weights.FMOW_RGB_GASSL")
    state_dict = weights.get_state_dict(progress=True)
    net.load_state_dict(state_dict, strict=False)

    # 3) Keep only convolutional part (remove avgpool + fc)
    feature_extractor = nn.Sequential(*list(net.children())[:-2])
    feature_extractor.eval().to(device)

    # 4) Preprocess image
    x = img.astype(np.float32)

    # scale raw Sentinel-like values to [0,1]
    if x.max() > 1.5:
        scale = np.percentile(x, 99.5)
        scale = max(scale, 1.0)
        x = np.clip(x / scale, 0.0, 1.0)

    x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
    x = torch.from_numpy(x).unsqueeze(0).to(device)  # (1,3,H,W)

    # TorchGeo transform
    preprocess = weights.transforms
    x = preprocess(x)

    # 5) Forward once on whole image
    with torch.no_grad():
        feat_map = feature_extractor(x)   # (1, C, Hf, Wf)

    feat_map = feat_map.squeeze(0).cpu().numpy()  # (C, Hf, Wf)

    C, Hf, Wf = feat_map.shape
    H, W, _ = img.shape

    # patch grid in original image
    grid_h = H // patch_size
    grid_w = W // patch_size
    Hc = grid_h * patch_size
    Wc = grid_w * patch_size

    embeddings = []

    for r in range(grid_h):
        for c in range(grid_w):
            r0 = r * patch_size
            r1 = (r + 1) * patch_size
            c0 = c * patch_size
            c1 = (c + 1) * patch_size

            # map patch bounds from image space -> feature map space
            fr0 = int(np.floor(r0 / Hc * Hf))
            fr1 = int(np.ceil(r1 / Hc * Hf))
            fc0 = int(np.floor(c0 / Wc * Wf))
            fc1 = int(np.ceil(c1 / Wc * Wf))

            fr1 = max(fr1, fr0 + 1)
            fc1 = max(fc1, fc0 + 1)

            region = feat_map[:, fr0:fr1, fc0:fc1]   # (C, h, w)
            emb = region.mean(axis=(1, 2))           # (C,)
            embeddings.append(emb)

    F = np.stack(embeddings, axis=0).astype(np.float32)
    return F