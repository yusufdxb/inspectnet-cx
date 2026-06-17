"""Smoke test for the reverse-distillation detector.

ponytail: one runnable check. Loss is finite/trainable, the anomaly map matches the
input resolution, and a defaced patch scores higher than the clean image after a
brief fit on the "normal" batch.
"""
import torch

from inspectnet_cx.models.reverse_distill import ReverseDistill


def test_loss_map_and_defect_ordering():
    torch.manual_seed(0)
    model = ReverseDistill()
    x = torch.rand(2, 3, 256, 256)

    loss = model.loss(x)
    assert loss.requires_grad and torch.isfinite(loss)
    assert model.anomaly_map(x).shape == (2, 256, 256)
    assert model.image_score(x).shape == (2,)

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=5e-3)
    model.train()
    model.teacher.eval()
    normal = torch.rand(4, 3, 256, 256)
    for _ in range(15):
        opt.zero_grad()
        model.loss(normal).backward()
        opt.step()
    model.eval()

    clean = normal[:1]
    defaced = clean.clone()
    defaced[:, :, 80:160, 80:160] = torch.rand(1, 3, 80, 80) * 5.0
    assert model.image_score(defaced).item() > model.image_score(clean).item()
