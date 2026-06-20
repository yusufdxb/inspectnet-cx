"""Smoke test for the InspectNet-CX student-teacher detector.

ponytail: one runnable check on random tensors. Asserts the loss is finite and
trainable, the anomaly map matches the input resolution, and a defaced image
scores higher than the clean one it was derived from.
"""
import pytest
import torch

pytest.importorskip("torchvision")  # baseline-tier dep; native-model test skips without it

from inspectnet_cx.models.student_teacher import StudentTeacher


def test_loss_and_map_shapes():
    model = StudentTeacher()
    x = torch.rand(2, 3, 64, 64)

    loss = model.loss(x)
    assert loss.requires_grad and torch.isfinite(loss)

    amap = model.anomaly_map(x)
    assert amap.shape == (2, 64, 64)

    scores = model.image_score(x)
    assert scores.shape == (2,) and torch.all(torch.isfinite(scores))


def test_defect_scores_higher_after_training():
    torch.manual_seed(0)
    model = StudentTeacher()
    normal = torch.rand(4, 3, 64, 64)

    # Briefly fit the student to the "normal" batch.
    opt = torch.optim.Adam(model.student.parameters(), lr=1e-3)
    model.student.train()
    for _ in range(20):
        opt.zero_grad()
        model.loss(normal).backward()
        opt.step()
    model.student.eval()

    clean = normal[:1]
    defaced = clean.clone()
    defaced[:, :, 20:40, 20:40] = torch.rand(1, 3, 20, 20) * 5.0  # out-of-distribution patch
    assert model.image_score(defaced).item() > model.image_score(clean).item()
