import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mineru_vl_utils import mlx_compat


@pytest.mark.parametrize("explicit_head, needs_patch", [(False, True), (True, True), (False, False)])
def test_prepared_model_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit_head: bool, needs_patch: bool
) -> None:
    """验证配置修正、显式 lm_head 分支以及异常退出清理，不改变原始文件。"""
    config = {"model_type": "qwen2_vl", "tie_word_embeddings": not needs_patch, "text_config": {"tie_word_embeddings": True}}
    source = json.dumps(config)
    (tmp_path / "config.json").write_text(source)
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(mlx_compat, "_model_has_explicit_lm_head", lambda _: explicit_head)
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", SimpleNamespace(get_model_path=lambda *a, **k: tmp_path))
    prepared = None
    with pytest.raises(RuntimeError):
        with mlx_compat.prepare_mlx_model_path(tmp_path) as prepared:
            if needs_patch and not explicit_head:
                assert prepared != tmp_path
                assert json.loads((prepared / "config.json").read_text())["tie_word_embeddings"] is True
                assert (prepared / "model.safetensors").read_bytes() == b"weights"
            else:
                assert prepared == tmp_path
            raise RuntimeError("service failed")
    assert (tmp_path / "config.json").read_text() == source
    assert (tmp_path / "model.safetensors").read_bytes() == b"weights"
    if prepared != tmp_path:
        assert not prepared.exists()
