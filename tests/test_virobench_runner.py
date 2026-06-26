from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from virobench_gamil.runner import load_embedder


class LoadEmbedderPathTests(unittest.TestCase):
    def _install_dummy_module(self, module_name: str, **attrs):
        old_parent_name = module_name.split(".")[0]
        old_parent = sys.modules.get(old_parent_name)
        old_module = sys.modules.get(module_name)
        parent = types.ModuleType(old_parent_name)
        parent.__path__ = []
        module = types.ModuleType(module_name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[old_parent_name] = parent
        sys.modules[module_name] = module
        return old_parent_name, old_parent, module_name, old_module

    def _restore_dummy_module(self, state) -> None:
        old_parent_name, old_parent, module_name, old_module = state
        if old_parent is None:
            sys.modules.pop(old_parent_name, None)
        else:
            sys.modules[old_parent_name] = old_parent
        if old_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = old_module

    def test_lucavirus_default_path_uses_public_dot_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            external_root = Path(tmp) / "external"
            virobench_root = external_root / "ViroBench"
            public_model_dir = external_root / "model_weight" / "LucaVirus-default-step3.8M"
            virobench_root.mkdir(parents=True)
            public_model_dir.mkdir(parents=True)

            captured = {}

            class DummyLucaVirusModel:
                def __init__(self, **kwargs):
                    captured.update(kwargs)

                def get_embedding(self, *args, **kwargs):
                    return np.zeros((1, 13), dtype=np.float32)

            module_state = self._install_dummy_module(
                "models.lucavirus",
                LucaVirusModel=DummyLucaVirusModel,
            )
            try:
                _, cfg = load_embedder(
                    virobench_root,
                    "LucaVirus-default-step3.8M",
                    model_dir=None,
                    device="cpu",
                )
            finally:
                self._restore_dummy_module(module_state)

            self.assertEqual(str(public_model_dir), captured["model_path"])
            self.assertEqual(13, cfg["hidden_size"])

    def test_omnireg_default_paths_are_repo_local(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            external_root = Path(tmp) / "external"
            virobench_root = external_root / "ViroBench"
            repo_dir = external_root / "official" / "OmniReg-GPT"
            asset_dir = external_root / "model_weight" / "OmniReg-GPT"
            virobench_root.mkdir(parents=True)
            repo_dir.mkdir(parents=True)
            asset_dir.mkdir(parents=True)
            (asset_dir / "pytorch_model.bin").write_bytes(b"")
            (asset_dir / "gena-lm-bert-large-t2t").mkdir()

            captured = {}

            class DummyOmniRegGPTModel:
                def __init__(self, **kwargs):
                    captured.update(kwargs)

            module_state = self._install_dummy_module(
                "models.omnireg_model",
                OmniRegGPTModel=DummyOmniRegGPTModel,
            )
            try:
                _, cfg = load_embedder(
                    virobench_root,
                    "OmniReg-GPT",
                    model_dir=None,
                    device="cpu",
                )
            finally:
                self._restore_dummy_module(module_state)

            self.assertEqual(str(asset_dir / "pytorch_model.bin"), captured["model_path"])
            self.assertEqual(str(asset_dir / "gena-lm-bert-large-t2t"), captured["tokenizer_path"])
            self.assertEqual(str(repo_dir), captured["omnireg_repo_path"])
            self.assertEqual(1024, cfg["hidden_size"])


if __name__ == "__main__":
    unittest.main()
