import json
import tempfile
import unittest
from pathlib import Path

import torch

from models.nanoGPT import GPT, GPTConfig
from models.weight_norms import load_target_norms


class WeightNormInitializationTests(unittest.TestCase):
    def test_loads_wrapped_json_and_canonicalizes_names(self):
        config = GPTConfig(vocab_size=16, n_layer=2, n_head=2, n_embd=8)
        payload = {
            "metadata": {
                "n_layer": 2,
                "n_embd": 8,
                "source": "unit test",
            },
            "target_norms": {
                "module._orig_mod.transformer.wte.weight": 3.5,
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target_norms.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = load_target_norms(path, config)

        self.assertEqual(result, {"transformer.wte.weight": 3.5})

    def test_rejects_architecture_mismatch(self):
        config = GPTConfig(vocab_size=16, n_layer=2, n_head=2, n_embd=8)
        payload = {
            "metadata": {"n_embd": 16},
            "target_norms": {"transformer.wte.weight": 3.5},
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target_norms.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "n_embd=16"):
                load_target_norms(path, config)

    def test_model_applies_every_target_norm_and_preserves_tied_weight(self):
        config = GPTConfig(vocab_size=16, n_layer=2, n_head=2, n_embd=8)
        reference_model = GPT(config)
        target_norms = {
            name: float(index + 1)
            for index, (name, _) in enumerate(reference_model.named_parameters())
        }

        model = GPT(config, target_weight_norms=target_norms)

        for name, parameter in model.named_parameters():
            self.assertTrue(
                torch.isclose(
                    parameter.detach().float().norm(),
                    torch.tensor(target_norms[name]),
                    rtol=1e-5,
                    atol=1e-6,
                ),
                msg=name,
            )

        self.assertIs(model.transformer.wte.weight, model.lm_head.weight)

    def test_model_can_use_separate_embedding_and_lm_head_weights(self):
        config = GPTConfig(
            vocab_size=16,
            n_layer=2,
            n_head=2,
            n_embd=8,
            tie_embeddings=False,
        )
        model = GPT(config)

        self.assertIsNot(model.transformer.wte.weight, model.lm_head.weight)
        self.assertNotEqual(
            model.transformer.wte.weight.untyped_storage().data_ptr(),
            model.lm_head.weight.untyped_storage().data_ptr(),
        )

        embedding_before = model.transformer.wte.weight.detach().clone()
        with torch.no_grad():
            model.lm_head.weight.zero_()
        self.assertTrue(torch.equal(model.transformer.wte.weight, embedding_before))

    def test_model_rejects_incomplete_target_table(self):
        config = GPTConfig(vocab_size=16, n_layer=2, n_head=2, n_embd=8)
        with self.assertRaisesRegex(KeyError, "Missing target norms"):
            GPT(
                config,
                target_weight_norms={"transformer.wte.weight": 3.5},
            )


if __name__ == "__main__":
    unittest.main()
