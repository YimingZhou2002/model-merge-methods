"""Algorithm parity, strict task alignment and real safetensors export checks."""
import ast
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from research.checkpoint_merge_14b import core, merge, study

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/"research/checkpoint_merge_14b/reference"


def reference_function(relative, name):
    parsed = ast.parse((REFERENCE/relative).read_text())
    node = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"np": np, "torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(relative), "exec"), namespace)
    return namespace[name]


class MergeAlgorithms(unittest.TestCase):
    def test_mmr_matches_original_8b_selection_and_weights(self):
        old = reference_function("quality_diversity_mmr_v2/run_mmr.py", "select_bank")
        rng = np.random.default_rng(32)
        vectors = {s: rng.integers(0, 9, 57)/8 for s in range(230, 301, 10)}
        expected, actual = old(vectors), core.select_bank(vectors)
        self.assertEqual(actual["greedy_order"], expected["greedy_order"])
        self.assertEqual(actual["weights"], expected["weights"])
        self.assertEqual(actual["trace"], expected["trace"])

    def test_mmr_prefers_diversity_when_quality_weight_is_zero(self):
        result = core.select_bank({230: [1., 1., 0.], 240: [1., 1., 0.], 250: [0., 0., 1.]}, k=2, lam=0.)
        self.assertEqual(result["greedy_order"], [230, 250])

    def test_all_trajectory_gates_match_original_including_flat_coordinates(self):
        old = reference_function("quality_diversity_v6_g47/build_extensions.py", "trajectory_gates")
        generator = torch.Generator().manual_seed(15)
        checkpoints = [torch.randn((7, 11), generator=generator) for _ in range(8)]
        for tensor in checkpoints:
            tensor[0] = 5.
        update = torch.randn((7, 11), generator=generator)
        quality = np.linspace(.3, .7, 8)
        for expected, actual in zip(old(checkpoints, quality, update), core.trajectory_gates(checkpoints, quality, update)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(actual).all())
            self.assertTrue(((actual >= 0) & (actual <= 1)).all())
            self.assertTrue((actual[0] == 0).all())

    def test_ties_elects_weighted_sign_and_renormalizes_only_retained_members(self):
        base = torch.tensor([10., 10., 10., 10.])
        donors = [torch.tensor([12., 8., 11., 10.]), torch.tensor([8., 12., 13., 10.])]
        torch.testing.assert_close(core.weighted_ties(base, donors, [.75, .25]), torch.tensor([12., 8., 11.5, 10.]))
        torch.testing.assert_close(core.weighted_ties(base, donors, [.5, .5]), torch.tensor([12., 12., 12., 10.]))

    def test_incomplete_or_misaligned_rollouts_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"rollouts.jsonl"
            rows = [{"input": f"task-{i}", "score": int(i == 0)} for i in range(3) for _ in range(8)]
            path.write_text("\n".join(map(json.dumps, rows)))
            result = core.read_rollouts(path, tasks=3)
            self.assertAlmostEqual(result["mean_at_8"], 1/3)
            rows[0]["input"] = "task-1"
            path.write_text("\n".join(map(json.dumps, rows)))
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                core.read_rollouts(path, tasks=3)
            other = dict(result, task_keys=["different"]*3)
            with self.assertRaisesRegex(ValueError, "Different task"):
                core.paired_comparison(result, other)

    def test_prompt_family_mapping_and_cluster_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"rollouts.jsonl"
            instructions = {"abc_1": "first instruction", "abc_2": "second instruction", "def_1": "third instruction"}
            rows = [{"input": f"system: {instruction}", "score": 1} for instruction in instructions.values() for _ in range(8)]
            path.write_text("\n".join(map(json.dumps, rows)))
            left = core.read_rollouts(path, tasks=3, task_instructions=instructions)
            self.assertEqual(set(left['task_families']), {'abc', 'def'})
            right = dict(left, vector=[0., 0., 0.])
            compared = core.paired_comparison(left, right, iterations=100)
            self.assertEqual(compared['families'], 2)
            self.assertEqual(compared['family_bootstrap_95_pp'], [100., 100.])

    def test_exported_sharded_models_and_norm_control_match_known_tensor_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = root/"pool"
            checkpoints = []
            steps = list(range(230, 301, 10))
            for index, step in enumerate(steps):
                path = pool/f"checkpoint-{step}"
                path.mkdir(parents=True)
                value = torch.tensor([[1.+index*.03, 2.+(-1)**index*.12], [3.-index*.02, 4.]])
                checkpoints.append(value)
                save_file({"weight": value}, str(path/"model.safetensors"))
                (path/"config.json").write_text('{"dtype":"float32"}')
                (path/"tokenizer.json").write_text('{}')
            base = root/"base"
            base.mkdir()
            raw = torch.ones((2, 2), dtype=torch.bfloat16)
            save_file({"weight": raw}, str(base/"model.safetensors"))
            (base/"config.json").write_text('{"dtype":"bfloat16"}')
            config = {"pool": str(pool), "base_model": str(base), "checkpoints": steps, "anchor": 300}
            selection = {"weights": {"260": .2, "270": .3, "300": .5}}
            out = root/"models"
            merge.build_primary(config, selection, out)
            merge.build_norm_control(config, out)
            a = checkpoints[-1]
            linear = .2*checkpoints[3]+.3*checkpoints[4]+.5*a
            u = linear-a
            gate = core.trajectory_gates(checkpoints, list(range(8)), u)[0]
            expected = {"linear": linear, "tsg_tv": a+gate*u,
                        "mmr_ties": core.weighted_ties(raw.float(), [checkpoints[3], checkpoints[4], a], [.2, .3, .5])}
            values = {}
            for name in [*expected, "tsg_tv_norm"]:
                index = json.loads((out/name/"model.safetensors.index.json").read_text())
                values[name] = load_file(str(out/name/index["weight_map"]["weight"]))["weight"]
                self.assertTrue((out/name/"EXPORT_READY.json").is_file())
                self.assertEqual(values[name].dtype, torch.float32)
            for name in expected:
                torch.testing.assert_close(values[name], expected[name])
            self.assertAlmostEqual(float(torch.linalg.vector_norm(values['tsg_tv']-a)),
                                   float(torch.linalg.vector_norm(values['tsg_tv_norm']-a)), places=5)
            with self.assertRaises(FileExistsError):
                merge.build_primary(config, selection, out)

    def test_frozen_recipe_requires_all_baseline_prompts_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {"checkpoints": list(range(230, 301, 10)), "mmr": {"k": 5, "lambda": .7, "tau": .02},
                      "anchor": 300, "run_root": tmp}
            rows = {str(s): {"task_keys": ["a", "b"], "vector": [.5, .6], "log_sha256": str(s)} for s in config["checkpoints"]}
            rows["300"]["task_keys"] = ["a", "changed-prompt"]
            with self.assertRaisesRegex(ValueError, "different task"):
                study.freeze_recipes(config, rows)
            self.assertFalse((Path(tmp)/"recipes.json").exists())


if __name__ == "__main__":
    unittest.main()
