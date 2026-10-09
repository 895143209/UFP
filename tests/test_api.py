import ast
import inspect
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ufp  # noqa: E402


class PublicApiTests(unittest.TestCase):
    def test_public_exports_are_unique_and_resolvable(self):
        self.assertEqual(len(ufp.__all__), len(set(ufp.__all__)))
        self.assertNotIn("user_function", ufp.__all__)
        for name in ufp.__all__:
            self.assertTrue(hasattr(ufp, name), name)

    def test_kj_critical_api_is_available(self):
        expected = {
            "apply_model_table",
            "beam_mesh",
            "column_mesh",
            "wall_mesh_zoned",
            "build_rect_section",
            "create_layershell",
            "analyze_modal_participation",
            "apply_uniform_damping",
            "read_ground_motion",
            "setup_full_native_dynamic_recorders",
            "postprocess_native_dynamic_recorders",
            "run_transient_analysis",
        }
        self.assertTrue(expected.issubset(set(ufp.__all__)))

    def test_no_duplicate_top_level_definitions(self):
        source = ROOT / "ufp" / "user_function.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        names = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        self.assertEqual(len(names), len(set(names)))

    def test_opstool_import_is_lazy(self):
        source = ROOT / "ufp" / "user_function.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        top_level_imports = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertNotIn("opstool", top_level_imports)

    def test_critical_signatures_remain_stable(self):
        self.assertEqual(
            list(inspect.signature(ufp.apply_model_table).parameters),
            ["root", "table", "show_sections"],
        )
        self.assertEqual(
            list(inspect.signature(ufp.run_transient_analysis).parameters)[:2],
            ["dt", "n_step"],
        )
        self.assertIn(
            "story_weight_summary",
            inspect.signature(ufp.setup_full_native_dynamic_recorders).parameters,
        )


if __name__ == "__main__":
    unittest.main()
