import contextlib
import io
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import openseespy.opensees as ops  # noqa: E402
import ufp  # noqa: E402
from ufp import model_table_generator as mtg  # noqa: E402


class PureHelperTests(unittest.TestCase):
    def test_rebar_geometry(self):
        expected_area = math.pi * 10.0**2 / 4.0
        self.assertAlmostEqual(ufp.rebar_area_from_dia(10), expected_area)
        self.assertAlmostEqual(
            ufp.rebar_equivalent_layer_thickness(10, 200),
            expected_area / 200.0,
        )
        with self.assertRaises(ValueError):
            ufp.rebar_area_from_dia(0)

    def test_table_parser_helpers(self):
        self.assertEqual(mtg.parse_bar_count_dia("12Φ16"), (12, 16.0))
        self.assertEqual(mtg.parse_coord_spec("0, 100, 300"), [0.0, 100.0, 300.0])
        vertical, horizontal = mtg.parse_wall_rebar_spec("V10@200; H8@200")
        self.assertEqual(vertical, (10.0, 200.0))
        self.assertEqual(horizontal, (8.0, 200.0))

    def test_explicit_tag_mappings(self):
        materials = {"C30": (1, "Concrete C30")}
        sections = {"W1": (10, "Wall W1")}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            ufp.print_all_material_tags(materials)
            ufp.print_all_section_tags(sections)
            ufp.search_tags("Wall", materials, sections)
        rendered = output.getvalue()
        self.assertIn("C30", rendered)
        self.assertIn("W1", rendered)


class OpenSeesSmokeTests(unittest.TestCase):
    def tearDown(self):
        ops.wipe()

    def test_node_registry_reuses_existing_node(self):
        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 6)
        ops.node(1, 0.0, 0.0, 0.0)
        node_map = ufp.build_node_bucket_map()

        tag, created, node_map = ufp.get_or_create_node(
            0.0, 0.0, 0.0, node_map=node_map
        )
        self.assertEqual(tag, 1)
        self.assertFalse(created)

        new_tag, created, _ = ufp.get_or_create_node(
            100.0, 0.0, 0.0, node_map=node_map
        )
        self.assertEqual(new_tag, 2)
        self.assertTrue(created)
        self.assertEqual(ops.getNodeTags(), [1, 2])


if __name__ == "__main__":
    unittest.main()
