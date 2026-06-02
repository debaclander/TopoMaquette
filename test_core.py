import unittest
import os
import io
import xml.etree.ElementTree as ET
import ezdxf
import shapely.geometry as sg
from shapely.affinity import rotate as rotate_geometry
from shapely.ops import unary_union
import networkx as nx

from core.parser import extract_polygons_from_dxf
from core.topology import build_topology_tree, assign_elevations
from core.slicer import generate_rings
from core.nesting import (
    _candidate_piece_sets,
    _effort_search_limits,
    _label_areas_for_piece,
    _optional_split_budget,
    split_polygon_if_needed,
    perform_nesting,
)
from core.exporter import (
    DEFAULT_LAYER_CONFIG,
    _clean_export_geometry,
    diagnose_exported_dxf,
    export_sheets_to_combined_dxf,
    export_sheets_to_dxf,
    export_sheets_to_svg,
)
from core.terrain_slicer import (
    _add_assembly_engraving_lines,
    _annotate_solid_slice_visibility,
    _clean_layer_geometry,
    _ensure_unique_and_complete_ids,
    _convert_solid_slices_to_hollow,
    _hidden_support_geometry,
    _prune_floating_hollow_components,
    build_contour_correction_table,
    generate_terrain_layers_from_points,
    generate_terrain_layers_from_contours,
)
from core.diagnostics import build_dxf_diagnostics
from core.ui_utils import (
    contour_id_sort_key,
    normalize_override_map,
    parse_bool,
    parse_curve_ids,
    parse_float_or_none,
    parse_int_or_none,
    resolve_boundary_and_contours,
    resolve_effective_overrides,
)

class TestCoreModules(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Ensure test_topography.dxf exists
        if not os.path.exists("test_topography.dxf"):
            from generate_test_dxf import create_test_dxf
            create_test_dxf("test_topography.dxf")
            
    def test_parser_valid_dxf(self):
        polygons = extract_polygons_from_dxf("test_topography.dxf")
        self.assertEqual(len(polygons), 7)
        for p in polygons:
            self.assertIn("id", p)
            self.assertIn("layer", p)
            self.assertIn("geometry", p)
            self.assertIn("elevation", p)
            self.assertTrue(p["geometry"].is_valid)
            self.assertFalse(p["geometry"].is_empty)
            
    def test_parser_nonexistent_file(self):
        polygons = extract_polygons_from_dxf("nonexistent_file.dxf")
        self.assertEqual(polygons, [])

    def test_topology_building(self):
        # Create a nested set of polygons
        # p1 (largest) contains p2, which contains p3
        p1 = {"id": 1, "geometry": sg.box(0, 0, 10, 10)}
        p2 = {"id": 2, "geometry": sg.box(2, 2, 8, 8)}
        p3 = {"id": 3, "geometry": sg.box(4, 4, 6, 6)}
        
        polygons_data = [p3, p1, p2] # unordered
        G = build_topology_tree(polygons_data)
        
        self.assertTrue(G.has_node(1))
        self.assertTrue(G.has_node(2))
        self.assertTrue(G.has_node(3))
        
        # Check parent-child hierarchy (p1 contains p2 directly, p2 contains p3 directly)
        self.assertTrue(G.has_edge(1, 2))
        self.assertTrue(G.has_edge(2, 3))
        self.assertFalse(G.has_edge(1, 3)) # Not direct
        
    def test_topology_assign_elevations(self):
        # Hierarchy: 1 -> 2 -> 3
        p1 = {"id": 1, "geometry": sg.box(0, 0, 10, 10)}
        p2 = {"id": 2, "geometry": sg.box(2, 2, 8, 8)}
        p3 = {"id": 3, "geometry": sg.box(4, 4, 6, 6)}
        
        G = build_topology_tree([p1, p2, p3])
        
        # Test standard elevations from base_id = 1
        G_res = assign_elevations(G, base_id=1, direction_changes=set(), interval=1.0)
        self.assertEqual(G_res.nodes[1]["data"]["elevation"], 0.0)
        self.assertEqual(G_res.nodes[2]["data"]["elevation"], 1.0)
        self.assertEqual(G_res.nodes[3]["data"]["elevation"], 2.0)
        
        # Test with direction change at node 2 (e.g. a peak at 2, so 3 goes down)
        G_res2 = assign_elevations(G, base_id=1, direction_changes={2}, interval=1.0)
        self.assertEqual(G_res2.nodes[1]["data"]["elevation"], 0.0)
        self.assertEqual(G_res2.nodes[2]["data"]["elevation"], 1.0)
        self.assertEqual(G_res2.nodes[3]["data"]["elevation"], 0.0) # Down from 2
        
        # Test starting elevation from intermediate base_id = 2
        G_res3 = assign_elevations(G, base_id=2, direction_changes=set(), interval=2.0)
        self.assertEqual(G_res3.nodes[2]["data"]["elevation"], 0.0)
        self.assertEqual(G_res3.nodes[3]["data"]["elevation"], 2.0)  # Child of base goes up
        self.assertEqual(G_res3.nodes[1]["data"]["elevation"], -2.0) # Parent of base goes down

    def test_slicer_generate_rings(self):
        # Create a simple nested tree
        # 1 contains 2 (which is a square inside 1)
        p1_geom = sg.box(0, 0, 10, 10)
        p2_geom = sg.box(2, 2, 8, 8)
        
        G = nx.DiGraph()
        G.add_node(1, data={"id": 1, "geometry": p1_geom, "elevation": 0.0})
        G.add_node(2, data={"id": 2, "geometry": p2_geom, "elevation": 1.0})
        G.add_edge(1, 2)
        
        # Generate rings
        slices = generate_rings(G, engrave_offset=0.5, scale_factor=2.0, direction_changes=set(), slicing_mode="hollow")
        
        self.assertEqual(len(slices), 2)
        
        # Slices are sorted bottom-up (lowest elevation first, so node 1 then node 2)
        slice1 = slices[0]
        self.assertEqual(slice1["id"], 1)
        self.assertEqual(slice1["elevation"], 0.0)
        
        # Geometry for slice 1 should be p1_geom - p2_geom, scaled by 2.0
        # p1_geom area = 100, p2_geom area = 36. Difference area = 64.
        # Scaled by 2.0 in x and y means area is multiplied by 4.0 = 256.0.
        self.assertAlmostEqual(slice1["slice_geom"].area, 256.0)
        
        # Engraving lines for slice 1: should be boundary of child (p2_geom) buffered by -0.5
        # p2_geom is (2,2) to (8,8) -> size 6. Buffered by -0.5 -> size 5 (coords 2.5 to 7.5).
        # Scaled by 2.0 -> coords 5.0 to 15.0 -> size 10. Perimeter = 40.
        self.assertEqual(len(slice1["engraving_lines"]), 1)
        self.assertAlmostEqual(slice1["engraving_lines"][0].length, 40.0)

    def test_hollow_slicer_can_add_glue_margin(self):
        boundary_geom = sg.box(0, 0, 10, 10)
        level_geom = sg.box(2, 2, 8, 8)

        G = nx.DiGraph()
        G.add_node(1, data={"id": 1, "geometry": boundary_geom, "elevation": 0.0})
        G.add_node(2, data={"id": 2, "geometry": level_geom, "elevation": 1.0})
        G.add_edge(1, 2)

        slices = generate_rings(
            G,
            engrave_offset=0.0,
            scale_factor=1.0,
            direction_changes=set(),
            slicing_mode="hollow",
            hollow_glue_margin=0.5,
        )

        self.assertEqual(len(slices), 2)
        self.assertAlmostEqual(slices[0]["visible_geom"].area, 64.0)
        self.assertAlmostEqual(slices[0]["support_geom"].area, 11.0)
        self.assertAlmostEqual(slices[0]["slice_geom"].area, 75.0)
        self.assertAlmostEqual(slices[1]["visible_geom"].area, 36.0)
        self.assertAlmostEqual(slices[1]["slice_geom"].area, 36.0)

    def test_hollow_slicer_glue_margin_uses_boundary_as_limit(self):
        boundary_geom = sg.box(0, 0, 10, 10)
        level_geom = sg.box(2, 2, 8, 8)
        top_geom = sg.box(4, 4, 6, 6)

        G = nx.DiGraph()
        G.add_node(1, data={"id": 1, "geometry": boundary_geom, "elevation": 0.0})
        G.add_node(2, data={"id": 2, "geometry": level_geom, "elevation": 1.0})
        G.add_node(3, data={"id": 3, "geometry": top_geom, "elevation": 2.0})
        G.add_edge(1, 2)
        G.add_edge(2, 3)

        slices = generate_rings(
            G,
            engrave_offset=0.0,
            scale_factor=1.0,
            direction_changes=set(),
            slicing_mode="hollow",
            hollow_glue_margin=3.0,
        )

        self.assertEqual(len(slices), 3)
        self.assertAlmostEqual(slices[0]["visible_geom"].area, 64.0)
        self.assertAlmostEqual(slices[0]["support_geom"].area, 36.0)
        self.assertAlmostEqual(slices[0]["slice_geom"].area, 100.0)
        self.assertAlmostEqual(slices[1]["visible_geom"].area, 32.0)
        self.assertAlmostEqual(slices[1]["support_geom"].area, 4.0)
        self.assertAlmostEqual(slices[1]["slice_geom"].area, 36.0)
        self.assertAlmostEqual(slices[2]["visible_geom"].area, 4.0)
        self.assertAlmostEqual(slices[2]["slice_geom"].area, 4.0)
        self.assertTrue(boundary_geom.covers(slices[0]["slice_geom"]))
        self.assertTrue(boundary_geom.covers(slices[1]["slice_geom"]))
        self.assertTrue(boundary_geom.covers(slices[2]["slice_geom"]))
        self.assertAlmostEqual(slices[0]["visible_geom"].intersection(level_geom).area, 0.0)
        self.assertAlmostEqual(slices[1]["visible_geom"].intersection(top_geom).area, 0.0)

    def test_hollow_slicer_glue_margin_is_contact_strip(self):
        boundary_geom = sg.box(0, 0, 20, 10)
        hidden_side_geom = sg.box(5, 0, 20, 10)

        G = nx.DiGraph()
        G.add_node(1, data={"id": 1, "geometry": boundary_geom, "elevation": 0.0})
        G.add_node(2, data={"id": 2, "geometry": hidden_side_geom, "elevation": 1.0})
        G.add_edge(1, 2)

        slices = generate_rings(
            G,
            engrave_offset=0.0,
            scale_factor=1.0,
            direction_changes=set(),
            slicing_mode="hollow",
            hollow_glue_margin=2.0,
        )

        self.assertAlmostEqual(slices[0]["visible_geom"].area, 50.0)
        self.assertAlmostEqual(slices[0]["hidden_base_geom"].area, 150.0)
        self.assertAlmostEqual(slices[0]["support_geom"].area, 20.0)
        self.assertAlmostEqual(slices[0]["slice_geom"].area, 70.0)

    def test_slicer_generate_solid_frames(self):
        # boundary (irregular base polygon)
        boundary_geom = sg.box(0, 0, 20, 20)
        p1_geom = sg.box(2, 2, 18, 18) # Level 1 contour (rising)
        p2_geom = sg.box(4, 4, 16, 16) # Level 2 contour (descending/direction change)
        
        G = nx.DiGraph()
        G.add_node(100, data={"id": 100, "geometry": boundary_geom, "layer": "boundary", "elevation": 0.0, "is_descending": False})
        G.add_node(101, data={"id": 101, "geometry": p1_geom, "layer": "topografia", "elevation": 1.0, "is_descending": False})
        G.add_node(102, data={"id": 102, "geometry": p2_geom, "layer": "topografia", "elevation": 2.0, "is_descending": True})
        
        G.add_edge(100, 101)
        G.add_edge(101, 102)
        
        slices = generate_rings(G, engrave_offset=1.0, scale_factor=1.0, direction_changes=set(), slicing_mode="solid")
        
        self.assertEqual(len(slices), 3)
        
        # Slices are sorted bottom-up (0.0, 1.0, 2.0)
        # 1. Base level (elevation 0.0): should be fully solid boundary
        self.assertAlmostEqual(slices[0]["slice_geom"].area, 400.0) # 20x20
        
        # 2. Level 1 (elevation 1.0, rising): should be boundary - p1_geom
        # boundary = 400. p1_geom = 16x16 = 256. Difference = 144.
        self.assertAlmostEqual(slices[1]["slice_geom"].area, 144.0)
        
        # 3. Level 2 (elevation 2.0, descending): should NOT subtract (remains fully solid boundary)
        self.assertAlmostEqual(slices[2]["slice_geom"].area, 400.0)

    def test_nesting_split_needed(self):
        # Slice geometry: 100 x 10, larger than bed (50 x 50)
        geom = sg.box(0, 0, 100, 10)
        slice_data = {
            'id': 1,
            'elevation': 0.0,
            'slice_geom': geom,
            'engraving_lines': [sg.LineString([(10, 5), (90, 5)])]
        }
        
        pieces = split_polygon_if_needed(slice_data, bed_w=50, bed_h=50)
        # Should be split at least once
        self.assertTrue(len(pieces) > 1)
        for piece in pieces:
            minx, miny, maxx, maxy = piece['slice_geom'].bounds
            self.assertTrue(maxx - minx <= 50)
            self.assertTrue(maxy - miny <= 50)
            
    def test_nesting_does_not_split_piece_that_fits_when_rotated(self):
        geom = sg.box(0, 0, 64, 20)
        slice_data = {
            "id": 1,
            "elevation": 0.0,
            "slice_geom": geom,
            "engraving_lines": [],
        }

        pieces = split_polygon_if_needed(slice_data, bed_w=60, bed_h=60)
        self.assertEqual(len(pieces), 1)
        self.assertEqual(pieces[0]["id"], 1)

    def test_preserve_whole_percent_100_keeps_fitting_piece_whole(self):
        slices = [
            {
                "id": "whole_piece",
                "elevation": 0.0,
                "slice_geom": sg.box(0, 0, 70, 55),
                "engraving_lines": [],
            }
        ]

        sheets = perform_nesting(
            slices,
            bed_w=100,
            bed_h=100,
            margin=5,
            priority="preserve_shapes",
            target_utilization=100,
            preserve_whole_percent=100,
        )

        pieces = [piece for sheet in sheets.values() for piece in sheet]
        self.assertEqual(len(pieces), 1)
        self.assertEqual(pieces[0]["id"], "whole_piece")

    def test_preserve_whole_uses_geometry_angle_before_splitting(self):
        geom = rotate_geometry(sg.box(0, 0, 100, 60), 17, origin="centroid")
        slices = [
            {
                "id": "angled_piece",
                "elevation": 0.0,
                "slice_geom": geom,
                "engraving_lines": [],
            }
        ]

        sheets = perform_nesting(
            slices,
            bed_w=101,
            bed_h=61,
            margin=0,
            priority="preserve_shapes",
            target_utilization=100,
            preserve_whole_percent=100,
        )

        pieces = [piece for sheet in sheets.values() for piece in sheet]
        self.assertEqual(len(pieces), 1)
        self.assertEqual(pieces[0]["id"], "angled_piece")
        self.assertFalse(pieces[0].get("was_split", False))

    def test_maximize_usage_does_not_optionally_split_fitting_pieces(self):
        whole_components = [
            {
                "id": "whole_piece",
                "elevation": 0.0,
                "slice_geom": sg.box(0, 0, 70, 55),
                "engraving_lines": [],
            }
        ]

        candidates = list(
            _candidate_piece_sets(
                required_pieces=[],
                whole_components=whole_components,
                usable_w=90,
                usable_h=90,
                priority="maximize_usage",
                target_utilization=100,
                preserve_whole_percent=0,
            )
        )

        self.assertEqual(len(candidates), 1)
        pieces, optional_split_count = candidates[0]
        self.assertEqual(optional_split_count, 0)
        self.assertEqual(len(pieces), 1)
        self.assertFalse(pieces[0].get("was_split", False))

    def test_optional_split_budget_respects_effort_in_maximize_mode(self):
        self.assertEqual(
            _optional_split_budget(
                whole_count=10,
                priority="maximize_usage",
                target_utilization=20,
                preserve_whole_percent=0,
            ),
            0,
        )
        self.assertEqual(
            _optional_split_budget(
                whole_count=10,
                priority="maximize_usage",
                target_utilization=50,
                preserve_whole_percent=0,
            ),
            1,
        )
        self.assertEqual(
            _optional_split_budget(
                whole_count=10,
                priority="maximize_usage",
                target_utilization=90,
                preserve_whole_percent=0,
            ),
            2,
        )

    def test_effort_search_limits_scale_in_maximize_mode(self):
        low_axis, low_positions = _effort_search_limits(10, priority="maximize_usage")
        mid_axis, mid_positions = _effort_search_limits(50, priority="maximize_usage")
        high_axis, high_positions = _effort_search_limits(100, priority="maximize_usage")

        self.assertLess(low_axis, mid_axis)
        self.assertLess(mid_axis, high_axis)
        self.assertLess(low_positions, mid_positions)
        self.assertLess(mid_positions, high_positions)

    def test_label_areas_prefer_support_geometry(self):
        piece = {
            "id": "support_label_piece",
            "slice_geom": sg.box(0, 0, 100, 60),
            "support_geom": sg.box(70, 10, 98, 26),
            "hidden_base_geom": sg.box(68, 8, 99, 28),
            "engraving_lines": [],
            "label_areas": [],
        }
        areas = _label_areas_for_piece(piece)
        self.assertTrue(areas)
        merged = unary_union(areas)
        self.assertGreater(merged.area, 0.0)
        self.assertTrue(piece["support_geom"].covers(merged))

    def test_candidate_piece_sets_can_emit_optional_split_candidates(self):
        whole_components = [
            {
                "id": "whole_piece",
                "elevation": 0.0,
                "slice_geom": sg.box(0, 0, 70, 55),
                "engraving_lines": [],
            }
        ]

        candidates = list(
            _candidate_piece_sets(
                required_pieces=[],
                whole_components=whole_components,
                usable_w=90,
                usable_h=90,
                priority="maximize_usage",
                target_utilization=90,
                preserve_whole_percent=0,
                include_optional_splits=True,
            )
        )

        optional_counts = [optional_split_count for _, optional_split_count in candidates]
        self.assertIn(0, optional_counts)
        self.assertIn(1, optional_counts)

    def test_nesting_perform(self):
        # 3 slices, each fits easily on a 100x100 bed
        slices = [
            {'id': 1, 'elevation': 0.0, 'slice_geom': sg.box(0, 0, 30, 30), 'engraving_lines': []},
            {'id': 2, 'elevation': 1.0, 'slice_geom': sg.box(0, 0, 40, 40), 'engraving_lines': []},
            {'id': 3, 'elevation': 2.0, 'slice_geom': sg.box(0, 0, 50, 50), 'engraving_lines': []},
        ]
        
        sheets = perform_nesting(slices, bed_w=100, bed_h=100, margin=5)
        self.assertTrue(len(sheets) > 0)
        
        # Check that all geometries lie inside the bed boundaries
        for sheet_id, pieces in sheets.items():
            for piece in pieces:
                minx, miny, maxx, maxy = piece['geom'].bounds
                self.assertTrue(minx >= 0)
                self.assertTrue(miny >= 0)
                self.assertTrue(maxx <= 100)
                self.assertTrue(maxy <= 100)

    def test_nesting_respects_effective_margin(self):
        slices = [
            {'id': 1, 'elevation': 0.0, 'slice_geom': sg.box(0, 0, 30, 30), 'engraving_lines': []},
        ]

        sheets = perform_nesting(slices, bed_w=100, bed_h=100, margin=6)
        piece = sheets[0][0]
        minx, miny, maxx, maxy = piece["geom"].bounds
        self.assertGreaterEqual(minx, 6 - 1e-6)
        self.assertGreaterEqual(miny, 6 - 1e-6)
        self.assertLessEqual(maxx, 94 + 1e-6)
        self.assertLessEqual(maxy, 94 + 1e-6)

    def test_nesting_uses_no_fit_candidates_for_internal_void(self):
        outer = [(0, 0), (70, 0), (70, 70), (0, 70), (0, 0)]
        hidden_void = [(19, 19), (51, 19), (51, 51), (19, 51), (19, 19)]
        large_piece = sg.Polygon(outer, [hidden_void])
        small_piece = sg.box(0, 0, 30, 30)
        slices = [
            {"id": "large", "elevation": 0.0, "slice_geom": large_piece, "engraving_lines": []},
            {"id": "small", "elevation": 0.0, "slice_geom": small_piece, "engraving_lines": []},
        ]

        sheets = perform_nesting(slices, bed_w=72, bed_h=72, margin=1)
        pieces = [piece for sheet in sheets.values() for piece in sheet]
        small = next(piece for piece in pieces if piece["id"] == "small")
        large = next(piece for piece in pieces if piece["id"] == "large")

        self.assertEqual(len(sheets), 1)
        self.assertGreaterEqual(small["geom"].distance(large["geom"]), 1 - 1e-6)
        minx, miny, maxx, maxy = small["geom"].bounds
        self.assertGreaterEqual(minx, 21 - 1e-6)
        self.assertGreaterEqual(miny, 21 - 1e-6)
        self.assertLessEqual(maxx, 51 + 1e-6)
        self.assertLessEqual(maxy, 51 + 1e-6)

    def test_nesting_preserves_model_geometry_for_split_preview(self):
        slices = [
            {
                "id": "level_1",
                "elevation": 1.0,
                "slice_geom": sg.box(0, 0, 100, 10),
                "engraving_lines": [],
            }
        ]

        sheets = perform_nesting(slices, bed_w=50, bed_h=50, margin=5)
        pieces = [piece for sheet in sheets.values() for piece in sheet]
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(piece.get("source_slice_id") == "level_1" for piece in pieces))
        self.assertTrue(all(piece.get("model_geom") is not None for piece in pieces))

    def test_nesting_expands_repeated_base_group_as_real_pieces(self):
        slices = [
            {
                "id": "base",
                "elevation": 0.0,
                "source_elevation": 0.0,
                "slice_geom": sg.box(0, 0, 20, 10),
                "engraving_lines": [],
                "is_zero_base": True,
                "repeat_count": 5,
                "repeat_start_elevation": 0.0,
                "repeat_interval": 0.5,
            }
        ]

        sheets = perform_nesting(slices, bed_w=60, bed_h=30, margin=2)
        pieces = [piece for sheet in sheets.values() for piece in sheet]

        self.assertEqual(len(pieces), 5)
        self.assertEqual(len(sheets), 2)
        self.assertEqual(
            [piece.get("source_elevation") for piece in pieces],
            [0.0, 0.5, 1.0, 1.5, 2.0],
        )

    def test_hollow_conversion_for_terrain_slices(self):
        solid_slices = [
            {"id": "base", "elevation": 0.0, "slice_geom": sg.box(0, 0, 10, 10), "engraving_lines": []},
            {"id": "level_1", "elevation": 1.0, "slice_geom": sg.box(2, 2, 8, 8), "engraving_lines": []},
            {"id": "level_2", "elevation": 2.0, "slice_geom": sg.box(4, 4, 6, 6), "engraving_lines": []},
        ]

        hollow = _convert_solid_slices_to_hollow(solid_slices, area_tolerance=0.1)
        self.assertEqual(len(hollow), 3)
        self.assertAlmostEqual(hollow[0]["slice_geom"].area, 64.0)  # 10x10 - 6x6
        self.assertAlmostEqual(hollow[1]["slice_geom"].area, 32.0)  # 6x6 - 2x2
        self.assertAlmostEqual(hollow[2]["slice_geom"].area, 4.0)   # top keeps core
        self.assertAlmostEqual(hollow[0]["visible_geom"].area, 64.0)
        self.assertTrue(hollow[0]["support_geom"].is_empty)

    def test_solid_visibility_metadata_marks_hidden_base(self):
        solid_slices = [
            {"id": "base", "elevation": 0.0, "slice_geom": sg.box(0, 0, 10, 10), "engraving_lines": []},
            {"id": "level_1", "elevation": 1.0, "slice_geom": sg.box(2, 2, 8, 8), "engraving_lines": []},
            {"id": "level_2", "elevation": 2.0, "slice_geom": sg.box(4, 4, 6, 6), "engraving_lines": []},
        ]

        solid = _annotate_solid_slice_visibility(solid_slices, area_tolerance=0.1)

        self.assertEqual(len(solid), 3)
        self.assertAlmostEqual(solid[0]["slice_geom"].area, 100.0)
        self.assertAlmostEqual(solid[0]["visible_geom"].area, 64.0)
        self.assertAlmostEqual(solid[0]["hidden_base_geom"].area, 36.0)
        self.assertAlmostEqual(solid[0]["support_geom"].area, 36.0)
        self.assertAlmostEqual(solid[1]["slice_geom"].area, 36.0)
        self.assertAlmostEqual(solid[1]["visible_geom"].area, 32.0)
        self.assertAlmostEqual(solid[1]["support_geom"].area, 4.0)
        self.assertAlmostEqual(solid[2]["visible_geom"].area, 4.0)
        self.assertTrue(solid[2]["support_geom"].is_empty)

    def test_solid_engraving_lines_follow_physical_upper_cut(self):
        slices = [
            {"id": "base", "elevation": 0.0, "slice_geom": sg.box(0, 0, 10, 10), "engraving_lines": []},
            {"id": "level_1", "elevation": 1.0, "slice_geom": sg.box(2, 2, 8, 8), "engraving_lines": []},
            {"id": "level_2", "elevation": 2.0, "slice_geom": sg.box(4, 4, 6, 6), "engraving_lines": []},
        ]
        solid = _annotate_solid_slice_visibility(slices, area_tolerance=0.1)
        _add_assembly_engraving_lines(solid, offset=0.5)

        base_lines = solid[0]["engraving_lines"]
        level_1_lines = solid[1]["engraving_lines"]
        self.assertEqual(len(base_lines), 1)
        self.assertEqual(len(level_1_lines), 1)
        self.assertAlmostEqual(base_lines[0].length, 20.0)
        self.assertAlmostEqual(level_1_lines[0].length, 4.0)

    def test_hollow_engraving_rule_is_independent_from_solid_metadata(self):
        support = sg.box(0, 0, 10, 2)
        visible = sg.box(0, 2, 10, 5)
        upper_visible = sg.box(0, 1, 10, 4)
        rows = [
            {
                "id": "lower",
                "elevation": 0.0,
                "slice_geom": unary_union([support, visible]),
                "visible_geom": visible,
                "support_geom": support,
                "support_rule_version": "solid_visibility_v1",
                "engraving_lines": [],
            },
            {
                "id": "upper",
                "elevation": 1.0,
                "slice_geom": upper_visible,
                "visible_geom": upper_visible,
                "support_geom": sg.Polygon(),
                "support_rule_version": "solid_visibility_v1",
                "engraving_lines": [],
            },
        ]

        _add_assembly_engraving_lines(rows, offset=0.0, slicing_mode="hollow")

        lines = rows[0]["engraving_lines"]
        self.assertEqual(len(lines), 1)
        self.assertAlmostEqual(lines[0].length, 10.0)
        self.assertTrue(support.buffer(1e-6).covers(lines[0]))
        self.assertTrue(visible.buffer(1e-6).covers(lines[0]))

    def test_solid_visibility_can_rebuild_full_shape_from_hollow_metadata(self):
        solid_slices = [
            {"id": "base", "elevation": 0.0, "slice_geom": sg.box(0, 0, 10, 10), "engraving_lines": []},
            {"id": "level_1", "elevation": 1.0, "slice_geom": sg.box(2, 2, 8, 8), "engraving_lines": []},
        ]
        hollow = _convert_solid_slices_to_hollow(
            solid_slices,
            area_tolerance=0.1,
            glue_margin=0.5,
        )

        solid = _annotate_solid_slice_visibility(hollow, area_tolerance=0.1)

        self.assertAlmostEqual(hollow[0]["slice_geom"].area, 75.0)
        self.assertAlmostEqual(solid[0]["slice_geom"].area, 100.0)
        self.assertAlmostEqual(solid[0]["visible_geom"].area, 64.0)
        self.assertAlmostEqual(solid[0]["support_geom"].area, 36.0)

    def test_hollow_conversion_can_add_glue_margin(self):
        solid_slices = [
            {"id": "base", "elevation": 0.0, "slice_geom": sg.box(0, 0, 10, 10), "engraving_lines": []},
            {"id": "level_1", "elevation": 1.0, "slice_geom": sg.box(2, 2, 8, 8), "engraving_lines": []},
            {"id": "level_2", "elevation": 2.0, "slice_geom": sg.box(4, 4, 6, 6), "engraving_lines": []},
        ]

        hollow = _convert_solid_slices_to_hollow(
            solid_slices,
            area_tolerance=0.1,
            glue_margin=0.5,
        )

        self.assertEqual(len(hollow), 3)
        self.assertAlmostEqual(hollow[0]["visible_geom"].area, 64.0)
        self.assertAlmostEqual(hollow[0]["support_geom"].area, 11.0)
        self.assertAlmostEqual(hollow[0]["slice_geom"].area, 75.0)
        self.assertAlmostEqual(hollow[1]["visible_geom"].area, 32.0)
        self.assertAlmostEqual(hollow[1]["support_geom"].area, 3.0)
        self.assertAlmostEqual(hollow[1]["slice_geom"].area, 35.0)
        self.assertAlmostEqual(hollow[2]["visible_geom"].area, 4.0)
        self.assertAlmostEqual(hollow[2]["slice_geom"].area, 4.0)

    def test_hollow_glue_margin_uses_boundary_as_limit(self):
        solid_slices = [
            {"id": "base", "elevation": 0.0, "slice_geom": sg.box(0, 0, 10, 10), "engraving_lines": []},
            {"id": "level_1", "elevation": 1.0, "slice_geom": sg.box(2, 2, 8, 8), "engraving_lines": []},
            {"id": "level_2", "elevation": 2.0, "slice_geom": sg.box(4, 4, 6, 6), "engraving_lines": []},
        ]

        hollow = _convert_solid_slices_to_hollow(
            solid_slices,
            area_tolerance=0.1,
            glue_margin=3.0,
            glue_limit_geom=sg.box(0, 0, 10, 10),
        )

        self.assertEqual(len(hollow), 3)
        self.assertAlmostEqual(hollow[0]["visible_geom"].area, 64.0)
        self.assertAlmostEqual(hollow[0]["support_geom"].area, 36.0)
        self.assertAlmostEqual(hollow[0]["slice_geom"].area, 100.0)
        self.assertAlmostEqual(hollow[1]["visible_geom"].area, 32.0)
        self.assertAlmostEqual(hollow[1]["support_geom"].area, 4.0)
        self.assertAlmostEqual(hollow[1]["slice_geom"].area, 36.0)
        self.assertAlmostEqual(hollow[2]["visible_geom"].area, 4.0)
        self.assertAlmostEqual(hollow[2]["slice_geom"].area, 4.0)
        boundary = sg.box(0, 0, 10, 10)
        self.assertTrue(boundary.covers(hollow[0]["slice_geom"]))
        self.assertTrue(boundary.covers(hollow[1]["slice_geom"]))
        self.assertTrue(boundary.covers(hollow[2]["slice_geom"]))
        level_geom = solid_slices[1]["slice_geom"]
        top_geom = solid_slices[2]["slice_geom"]
        self.assertAlmostEqual(hollow[0]["visible_geom"].intersection(level_geom).area, 0.0)
        self.assertAlmostEqual(hollow[1]["visible_geom"].intersection(top_geom).area, 0.0)

    def test_hollow_glue_margin_is_contact_strip(self):
        solid_slices = [
            {"id": "base", "elevation": 0.0, "slice_geom": sg.box(0, 0, 20, 10), "engraving_lines": []},
            {"id": "level_1", "elevation": 1.0, "slice_geom": sg.box(5, 0, 20, 10), "engraving_lines": []},
        ]

        hollow = _convert_solid_slices_to_hollow(
            solid_slices,
            area_tolerance=0.1,
            glue_margin=2.0,
            glue_limit_geom=sg.box(0, 0, 20, 10),
        )

        self.assertAlmostEqual(hollow[0]["visible_geom"].area, 50.0)
        self.assertAlmostEqual(hollow[0]["hidden_base_geom"].area, 150.0)
        self.assertAlmostEqual(hollow[0]["support_geom"].area, 20.0)
        self.assertAlmostEqual(hollow[0]["slice_geom"].area, 70.0)

    def test_hollow_glue_margin_ignores_detached_islands(self):
        visible_geom = sg.box(0, 0, 10, 2)
        attached_base = sg.box(0, 2, 4, 6)
        detached_island = sg.box(6, 2.2, 7, 3.2)
        hidden_geom = unary_union([attached_base, detached_island])

        support = _hidden_support_geometry(
            visible_geom,
            hidden_geom,
            glue_margin=1.0,
            area_tolerance=0.01,
        )

        self.assertAlmostEqual(support.area, 4.0)
        self.assertTrue(support.intersection(detached_island).is_empty)

    def test_hollow_glue_margin_drops_very_thin_regions(self):
        visible_geom = sg.box(0, 0, 10, 2)
        thin_base = sg.box(0, 2, 10, 2.05)

        support = _hidden_support_geometry(
            visible_geom,
            thin_base,
            glue_margin=1.0,
            area_tolerance=0.01,
        )

        self.assertTrue(support.is_empty)

    def test_hollow_conversion_removes_support_orphans_after_visible_cleanup(self):
        solid_slices = [
            {
                "id": "base",
                "elevation": 0.0,
                "slice_geom": unary_union([
                    sg.box(0, 0, 20, 20),
                    sg.box(100.0, 100.0, 100.55, 100.55),
                ]),
                "engraving_lines": [],
            },
            {
                "id": "level_1",
                "elevation": 1.0,
                "slice_geom": unary_union([
                    sg.box(2, 2, 18, 18),
                    sg.box(100.02, 100.02, 100.52, 100.52),
                ]),
                "engraving_lines": [],
            },
        ]

        hollow = _convert_solid_slices_to_hollow(
            solid_slices,
            area_tolerance=0.1,
            glue_margin=2.0,
        )
        support = hollow[0]["support_geom"]
        visible = hollow[0]["visible_geom"]

        self.assertFalse(support.is_empty)
        self.assertAlmostEqual(support.distance(visible), 0.0, places=6)
        self.assertTrue(visible.buffer(2.0 + 1e-6).covers(support))

    def test_hollow_prune_removes_components_not_connected_to_lower_layer(self):
        lower = sg.box(0, 0, 10, 10)
        upper_connected = sg.box(1, 1, 4, 4)
        upper_floating = sg.box(20, 20, 24, 24)
        upper = unary_union([upper_connected, upper_floating])

        slices = [
            {
                "id": "base",
                "elevation": 0.0,
                "slice_geom": lower,
                "visible_geom": lower,
                "support_geom": sg.Polygon(),
                "hidden_base_geom": sg.Polygon(),
            },
            {
                "id": "level_1",
                "elevation": 1.0,
                "slice_geom": upper,
                "visible_geom": upper,
                "support_geom": sg.Polygon(),
                "hidden_base_geom": sg.Polygon(),
            },
        ]

        _prune_floating_hollow_components(
            slices,
            area_tolerance=0.01,
            glue_margin=1.0,
        )

        cleaned = slices[1]["slice_geom"]
        self.assertTrue(lower.buffer(1e-6).intersects(cleaned))
        self.assertTrue(cleaned.intersection(upper_floating).is_empty)
        self.assertAlmostEqual(cleaned.area, upper_connected.area)

    def test_build_contour_correction_table(self):
        contours = [
            {"id": 1, "layer": "topo", "geometry": sg.box(10, 10, 40, 40)},
            {"id": 2, "layer": "topo", "geometry": sg.box(60, 60, 90, 90)},
        ]
        points = [
            {"x": 10, "y": 25, "z": 100, "geometry": sg.Point(10, 25)},
            {"x": 40, "y": 25, "z": 100, "geometry": sg.Point(40, 25)},
            {"x": 25, "y": 10, "z": 100, "geometry": sg.Point(25, 10)},
            {"x": 60, "y": 75, "z": 120, "geometry": sg.Point(60, 75)},
            {"x": 90, "y": 75, "z": 120, "geometry": sg.Point(90, 75)},
            {"x": 75, "y": 60, "z": 120, "geometry": sg.Point(75, 60)},
        ]
        table = build_contour_correction_table(contours, points, near_distance=0.1)
        self.assertEqual(len(table), 2)
        self.assertEqual(table[0]["keep"], 1)
        self.assertIn("forced_elevation", table[0])
        self.assertIn("forced_source_elevation", table[0])
        self.assertIn("forced_relative_elevation", table[0])
        self.assertIn("forced_layer_index", table[0])

    def test_manual_override_keep_contour(self):
        boundary = sg.box(0, 0, 100, 100)
        contours = [
            {"id": 1, "layer": "topo", "geometry": sg.box(10, 10, 40, 40)},
            {"id": 2, "layer": "topo", "geometry": sg.box(60, 60, 90, 90)},
        ]
        points = [
            {"x": 10, "y": 25, "z": 100, "geometry": sg.Point(10, 25)},
            {"x": 40, "y": 25, "z": 100, "geometry": sg.Point(40, 25)},
            {"x": 25, "y": 10, "z": 100, "geometry": sg.Point(25, 10)},
            {"x": 60, "y": 75, "z": 120, "geometry": sg.Point(60, 75)},
            {"x": 90, "y": 75, "z": 120, "geometry": sg.Point(90, 75)},
            {"x": 75, "y": 60, "z": 120, "geometry": sg.Point(75, 60)},
        ]

        base = generate_terrain_layers_from_contours(
            contours,
            points,
            boundary,
            interval=1.0,
            near_distance=0.1,
            scale_factor=1.0,
            assembly_offset_mm=0.0,
        )
        without_first = generate_terrain_layers_from_contours(
            contours,
            points,
            boundary,
            interval=1.0,
            near_distance=0.1,
            scale_factor=1.0,
            manual_overrides={"1": {"keep": "0"}},
            assembly_offset_mm=0.0,
        )

        base_ids = {
            source_id
            for slice_data in base
            for source_id in (slice_data.get("source_ids") or [])
        }
        without_first_ids = {
            source_id
            for slice_data in without_first
            for source_id in (slice_data.get("source_ids") or [])
        }
        self.assertIn(1, base_ids)
        self.assertNotIn(1, without_first_ids)

    def test_contours_on_same_topographic_level_share_one_layer(self):
        boundary = sg.box(0, 0, 100, 100)
        contours = [
            {"id": 1, "layer": "topo", "geometry": sg.box(10, 10, 20, 20)},
            {"id": 2, "layer": "topo", "geometry": sg.box(60, 60, 70, 70)},
        ]
        points = [
            {"x": 0, "y": 0, "z": 80.0, "geometry": sg.Point(0, 0)},
            {"x": 10, "y": 15, "z": 81.0, "geometry": sg.Point(10, 15)},
            {"x": 20, "y": 15, "z": 81.0, "geometry": sg.Point(20, 15)},
            {"x": 15, "y": 10, "z": 81.0, "geometry": sg.Point(15, 10)},
            {"x": 60, "y": 65, "z": 81.0, "geometry": sg.Point(60, 65)},
            {"x": 70, "y": 65, "z": 81.0, "geometry": sg.Point(70, 65)},
            {"x": 65, "y": 60, "z": 81.0, "geometry": sg.Point(65, 60)},
        ]

        slices = generate_terrain_layers_from_contours(
            contours,
            points,
            boundary,
            interval=0.5,
            target_layer_count=10,
            scale_factor=1.0,
            assembly_offset_mm=0.0,
        )

        self.assertEqual(len(slices), 2)
        self.assertEqual(sorted(slices[1].get("source_ids", [])), [1, 2])
        self.assertAlmostEqual(slices[1]["source_elevation"], 81.0, places=3)
        self.assertAlmostEqual(slices[1]["slice_geom"].area, 200.0, places=3)

    def test_nested_ascending_contours_create_each_level(self):
        boundary = sg.box(0, 0, 100, 100)
        contours = [
            {"id": "outer_81", "layer": "topo", "geometry": sg.box(10, 10, 90, 90)},
            {"id": "middle_82", "layer": "topo", "geometry": sg.box(30, 30, 70, 70)},
            {"id": "inner_83", "layer": "topo", "geometry": sg.box(45, 45, 55, 55)},
        ]
        points = [
            {"x": 0, "y": 0, "z": 80.0, "geometry": sg.Point(0, 0)},
            {"x": 10, "y": 50, "z": 81.0, "geometry": sg.Point(10, 50)},
            {"x": 90, "y": 50, "z": 81.0, "geometry": sg.Point(90, 50)},
            {"x": 50, "y": 10, "z": 81.0, "geometry": sg.Point(50, 10)},
            {"x": 30, "y": 50, "z": 82.0, "geometry": sg.Point(30, 50)},
            {"x": 70, "y": 50, "z": 82.0, "geometry": sg.Point(70, 50)},
            {"x": 50, "y": 30, "z": 82.0, "geometry": sg.Point(50, 30)},
            {"x": 45, "y": 50, "z": 83.0, "geometry": sg.Point(45, 50)},
            {"x": 55, "y": 50, "z": 83.0, "geometry": sg.Point(55, 50)},
            {"x": 50, "y": 45, "z": 83.0, "geometry": sg.Point(50, 45)},
        ]

        slices = generate_terrain_layers_from_contours(
            contours,
            points,
            boundary,
            interval=0.0,
            scale_factor=1.0,
            assembly_offset_mm=0.0,
        )

        self.assertEqual([item.get("source_elevation") for item in slices], [80.0, 81.0, 82.0, 83.0])
        self.assertEqual([round(item["slice_geom"].area, 3) for item in slices], [10000.0, 6400.0, 1600.0, 100.0])
        self.assertEqual([item.get("source_ids") for item in slices[1:]], [["outer_81"], ["middle_82"], ["inner_83"]])

    def test_point_elevations_snap_to_regular_interval(self):
        boundary = sg.box(0, 0, 100, 100)
        contours = [
            {"id": "level_80", "layer": "topo", "geometry": sg.box(20, 20, 80, 80)},
        ]
        points = [
            {"x": 0, "y": 0, "z": 66.942, "geometry": sg.Point(0, 0)},
            {"x": 20, "y": 50, "z": 79.991, "geometry": sg.Point(20, 50)},
            {"x": 80, "y": 50, "z": 79.991, "geometry": sg.Point(80, 50)},
            {"x": 50, "y": 20, "z": 79.991, "geometry": sg.Point(50, 20)},
        ]

        slices = generate_terrain_layers_from_contours(
            contours,
            points,
            boundary,
            interval=0.5,
            scale_factor=1.0,
            assembly_offset_mm=0.0,
        )

        self.assertEqual(len(slices), 2)
        self.assertAlmostEqual(slices[0]["source_elevation"], 66.5, places=3)
        self.assertAlmostEqual(slices[1]["source_elevation"], 79.5, places=3)
        self.assertAlmostEqual(slices[1]["elevation"], 13.0, places=3)

    def test_extend_base_to_real_zero_adds_regular_base_layers(self):
        boundary = sg.box(0, 0, 100, 100)
        contours = [
            {"id": "level_2_5", "layer": "topo", "geometry": sg.box(20, 20, 80, 80)},
        ]
        points = [
            {"x": 0, "y": 0, "z": 2.0, "geometry": sg.Point(0, 0)},
            {"x": 20, "y": 50, "z": 2.5, "geometry": sg.Point(20, 50)},
            {"x": 80, "y": 50, "z": 2.5, "geometry": sg.Point(80, 50)},
            {"x": 50, "y": 20, "z": 2.5, "geometry": sg.Point(50, 20)},
        ]

        slices = generate_terrain_layers_from_contours(
            contours,
            points,
            boundary,
            interval=0.5,
            scale_factor=1.0,
            assembly_offset_mm=0.0,
            extend_base_to_zero=True,
        )

        self.assertEqual(
            [item.get("source_elevation") for item in slices],
            [0.0, 2.0, 2.5],
        )
        self.assertEqual(sum(1 for item in slices if item.get("is_zero_base")), 1)
        self.assertEqual(slices[0].get("repeat_count"), 4)
        self.assertEqual(slices[0].get("repeat_start_elevation"), 0.0)
        self.assertEqual(slices[0].get("repeat_interval"), 0.5)
        self.assertEqual(slices[0].get("repeat_end_elevation"), 1.5)
        self.assertEqual([item.get("elevation") for item in slices], [0.0, 2.0, 2.5])
        self.assertAlmostEqual(slices[0]["slice_geom"].area, boundary.area)

    def test_tiny_boolean_holes_are_removed_from_layer_geometry(self):
        geom = sg.Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
            [[(1, 1), (1.01, 1), (1.01, 1.01), (1, 1.01), (1, 1)]],
        )

        cleaned = _clean_layer_geometry(geom, area_tolerance=0.1)

        self.assertEqual(len(cleaned.interiors), 0)
        self.assertAlmostEqual(cleaned.area, 100.0, places=3)

    def test_zero_area_exterior_spikes_are_removed(self):
        geom = sg.Polygon([
            (0, 0), (4, 0), (4, 4), (2, 4), (2, 5), (2, 4), (0, 4), (0, 0)
        ])

        cleaned = _clean_layer_geometry(geom, area_tolerance=0.1)
        coords = list(cleaned.exterior.coords)
        has_spike = any(
            coords[index - 1] == coords[index + 1]
            for index in range(1, len(coords) - 1)
        )

        self.assertFalse(has_spike)
        self.assertAlmostEqual(cleaned.area, 16.0, places=3)

    def test_contour_generation_keeps_dxf_units_before_output_scaling(self):
        boundary = sg.box(0, 0, 100, 100)
        contours = [
            {"id": "level_80", "layer": "topo", "geometry": sg.box(20, 20, 80, 80)},
        ]
        points = [
            {"x": 0, "y": 0, "z": 66.942, "geometry": sg.Point(0, 0)},
            {"x": 20, "y": 50, "z": 79.991, "geometry": sg.Point(20, 50)},
            {"x": 80, "y": 50, "z": 79.991, "geometry": sg.Point(80, 50)},
            {"x": 50, "y": 20, "z": 79.991, "geometry": sg.Point(50, 20)},
        ]

        slices = generate_terrain_layers_from_contours(
            contours,
            points,
            boundary,
            interval=0.5,
            scale_factor=2.0,
            assembly_offset_mm=0.0,
        )

        self.assertAlmostEqual(slices[0]["slice_geom"].area, 10000.0, places=3)
        self.assertAlmostEqual(slices[1]["slice_geom"].area, 3600.0, places=3)

    def test_target_layer_count_does_not_move_direct_point_levels(self):
        boundary = sg.box(0, 0, 100, 100)
        contours = [
            {"id": "a", "layer": "topo", "geometry": sg.box(10, 10, 30, 30)},
            {"id": "b", "layer": "topo", "geometry": sg.box(60, 60, 80, 80)},
        ]
        points = [
            {"x": 0, "y": 0, "z": 80.0, "geometry": sg.Point(0, 0)},
            {"x": 10, "y": 20, "z": 83.0, "geometry": sg.Point(10, 20)},
            {"x": 30, "y": 20, "z": 83.0, "geometry": sg.Point(30, 20)},
            {"x": 60, "y": 70, "z": 83.0, "geometry": sg.Point(60, 70)},
            {"x": 80, "y": 70, "z": 83.0, "geometry": sg.Point(80, 70)},
        ]

        slices = generate_terrain_layers_from_contours(
            contours,
            points,
            boundary,
            interval=0.5,
            target_layer_count=12,
            scale_factor=1.0,
            assembly_offset_mm=0.0,
        )

        self.assertEqual(len(slices), 2)
        self.assertAlmostEqual(slices[1]["source_elevation"], 83.0, places=3)
        self.assertEqual(sorted(slices[1].get("source_ids", [])), ["a", "b"])

    def test_target_layer_count_does_not_move_nearest_point_levels(self):
        boundary = sg.box(0, 0, 100, 100)
        contours = [
            {"id": "far", "layer": "topo", "geometry": sg.box(10, 10, 30, 30)},
        ]
        points = [
            {"x": 0, "y": 0, "z": 79.0, "geometry": sg.Point(0, 0)},
            {"x": 15, "y": 40, "z": 83.0, "geometry": sg.Point(15, 40)},
        ]

        slices = generate_terrain_layers_from_contours(
            contours,
            points,
            boundary,
            interval=0.5,
            near_distance=0.01,
            target_layer_count=12,
            scale_factor=1.0,
            assembly_offset_mm=0.0,
        )

        self.assertEqual(len(slices), 2)
        self.assertAlmostEqual(slices[1]["source_elevation"], 83.0, places=3)

    def test_adjacent_unlabelled_contours_fill_regular_levels(self):
        boundary = sg.box(0, 0, 40, 10)
        contours = [
            {"id": "a", "layer": "topo", "geometry": sg.box(0, 0, 10, 10)},
            {"id": "b", "layer": "topo", "geometry": sg.box(10, 0, 20, 10)},
            {"id": "c", "layer": "topo", "geometry": sg.box(20, 0, 30, 10)},
            {"id": "d", "layer": "topo", "geometry": sg.box(30, 0, 40, 10)},
        ]
        points = [
            {"x": 0, "y": 5, "z": 0.0, "geometry": sg.Point(0, 5)},
            {"x": 40, "y": 5, "z": 1.5, "geometry": sg.Point(40, 5)},
        ]

        slices = generate_terrain_layers_from_contours(
            contours,
            points,
            boundary,
            interval=0.5,
            near_distance=0.01,
            scale_factor=1.0,
            assembly_offset_mm=0.0,
        )

        self.assertEqual([item.get("source_elevation") for item in slices], [0.0, 0.5, 1.0, 1.5])
        self.assertEqual([item.get("source_ids") for item in slices[1:]], [["b"], ["c"], ["d"]])

    def test_point_generation_keeps_dxf_units_before_output_scaling(self):
        boundary = sg.box(0, 0, 10, 10)
        points = [
            {"x": 0, "y": 0, "z": 0.0, "geometry": sg.Point(0, 0)},
            {"x": 10, "y": 0, "z": 1.0, "geometry": sg.Point(10, 0)},
            {"x": 0, "y": 10, "z": 1.0, "geometry": sg.Point(0, 10)},
        ]

        slices = generate_terrain_layers_from_points(
            points,
            boundary,
            interval=0.5,
            scale_factor=2.0,
        )

        self.assertAlmostEqual(slices[0]["slice_geom"].area, 100.0, places=3)

    def test_nested_descending_contours_cut_downward(self):
        boundary = sg.box(0, 0, 100, 100)
        contours = [
            {"id": "outer_83", "layer": "topo", "geometry": sg.box(10, 10, 90, 90)},
            {"id": "middle_82", "layer": "topo", "geometry": sg.box(30, 30, 70, 70)},
            {"id": "inner_81", "layer": "topo", "geometry": sg.box(45, 45, 55, 55)},
        ]
        points = [
            {"x": 10, "y": 50, "z": 83.0, "geometry": sg.Point(10, 50)},
            {"x": 90, "y": 50, "z": 83.0, "geometry": sg.Point(90, 50)},
            {"x": 50, "y": 10, "z": 83.0, "geometry": sg.Point(50, 10)},
            {"x": 30, "y": 50, "z": 82.0, "geometry": sg.Point(30, 50)},
            {"x": 70, "y": 50, "z": 82.0, "geometry": sg.Point(70, 50)},
            {"x": 50, "y": 30, "z": 82.0, "geometry": sg.Point(50, 30)},
            {"x": 45, "y": 50, "z": 81.0, "geometry": sg.Point(45, 50)},
            {"x": 55, "y": 50, "z": 81.0, "geometry": sg.Point(55, 50)},
            {"x": 50, "y": 45, "z": 81.0, "geometry": sg.Point(50, 45)},
        ]

        slices = generate_terrain_layers_from_contours(
            contours,
            points,
            boundary,
            interval=0.0,
            scale_factor=1.0,
            assembly_offset_mm=0.0,
        )

        self.assertEqual([item.get("source_elevation") for item in slices], [81.0, 82.0, 83.0])
        self.assertEqual([round(item["slice_geom"].area, 3) for item in slices], [10000.0, 8400.0, 3600.0])
        self.assertEqual(slices[1].get("source_ids"), ["middle_82"])
        self.assertEqual(slices[2].get("source_ids"), ["outer_83"])

    def test_manual_override_force_layer_index(self):
        boundary = sg.box(0, 0, 100, 100)
        contours = [
            {"id": 1, "layer": "topo", "geometry": sg.box(10, 10, 40, 40)},
            {"id": 2, "layer": "topo", "geometry": sg.box(60, 60, 90, 90)},
        ]
        points = [
            {"x": 10, "y": 25, "z": 100, "geometry": sg.Point(10, 25)},
            {"x": 40, "y": 25, "z": 100, "geometry": sg.Point(40, 25)},
            {"x": 25, "y": 10, "z": 100, "geometry": sg.Point(25, 10)},
            {"x": 60, "y": 75, "z": 120, "geometry": sg.Point(60, 75)},
            {"x": 90, "y": 75, "z": 120, "geometry": sg.Point(90, 75)},
            {"x": 75, "y": 60, "z": 120, "geometry": sg.Point(75, 60)},
        ]

        slices = generate_terrain_layers_from_contours(
            contours,
            points,
            boundary,
            interval=1.0,
            near_distance=0.1,
            scale_factor=1.0,
            manual_overrides={"2": {"forced_layer_index": 3}},
            assembly_offset_mm=0.0,
        )

        source_by_id = {}
        for slice_data in slices:
            for source_id in slice_data.get("source_ids", []):
                source_by_id[source_id] = slice_data.get("source_elevation")

        # Contour 2 should move near base + 3*interval (base=100, interval=1.0).
        self.assertIn(2, source_by_id)
        self.assertAlmostEqual(source_by_id[2], 103.0, places=3)

    def test_manual_override_force_relative_elevation(self):
        boundary = sg.box(0, 0, 100, 100)
        contours = [
            {"id": 1, "layer": "topo", "geometry": sg.box(10, 10, 40, 40)},
            {"id": 2, "layer": "topo", "geometry": sg.box(60, 60, 90, 90)},
        ]
        points = [
            {"x": 10, "y": 25, "z": 100, "geometry": sg.Point(10, 25)},
            {"x": 40, "y": 25, "z": 100, "geometry": sg.Point(40, 25)},
            {"x": 25, "y": 10, "z": 100, "geometry": sg.Point(25, 10)},
            {"x": 60, "y": 75, "z": 120, "geometry": sg.Point(60, 75)},
            {"x": 90, "y": 75, "z": 120, "geometry": sg.Point(90, 75)},
            {"x": 75, "y": 60, "z": 120, "geometry": sg.Point(75, 60)},
        ]

        slices = generate_terrain_layers_from_contours(
            contours,
            points,
            boundary,
            interval=1.0,
            near_distance=0.1,
            scale_factor=1.0,
            manual_overrides={"1": {"forced_relative_elevation": 2.5}},
            assembly_offset_mm=0.0,
        )

        source_by_id = {}
        for slice_data in slices:
            for source_id in slice_data.get("source_ids", []):
                source_by_id[source_id] = slice_data.get("source_elevation")

        # Point-based cotas use the lower bound of the active topographic interval.
        self.assertIn(1, source_by_id)
        self.assertAlmostEqual(source_by_id[1], 102.0, places=3)

    def test_source_ids_are_unique_and_complete(self):
        slices = [
            {"id": "base", "source_elevation": 100.0, "slice_geom": sg.box(0, 0, 10, 10), "engraving_lines": []},
            {
                "id": "level_1",
                "source_elevation": 101.0,
                "source_ids": [1, 2],
                "source_layers": ["L1", "L2"],
                "slice_geom": sg.box(1, 1, 9, 9),
                "engraving_lines": [],
            },
            {
                "id": "level_2",
                "source_elevation": 102.0,
                "source_ids": [2],
                "source_layers": ["L2"],
                "slice_geom": sg.box(2, 2, 8, 8),
                "engraving_lines": [],
            },
        ]
        contour_sources_by_id = {
            1: {"source_elevation": 101.0, "layer": "L1"},
            2: {"source_elevation": 102.0, "layer": "L2"},
            3: {"source_elevation": 103.0, "layer": "L3"},
        }

        _ensure_unique_and_complete_ids(slices, contour_sources_by_id, min_z=100.0)

        all_ids = []
        for slice_data in slices[1:]:
            all_ids.extend(slice_data.get("source_ids", []))

        self.assertEqual(sorted(all_ids), [1, 2])
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertFalse(any(3 in slice_data.get("source_ids", []) for slice_data in slices[1:]))

    def test_exporter_to_svg(self):
        # Create a sheets dict
        sheets = {
            0: [
                {
                    'id': '1_0',
                    'elevation': 1.5,
                    'geom': sg.box(10, 10, 40, 40),
                    'engravings': [sg.LineString([(15, 15), (35, 35)])]
                }
            ]
        }
        
        svgs = export_sheets_to_svg(sheets, bed_w=100, bed_h=100, margin=5)
        self.assertIn(0, svgs)
        svg_content = svgs[0]
        
        # Verify SVG contains correct tags and attributes
        self.assertIn('<svg width="100mm" height="100mm"', svg_content)
        self.assertIn('stroke="#ff0000"', svg_content) # for cut
        self.assertIn('stroke="#0057ff"', svg_content) # for engraving
        self.assertIn("Placa", svg_content)
        self.assertIn("MargemSeguranca", svg_content)
        self.assertIn('1.5', svg_content) # elevation label metadata
        self.assertNotIn('Elev:', svg_content)
        self.assertNotIn('<text', svg_content) # labels are exported as line outlines
        self.assertNotIn('<polyline', svg_content)
        self.assertIn('<line ', svg_content)
        ET.fromstring(svg_content)

    def test_exporter_label_uses_source_elevation_number_only(self):
        sheets = {
            0: [
                {
                    "id": "1_0",
                    "elevation": 15.5,
                    "source_elevation": 82.0,
                    "geom": sg.box(10, 10, 40, 40),
                    "engravings": [],
                }
            ]
        }

        svgs = export_sheets_to_svg(sheets, bed_w=100, bed_h=100, margin=5)
        self.assertIn("82", svgs[0])
        self.assertNotIn("Elev", svgs[0])
        self.assertNotIn("15.5", svgs[0])

    def test_exporter_to_dxf_layers(self):
        sheets = {
            0: [
                {
                    'id': '1_0',
                    'elevation': 1.5,
                    'geom': sg.box(10, 10, 40, 40),
                    'engravings': [sg.LineString([(15, 15), (35, 35)])]
                }
            ]
        }

        dxfs = export_sheets_to_dxf(sheets, bed_w=100, bed_h=100, margin=5)
        self.assertIn(0, dxfs)
        dxf_content = dxfs[0].decode("utf-8", errors="ignore")

        self.assertIn("CORTE", dxf_content)
        self.assertIn("GRAVAÇÃO", dxf_content)
        self.assertIn("MANCHA", dxf_content)
        self.assertIn("Placa", dxf_content)
        self.assertIn("MargemSeguranca", dxf_content)

    def test_exported_dxf_matches_laser_linework_rules(self):
        sheets = {
            0: [
                {
                    "id": "1_0",
                    "elevation": 1.5,
                    "geom": sg.box(10, 10, 40, 40),
                    "engravings": [],
                }
            ]
        }

        dxfs = export_sheets_to_dxf(sheets, bed_w=100, bed_h=100, margin=5)
        diagnostics = diagnose_exported_dxf(dxfs[0])
        self.assertTrue(diagnostics["ok"], diagnostics["issues"])
        self.assertEqual(diagnostics["entity_types"], {"LINE": diagnostics["entity_count"]})
        self.assertEqual(diagnostics["overlap_count"], 0)

        doc = ezdxf.read(io.StringIO(dxfs[0].decode("utf-8", errors="ignore")))
        for entity in doc.modelspace():
            self.assertEqual(entity.dxftype(), "LINE")
            self.assertEqual(entity.dxf.color, 256)
            self.assertEqual(entity.dxf.linetype.upper(), "BYLAYER")
            self.assertEqual(entity.dxf.lineweight, -1)

        cut_lines = [
            entity
            for entity in doc.modelspace().query("LINE")
            if entity.dxf.layer == "CORTE"
        ]
        self.assertTrue(cut_lines)

    def test_exported_dxf_diagnostics_detect_overlapping_lines(self):
        sheets = {
            0: [
                {
                    "id": "1_0",
                    "elevation": 1.5,
                    "geom": sg.box(10, 10, 40, 40),
                    "engravings": [],
                }
            ]
        }

        dxfs = export_sheets_to_dxf(sheets, bed_w=100, bed_h=100, margin=5)
        doc = ezdxf.read(io.StringIO(dxfs[0].decode("utf-8", errors="ignore")))
        cut_line = next(
            entity
            for entity in doc.modelspace().query("LINE")
            if entity.dxf.layer == "CORTE"
        )
        doc.modelspace().add_line(
            cut_line.dxf.start,
            cut_line.dxf.end,
            dxfattribs={
                "layer": cut_line.dxf.layer,
                "color": 256,
                "linetype": "BYLAYER",
                "lineweight": -1,
            },
        )
        stream = io.StringIO()
        doc.write(stream)

        diagnostics = diagnose_exported_dxf(doc.encode(stream.getvalue()))
        self.assertFalse(diagnostics["ok"])
        self.assertTrue(
            any(issue["type"] == "overlapping_segments" for issue in diagnostics["issues"])
        )

    def test_exporter_trims_engraving_segments_over_cut_lines(self):
        sheets = {
            0: [
                {
                    "id": "1_0",
                    "elevation": 1.5,
                    "geom": sg.box(10, 10, 40, 40),
                    "engravings": [
                        sg.LineString([(5, 10), (25, 10), (25, 25)]),
                    ],
                }
            ]
        }

        dxfs = export_sheets_to_dxf(sheets, bed_w=100, bed_h=100, margin=5)
        diagnostics = diagnose_exported_dxf(dxfs[0])
        self.assertTrue(diagnostics["ok"], diagnostics["issues"])

        doc = ezdxf.read(io.StringIO(dxfs[0].decode("utf-8", errors="ignore")))
        engrave_segments = []
        for entity in doc.modelspace().query("LINE"):
            if entity.dxf.layer != "GRAVAÇÃO":
                continue
            start = entity.dxf.start
            end = entity.dxf.end
            engrave_segments.append(
                ((float(start.x), float(start.y)), (float(end.x), float(end.y)))
            )

        self.assertTrue(engrave_segments)
        self.assertFalse(
            any(
                abs(start[1] - 10) <= 1e-6
                and abs(end[1] - 10) <= 1e-6
                and max(start[0], end[0]) > 10
                and min(start[0], end[0]) < 25
                for start, end in engrave_segments
            )
        )

    def test_exporter_removes_same_layer_overlapping_engraving_segments(self):
        sheets = {
            0: [
                {
                    "id": "1_0",
                    "elevation": 1.5,
                    "geom": sg.box(10, 10, 70, 30),
                    "engravings": [
                        sg.LineString([(20, 20), (50, 20)]),
                        sg.LineString([(30, 20), (60, 20)]),
                    ],
                }
            ]
        }

        dxfs = export_sheets_to_dxf(sheets, bed_w=100, bed_h=100, margin=5)
        diagnostics = diagnose_exported_dxf(dxfs[0])
        self.assertTrue(diagnostics["ok"], diagnostics["issues"])

        doc = ezdxf.read(io.StringIO(dxfs[0].decode("utf-8", errors="ignore")))
        horizontal_lengths = []
        for entity in doc.modelspace().query("LINE"):
            if entity.dxf.layer != "GRAVAÇÃO":
                continue
            start = entity.dxf.start
            end = entity.dxf.end
            if abs(float(start.y) - 20) <= 1e-6 and abs(float(end.y) - 20) <= 1e-6:
                horizontal_lengths.append(abs(float(end.x) - float(start.x)))

        self.assertAlmostEqual(sum(horizontal_lengths), 40.0)

    def test_exporter_trims_guide_lines_that_overlap_cut_lines(self):
        sheets = {
            0: [
                {
                    "id": "1_0",
                    "elevation": 1.5,
                    "geom": sg.box(5, 5, 40, 40),
                    "engravings": [],
                }
            ]
        }

        dxfs = export_sheets_to_dxf(sheets, bed_w=100, bed_h=100, margin=5)
        diagnostics = diagnose_exported_dxf(dxfs[0])
        self.assertTrue(diagnostics["ok"], diagnostics["issues"])
        self.assertEqual(diagnostics["overlap_count"], 0)

        doc = ezdxf.read(io.StringIO(dxfs[0].decode("utf-8", errors="ignore")))
        self.assertTrue(
            any(entity.dxf.layer == "CORTE" for entity in doc.modelspace().query("LINE"))
        )

    def test_exporter_keeps_guide_lines_one_mm_away_from_piece(self):
        piece_geom = sg.box(5, 5, 40, 40)
        sheets = {
            0: [
                {
                    "id": "1_0",
                    "elevation": 1.5,
                    "geom": piece_geom,
                    "engravings": [],
                }
            ]
        }

        dxfs = export_sheets_to_dxf(sheets, bed_w=100, bed_h=100, margin=5)
        doc = ezdxf.read(io.StringIO(dxfs[0].decode("utf-8", errors="ignore")))
        clearance = piece_geom.buffer(1.0, join_style=2)

        for entity in doc.modelspace().query("LINE"):
            if entity.dxf.layer not in {"Placa", "MargemSeguranca"}:
                continue
            line = sg.LineString([
                (float(entity.dxf.start.x), float(entity.dxf.start.y)),
                (float(entity.dxf.end.x), float(entity.dxf.end.y)),
            ])
            self.assertLessEqual(line.intersection(clearance).length, 1e-6)

    def test_exporter_cleans_small_inward_polygon_notches_before_linework(self):
        polygon = sg.Polygon([
            (0, 0),
            (30, 0),
            (30, 20),
            (17, 20),
            (17, 19.6),
            (16.2, 19.6),
            (16.2, 20),
            (0, 20),
            (0, 0),
        ])

        cleaned = _clean_export_geometry(polygon, tolerance=0.5)
        self.assertTrue(cleaned.is_valid)
        self.assertTrue(cleaned.covers(sg.Point(16.6, 19.8)))

        sheets = {
            0: [
                {
                    "id": "1_0",
                    "elevation": 1.5,
                    "geom": polygon,
                    "engravings": [],
                }
            ]
        }
        dxfs = export_sheets_to_dxf(sheets, bed_w=60, bed_h=50, margin=2)
        diagnostics = diagnose_exported_dxf(dxfs[0])
        self.assertTrue(diagnostics["ok"], diagnostics["issues"])

    def test_exporter_allows_configuring_small_notch_cleanup(self):
        polygon = sg.Polygon([
            (0, 0),
            (30, 0),
            (30, 20),
            (17, 20),
            (17, 19.6),
            (16.2, 19.6),
            (16.2, 20),
            (0, 20),
            (0, 0),
        ])
        sheets = {
            0: [
                {
                    "id": "1_0",
                    "elevation": 1.5,
                    "geom": polygon,
                    "engravings": [],
                }
            ]
        }

        raw_dxfs = export_sheets_to_dxf(
            sheets,
            bed_w=60,
            bed_h=50,
            margin=2,
            geometry_clean_tolerance=0.0,
        )
        clean_dxfs = export_sheets_to_dxf(
            sheets,
            bed_w=60,
            bed_h=50,
            margin=2,
            geometry_clean_tolerance=0.5,
        )
        raw_doc = ezdxf.read(io.StringIO(raw_dxfs[0].decode("utf-8", errors="ignore")))
        clean_doc = ezdxf.read(io.StringIO(clean_dxfs[0].decode("utf-8", errors="ignore")))
        raw_cut = [e for e in raw_doc.modelspace().query("LINE") if e.dxf.layer == "CORTE"]
        clean_cut = [e for e in clean_doc.modelspace().query("LINE") if e.dxf.layer == "CORTE"]
        self.assertGreater(len(raw_cut), len(clean_cut))

    def test_exporter_simplifies_nearly_collinear_laser_linework(self):
        sheets = {
            0: [
                {
                    "id": "1_0",
                    "geom": sg.box(0, 0, 50, 20),
                    "engravings": [
                        sg.LineString([(5, 10), (15, 10.006), (25, 9.997), (35, 10)])
                    ],
                }
            ]
        }

        raw_dxfs = export_sheets_to_dxf(
            sheets,
            bed_w=60,
            bed_h=30,
            margin=2,
            geometry_clean_tolerance=0.0,
            linework_simplify_tolerance=0.0,
        )
        clean_dxfs = export_sheets_to_dxf(
            sheets,
            bed_w=60,
            bed_h=30,
            margin=2,
            geometry_clean_tolerance=0.0,
            linework_simplify_tolerance=0.02,
        )
        raw_doc = ezdxf.read(io.StringIO(raw_dxfs[0].decode("utf-8", errors="ignore")))
        clean_doc = ezdxf.read(io.StringIO(clean_dxfs[0].decode("utf-8", errors="ignore")))
        engrave_layer = DEFAULT_LAYER_CONFIG["engrave"]["name"]
        raw_engrave = [e for e in raw_doc.modelspace().query("LINE") if e.dxf.layer == engrave_layer]
        clean_engrave = [e for e in clean_doc.modelspace().query("LINE") if e.dxf.layer == engrave_layer]

        self.assertGreater(len(raw_engrave), len(clean_engrave))
        self.assertEqual(len(clean_engrave), 1)

    def test_exporter_combined_dxf_contains_all_sheets_as_lines(self):
        sheets = {
            0: [
                {
                    "id": "1_0",
                    "elevation": 1.5,
                    "geom": sg.box(10, 10, 40, 40),
                    "engravings": [],
                }
            ],
            1: [
                {
                    "id": "2_0",
                    "elevation": 2.0,
                    "geom": sg.box(10, 10, 50, 50),
                    "engravings": [],
                }
            ],
        }

        dxf = export_sheets_to_combined_dxf(sheets, bed_w=100, bed_h=100, margin=5)
        diagnostics = diagnose_exported_dxf(dxf)
        self.assertTrue(diagnostics["ok"], diagnostics["issues"])
        doc = ezdxf.read(io.StringIO(dxf.decode("utf-8", errors="ignore")))
        self.assertEqual({entity.dxftype() for entity in doc.modelspace()}, {"LINE"})
        self.assertGreater(max(entity.dxf.start.x for entity in doc.modelspace().query("LINE")), 100)

    def test_exporter_places_label_inside_large_engraving_area(self):
        engraving_area = sg.box(20, 20, 90, 60)
        sheets = {
            0: [
                {
                    "id": "1_0",
                    "elevation": 1.5,
                    "geom": sg.box(10, 10, 100, 80),
                    "engravings": [sg.LinearRing(engraving_area.exterior.coords)],
                }
            ]
        }

        dxfs = export_sheets_to_dxf(sheets, bed_w=120, bed_h=100, margin=5)
        doc = ezdxf.read(io.StringIO(dxfs[0].decode("utf-8", errors="ignore")))
        label_zone = engraving_area.buffer(-2.0)
        interior_engraving_segments = 0

        for entity in doc.modelspace().query("LINE"):
            if entity.dxf.layer != "GRAVAÇÃO":
                continue
            midpoint = sg.Point(
                (float(entity.dxf.start.x) + float(entity.dxf.end.x)) / 2.0,
                (float(entity.dxf.start.y) + float(entity.dxf.end.y)) / 2.0,
            )
            if label_zone.contains(midpoint):
                interior_engraving_segments += 1

        self.assertGreater(interior_engraving_segments, 0)

    def test_exporter_places_split_label_inside_hidden_area_metadata(self):
        hidden_area = sg.box(30, 20, 80, 55)
        sheets = {
            0: [
                {
                    "id": "split_1",
                    "elevation": 1.5,
                    "source_elevation": 82.0,
                    "geom": sg.box(10, 10, 100, 80),
                    "engravings": [],
                    "label_areas": [hidden_area],
                }
            ]
        }

        dxfs = export_sheets_to_dxf(sheets, bed_w=120, bed_h=100, margin=5)
        doc = ezdxf.read(io.StringIO(dxfs[0].decode("utf-8", errors="ignore")))
        label_zone = hidden_area.buffer(-2.0)
        interior_segments = 0
        engrave_layer = DEFAULT_LAYER_CONFIG["engrave"]["name"]

        for entity in doc.modelspace().query("LINE"):
            if entity.dxf.layer != engrave_layer:
                continue
            midpoint = sg.Point(
                (float(entity.dxf.start.x) + float(entity.dxf.end.x)) / 2.0,
                (float(entity.dxf.start.y) + float(entity.dxf.end.y)) / 2.0,
            )
            if label_zone.contains(midpoint):
                interior_segments += 1

        self.assertGreater(interior_segments, 0)

    def test_exporter_places_external_label_three_mm_top_right(self):
        piece_geom = sg.box(20, 20, 50, 50)
        sheets = {
            0: [
                {
                    "id": "plain",
                    "elevation": 1.5,
                    "source_elevation": 82.0,
                    "geom": piece_geom,
                    "engravings": [],
                }
            ]
        }

        dxfs = export_sheets_to_dxf(sheets, bed_w=100, bed_h=100, margin=5)
        doc = ezdxf.read(io.StringIO(dxfs[0].decode("utf-8", errors="ignore")))
        engrave_layer = DEFAULT_LAYER_CONFIG["engrave"]["name"]
        engrave_lines = [
            entity
            for entity in doc.modelspace().query("LINE")
            if entity.dxf.layer == engrave_layer
        ]
        self.assertTrue(engrave_lines)

        minx = min(min(float(entity.dxf.start.x), float(entity.dxf.end.x)) for entity in engrave_lines)
        maxy = max(max(float(entity.dxf.start.y), float(entity.dxf.end.y)) for entity in engrave_lines)
        self.assertGreaterEqual(minx, 53.0 - 1e-6)
        self.assertLessEqual(maxy, 17.0 + 1e-6)

    def test_exporter_custom_layers_and_colors(self):
        sheets = {
            0: [
                {
                    'id': '1_0',
                    'elevation': 1.5,
                    'geom': sg.box(10, 10, 40, 40),
                    'engravings': []
                }
            ]
        }
        layer_config = {
            "cut": {"name": "CORTA_AQUI", "dxf_color": 2, "svg_color": "#00ff00"},
            "engrave": {"name": "GRAVA_AQUI", "dxf_color": 4, "svg_color": "#0000ff"},
            "mark": {"name": "MANCHA_AQUI", "dxf_color": 7, "svg_color": "#000000"},
            "bed": {"name": "PLACA_AQUI", "dxf_color": 8, "svg_color": "#888888"},
            "margin": {"name": "MARGEM_AQUI", "dxf_color": 9, "svg_color": "#999999"},
        }

        svgs = export_sheets_to_svg(sheets, bed_w=100, bed_h=100, margin=3, layer_config=layer_config)
        self.assertIn("CORTA_AQUI", svgs[0])
        self.assertIn("PLACA_AQUI", svgs[0])

        dxfs = export_sheets_to_dxf(sheets, bed_w=100, bed_h=100, margin=3, layer_config=layer_config)
        dxf_content = dxfs[0].decode("utf-8", errors="ignore")
        self.assertIn("CORTA_AQUI", dxf_content)
        self.assertIn("GRAVA_AQUI", dxf_content)
        self.assertIn("PLACA_AQUI", dxf_content)

    def test_exporter_units_svg_cm(self):
        sheets = {
            0: [
                {
                    "id": "1_0",
                    "elevation": 1.5,
                    "geom": sg.box(10, 10, 40, 40),
                    "engravings": [],
                }
            ]
        }

        svgs = export_sheets_to_svg(sheets, bed_w=100, bed_h=200, margin=5, export_units="cm")
        svg_content = svgs[0]
        self.assertIn('<svg width="10cm" height="20cm"', svg_content)

    def test_exporter_units_dxf_cm(self):
        sheets = {
            0: [
                {
                    "id": "1_0",
                    "elevation": 1.5,
                    "geom": sg.box(10, 10, 40, 40),
                    "engravings": [],
                }
            ]
        }

        dxfs = export_sheets_to_dxf(sheets, bed_w=100, bed_h=100, margin=5, export_units="cm")
        text = dxfs[0].decode("utf-8", errors="ignore")
        doc = ezdxf.read(io.StringIO(text))
        self.assertEqual(doc.units, ezdxf.units.CM)

        x_values = []
        for entity in doc.modelspace().query("LINE"):
            x_values.append(float(entity.dxf.start.x))
            x_values.append(float(entity.dxf.end.x))

        self.assertTrue(x_values)
        self.assertLessEqual(max(x_values), 10.0001)

    def test_ui_utils_override_resolution(self):
        points = {
            "1": {"forced_source_elevation": 100.0},
            "2": {"forced_layer_index": 4},
        }
        summary = {
            "2": {"forced_layer_index": 7},
            "3": {"keep": False},
        }
        effective_points = resolve_effective_overrides(points, summary, "points")
        self.assertEqual(effective_points["2"]["forced_layer_index"], 4)

        effective_summary = resolve_effective_overrides(points, summary, "summary")
        self.assertEqual(effective_summary["2"]["forced_layer_index"], 7)
        self.assertEqual(effective_summary["3"]["keep"], False)

    def test_ui_utils_parse_helpers(self):
        self.assertEqual(parse_float_or_none("12,5"), 12.5)
        self.assertIsNone(parse_float_or_none("nan"))
        self.assertEqual(parse_int_or_none("42"), 42)
        self.assertIsNone(parse_int_or_none("4.2"))
        self.assertEqual(parse_curve_ids("1, 2,3"), [1, 2, 3])
        self.assertIsNone(parse_curve_ids("1,a"))
        self.assertTrue(parse_bool("sim"))
        self.assertFalse(parse_bool("nao"))
        self.assertEqual(contour_id_sort_key("10"), (0, 10))

    def test_ui_utils_boundary_resolution(self):
        polygons = [
            {"id": 1, "layer": "boundary", "geometry": sg.box(0, 0, 10, 10)},
            {"id": 2, "layer": "boundary", "geometry": sg.box(1, 1, 9, 9)},
            {"id": 3, "layer": "topo", "geometry": sg.box(2, 2, 8, 8)},
        ]
        boundary_geom, contours, boundary_polygons, outer = resolve_boundary_and_contours(polygons, "boundary")
        self.assertEqual(round(boundary_geom.area, 3), 100.0)
        self.assertEqual(len(boundary_polygons), 2)
        self.assertEqual(outer["id"], 1)
        self.assertIn(2, [item["id"] for item in contours])
        self.assertIn(3, [item["id"] for item in contours])

    def test_ui_utils_normalize_override_map(self):
        raw = {
            1: {"forced_source_elevation": 10.0, "forced_relative_elevation": ""},
            " ": {"keep": False},
            "3": {"keep": True},
        }
        normalized = normalize_override_map(raw)
        self.assertEqual(normalized["1"]["forced_source_elevation"], 10.0)
        self.assertNotIn("forced_relative_elevation", normalized["1"])
        self.assertEqual(normalized["3"]["keep"], True)

    def test_dxf_diagnostics_for_known_fixture(self):
        diagnostics = build_dxf_diagnostics("test_topography.dxf")
        self.assertEqual(diagnostics["closed_total"], 7)
        self.assertEqual(diagnostics["valid_total"], 7)
        self.assertEqual(diagnostics["invalid_total"], 0)
        self.assertIn("layers", diagnostics)
        self.assertTrue(any(row["layer"] == "0" for row in diagnostics["layers"]))
        self.assertIn("overlap_summary", diagnostics)

if __name__ == "__main__":
    unittest.main()
