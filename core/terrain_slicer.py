import math
from collections import defaultdict

from shapely.affinity import scale
from shapely.geometry import LineString, MultiPoint, Polygon
from shapely.ops import linemerge, snap, triangulate, unary_union

from core.topology import build_topology_tree

HOLLOW_SUPPORT_RULE_VERSION = "visible_buffer_clip_clean_restore_v1"
SOLID_VISIBILITY_RULE_VERSION = "solid_visibility_v1"
OFFSET_SIMPLIFY_RATIO = 0.02


def generate_terrain_layers_from_points(points, boundary_geom, interval=0.5, scale_factor=1.0):
    """
    Generate stacked terrain layers from 3D survey points using a linear TIN.

    Each layer is the area where the interpolated terrain is above a height
    threshold, clipped to the boundary. The first layer is always the full
    boundary, matching the physical base sheet.
    """
    if not points or boundary_geom is None or boundary_geom.is_empty:
        return []

    _ = scale_factor

    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    zs = [p["z"] for p in points]
    if len(points) < 3:
        return []

    min_z = min(zs)
    max_z = max(zs)
    # The physical base is already the full boundary. Start one step above the
    # rounded minimum height to avoid duplicating a near-identical base layer.
    start_level = math.ceil(min_z / interval) * interval + interval

    triangles = _triangulate_points(points)
    slices = [{
        "id": "base",
        "elevation": 0.0,
        "source_elevation": min_z,
        "slice_geom": boundary_geom,
        "engraving_lines": [],
    }]

    level = start_level
    index = 1
    while level <= max_z + 1e-9:
        layer_geom = _area_above_level(triangles, level)
        if not layer_geom.is_empty:
            layer_geom = layer_geom.intersection(boundary_geom)

        if not layer_geom.is_empty and layer_geom.area > 0.1:
            slices.append({
                "id": f"level_{index}",
                "elevation": round(level - min_z, 3),
                "source_elevation": round(level, 3),
                "slice_geom": layer_geom,
                "engraving_lines": [],
            })
            index += 1

        level += interval

    return _annotate_solid_slice_visibility(slices, area_tolerance=0.1)


def build_contour_correction_table(
    contours,
    points,
    nearest_count=12,
    near_distance=None,
    elevation_precision=0.0,
    interval=1.0,
    target_layer_count=None,
):
    """
    Build a flat table for manual elevation correction in Excel/CSV.
    """
    _ = nearest_count
    _ = near_distance

    rows = []
    if not contours or not points:
        return rows

    rough_elevations = [
        elevation
        for contour in contours
        if contour.get("geometry") is not None and not contour.get("geometry").is_empty
        for elevation in [_estimate_contour_elevation(contour["geometry"], points)]
        if elevation is not None
    ]
    grouping_precision = _resolve_topographic_precision(
        interval,
        elevation_precision,
        rough_elevations,
        target_layer_count,
    )
    base_elevation = _base_elevation_from_values(
        [point["z"] for point in points],
        grouping_precision,
    )

    for contour in contours:
        geom = contour.get("geometry")
        if geom is None or geom.is_empty:
            continue

        elevation = _estimate_contour_elevation(geom, points)
        if elevation is None:
            continue

        snapped = _estimate_contour_topographic_elevation(geom, points, grouping_precision)
        if snapped is None:
            continue

        rows.append({
            "contour_id": contour.get("id"),
            "layer": contour.get("layer"),
            "estimated_elevation": round(elevation, 6),
            "snapped_elevation": round(snapped, 6),
            "estimated_relative_elevation": round(snapped - base_elevation, 6),
            "estimated_layer_index": int(round((snapped - base_elevation) / grouping_precision)) if grouping_precision > 0 else "",
            "forced_elevation": "",
            "forced_source_elevation": "",
            "forced_relative_elevation": "",
            "forced_layer_index": "",
            "keep": 1,
            "note": "",
        })

    rows.sort(key=lambda row: (row["snapped_elevation"], row["contour_id"]))
    return rows


def generate_terrain_layers_from_contours(
    contours,
    points,
    boundary_geom,
    interval=1.0,
    elevation_precision=0.0,
    nearest_count=12,
    near_distance=0.2,
    scale_factor=1.0,
    slicing_mode="solid",
    manual_overrides=None,
    target_layer_count=None,
    assembly_offset_mm=5.0,
    hollow_glue_margin=0.0,
    adjacency_tolerance=1e-6,
    area_tolerance=0.1,
    extend_base_to_zero=False,
):
    """
    Generate stacked layers using the original DXF contour polylines.

    Survey points are used only to estimate each contour's elevation. The output
    geometry is made from the original contour polygons, so it preserves the CAD
    polyline vertices instead of creating interpolated TIN contour lines.
    """
    if not contours or not points or boundary_geom is None or boundary_geom.is_empty:
        return []

    _ = nearest_count
    _ = scale_factor
    _ = target_layer_count

    manual_overrides = manual_overrides or {}
    rough_elevations = [
        elevation
        for contour in contours
        if contour.get("geometry") is not None and not contour.get("geometry").is_empty
        for elevation in [_estimate_contour_elevation(contour["geometry"], points)]
        if elevation is not None
    ]
    grouping_precision = _resolve_topographic_precision(
        interval,
        elevation_precision,
        rough_elevations,
        target_layer_count,
    )
    base_elevation = _base_elevation_from_values(
        [point["z"] for point in points],
        grouping_precision,
    )
    point_match_distance = near_distance if near_distance and near_distance > 0 else None
    contour_info = []
    for contour in contours:
        geom = contour["geometry"]
        if geom is None or geom.is_empty or geom.area <= area_tolerance:
            continue

        contour_id = contour.get("id")
        override = _resolve_contour_override(manual_overrides, contour_id)
        if not override["keep"]:
            continue

        forced_elevation = _resolve_override_source_elevation(
            override,
            base_elevation=base_elevation,
            interval=interval,
        )
        if forced_elevation is None:
            nearest_point_distance = _nearest_boundary_point_distance(geom, points)
            elevation = None
            elevation_source = "point"
            if point_match_distance is not None:
                elevation = _estimate_contour_topographic_elevation(
                    geom,
                    points,
                    grouping_precision,
                    base_elevation=base_elevation,
                    max_distance=point_match_distance,
                )
                elevation_source = "direct"
            if elevation is None:
                elevation = _estimate_contour_topographic_elevation(
                    geom,
                    points,
                    grouping_precision,
                    base_elevation=base_elevation,
                )
                if point_match_distance is not None:
                    elevation_source = "nearest"
            if elevation is None:
                continue
            estimated_elevation = elevation
        else:
            elevation = _bucket_elevation(
                forced_elevation,
                grouping_precision,
                base_elevation=base_elevation,
            )
            estimated_elevation = forced_elevation
            elevation_source = "forced"

        contour_info.append({
            "id": contour_id,
            "ids": [contour_id],
            "layers": [contour.get("layer")],
            "geometry": geom,
            "elevation": elevation,
            "source_elevation": elevation,
            "estimated_elevation": estimated_elevation,
            "elevation_source": elevation_source,
            "point_match_distance": nearest_point_distance if forced_elevation is None else None,
        })

    if not contour_info:
        return []

    _spread_contours_to_target_levels(
        contour_info,
        base_elevation=base_elevation,
        interval=grouping_precision,
        target_layer_count=target_layer_count,
    )
    _infer_topological_contour_elevations(
        contour_info,
        base_elevation=base_elevation,
        interval=grouping_precision,
        adjacency_tolerance=adjacency_tolerance,
    )
    contour_info = _merge_duplicate_contours(contour_info, area_tolerance)
    contour_groups = _group_contours_by_elevation(contour_info, grouping_precision)

    contour_sources_by_id = _build_contour_sources_by_id(contour_groups)
    topology_tree = build_topology_tree(contour_info)
    terrain_cells = _build_topographic_cells(
        contour_info,
        topology_tree,
        boundary_geom,
        min_z=base_elevation,
        area_tolerance=area_tolerance,
    )

    slices = [{
        "id": "base",
        "elevation": 0.0,
        "source_elevation": base_elevation,
        "slice_geom": boundary_geom,
        "engraving_lines": [],
    }]

    previous_geom = boundary_geom
    index = 1
    for group in contour_groups:
        group_elevation = _group_elevation(group)
        layer_geom = _geometry_above_level(terrain_cells, group_elevation, boundary_geom)
        layer_geom = _clean_layer_geometry(layer_geom, area_tolerance)
        if layer_geom.is_empty or layer_geom.area <= area_tolerance:
            continue

        # Avoid duplicate sheets when a level does not change the cut geometry.
        if _geometries_equivalent(previous_geom, layer_geom, area_tolerance):
            _merge_slice_source_metadata(slices[-1], group)
            continue

        source_ids = []
        source_layers = []
        source_elevations = []
        for item in group:
            source_ids.extend(item["ids"])
            source_layers.extend(item["layers"])
            source_elevations.append(item.get("estimated_elevation", item["source_elevation"]))

        source_ids, source_layers = _dedupe_id_layer_pairs(source_ids, source_layers)

        slices.append({
            "id": f"level_{index}",
            "elevation": round(group_elevation - base_elevation, 3),
            "source_elevation": round(group_elevation, 3),
            "source_ids": source_ids,
            "source_layers": source_layers,
            "estimated_elevations": [round(value, 3) for value in source_elevations],
            "removed_area": round(previous_geom.area - layer_geom.area, 3),
            "slice_geom": layer_geom,
            "engraving_lines": [],
        })
        previous_geom = layer_geom
        index += 1

    if slicing_mode == "hollow":
        slices = _convert_solid_slices_to_hollow(
            slices,
            area_tolerance,
            glue_margin=hollow_glue_margin,
            glue_limit_geom=boundary_geom,
        )

    if extend_base_to_zero:
        slices = _prepend_real_zero_base_layers(
            slices,
            boundary_geom,
            base_elevation=base_elevation,
            interval=grouping_precision,
            area_tolerance=area_tolerance,
        )

    if slicing_mode == "solid":
        slices = _annotate_solid_slice_visibility(slices, area_tolerance)

    _add_assembly_engraving_lines(slices, offset=assembly_offset_mm, slicing_mode=slicing_mode)
    _ensure_unique_and_complete_ids(
        slices,
        contour_sources_by_id,
        min_z=base_elevation,
    )
    _renumber_layers(slices, interval)
    return slices


def _annotate_solid_slice_visibility(slices, area_tolerance=0.1):
    if not slices:
        return []

    physical_geoms = [
        _solid_physical_geometry_from_metadata(item, area_tolerance)
        for item in slices
    ]

    annotated = []
    for index, current in enumerate(slices):
        current_geom = physical_geoms[index]
        if current_geom is None or current_geom.is_empty:
            continue

        if index < len(physical_geoms) - 1:
            next_geom = physical_geoms[index + 1]
        else:
            next_geom = Polygon()

        if next_geom is not None and not next_geom.is_empty:
            hidden_base_geom = _clean_layer_geometry(current_geom.intersection(next_geom), area_tolerance)
            visible_geom = _clean_layer_geometry(current_geom.difference(next_geom), area_tolerance)
        else:
            hidden_base_geom = Polygon()
            visible_geom = current_geom

        item = dict(current)
        item["slice_geom"] = current_geom
        item["visible_geom"] = visible_geom
        item["hidden_base_geom"] = hidden_base_geom
        item["support_geom"] = hidden_base_geom
        item["support_rule_version"] = SOLID_VISIBILITY_RULE_VERSION
        item["support_glue_margin"] = 0.0
        annotated.append(item)

    return annotated


def _solid_physical_geometry_from_metadata(item, area_tolerance):
    visible_geom = item.get("visible_geom")
    hidden_base_geom = item.get("hidden_base_geom")
    if (
        visible_geom is not None
        and not visible_geom.is_empty
        and hidden_base_geom is not None
        and not hidden_base_geom.is_empty
    ):
        return _clean_layer_geometry(unary_union([visible_geom, hidden_base_geom]), area_tolerance)

    return _clean_layer_geometry(item.get("slice_geom"), area_tolerance)


def _convert_solid_slices_to_hollow(slices, area_tolerance, glue_margin=0.0, glue_limit_geom=None):
    if not slices:
        return []

    limit_geom = glue_limit_geom
    if limit_geom is None or limit_geom.is_empty:
        limit_geom = next(
            (
                item.get("slice_geom")
                for item in slices
                if item.get("slice_geom") is not None and not item.get("slice_geom").is_empty
            ),
            None,
        )

    converted = []
    for index, current in enumerate(slices):
        current_geom = current.get("slice_geom")
        if current_geom is None or current_geom.is_empty:
            continue

        if index < len(slices) - 1:
            next_geom = slices[index + 1].get("slice_geom")
            if next_geom is not None and not next_geom.is_empty:
                ring_geom, visible_geom, support_geom, hidden_base_geom = _hollow_layer_geometry(
                    current_geom,
                    next_geom,
                    limit_geom,
                    glue_margin,
                    area_tolerance,
                )
            else:
                ring_geom, visible_geom, support_geom, hidden_base_geom = _hollow_layer_geometry(
                    current_geom,
                    None,
                    limit_geom,
                    glue_margin,
                    area_tolerance,
                )
        else:
            ring_geom, visible_geom, support_geom, hidden_base_geom = _hollow_layer_geometry(
                current_geom,
                None,
                limit_geom,
                glue_margin,
                area_tolerance,
            )

        ring_geom = _clean_layer_geometry(ring_geom, area_tolerance)
        if ring_geom.is_empty or ring_geom.area <= area_tolerance:
            continue

        clean_visible = _clean_layer_geometry(visible_geom, area_tolerance)
        clean_support = _clean_layer_geometry(support_geom, area_tolerance)
        if not clean_support.is_empty and glue_margin and glue_margin > 0:
            # Re-check support against the final visible geometry because
            # cleaning may remove tiny visible fragments used in a first pass.
            clean_support = _clean_support_geometry(
                clean_support,
                clean_visible,
                glue_margin,
                area_tolerance,
            )
        clean_hidden = _clean_layer_geometry(hidden_base_geom, area_tolerance)
        clean_ring = clean_visible if clean_support.is_empty else _clean_layer_geometry(
            unary_union([clean_visible, clean_support]),
            area_tolerance,
        )
        if clean_ring.is_empty or clean_ring.area <= area_tolerance:
            continue

        item = dict(current)
        item["slice_geom"] = clean_ring
        item["visible_geom"] = clean_visible
        item["support_geom"] = clean_support
        item["hidden_base_geom"] = clean_hidden
        item["support_rule_version"] = HOLLOW_SUPPORT_RULE_VERSION
        item["support_glue_margin"] = float(glue_margin or 0.0)
        converted.append(item)

    return converted


def _prune_floating_hollow_components(slices, area_tolerance=0.1, glue_margin=0.0):
    if not slices or len(slices) < 2:
        return

    contact_tolerance = max(1e-6, min(max(float(glue_margin or 0.0) * 0.02, 0.01), 0.2))
    for index in range(1, len(slices)):
        lower_row = slices[index - 1]
        row = slices[index]

        lower_geom = _clean_layer_geometry(lower_row.get("slice_geom"), area_tolerance)
        ring_geom = _clean_layer_geometry(row.get("slice_geom"), area_tolerance)
        if lower_geom.is_empty or ring_geom.is_empty:
            continue

        visible_geom = _clean_layer_geometry(_polygonal_geometry(row.get("visible_geom")), area_tolerance)
        if visible_geom.is_empty:
            continue

        hidden_candidate = _clean_layer_geometry(_polygonal_geometry(row.get("hidden_base_geom")), area_tolerance)
        if hidden_candidate.is_empty:
            inferred_hidden = _clean_layer_geometry(ring_geom.difference(visible_geom), area_tolerance)
            hidden_candidate = inferred_hidden if not inferred_hidden.is_empty else Polygon()

        lower_contact_band = lower_geom.buffer(
            contact_tolerance,
            cap_style=2,
            join_style=2,
            mitre_limit=2.0,
        )
        pre_support = _hidden_support_geometry(
            visible_geom,
            hidden_candidate,
            glue_margin,
            area_tolerance,
        )
        pre_ring = _clean_layer_geometry(
            visible_geom if pre_support.is_empty else unary_union([visible_geom, pre_support]),
            area_tolerance,
        )
        kept_parts = [part for part in _polygon_parts(pre_ring) if part.intersects(lower_contact_band)]

        # Keep only components that can physically connect to the layer below.
        # This avoids "floating islands" in Option B when offsets create detached regions.
        if not kept_parts:
            continue

        pruned_ring = _clean_layer_geometry(unary_union(kept_parts), area_tolerance)
        if pruned_ring.is_empty or pruned_ring.area <= area_tolerance:
            continue

        visible_geom = _clean_layer_geometry(
            visible_geom.intersection(pruned_ring),
            area_tolerance,
        )
        hidden_geom = _clean_layer_geometry(
            hidden_candidate.intersection(pruned_ring),
            area_tolerance,
        )
        support_geom = _hidden_support_geometry(
            visible_geom,
            hidden_geom,
            glue_margin,
            area_tolerance,
        )
        rebuilt_ring = _clean_layer_geometry(
            visible_geom if support_geom.is_empty else unary_union([visible_geom, support_geom]),
            area_tolerance,
        )
        if rebuilt_ring.is_empty or rebuilt_ring.area <= area_tolerance:
            continue

        row["slice_geom"] = rebuilt_ring
        row["visible_geom"] = visible_geom
        row["support_geom"] = support_geom
        row["hidden_base_geom"] = hidden_geom


def _hollow_layer_geometry(base_geom, cut_geom, limit_geom, glue_margin, area_tolerance=0.1):
    if limit_geom is not None and not limit_geom.is_empty:
        base_geom = base_geom.intersection(limit_geom)

    if cut_geom is None or cut_geom.is_empty:
        visible_geom = base_geom
        hidden_geom = Polygon()
    else:
        visible_geom = base_geom.difference(cut_geom)
        hidden_geom = base_geom.intersection(cut_geom)

    if glue_margin is None or glue_margin <= 0:
        return visible_geom, visible_geom, Polygon(), hidden_geom

    support_geom = _hidden_support_geometry(
        visible_geom,
        hidden_geom,
        glue_margin,
        area_tolerance,
    )
    if support_geom.is_empty or support_geom.area <= area_tolerance:
        return visible_geom, visible_geom, support_geom, hidden_geom

    return unary_union([visible_geom, support_geom]), visible_geom, support_geom, hidden_geom


def _hidden_support_geometry(visible_geom, hidden_geom, glue_margin, area_tolerance=0.1):
    if hidden_geom is None or hidden_geom.is_empty or glue_margin is None or glue_margin <= 0:
        return Polygon()
    if visible_geom is None or visible_geom.is_empty:
        return Polygon()
    if hidden_geom.area <= area_tolerance:
        return Polygon()

    support_geom = visible_geom.buffer(
        float(glue_margin),
        join_style=2,
        mitre_limit=2.0,
    ).intersection(hidden_geom)

    support_geom = _clean_support_geometry(
        support_geom,
        visible_geom,
        glue_margin,
        area_tolerance,
    )
    return support_geom if not support_geom.is_empty else Polygon()


def _shared_visible_hidden_contact_line(visible_geom, hidden_geom, glue_margin):
    if visible_geom is None or visible_geom.is_empty or hidden_geom is None or hidden_geom.is_empty:
        return LineString()

    shared = visible_geom.boundary.intersection(hidden_geom.boundary)
    if shared.is_empty:
        shared = visible_geom.boundary.intersection(hidden_geom)
    if shared.is_empty:
        return LineString()

    return _simplify_contact_line(shared, glue_margin)


def _clean_support_geometry(support_geom, visible_geom, glue_margin, area_tolerance=0.1):
    support_geom = _polygonal_geometry(support_geom)
    if support_geom.is_empty:
        return Polygon()

    glue_margin = float(glue_margin or 0.0)
    contact_tolerance = max(1e-6, min(glue_margin * 0.02, 0.1))
    min_contact_length = max(0.25, glue_margin * 0.05)
    min_width = max(0.2, glue_margin * 0.15)
    min_area = max(area_tolerance, glue_margin * glue_margin * 0.05)
    visible_contact_band = visible_geom.boundary.buffer(
        contact_tolerance,
        cap_style=2,
        join_style=2,
        mitre_limit=2.0,
    )

    kept_parts = []
    for part in _polygon_parts(support_geom):
        if part.area <= min_area:
            continue
        if _support_part_contact_length(part, visible_contact_band) < min_contact_length:
            continue
        if _support_part_effective_width(part) < min_width:
            continue
        kept_parts.append(part)

    if not kept_parts:
        return Polygon()

    cleaned = unary_union(kept_parts)
    if not cleaned.is_valid:
        cleaned = cleaned.buffer(0)
    return _polygonal_geometry(cleaned)


def _support_part_contact_length(part, visible_contact_band):
    if part is None or part.is_empty or visible_contact_band is None or visible_contact_band.is_empty:
        return 0.0
    contact = part.boundary.intersection(visible_contact_band)
    return sum(line.length for line in _linear_parts(contact))


def _support_part_effective_width(part):
    if part is None or part.is_empty or part.area <= 0:
        return 0.0

    widths = []
    if part.length > 1e-9:
        widths.append((2.0 * part.area) / part.length)

    rect = part.minimum_rotated_rectangle
    if not rect.is_empty and rect.geom_type == "Polygon":
        coords = list(rect.exterior.coords)
        edge_lengths = [
            math.hypot(coords[idx + 1][0] - coords[idx][0], coords[idx + 1][1] - coords[idx][1])
            for idx in range(min(4, len(coords) - 1))
        ]
        positive_lengths = [length for length in edge_lengths if length > 1e-9]
        if positive_lengths:
            widths.append(min(positive_lengths))

    return min(widths) if widths else 0.0


def _offset_contact_line_to_hidden_side(contact_line, hidden_geom, glue_margin):
    if _contact_margin_covers_base(contact_line, hidden_geom, glue_margin):
        return hidden_geom

    supports = []
    for line in _linear_parts(contact_line):
        support = _best_hidden_side_single_offset(line, hidden_geom, glue_margin)
        if support is not None and not support.is_empty:
            supports.append(support)

    if not supports:
        return Polygon()

    support = unary_union(supports).intersection(hidden_geom)
    return _polygonal_geometry(support)


def _contact_margin_covers_base(contact_line, hidden_geom, glue_margin):
    if contact_line is None or contact_line.is_empty or hidden_geom is None or hidden_geom.is_empty:
        return False
    coverage = contact_line.buffer(
        float(glue_margin),
        cap_style=2,
        join_style=2,
        mitre_limit=2.0,
    )
    if coverage.is_empty:
        return False
    remainder = hidden_geom.difference(coverage)
    return remainder.is_empty or remainder.area <= max(1e-6, hidden_geom.area * 1e-6)


def _best_hidden_side_single_offset(line, hidden_geom, glue_margin):
    line = _normalized_line_for_offset(line)
    if line is None or line.is_empty or line.length <= 1e-9:
        return None

    best_support = None
    best_score = None
    expected_area = max(line.length * float(glue_margin), 1e-9)
    for distance in (float(glue_margin), -float(glue_margin)):
        try:
            strip = line.buffer(
                distance,
                single_sided=True,
                join_style=2,
                mitre_limit=2.0,
            )
        except (AttributeError, ValueError, TypeError):
            continue

        if strip.is_empty or strip.area <= 1e-9:
            continue
        support = strip.intersection(hidden_geom)
        area = support.area if not support.is_empty else 0.0
        if area <= 1e-9:
            continue
        inside_ratio = area / max(strip.area, 1e-9)
        area_error = abs(area - expected_area) / expected_area
        score = (inside_ratio, -area_error, area)
        if best_score is None or score > best_score:
            best_score = score
            best_support = support

    return best_support


def _normalized_line_for_offset(line, tolerance=1e-9):
    if line is None or line.is_empty:
        return None
    try:
        coords = list(line.coords)
    except Exception:
        return None
    if len(coords) < 2:
        return None

    cleaned = [coords[0]]
    for point in coords[1:]:
        if (
            abs(point[0] - cleaned[-1][0]) <= tolerance
            and abs(point[1] - cleaned[-1][1]) <= tolerance
        ):
            continue
        cleaned.append(point)

    if len(cleaned) < 2:
        return None
    if (
        len(cleaned) == 2
        and abs(cleaned[0][0] - cleaned[1][0]) <= tolerance
        and abs(cleaned[0][1] - cleaned[1][1]) <= tolerance
    ):
        return None

    try:
        normalized = LineString(cleaned)
    except (TypeError, ValueError):
        return None

    if normalized.is_empty or normalized.length <= tolerance:
        return None
    return normalized


def _line_inside_ratio(line, geom):
    if line is None or line.is_empty or geom is None or geom.is_empty:
        return 0.0
    total_length = line.length
    if total_length <= 1e-9:
        return 0.0
    inside = line.intersection(geom)
    return max(0.0, min(1.0, inside.length / total_length))


def _strip_polygons_between_lines(line, offset_line):
    if line is None or offset_line is None or line.is_empty or offset_line.is_empty:
        return []

    line_coords = list(line.coords)
    offset_coords = list(offset_line.coords)
    if len(line_coords) < 2 or len(offset_coords) < 2:
        return []

    if _is_closed_line(line) and _is_closed_line(offset_line):
        original_poly = _safe_polygon(line_coords)
        offset_poly = _safe_polygon(offset_coords)
        if original_poly.is_empty or offset_poly.is_empty:
            return []
        return _polygon_parts(original_poly.symmetric_difference(offset_poly))

    candidates = []
    for candidate_offset_coords in (offset_coords, list(reversed(offset_coords))):
        shell = line_coords + list(reversed(candidate_offset_coords))
        if len(shell) < 4:
            continue
        poly = _safe_polygon(shell)
        candidates.extend(_polygon_parts(poly))
    return candidates


def _safe_polygon(coords):
    try:
        poly = Polygon(coords)
    except (TypeError, ValueError):
        return Polygon()
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if not poly.is_empty else Polygon()


def _is_closed_line(line):
    coords = list(line.coords)
    if len(coords) < 4:
        return False
    return bool(getattr(line, "is_ring", False)) or (
        abs(coords[0][0] - coords[-1][0]) <= 1e-9
        and abs(coords[0][1] - coords[-1][1]) <= 1e-9
    )


def _polygonal_geometry(geom):
    parts = _polygon_parts(geom)
    return _safe_polygonal_union(parts)


def _polygon_parts(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        parts = []
        for child in geom.geoms:
            parts.extend(_polygon_parts(child))
        return parts
    return []


def _safe_polygonal_union(geoms, area_tolerance=0.0):
    parts = []
    for geom in geoms or []:
        for part in _polygon_parts(geom):
            if part.is_empty or part.area <= area_tolerance:
                continue
            if not part.is_valid:
                part = part.buffer(0)
            for cleaned_part in _polygon_parts(part):
                if cleaned_part.is_empty or cleaned_part.area <= area_tolerance:
                    continue
                parts.append(cleaned_part)

    if not parts:
        return Polygon()
    if len(parts) == 1:
        return parts[0]
    try:
        return unary_union(parts)
    except TypeError:
        merged = parts[0]
        for part in parts[1:]:
            merged = merged.union(part)
        return merged


def _simplify_contact_line(geom, glue_margin):
    if geom is None or geom.is_empty or glue_margin is None or glue_margin <= 0:
        return LineString()

    tolerance = min(float(glue_margin) * OFFSET_SIMPLIFY_RATIO, 0.25)
    min_length = max(tolerance, 1e-6)
    source_lines = [
        line
        for line in _linear_parts(geom)
        if line.length > min_length
    ]
    if not source_lines:
        return LineString()

    merged = unary_union(source_lines)
    merged = snap(merged, merged, max(tolerance, 1e-6))
    merged = unary_union(_linear_parts(merged))
    try:
        merged = linemerge(merged)
    except ValueError:
        pass

    simplified_lines = []
    for line in _linear_parts(merged):
        if line.length <= min_length:
            continue
        simplified = line.simplify(tolerance, preserve_topology=False) if tolerance > 0 else line
        if simplified.is_empty:
            simplified = line
        simplified_lines.extend(
            part for part in _linear_parts(simplified) if part.length > min_length
        )

    if not simplified_lines:
        return LineString()

    merged = unary_union(simplified_lines)
    merged = snap(merged, merged, max(tolerance, 1e-6))
    merged = unary_union(_linear_parts(merged))
    try:
        merged = linemerge(merged)
    except ValueError:
        pass
    return merged


def _linear_parts(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type in ("LineString", "LinearRing"):
        return [geom]
    if geom.geom_type == "MultiLineString":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        parts = []
        for child in geom.geoms:
            parts.extend(_linear_parts(child))
        return parts
    return []


def _prepend_real_zero_base_layers(
    slices,
    boundary_geom,
    base_elevation,
    interval,
    area_tolerance=0.1,
):
    if not slices or boundary_geom is None or boundary_geom.is_empty:
        return slices
    if interval is None or interval <= 0:
        return slices
    if base_elevation is None or base_elevation <= interval / 2.0:
        return slices

    base_geom = _clean_layer_geometry(boundary_geom, area_tolerance)
    if base_geom.is_empty or base_geom.area <= area_tolerance:
        return slices

    levels = []
    level = 0.0
    while level < base_elevation - 1e-9:
        levels.append(round(level, 6))
        level += interval

    repeat_count = len(levels)
    if repeat_count <= 0:
        return slices

    base_layer = {
        "id": "real_zero_base",
        "elevation": 0.0,
        "source_elevation": 0.0,
        "slice_geom": base_geom,
        "engraving_lines": [],
        "is_zero_base": True,
        "repeat_count": repeat_count,
        "repeat_start_elevation": 0.0,
        "repeat_interval": round(float(interval), 6),
        "repeat_end_elevation": round(levels[-1], 6),
    }

    shifted_slices = []
    for item in slices:
        row = dict(item)
        source_elevation = row.get("source_elevation")
        if source_elevation is not None:
            row["elevation"] = round(float(source_elevation), 3)
        else:
            row["elevation"] = round(float(row.get("elevation", 0.0)) + float(base_elevation), 3)
        shifted_slices.append(row)

    return [base_layer] + shifted_slices


def _build_topographic_cells(contours, topology_tree, boundary_geom, min_z, area_tolerance):
    cells = []

    top_nodes = [node for node, degree in topology_tree.in_degree() if degree == 0]
    top_geoms = [
        topology_tree.nodes[node]["data"]["geometry"]
        for node in top_nodes
    ]
    top_union = _safe_polygonal_union(top_geoms, area_tolerance)
    if not top_union.is_empty:
        outside_geom = boundary_geom.difference(top_union)
    else:
        outside_geom = boundary_geom

    outside_geom = _clean_layer_geometry(outside_geom, area_tolerance)
    if not outside_geom.is_empty and outside_geom.area > area_tolerance:
        cells.append({
            "geometry": outside_geom,
            "elevation": _outside_region_elevation(topology_tree, top_nodes, min_z),
        })

    for contour in contours:
        node = contour.get("id")
        if node not in topology_tree:
            continue

        children = list(topology_tree.successors(node))
        child_geoms = [
            topology_tree.nodes[child]["data"]["geometry"]
            for child in children
        ]
        geom = contour["geometry"].intersection(boundary_geom)
        child_union = _safe_polygonal_union(child_geoms, area_tolerance)
        if not child_union.is_empty:
            geom = geom.difference(child_union)
        geom = _clean_layer_geometry(geom, area_tolerance)
        if geom.is_empty or geom.area <= area_tolerance:
            continue

        cells.append({
            "geometry": geom,
            "elevation": _contour_band_elevation(topology_tree, node),
        })

    return cells


def _outside_region_elevation(topology_tree, top_nodes, min_z):
    descending_top_levels = [
        topology_tree.nodes[node]["data"]["elevation"]
        for node in top_nodes
        if _has_lower_child(topology_tree, node)
    ]
    if descending_top_levels:
        return max(descending_top_levels)
    return min_z


def _contour_band_elevation(topology_tree, node):
    node_elev = topology_tree.nodes[node]["data"]["elevation"]
    child_elevs = [
        topology_tree.nodes[child]["data"]["elevation"]
        for child in topology_tree.successors(node)
    ]
    lower_children = [elev for elev in child_elevs if elev < node_elev]
    higher_children = [elev for elev in child_elevs if elev > node_elev]

    if lower_children and not higher_children:
        return max(lower_children)
    return node_elev


def _has_lower_child(topology_tree, node):
    node_elev = topology_tree.nodes[node]["data"]["elevation"]
    return any(
        topology_tree.nodes[child]["data"]["elevation"] < node_elev
        for child in topology_tree.successors(node)
    )


def _geometry_above_level(cells, level, boundary_geom):
    geoms = [
        cell["geometry"]
        for cell in cells
        if cell["elevation"] >= level - 1e-9
    ]
    merged = _safe_polygonal_union(geoms)
    if merged.is_empty:
        return Polygon()
    return merged.intersection(boundary_geom)


def _geometries_equivalent(geom_a, geom_b, area_tolerance):
    if geom_a is None or geom_b is None:
        return False
    if geom_a.is_empty and geom_b.is_empty:
        return True
    if geom_a.is_empty or geom_b.is_empty:
        return False

    tolerance = max(area_tolerance, min(geom_a.area, geom_b.area) * 1e-6)
    return geom_a.symmetric_difference(geom_b).area <= tolerance


def _merge_slice_source_metadata(slice_data, group):
    source_ids = list(slice_data.get("source_ids") or [])
    source_layers = list(slice_data.get("source_layers") or [])
    estimated_elevations = list(slice_data.get("estimated_elevations") or [])

    for item in group:
        source_ids.extend(item["ids"])
        source_layers.extend(item["layers"])
        estimated_elevations.append(round(item.get("estimated_elevation", item["source_elevation"]), 3))

    source_ids, source_layers = _dedupe_id_layer_pairs(source_ids, source_layers)
    slice_data["source_ids"] = source_ids
    slice_data["source_layers"] = source_layers
    slice_data["estimated_elevations"] = estimated_elevations


def _group_elevation(group):
    return _representative_elevation(item["elevation"] for item in group)


def _representative_elevation(values):
    counts = defaultdict(int)
    for value in values:
        counts[round(float(value), 6)] += 1

    if not counts:
        return 0.0

    return min(counts.items(), key=lambda item: (-item[1], item[0]))[0]

def _clean_layer_geometry(geom, area_tolerance):
    if geom is None or geom.is_empty:
        return Polygon()

    if geom.geom_type == "Polygon":
        return _clean_polygon_geometry(geom, area_tolerance)

    if geom.geom_type == "GeometryCollection":
        polygon_parts = []
        for part in geom.geoms:
            cleaned = _clean_layer_geometry(part, area_tolerance)
            if cleaned.is_empty:
                continue
            if cleaned.geom_type == "Polygon":
                polygon_parts.append(cleaned)
            elif cleaned.geom_type == "MultiPolygon":
                polygon_parts.extend(cleaned.geoms)
        if not polygon_parts:
            return Polygon()
        return _safe_polygonal_union(polygon_parts, area_tolerance)

    if geom.geom_type != "MultiPolygon":
        return Polygon()

    parts = [
        cleaned
        for part in geom.geoms
        for cleaned in [_clean_polygon_geometry(part, area_tolerance)]
        if not cleaned.is_empty and cleaned.area > area_tolerance
    ]
    if not parts:
        return Polygon()

    if len(parts) == 1:
        return parts[0]

    return _safe_polygonal_union(parts, area_tolerance)


def _clean_polygon_geometry(geom, area_tolerance):
    if geom.is_empty or geom.area <= area_tolerance:
        return Polygon()

    exterior = _clean_ring_coords(geom.exterior.coords)
    if len(exterior) < 4:
        return Polygon()

    holes = []
    for ring in geom.interiors:
        ring_polygon = Polygon(ring)
        if ring_polygon.area > area_tolerance:
            cleaned_ring = _clean_ring_coords(ring.coords)
            if len(cleaned_ring) >= 4:
                holes.append(cleaned_ring)

    if len(holes) == len(geom.interiors):
        original_exterior = list(geom.exterior.coords)
        if len(exterior) == len(original_exterior) and all(
            _coords_close(a, b)
            for a, b in zip(exterior, original_exterior)
        ):
            return geom

    cleaned = Polygon(exterior, holes)
    if not cleaned.is_valid:
        cleaned = cleaned.buffer(0)
    if cleaned.is_empty or cleaned.area <= area_tolerance:
        return Polygon()
    return cleaned


def _clean_ring_coords(coords, tolerance=1e-7):
    points = list(coords)
    if len(points) < 4:
        return points

    if _coords_close(points[0], points[-1], tolerance):
        points = points[:-1]

    changed = True
    while changed and len(points) >= 4:
        changed = False
        deduped = []
        for point in points:
            if deduped and _coords_close(point, deduped[-1], tolerance):
                changed = True
                continue
            deduped.append(point)

        if len(deduped) >= 2 and _coords_close(deduped[0], deduped[-1], tolerance):
            deduped.pop()
            changed = True

        points = deduped
        if len(points) < 4:
            break

        spike_index = None
        for index, point in enumerate(points):
            previous_point = points[index - 1]
            next_point = points[(index + 1) % len(points)]
            if _coords_close(previous_point, next_point, tolerance):
                spike_index = index
                break

        if spike_index is not None:
            points.pop(spike_index)
            changed = True

    if len(points) < 3:
        return []

    points.append(points[0])
    return points


def _coords_close(point_a, point_b, tolerance=1e-7):
    return (
        abs(point_a[0] - point_b[0]) <= tolerance
        and abs(point_a[1] - point_b[1]) <= tolerance
    )


def _add_assembly_engraving_lines(slices, offset=5.0, slicing_mode=None):
    def _is_solid_row(row):
        return row.get("support_rule_version") == SOLID_VISIBILITY_RULE_VERSION

    def _is_hollow_row(row):
        return row.get("support_rule_version") == HOLLOW_SUPPORT_RULE_VERSION

    def _hollow_contact_lines(row):
        visible = row.get("visible_geom")
        support = row.get("support_geom")
        if (
            visible is None
            or visible.is_empty
            or support is None
            or support.is_empty
        ):
            return []

        contact = visible.boundary.intersection(support.boundary)
        if contact.is_empty:
            contact = visible.boundary.intersection(support)
        return _clean_engraving_lines(_extract_linework(contact))

    def _engraving_reference_geom(row):
        visible = row.get("visible_geom")
        if visible is not None and not visible.is_empty:
            return visible
        return row.get("slice_geom")

    def _engraving_target_geom(row):
        support = row.get("support_geom")
        if support is not None and not support.is_empty:
            return support
        return row.get("slice_geom")

    use_hollow_contact_rule = (
        slicing_mode == "hollow"
        or (slicing_mode is None and any(_is_hollow_row(row) for row in slices))
    )
    if use_hollow_contact_rule:
        for row in slices:
            lines = _hollow_contact_lines(row)
            if lines:
                row.setdefault("engraving_lines", [])
                row["engraving_lines"].extend(lines)
        return

    for index in range(len(slices) - 1):
        current_row = slices[index]
        upper_row = slices[index + 1]

        use_solid_rule = (
            slicing_mode == "solid"
            or (
                slicing_mode is None
                and (_is_solid_row(current_row) or _is_solid_row(upper_row))
            )
        )

        # Keep Option A stable: engraving follows physical cut geometries.
        # Option B is selected explicitly and stays role-aware (visible/support).
        if use_solid_rule:
            current_geom = current_row.get("slice_geom")
            upper_reference_geom = upper_row.get("slice_geom")
        else:
            current_geom = _engraving_target_geom(current_row)
            upper_reference_geom = _engraving_reference_geom(upper_row)

        upper_physical_geom = upper_row.get("slice_geom")
        if (
            current_geom is None
            or current_geom.is_empty
            or upper_reference_geom is None
            or upper_reference_geom.is_empty
            or upper_physical_geom is None
            or upper_physical_geom.is_empty
        ):
            continue

        guide_source = (
            upper_reference_geom.buffer(-offset, join_style=2)
            if offset > 0
            else upper_reference_geom
        )
        if guide_source.is_empty:
            continue

        guide = guide_source.boundary.intersection(current_geom).intersection(upper_physical_geom)
        lines = _clean_engraving_lines(_extract_linework(guide))

        slices[index].setdefault("engraving_lines", [])
        slices[index]["engraving_lines"].extend(lines)


def _extract_linework(geom):
    if geom.is_empty:
        return []

    if geom.geom_type in ("LineString", "LinearRing"):
        return [geom]

    if geom.geom_type == "MultiLineString":
        return list(geom.geoms)

    if geom.geom_type == "GeometryCollection":
        lines = []
        for item in geom.geoms:
            lines.extend(_extract_linework(item))
        return lines

    if geom.geom_type == "Polygon":
        return [geom.exterior]

    if geom.geom_type == "MultiPolygon":
        return [poly.exterior for poly in geom.geoms]

    return []


def _clean_engraving_lines(lines, minimum_length=2.0):
    if not lines:
        return []

    merged = unary_union(lines)
    if merged.geom_type == "MultiLineString":
        merged = linemerge(merged)

    return [
        line
        for line in _extract_linework(merged)
        if line.length >= minimum_length
    ]


def _resolve_topographic_precision(interval, elevation_precision, elevations=None, target_layer_count=None):
    if elevation_precision and elevation_precision > 0:
        return elevation_precision
    if interval and interval > 0:
        return interval
    inferred = _infer_topographic_precision(elevations or [], target_layer_count)
    if inferred is not None:
        return inferred
    return 0.0


def _infer_topographic_precision(elevations, target_layer_count=None):
    values = sorted(float(value) for value in elevations if value is not None)
    if len(values) < 2:
        return None

    span = max(values) - min(values)
    if span <= 1e-9:
        return None

    candidates = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    target = int(target_layer_count or 0)
    if target > 1:
        scored = []
        for candidate in candidates:
            count = int(round(span / candidate)) + 1
            scored.append((abs(count - target), abs((span / max(target - 1, 1)) - candidate), candidate))
        return min(scored)[2]

    scored = []
    for candidate in candidates:
        residuals = [
            abs(value - _snap_elevation(value, candidate))
            for value in values
        ]
        mean_residual = sum(residuals) / len(residuals)
        unique_count = len({_snap_elevation(value, candidate) for value in values})
        scored.append((mean_residual, -unique_count, candidate))
    return min(scored)[2]


def _base_elevation_from_values(elevations, precision):
    values = [float(value) for value in elevations if value is not None]
    if not values:
        return 0.0
    min_value = min(values)
    return _bucket_elevation(min_value, precision) if precision > 0 else round(min_value, 6)


def _spread_contours_to_target_levels(contours, base_elevation, interval, target_layer_count):
    # The requested layer count is only a validation target. Moving contours to
    # make the count match invents cotas and breaks the survey logic.
    _ = contours
    _ = base_elevation
    _ = interval
    _ = target_layer_count
    return


def _infer_topological_contour_elevations(contours, base_elevation, interval, adjacency_tolerance):
    if interval <= 0 or len(contours) < 2:
        return

    adjacency = _build_contour_adjacency(contours, adjacency_tolerance)
    if not adjacency:
        return

    by_id = {contour["id"]: contour for contour in contours}
    visited = set()
    for contour in contours:
        node = contour["id"]
        if node in visited:
            continue

        component = _collect_adjacency_component(node, adjacency)
        visited.update(component)
        if len(component) < 3:
            continue

        observations = _build_level_observations(component, by_id, base_elevation, interval)
        if not observations:
            continue

        assigned = _solve_component_level_indices(component, adjacency, observations)
        if not assigned:
            continue

        for contour_id, level_index in assigned.items():
            contour = by_id.get(contour_id)
            if contour is None:
                continue
            level = round(base_elevation + (level_index * interval), 6)
            if contour.get("elevation_source") == "forced":
                continue
            if abs(level - contour["source_elevation"]) > 1e-6:
                contour["elevation_source"] = "topology"
            contour["elevation"] = level
            contour["source_elevation"] = level


def _build_contour_adjacency(contours, adjacency_tolerance):
    adjacency = defaultdict(list)
    min_length = max(float(adjacency_tolerance or 0.0), 1e-6)

    for index, contour in enumerate(contours):
        geom = contour["geometry"]
        for other in contours[index + 1:]:
            other_geom = other["geometry"]
            shared = geom.boundary.intersection(other_geom.boundary)
            shared_length = 0.0 if shared.is_empty else shared.length
            if shared_length <= min_length:
                continue

            adjacency[contour["id"]].append(other["id"])
            adjacency[other["id"]].append(contour["id"])

    return adjacency


def _collect_adjacency_component(start, adjacency):
    stack = [start]
    component = set()
    while stack:
        node = stack.pop()
        if node in component:
            continue
        component.add(node)
        stack.extend(neighbor for neighbor in adjacency.get(node, []) if neighbor not in component)
    return component


def _build_level_observations(component, by_id, base_elevation, interval):
    observations = {}
    for contour_id in component:
        contour = by_id.get(contour_id)
        if contour is None:
            continue

        source = contour.get("elevation_source")
        source_elevation = contour.get("source_elevation")
        if source_elevation is None:
            continue

        level_index = int(round((source_elevation - base_elevation) / interval))
        if level_index < 0:
            continue

        if source == "forced":
            weight = 1000.0
        elif source in {"direct", "point"}:
            weight = _topology_observation_weight(contour)
        elif source == "nearest":
            weight = 0.25
        else:
            weight = 0.5

        observations[contour_id] = (level_index, weight)
    return observations


def _topology_observation_weight(contour):
    distance = contour.get("point_match_distance")
    if distance is None:
        return 1.0
    return max(0.5, min(8.0, 0.25 / (float(distance) + 0.02)))


def _solve_component_level_indices(component, adjacency, observations):
    root = min(
        component,
        key=lambda node: (
            observations.get(node, (10**9, 0.0))[0],
            _id_sort_key(node),
        ),
    )
    parent = {root: None}
    order = [root]
    queue = [root]
    while queue:
        node = queue.pop(0)
        for neighbor in sorted(adjacency.get(node, []), key=_id_sort_key):
            if neighbor in parent or neighbor not in component:
                continue
            parent[neighbor] = node
            order.append(neighbor)
            queue.append(neighbor)

    children = defaultdict(list)
    for node, parent_node in parent.items():
        if parent_node is not None:
            children[parent_node].append(node)

    observed_indices = [value for value, _ in observations.values()]
    min_label = max(0, min(observed_indices) - len(component))
    max_label = max(observed_indices) + len(component)
    labels = list(range(min_label, max_label + 1))
    if not labels:
        return {}

    child_choice = {}
    costs = {}
    for node in reversed(order):
        costs[node] = {}
        for label in labels:
            cost = _node_level_cost(node, label, observations)
            for child in children.get(node, []):
                best_label = min(
                    labels,
                    key=lambda child_label: (
                        costs[child][child_label] + _level_edge_cost(label, child_label),
                        child_label,
                    ),
                )
                child_choice[(node, child, label)] = best_label
                cost += costs[child][best_label] + _level_edge_cost(label, best_label)
            costs[node][label] = cost

    root_observation = observations.get(root)
    root_label = min(
        labels,
        key=lambda label: (
            costs[root][label]
            + (1000.0 * (label - root_observation[0]) ** 2 if root_observation else 0.0),
            label,
        ),
    )

    assigned = {root: root_label}
    for node in order:
        for child in children.get(node, []):
            assigned[child] = child_choice[(node, child, assigned[node])]

    return assigned


def _node_level_cost(node, label, observations):
    observation = observations.get(node)
    if observation is None:
        return 0.0
    observed_label, weight = observation
    return weight * (label - observed_label) ** 2


def _level_edge_cost(label_a, label_b):
    difference = abs(label_a - label_b)
    if difference == 1:
        return 0.0
    if difference == 0:
        return 2.0
    return 80.0 * (difference - 1) ** 2


def _merge_duplicate_contours(contours, area_tolerance):
    merged = []
    used = [False] * len(contours)

    for index, contour in enumerate(contours):
        if used[index]:
            continue

        used[index] = True
        group = [contour]
        for other_index in range(index + 1, len(contours)):
            if used[other_index]:
                continue

            other = contours[other_index]
            tolerance = max(area_tolerance, min(contour["geometry"].area, other["geometry"].area) * 0.001)
            if (
                _same_elevation(contour["elevation"], other["elevation"])
                and contour["geometry"].symmetric_difference(other["geometry"]).area <= tolerance
            ):
                used[other_index] = True
                group.append(other)

        merged.append(_merge_contour_group(group))

    return merged


def _same_elevation(value_a, value_b):
    return abs(float(value_a) - float(value_b)) <= 1e-6


def _merge_contour_group(group):
    if len(group) == 1:
        return group[0]

    ids = []
    layers = []
    elevations = []
    estimated_elevations = []
    elevation_sources = []
    geoms = []
    for item in group:
        ids.extend(item["ids"])
        layers.extend(item["layers"])
        elevations.append(item["elevation"])
        estimated_elevations.append(item.get("estimated_elevation", item["source_elevation"]))
        elevation_sources.append(item.get("elevation_source"))
        geoms.append(item["geometry"])

    elevation = _representative_elevation(elevations)
    return {
        "id": ids[0] if ids else None,
        "ids": ids,
        "layers": layers,
        "geometry": unary_union(geoms),
        "elevation": elevation,
        "source_elevation": elevation,
        "estimated_elevation": _representative_elevation(estimated_elevations),
        "elevation_source": _representative_source(elevation_sources),
    }


def _representative_source(sources):
    cleaned = [source for source in sources if source]
    if not cleaned:
        return None
    if "forced" in cleaned:
        return "forced"
    if "direct" in cleaned:
        return "direct"
    return cleaned[0]


def _group_contours_by_elevation(contours, elevation_precision):
    groups_by_level = defaultdict(list)
    for contour in contours:
        level = _snap_elevation(contour["elevation"], elevation_precision)
        contour["elevation"] = level
        contour["source_elevation"] = level
        groups_by_level[level].append(contour)

    return [
        groups_by_level[level]
        for level in sorted(groups_by_level)
    ]


def _build_contour_sources_by_id(contour_groups):
    sources = {}
    for group in contour_groups:
        for item in group:
            ids = item.get("ids") or []
            layers = item.get("layers") or []
            source_elevation = item.get("source_elevation")
            fallback_layer = layers[0] if layers else None
            for idx, contour_id in enumerate(ids):
                key = contour_id
                layer = layers[idx] if idx < len(layers) else fallback_layer
                sources[key] = {
                    "source_elevation": source_elevation,
                    "layer": layer,
                }
    return sources


def _dedupe_id_layer_pairs(source_ids, source_layers):
    if not source_ids:
        return [], []

    paired = []
    for idx, contour_id in enumerate(source_ids):
        layer = source_layers[idx] if idx < len(source_layers) else None
        paired.append((contour_id, layer))

    seen = set()
    unique = []
    for contour_id, layer in paired:
        if contour_id in seen:
            continue
        seen.add(contour_id)
        unique.append((contour_id, layer))

    return [item[0] for item in unique], [item[1] for item in unique]


def _ensure_unique_and_complete_ids(slices, contour_sources_by_id, min_z):
    if not contour_sources_by_id:
        return

    assignments = {}
    id_to_source = contour_sources_by_id

    for slice_index, slice_data in enumerate(slices):
        if slice_index == 0:
            continue
        source_ids = slice_data.get("source_ids") or []
        source_layers = slice_data.get("source_layers") or []
        for idx, contour_id in enumerate(source_ids):
            layer = source_layers[idx] if idx < len(source_layers) else None
            assignments.setdefault(contour_id, []).append((slice_index, layer))

    by_slice = {index: [] for index in range(1, len(slices))}
    for contour_id, rows in assignments.items():
        if len(rows) == 1:
            slice_index, layer = rows[0]
            by_slice[slice_index].append((contour_id, layer))
            continue

        target_elev = id_to_source.get(contour_id, {}).get("source_elevation")
        best = min(
            rows,
            key=lambda row: (
                abs((slices[row[0]].get("source_elevation") or min_z) - target_elev)
                if target_elev is not None
                else row[0],
                row[0],
            ),
        )
        by_slice[best[0]].append((contour_id, best[1]))

    for slice_index in range(1, len(slices)):
        current_pairs = by_slice.get(slice_index, [])
        current_pairs.sort(key=lambda row: _id_sort_key(row[0]))
        deduped = []
        seen = set()
        for contour_id, layer in current_pairs:
            if contour_id in seen:
                continue
            seen.add(contour_id)
            deduped.append((contour_id, layer))
        slices[slice_index]["source_ids"] = [row[0] for row in deduped]
        slices[slice_index]["source_layers"] = [row[1] for row in deduped]

    used_ids = set()
    for slice_data in slices[1:]:
        used_ids.update(slice_data.get("source_ids") or [])


def _id_sort_key(value):
    text = str(value).strip()
    try:
        return (0, int(text))
    except (TypeError, ValueError):
        return (1, text.lower())


def _renumber_layers(slices, interval):
    _ = interval
    for index, slice_data in enumerate(slices):
        if index == 0:
            slice_data["id"] = "base"
        else:
            slice_data["id"] = f"level_{index}"


def _resolve_contour_override(manual_overrides, contour_id):
    if contour_id in manual_overrides:
        raw = manual_overrides[contour_id]
    else:
        raw = manual_overrides.get(str(contour_id), {})

    keep = _to_bool(raw.get("keep"), default=True)
    forced = _to_float(raw.get("forced_elevation"))
    forced_source = _to_float(raw.get("forced_source_elevation"))
    forced_relative = _to_float(raw.get("forced_relative_elevation"))
    forced_layer = _to_int(raw.get("forced_layer_index"))
    return {
        "keep": keep,
        "forced_elevation": forced,
        "forced_source_elevation": forced_source,
        "forced_relative_elevation": forced_relative,
        "forced_layer_index": forced_layer,
    }


def _resolve_override_source_elevation(override, base_elevation, interval):
    forced_source = override.get("forced_source_elevation")
    if forced_source is not None:
        return forced_source

    # Backward compatibility with previous UI/API name.
    forced_absolute = override.get("forced_elevation")
    if forced_absolute is not None:
        return forced_absolute

    forced_relative = override.get("forced_relative_elevation")
    if forced_relative is not None:
        return base_elevation + forced_relative

    forced_layer = override.get("forced_layer_index")
    if forced_layer is not None and forced_layer >= 0 and interval > 0:
        return base_elevation + (forced_layer * interval)

    return None


def _to_bool(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0

    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "sim"}:
        return True
    if text in {"0", "false", "f", "no", "n", "nao", "não"}:
        return False
    return default


def _to_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(value):
    number = _to_float(value)
    if number is None:
        return None
    if not number.is_integer():
        return None
    return int(number)


def _estimate_contour_elevation(geom, points):
    samples = _select_boundary_point_samples(geom, points)
    zs = [z for _, z in samples]
    if not zs:
        return None

    return sum(zs) / len(zs)


def _estimate_contour_topographic_elevation(
    geom,
    points,
    precision,
    base_elevation=None,
    max_distance=None,
):
    samples = _select_boundary_point_samples(geom, points, max_distance=max_distance)
    if not samples:
        return None

    buckets = defaultdict(list)
    for distance, z in samples:
        level = (
            _bucket_elevation(z, precision, base_elevation=base_elevation)
            if precision > 0
            else round(float(z), 6)
        )
        buckets[level].append(distance)

    return min(
        buckets.items(),
        key=lambda item: (
            -len(item[1]),
            min(item[1]),
            sum(item[1]) / len(item[1]),
            -item[0],
        ),
    )[0]


def _nearest_boundary_point_distance(geom, points):
    if not points:
        return None
    return min(
        geom.exterior.distance(point["geometry"])
        for point in points
    )


def _select_boundary_point_samples(geom, points, max_distance=None):
    distances = sorted(
        (geom.exterior.distance(point["geometry"]), point["z"])
        for point in points
    )
    return _select_boundary_point_samples_from_distances(
        distances,
        geom,
        max_distance=max_distance,
    )


def _select_boundary_point_zs(distance_z_pairs, geom):
    return [z for _, z in _select_boundary_point_samples_from_distances(distance_z_pairs, geom)]


def _select_boundary_point_samples_from_distances(distance_z_pairs, geom, max_distance=None):
    if not distance_z_pairs:
        return []

    if max_distance is not None and max_distance > 0:
        return [
            (distance, z)
            for distance, z in distance_z_pairs
            if distance <= max_distance
        ]

    minx, miny, maxx, maxy = geom.bounds
    span = max(maxx - minx, maxy - miny, 1.0)
    base_tol = max(span * 1e-6, 1e-6)

    min_distance = distance_z_pairs[0][0]
    tolerance = max(base_tol, (min_distance * 2.5) + base_tol)
    samples = [(distance, z) for distance, z in distance_z_pairs if distance <= tolerance]

    if not samples:
        samples = [distance_z_pairs[0]]
    return samples


def _snap_elevation(elevation, precision):
    if precision <= 0:
        return elevation
    scaled = elevation / precision
    if scaled >= 0:
        snapped = math.floor(scaled + 0.5) * precision
    else:
        snapped = math.ceil(scaled - 0.5) * precision
    return round(snapped, 6)


def _bucket_elevation(elevation, precision, base_elevation=None):
    if precision <= 0:
        return round(float(elevation), 6)

    scaled = (float(elevation) + 1e-9) / precision
    bucket = math.floor(scaled) * precision
    bucket = round(bucket, 6)
    if base_elevation is not None and bucket < base_elevation:
        return round(float(base_elevation), 6)
    return bucket


def _triangulate_points(points):
    z_by_xy = {
        _coord_key(point["x"], point["y"]): point["z"]
        for point in points
    }
    multipoint = MultiPoint([(point["x"], point["y"]) for point in points])
    triangles = []

    for triangle in triangulate(multipoint):
        coords = list(triangle.exterior.coords)[:-1]
        values = []
        for x, y in coords:
            z = z_by_xy.get(_coord_key(x, y))
            if z is None:
                break
            values.append(z)

        if len(coords) == 3 and len(values) == 3:
            triangles.append((coords, values))

    return triangles


def _coord_key(x, y):
    return (round(x, 8), round(y, 8))


def _area_above_level(triangles, level):
    pieces = []
    for coords, values in triangles:
        piece = _clip_triangle_above_level(coords, values, level)
        if piece is not None and piece.is_valid and piece.area > 0:
            pieces.append(piece)

    if not pieces:
        return Polygon()

    return unary_union(pieces)


def _clip_triangle_above_level(coords, values, level):
    above = [z >= level for z in values]
    if all(above):
        return Polygon(coords)
    if not any(above):
        return None

    output = []
    for i in range(3):
        j = (i + 1) % 3
        point_i = coords[i]
        point_j = coords[j]
        z_i = values[i]
        z_j = values[j]

        if z_i >= level:
            output.append(point_i)

        if (z_i >= level) != (z_j >= level):
            t = (level - z_i) / (z_j - z_i)
            output.append((
                point_i[0] + t * (point_j[0] - point_i[0]),
                point_i[1] + t * (point_j[1] - point_i[1]),
            ))

    if len(output) < 3:
        return None

    center_x = sum(x for x, _ in output) / len(output)
    center_y = sum(y for _, y in output) / len(output)
    output = sorted(output, key=lambda p: math.atan2(p[1] - center_y, p[0] - center_x))
    return Polygon(output)


def _scale_geom(geom, scale_factor):
    if scale_factor == 1.0 or geom.is_empty:
        return geom
    return scale(geom, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
