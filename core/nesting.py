import math

import shapely.geometry as sg
from shapely.affinity import rotate, translate


DEFAULT_ROTATION_STEP = 90
USAGE_ROTATION_STEP = 30
MEDIUM_ROTATION_STEP = 45
FINE_ROTATION_STEP = 15
EPSILON = 1e-6
MIN_GEOMETRY_AREA = 0.1
MAX_OPTIONAL_SPLIT_CANDIDATES = 2
MAX_AXIS_CANDIDATES = 24
MAX_CANDIDATE_POSITIONS = 80
MAX_AXIS_CANDIDATES_HIGH_EFFORT = 40
MAX_CANDIDATE_POSITIONS_HIGH_EFFORT = 180


def split_polygon_if_needed(slice_data, bed_w, bed_h):
    """
    Divide uma fatia apenas quando nenhuma rotacao de nesting a consegue fazer caber.
    """
    geom = slice_data["slice_geom"]
    if geom.is_empty:
        return []
    source_slice_id = slice_data.get("source_slice_id", slice_data["id"])
    label_areas = _label_areas_for_piece(slice_data)

    if _fits_any_rotation(geom, bed_w, bed_h, _rotation_angles_for_geom(geom, 5)):
        piece = dict(slice_data)
        piece["source_slice_id"] = source_slice_id
        piece["model_geom"] = slice_data.get("model_geom", geom)
        piece["label_areas"] = label_areas
        piece["was_split"] = piece.get("was_split", False)
        return [piece]

    pieces = []
    boxes = _best_split_boxes(geom, bed_w, bed_h)

    for index, cut_box in enumerate(boxes):
        new_geom = geom.intersection(cut_box)
        if new_geom.is_empty or new_geom.area <= MIN_GEOMETRY_AREA:
            continue

        new_engravings = []
        for line in slice_data["engraving_lines"]:
            new_line = line.intersection(cut_box)
            if new_line.is_empty:
                continue

            if new_line.geom_type == "MultiLineString":
                new_engravings.extend(new_line.geoms)
            elif new_line.geom_type in ("LineString", "LinearRing"):
                new_engravings.append(new_line)

        new_slice = {
            "id": f"{slice_data['id']}_{index}",
            "source_slice_id": source_slice_id,
            "elevation": slice_data["elevation"],
            "source_elevation": slice_data.get("source_elevation"),
            "slice_geom": new_geom,
            "model_geom": new_geom,
            "engraving_lines": new_engravings,
            "label_areas": _clip_label_areas(label_areas, new_geom),
            "was_split": True,
            "split_reason": slice_data.get("split_reason", "required"),
        }
        pieces.extend(split_polygon_if_needed(new_slice, bed_w, bed_h))

    return pieces


def _split_polygon_once(slice_data, bed_w, bed_h):
    geom = slice_data["slice_geom"]
    if geom.is_empty or geom.area <= MIN_GEOMETRY_AREA:
        return []

    pieces = []
    boxes = _best_split_boxes(geom, bed_w, bed_h)
    source_slice_id = slice_data.get("source_slice_id", slice_data["id"])
    label_areas = _label_areas_for_piece(slice_data)

    for index, cut_box in enumerate(boxes):
        new_geom = geom.intersection(cut_box)
        if new_geom.is_empty or new_geom.area <= MIN_GEOMETRY_AREA:
            continue

        new_engravings = []
        for line in slice_data.get("engraving_lines", []):
            new_line = line.intersection(cut_box)
            if new_line.is_empty:
                continue

            if new_line.geom_type == "MultiLineString":
                new_engravings.extend(new_line.geoms)
            elif new_line.geom_type in ("LineString", "LinearRing"):
                new_engravings.append(new_line)

        new_slice = {
            "id": f"{slice_data['id']}_{index}",
            "source_slice_id": source_slice_id,
            "elevation": slice_data["elevation"],
            "source_elevation": slice_data.get("source_elevation"),
            "slice_geom": new_geom,
            "model_geom": new_geom,
            "engraving_lines": new_engravings,
            "label_areas": _clip_label_areas(label_areas, new_geom),
            "was_split": True,
            "split_reason": slice_data.get("split_reason", "optional"),
        }
        pieces.extend(split_polygon_if_needed(new_slice, bed_w, bed_h))

    if len(pieces) <= 1:
        item = dict(slice_data)
        item["source_slice_id"] = source_slice_id
        item["model_geom"] = item.get("model_geom", item["slice_geom"])
        item["label_areas"] = label_areas
        item["was_split"] = item.get("was_split", False)
        return [item]

    return pieces


def perform_nesting(
    slices,
    bed_w=600,
    bed_h=400,
    margin=5,
    priority="preserve_shapes",
    target_utilization=65,
    preserve_whole_percent=100,
    split_from_sheet_count=0,
):
    """
    Organiza as fatias usando a forma real dos poligonos.

    Este metodo testa rotacoes reais da geometria e valida sempre que a peca
    fica dentro da folha e afastada das outras pecas pela margem indicada.
    """
    usable_w = max(bed_w - 2 * margin, 1)
    usable_h = max(bed_h - 2 * margin, 1)
    regular_slices, repeated_groups = _split_repeated_slice_groups(slices)
    required_pieces = []
    whole_components = []

    for slice_data in regular_slices:
        geom = slice_data["slice_geom"]
        if geom.is_empty or geom.area < 0.1:
            continue

        for component in _explode_multipolygon_piece(slice_data):
            if _fits_any_rotation(
                component["slice_geom"],
                usable_w,
                usable_h,
                _rotation_angles_for_geom(component["slice_geom"], 5),
            ):
                whole_components.append(component)
                continue

            for piece in split_polygon_if_needed(component, usable_w, usable_h):
                required_pieces.extend(_explode_multipolygon_piece(piece))

    if not required_pieces and not whole_components and not repeated_groups:
        return {}

    effort_percent = _clamp(float(target_utilization or 0), 0.0, 100.0)
    split_from_sheet_count = max(0, int(split_from_sheet_count or 0))

    best = None
    last_error = None
    if required_pieces or whole_components:
        # 1) Base rule: keep pieces whole when they fit.
        for pieces, optional_split_count in _candidate_piece_sets(
            required_pieces,
            whole_components,
            usable_w,
            usable_h,
            priority=priority,
            target_utilization=target_utilization,
            preserve_whole_percent=preserve_whole_percent,
            include_optional_splits=False,
        ):
            if not pieces:
                continue

            try:
                sheets = _nest_with_best_order(
                    pieces,
                    bed_w,
                    bed_h,
                    margin,
                    priority=priority,
                    effort_percent=effort_percent,
                )
                _validate_sheets(sheets, bed_w, bed_h, margin)
            except ValueError as error:
                last_error = error
                continue

            score = _nesting_candidate_score(
                sheets,
                bed_w,
                bed_h,
                margin,
                priority=priority,
                target_utilization=target_utilization,
                optional_split_count=optional_split_count,
            )
            if best is None or score < best[0]:
                best = (score, sheets)
            if _base_candidate_is_good_enough(
                pieces,
                sheets,
                bed_w,
                bed_h,
                margin,
                priority=priority,
                target_utilization=target_utilization,
            ):
                break

        # 2) Optional rule: only test split candidates when requested and when
        # base solution reaches the configured sheet-count threshold.
        if (
            best is not None
            and split_from_sheet_count > 0
            and len(best[1]) >= split_from_sheet_count
        ):
            for pieces, optional_split_count in _candidate_piece_sets(
                required_pieces,
                whole_components,
                usable_w,
                usable_h,
                priority=priority,
                target_utilization=target_utilization,
                preserve_whole_percent=preserve_whole_percent,
                include_optional_splits=True,
            ):
                if optional_split_count <= 0 or not pieces:
                    continue

                try:
                    sheets = _nest_with_best_order(
                        pieces,
                        bed_w,
                        bed_h,
                        margin,
                        priority=priority,
                        effort_percent=effort_percent,
                    )
                    _validate_sheets(sheets, bed_w, bed_h, margin)
                except ValueError as error:
                    last_error = error
                    continue

                score = _nesting_candidate_score(
                    sheets,
                    bed_w,
                    bed_h,
                    margin,
                    priority=priority,
                    target_utilization=target_utilization,
                    optional_split_count=optional_split_count,
                )
                if best is None or score < best[0]:
                    best = (score, sheets)

    if best is None and (required_pieces or whole_components):
        if last_error is not None:
            raise last_error
        return {}

    sheets = list(best[1]) if best is not None else []
    for group in repeated_groups:
        _append_repeated_group_sheets(sheets, group, bed_w, bed_h, margin)

    return {index: sheet for index, sheet in enumerate(sheets)}


def _split_repeated_slice_groups(slices):
    regular = []
    repeated = []
    for slice_data in slices:
        count = _repeat_count(slice_data)
        if slice_data.get("is_zero_base") and count > 1:
            repeated.append(slice_data)
        else:
            regular.append(slice_data)
    return regular, repeated


def _repeat_count(slice_data):
    try:
        return max(1, int(slice_data.get("repeat_count", 1) or 1))
    except (TypeError, ValueError):
        return 1


def _append_repeated_group_sheets(sheets, group, bed_w, bed_h, margin):
    layout = _best_repeated_group_layout(group, bed_w, bed_h, margin)
    if layout is None:
        expanded = _expanded_repeated_group_slices(group)
        nested = perform_nesting(
            [dict(item, repeat_count=1, is_zero_base=False) for item in expanded],
            bed_w=bed_w,
            bed_h=bed_h,
            margin=margin,
            priority="preserve_shapes",
            target_utilization=100,
            preserve_whole_percent=100,
        )
        sheets.extend(nested[index] for index in sorted(nested))
        return

    rotated, cols, rows, step_x, step_y = layout
    per_sheet = max(cols * rows, 1)
    current_sheet = []
    for index, item in enumerate(_expanded_repeated_group_slices(group)):
        if index % per_sheet == 0:
            if current_sheet:
                sheets.append(current_sheet)
            current_sheet = []

        slot = index % per_sheet
        col = slot % cols
        row = slot // cols
        repeated_rotated = dict(rotated)
        repeated_rotated.update({
            "id": item["id"],
            "elevation": item["elevation"],
            "source_elevation": item.get("source_elevation"),
        })
        current_sheet.append(
            _translate_piece(
                repeated_rotated,
                margin + col * step_x,
                margin + row * step_y,
            )
        )

    if current_sheet:
        sheets.append(current_sheet)


def _best_repeated_group_layout(group, bed_w, bed_h, margin):
    best = None
    angles = _merge_rotation_angles(_rotation_angles(90), _fit_rotation_angles(group["slice_geom"]))
    for angle in angles:
        rotated = _rotate_piece_to_origin(group, angle)
        rotated_bounds = _piece_bounds(rotated)
        if not _bounds_can_fit(rotated_bounds, bed_w, bed_h, margin):
            continue

        minx, miny, maxx, maxy = rotated_bounds
        width = max(maxx - minx, EPSILON)
        height = max(maxy - miny, EPSILON)
        usable_w = max(bed_w - 2 * margin, EPSILON)
        usable_h = max(bed_h - 2 * margin, EPSILON)
        cols = max(1, int(math.floor((usable_w + margin) / (width + margin))))
        rows = max(1, int(math.floor((usable_h + margin) / (height + margin))))
        count = cols * rows
        score = (-count, width * height, angle)
        if best is None or score < best[0]:
            best = (score, rotated, cols, rows, width + margin, height + margin)

    if best is None:
        return None
    return best[1:]


def _expanded_repeated_group_slices(group):
    count = _repeat_count(group)
    try:
        start = float(group.get("repeat_start_elevation", group.get("source_elevation", 0.0)) or 0.0)
    except (TypeError, ValueError):
        start = 0.0
    try:
        interval = float(group.get("repeat_interval", 0.0) or 0.0)
    except (TypeError, ValueError):
        interval = 0.0

    for index in range(count):
        elevation = round(start + index * interval, 3)
        item = dict(group)
        item["id"] = f"{group.get('id', 'base')}_{index + 1}"
        item["elevation"] = elevation
        item["source_elevation"] = elevation
        item["repeat_count"] = 1
        item["repeat_index"] = index + 1
        item["repeat_total"] = count
        yield item


def _candidate_piece_sets(
    required_pieces,
    whole_components,
    usable_w,
    usable_h,
    priority="preserve_shapes",
    target_utilization=65,
    preserve_whole_percent=100,
    include_optional_splits=False,
):
    whole_pieces = []
    for component in whole_components:
        whole_pieces.extend(_explode_multipolygon_piece(component))

    yield required_pieces + whole_pieces, 0

    if not include_optional_splits:
        return

    budget = _optional_split_budget(
        len(whole_components),
        priority=priority,
        target_utilization=target_utilization,
        preserve_whole_percent=preserve_whole_percent,
    )
    if budget <= 0:
        return

    ordered_components = sorted(
        whole_components,
        key=lambda component: _piece_sort_key(component),
    )
    split_ids = set()
    split_cache = {}

    for component in ordered_components[:budget]:
        split_pieces = _split_polygon_once(component, usable_w, usable_h)
        split_pieces = [
            piece
            for split_piece in split_pieces
            for piece in _explode_multipolygon_piece(split_piece)
        ]
        if len(split_pieces) <= 1:
            continue

        split_cache[id(component)] = split_pieces
        split_ids.add(id(component))

        candidate = list(required_pieces)
        for item in whole_components:
            if id(item) in split_ids:
                candidate.extend(split_cache[id(item)])
            else:
                candidate.extend(_explode_multipolygon_piece(item))
        yield candidate, len(split_ids)


def _optional_split_budget(
    whole_count,
    priority="preserve_shapes",
    target_utilization=65,
    preserve_whole_percent=100,
):
    if whole_count <= 0:
        return 0

    if _normalize_priority(priority) == "maximize_usage":
        if whole_count > 60:
            return 0
        effort = _clamp(float(target_utilization or 0), 0.0, 100.0)
        if effort < 35:
            return 0
        if effort < 70:
            return min(1, whole_count)
        return min(MAX_OPTIONAL_SPLIT_CANDIDATES, whole_count)

    preserve = _clamp(float(preserve_whole_percent or 0), 0.0, 100.0)
    allowed_ratio = (100.0 - preserve) / 100.0
    if allowed_ratio <= 0:
        return 0
    return min(MAX_OPTIONAL_SPLIT_CANDIDATES, whole_count, math.floor(whole_count * allowed_ratio))


def _base_candidate_is_good_enough(
    pieces,
    sheets,
    bed_w,
    bed_h,
    margin,
    priority="preserve_shapes",
    target_utilization=65,
):
    usable_area = max((bed_w - 2 * margin) * (bed_h - 2 * margin), 1.0)
    total_area = sum(piece["slice_geom"].area for piece in pieces)
    sheet_lower_bound = max(1, int(math.ceil(total_area / usable_area)))
    if len(sheets) <= sheet_lower_bound:
        return True

    if _normalize_priority(priority) == "maximize_usage":
        target = _clamp(float(target_utilization or 0), 0.0, 100.0) / 100.0
        utilization = total_area / max(len(sheets) * usable_area, 1.0)
        return utilization >= target

    return False


def _nesting_candidate_score(
    sheets,
    bed_w,
    bed_h,
    margin,
    priority="preserve_shapes",
    target_utilization=65,
    optional_split_count=0,
):
    usable_area = max((bed_w - 2 * margin) * (bed_h - 2 * margin), 1.0)
    total_area = sum(_piece_area(piece) for sheet in sheets for piece in sheet)
    utilization = total_area / max(len(sheets) * usable_area, 1.0)
    target = _clamp(float(target_utilization or 0), 0.0, 100.0) / 100.0
    utilization_gap = max(target - utilization, 0.0)

    if _normalize_priority(priority) == "maximize_usage":
        return (
            len(sheets),
            utilization_gap,
            -utilization,
            optional_split_count,
            _sheet_total_bbox_area(sheets),
        )

    return (
        len(sheets),
        optional_split_count,
        _sheet_total_bbox_area(sheets),
        -utilization,
    )


def _sheet_total_bbox_area(sheets):
    return sum(_sheet_used_bbox_area(sheet) for sheet in sheets)


def _nest_with_best_order(
    pieces,
    bed_w,
    bed_h,
    margin,
    priority="preserve_shapes",
    effort_percent=65,
):
    best = None
    usable_area = max((bed_w - 2 * margin) * (bed_h - 2 * margin), 1.0)
    total_area = sum(piece["slice_geom"].area for piece in pieces)
    sheet_lower_bound = max(1, int(math.ceil(total_area / usable_area)))
    for ordered_pieces in _candidate_piece_orders(
        pieces,
        priority=priority,
        effort_percent=effort_percent,
    ):
        sheets = _pack_pieces_greedy(
            ordered_pieces,
            bed_w,
            bed_h,
            margin,
            priority=priority,
            effort_percent=effort_percent,
        )
        score = _sheets_score(sheets, bed_w, bed_h, priority=priority)
        if best is None or score < best[0]:
            best = (score, sheets)
            if len(sheets) <= sheet_lower_bound:
                break

    if best is None:
        return []
    return best[1]


def _pack_pieces_greedy(
    pieces,
    bed_w,
    bed_h,
    margin,
    priority="preserve_shapes",
    effort_percent=65,
):
    sheets = []
    high_usage = _normalize_priority(priority) == "maximize_usage"
    effort = _clamp(float(effort_percent or 0), 0.0, 100.0)
    rotation_step = DEFAULT_ROTATION_STEP
    fallback_step = MEDIUM_ROTATION_STEP
    if high_usage:
        if effort >= 75:
            rotation_step = USAGE_ROTATION_STEP
            fallback_step = FINE_ROTATION_STEP
        elif effort >= 45:
            rotation_step = MEDIUM_ROTATION_STEP
            fallback_step = USAGE_ROTATION_STEP
    rotation_angles = _rotation_angles(rotation_step)
    fallback_angles = _rotation_angles(fallback_step) if fallback_step else []

    for piece in pieces:
        placement = _find_best_placement(
            piece,
            sheets,
            bed_w,
            bed_h,
            margin,
            rotation_angles,
            priority=priority,
            effort_percent=effort,
        )
        if placement is None and fallback_angles:
            placement = _find_best_placement(
                piece,
                sheets,
                bed_w,
                bed_h,
                margin,
                fallback_angles,
                priority=priority,
                effort_percent=effort,
            )

        if placement is None:
            raise ValueError(f"A peca {piece['id']} nao cabe na folha com a margem definida.")

        sheet_index, placed_piece = placement
        if sheet_index == len(sheets):
            sheets.append([])
        sheets[sheet_index].append(placed_piece)

    return sheets


def _candidate_piece_orders(pieces, priority="preserve_shapes", effort_percent=65):
    indexed = list(enumerate(pieces))

    def bounds(piece):
        minx, miny, maxx, maxy = piece["slice_geom"].bounds
        return maxx - minx, maxy - miny

    order_keys = [
        lambda pair: _piece_sort_key(pair[1], pair[0]),
    ]
    effort = _clamp(float(effort_percent or 0), 0.0, 100.0)
    if _normalize_priority(priority) == "maximize_usage" and effort >= 35:
        order_piece_limit = 60
        if effort >= 85:
            order_piece_limit = 140
        elif effort >= 65:
            order_piece_limit = 100
        elif effort >= 45:
            order_piece_limit = 80
        if len(indexed) > order_piece_limit:
            order_piece_limit = 0
    else:
        order_piece_limit = 0

    if order_piece_limit > 0:
        order_keys.extend([
            lambda pair: (-max(bounds(pair[1])), -pair[1]["slice_geom"].area, pair[0]),
            lambda pair: (-pair[1]["slice_geom"].area, _bounds_aspect(pair[1]["slice_geom"]), pair[0]),
        ])
        if len(indexed) <= 60 and effort >= 75:
            order_keys.append(
                lambda pair: (_bounds_aspect(pair[1]["slice_geom"]), -pair[1]["slice_geom"].area, pair[0])
            )

    seen = set()
    for key in order_keys:
        ordered = sorted(indexed, key=key)
        order_signature = tuple(index for index, _ in ordered)
        if order_signature in seen:
            continue
        seen.add(order_signature)
        yield [piece for _, piece in ordered]


def _sheets_score(sheets, bed_w, bed_h, priority="preserve_shapes"):
    used_bbox_area = sum(_sheet_used_bbox_area(sheet) for sheet in sheets)
    tallest_used_h = max((_sheet_used_height(sheet) for sheet in sheets), default=0.0)
    widest_used_w = max((_sheet_used_width(sheet) for sheet in sheets), default=0.0)
    total_area = sum(_piece_area(piece) for sheet in sheets for piece in sheet)
    utilization = total_area / max(len(sheets) * bed_w * bed_h, 1.0)
    if _normalize_priority(priority) == "maximize_usage":
        return (
            len(sheets),
            -utilization,
            used_bbox_area,
            widest_used_w,
            tallest_used_h,
        )
    return (
        len(sheets),
        used_bbox_area,
        tallest_used_h,
        widest_used_w,
    )


def _sheet_used_bbox_area(sheet):
    if not sheet:
        return 0.0
    minx = min(_piece_bounds(piece)[0] for piece in sheet)
    miny = min(_piece_bounds(piece)[1] for piece in sheet)
    maxx = max(_piece_bounds(piece)[2] for piece in sheet)
    maxy = max(_piece_bounds(piece)[3] for piece in sheet)
    return max(maxx - minx, 0.0) * max(maxy - miny, 0.0)


def _sheet_used_width(sheet):
    if not sheet:
        return 0.0
    minx = min(_piece_bounds(piece)[0] for piece in sheet)
    maxx = max(_piece_bounds(piece)[2] for piece in sheet)
    return max(maxx - minx, 0.0)


def _sheet_used_height(sheet):
    if not sheet:
        return 0.0
    miny = min(_piece_bounds(piece)[1] for piece in sheet)
    maxy = max(_piece_bounds(piece)[3] for piece in sheet)
    return max(maxy - miny, 0.0)


def _bounds_area(geom):
    minx, miny, maxx, maxy = geom.bounds
    return max(maxx - minx, 0.0) * max(maxy - miny, 0.0)


def _bounds_aspect(geom):
    minx, miny, maxx, maxy = geom.bounds
    width = max(maxx - minx, EPSILON)
    height = max(maxy - miny, EPSILON)
    ratio = width / height
    return max(ratio, 1 / ratio)


def _normalize_priority(priority):
    raw = str(priority or "preserve_shapes").strip().lower()
    if raw in {"maximize_usage", "usage", "area", "ocupacao", "ocupação"}:
        return "maximize_usage"
    return "preserve_shapes"


def _clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def _piece_bounds(piece):
    bounds = piece.get("_bounds")
    if bounds is not None:
        return bounds
    return piece["geom"].bounds


def _piece_area(piece):
    area = piece.get("_area")
    if area is not None:
        return area
    return piece["geom"].area


def _offset_bounds(bounds, x, y):
    minx, miny, maxx, maxy = bounds
    return minx + x, miny + y, maxx + x, maxy + y


def _combine_bounds(bounds_a, bounds_b):
    if bounds_a is None:
        return bounds_b
    return (
        min(bounds_a[0], bounds_b[0]),
        min(bounds_a[1], bounds_b[1]),
        max(bounds_a[2], bounds_b[2]),
        max(bounds_a[3], bounds_b[3]),
    )


def _sheet_stats(sheet):
    if not sheet:
        return None, 0.0

    bounds = None
    total_area = 0.0
    for piece in sheet:
        bounds = _combine_bounds(bounds, _piece_bounds(piece))
        total_area += _piece_area(piece)
    return bounds, total_area


def _best_split_boxes(geom, bed_w, bed_h):
    minx, miny, maxx, maxy = geom.bounds
    w = maxx - minx
    h = maxy - miny
    axis = "x" if (w / max(bed_w, 1)) >= (h / max(bed_h, 1)) else "y"

    best = None
    ratios = [0.2 + index * 0.03 for index in range(21)]
    for ratio in ratios:
        if axis == "x":
            cut = minx + w * ratio
            boxes = [
                sg.box(minx, miny, cut, maxy),
                sg.box(cut, miny, maxx, maxy),
            ]
        else:
            cut = miny + h * ratio
            boxes = [
                sg.box(minx, miny, maxx, cut),
                sg.box(minx, cut, maxx, maxy),
            ]

        parts = [geom.intersection(box) for box in boxes]
        parts = [part for part in parts if not part.is_empty and part.area > MIN_GEOMETRY_AREA]
        if len(parts) != 2:
            continue

        score = _split_score(parts, bed_w, bed_h)
        if best is None or score < best[0]:
            best = (score, boxes)

    if best is not None:
        return best[1]

    if axis == "x":
        mid_x = minx + w / 2.0
        return [
            sg.box(minx, miny, mid_x, maxy),
            sg.box(mid_x, miny, maxx, maxy),
        ]

    mid_y = miny + h / 2.0
    return [
        sg.box(minx, miny, maxx, mid_y),
        sg.box(minx, mid_y, maxx, maxy),
    ]


def _split_score(parts, bed_w, bed_h):
    areas = [part.area for part in parts]
    balance = abs(areas[0] - areas[1]) / max(sum(areas), 1)
    smallest_ratio = min(areas) / max(sum(areas), 1)
    overflows = [_minimum_rotation_overflow(part, bed_w, bed_h) for part in parts]

    return (
        max(overflows),
        balance,
        -smallest_ratio,
        max(areas),
    )


def _minimum_rotation_overflow(geom, bed_w, bed_h):
    best = None
    for angle in _rotation_angles_for_geom(geom, 15):
        rotated = rotate(geom, angle, origin=geom.centroid.coords[0], use_radians=False)
        minx, miny, maxx, maxy = rotated.bounds
        overflow = max(maxx - minx - bed_w, maxy - miny - bed_h, 0)
        best = overflow if best is None else min(best, overflow)

    return best if best is not None else 0


def _explode_multipolygon_piece(piece):
    geom = piece["slice_geom"]
    if geom.geom_type != "MultiPolygon":
        item = dict(piece)
        item["source_slice_id"] = item.get("source_slice_id", item["id"])
        item["model_geom"] = item.get("model_geom", item["slice_geom"])
        item["label_areas"] = _label_areas_for_piece(item)
        item["was_split"] = item.get("was_split", False)
        return [item]

    pieces = []
    label_areas = _label_areas_for_piece(piece)
    for index, poly in enumerate(geom.geoms):
        if poly.area <= MIN_GEOMETRY_AREA:
            continue

        pieces.append({
            "id": f"{piece['id']}_{index}",
            "source_slice_id": piece.get("source_slice_id", piece["id"]),
            "elevation": piece["elevation"],
            "source_elevation": piece.get("source_elevation"),
            "slice_geom": poly,
            "model_geom": poly,
            "engraving_lines": _clip_engraving_lines_to_polygon(piece["engraving_lines"], poly),
            "label_areas": _clip_label_areas(label_areas, poly),
            "was_split": piece.get("was_split", False),
            "split_reason": piece.get("split_reason"),
        })

    return pieces


def _clip_engraving_lines_to_polygon(lines, polygon):
    clipped = []
    for line in lines:
        section = line.intersection(polygon)
        if section.is_empty:
            continue
        clipped.append(section)
    return clipped


def _label_areas_for_piece(piece):
    explicit = [
        area
        for area in piece.get("label_areas", []) or []
        if area is not None and not area.is_empty and area.area > MIN_GEOMETRY_AREA
    ]
    if explicit:
        return _clip_label_areas(explicit, piece.get("slice_geom"))

    # Prefer explicit support/base regions (Option B) for assembly labels.
    # This keeps numbers on stable glue/base areas instead of falling back
    # to outer-bounds placement when engraving lines are open segments.
    support_geom = piece.get("support_geom")
    if support_geom is not None and not support_geom.is_empty and support_geom.area > MIN_GEOMETRY_AREA:
        return _clip_label_areas(_polygon_parts_for_label_area(support_geom), piece.get("slice_geom"))

    hidden_geom = piece.get("hidden_base_geom")
    if hidden_geom is not None and not hidden_geom.is_empty and hidden_geom.area > MIN_GEOMETRY_AREA:
        return _clip_label_areas(_polygon_parts_for_label_area(hidden_geom), piece.get("slice_geom"))

    return _closed_engraving_areas(piece)


def _closed_engraving_areas(piece):
    areas = []
    for engraving in piece.get("engraving_lines", []) or []:
        for part in _line_parts_for_label_area(engraving):
            coords = [(round(float(x), 6), round(float(y), 6)) for x, y in part.coords]
            if len(coords) < 4:
                continue
            if not (_same_point(coords[0], coords[-1]) or bool(getattr(part, "is_ring", False))):
                continue
            try:
                polygon = sg.Polygon(coords)
            except (TypeError, ValueError):
                continue
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if polygon.is_empty or polygon.area <= MIN_GEOMETRY_AREA:
                continue
            areas.extend(_polygon_parts_for_label_area(polygon))
    return _clip_label_areas(areas, piece.get("slice_geom"))


def _clip_label_areas(areas, geom):
    if geom is None or geom.is_empty:
        return []

    clipped = []
    for area in areas or []:
        if area is None or area.is_empty:
            continue
        section = area.intersection(geom)
        for polygon in _polygon_parts_for_label_area(section):
            if polygon.area > MIN_GEOMETRY_AREA:
                clipped.append(polygon)
    return clipped


def _line_parts_for_label_area(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type in ("LineString", "LinearRing"):
        return [geom]
    if geom.geom_type == "MultiLineString":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        lines = []
        for part in geom.geoms:
            lines.extend(_line_parts_for_label_area(part))
        return lines
    return []


def _same_point(a, b):
    return abs(a[0] - b[0]) <= EPSILON and abs(a[1] - b[1]) <= EPSILON


def _polygon_parts_for_label_area(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        polygons = []
        for part in geom.geoms:
            polygons.extend(_polygon_parts_for_label_area(part))
        return polygons
    return []


def _piece_sort_key(piece, original_index=0):
    geom = piece["slice_geom"]
    minx, miny, maxx, maxy = geom.bounds
    w = maxx - minx
    h = maxy - miny
    return (-geom.area, -max(w, h), -min(w, h), original_index, str(piece["id"]))


def _find_best_placement(
    piece,
    sheets,
    bed_w,
    bed_h,
    margin,
    rotation_angles,
    priority="preserve_shapes",
    effort_percent=65,
):
    best = None
    sheet_count = len(sheets)
    effort = _clamp(float(effort_percent or 0), 0.0, 100.0)
    max_axis_candidates, max_candidate_positions = _effort_search_limits(effort, priority)
    fit_angles = _merge_rotation_angles(rotation_angles, _fit_rotation_angles(piece["slice_geom"]))
    rotated_options = []
    for angle in fit_angles:
        rotated = _rotate_piece_to_origin(piece, angle)
        rotated_bounds = _piece_bounds(rotated)
        if _bounds_can_fit(rotated_bounds, bed_w, bed_h, margin):
            rotated_options.append((rotated, rotated_bounds, _piece_area(rotated)))

    if not rotated_options:
        return None

    for sheet_index in range(sheet_count + 1):
        sheet = sheets[sheet_index] if sheet_index < sheet_count else []
        sheet_bounds, sheet_area = _sheet_stats(sheet)
        for rotated, rotated_bounds, rotated_area in rotated_options:
            for x, y in _candidate_positions(
                rotated["geom"],
                sheet,
                bed_w,
                bed_h,
                margin,
                bounds=rotated_bounds,
                max_axis_candidates=max_axis_candidates,
                max_candidate_positions=max_candidate_positions,
            ):
                placed_bounds = _offset_bounds(rotated_bounds, x, y)
                if not _placement_is_valid_at_offset(
                    rotated["geom"],
                    x,
                    y,
                    placed_bounds,
                    sheet,
                    bed_w,
                    bed_h,
                    margin,
                ):
                    continue

                score = _placement_score(
                    sheet_index,
                    sheet_count,
                    placed_bounds,
                    rotated_area,
                    sheet_bounds,
                    sheet_area,
                    bed_w,
                    bed_h,
                    priority=priority,
                )
                if best is None or score < best[0]:
                    best = (score, sheet_index, rotated, x, y)

    if best is None:
        return None

    return best[1], _translate_piece(best[2], best[3], best[4])


def _rotate_piece_to_origin(piece, angle):
    geom = piece["slice_geom"]
    origin = geom.centroid.coords[0]
    rotated_geom = rotate(geom, angle, origin=origin, use_radians=False)
    minx, miny, maxx, maxy = rotated_geom.bounds
    rotated_geom = translate(rotated_geom, xoff=-minx, yoff=-miny)
    rotated_bounds = (0.0, 0.0, maxx - minx, maxy - miny)

    rotated_engravings = []
    for line in piece["engraving_lines"]:
        rotated_line = rotate(line, angle, origin=origin, use_radians=False)
        rotated_engravings.append(translate(rotated_line, xoff=-minx, yoff=-miny))

    rotated_label_areas = []
    for area in piece.get("label_areas", []) or []:
        rotated_area = rotate(area, angle, origin=origin, use_radians=False)
        rotated_label_areas.append(translate(rotated_area, xoff=-minx, yoff=-miny))

    return {
        "id": piece["id"],
        "source_slice_id": piece.get("source_slice_id", piece["id"]),
        "model_geom": piece.get("model_geom", piece["slice_geom"]),
        "was_split": piece.get("was_split", False),
        "split_reason": piece.get("split_reason"),
        "elevation": piece["elevation"],
        "source_elevation": piece.get("source_elevation"),
        "geom": rotated_geom,
        "engravings": rotated_engravings,
        "label_areas": rotated_label_areas,
        "angle": angle,
        "_bounds": rotated_bounds,
        "_area": geom.area,
    }


def _translate_piece(piece, x, y):
    bounds = _offset_bounds(_piece_bounds(piece), x, y)
    return {
        "id": piece["id"],
        "source_slice_id": piece.get("source_slice_id", piece["id"]),
        "model_geom": piece.get("model_geom", piece["geom"]),
        "was_split": piece.get("was_split", False),
        "split_reason": piece.get("split_reason"),
        "elevation": piece["elevation"],
        "source_elevation": piece.get("source_elevation"),
        "geom": translate(piece["geom"], xoff=x, yoff=y),
        "engravings": [translate(line, xoff=x, yoff=y) for line in piece["engravings"]],
        "label_areas": [
            translate(area, xoff=x, yoff=y)
            for area in piece.get("label_areas", []) or []
        ],
        "angle": piece["angle"],
        "_bounds": bounds,
        "_area": _piece_area(piece),
    }


def _candidate_positions(
    geom,
    sheet,
    bed_w,
    bed_h,
    margin,
    bounds=None,
    max_axis_candidates=MAX_AXIS_CANDIDATES,
    max_candidate_positions=MAX_CANDIDATE_POSITIONS,
):
    minx, miny, maxx, maxy = bounds if bounds is not None else geom.bounds
    width = maxx - minx
    height = maxy - miny
    max_x = bed_w - margin - width
    max_y = bed_h - margin - height

    if max_x < margin - EPSILON or max_y < margin - EPSILON:
        return []

    x_candidates = {margin, max_x}
    y_candidates = {margin, max_y}
    positions = {(margin, margin), (max_x, margin), (margin, max_y)}
    for placed in sheet:
        pminx, pminy, pmaxx, pmaxy = _piece_bounds(placed)
        x_candidates.update({
            pminx,
            pmaxx + margin,
            pmaxx - width,
            pminx - margin - width,
        })
        y_candidates.update({
            pminy,
            pmaxy + margin,
            pmaxy - height,
            pminy - margin - height,
        })
        positions.update({
            (pmaxx + margin, pminy),
            (pminx, pmaxy + margin),
            (pmaxx + margin, pmaxy + margin),
            (pminx - margin - width, pminy),
            (pminx, pminy - margin - height),
            (pmaxx - width, pminy),
            (pminx, pmaxy - height),
            (pmaxx + margin, margin),
            (margin, pmaxy + margin),
        })

    positions.update(_interior_void_candidate_positions(width, height, sheet, margin, max_x, max_y))

    for x in _limited_axis_candidates(x_candidates, margin, max_x, max_axis_candidates):
        for y in _limited_axis_candidates(y_candidates, margin, max_y, max_axis_candidates):
            positions.add((x, y))

    clean = []
    seen = set()
    for x, y in positions:
        x = round(x, 6)
        y = round(y, 6)
        if x < margin - EPSILON or y < margin - EPSILON:
            continue
        if x > max_x + EPSILON or y > max_y + EPSILON:
            continue

        key = (max(margin, min(max_x, x)), max(margin, min(max_y, y)))
        if key not in seen:
            seen.add(key)
            clean.append(key)

    clean.sort(key=lambda point: (point[1], point[0]))
    return clean[:max(1, int(max_candidate_positions))]


def _limited_axis_candidates(candidates, min_value, max_value, max_axis_candidates=MAX_AXIS_CANDIDATES):
    values = sorted({
        round(float(value), 6)
        for value in candidates
        if min_value - EPSILON <= value <= max_value + EPSILON
    })
    max_axis_candidates = max(2, int(max_axis_candidates or MAX_AXIS_CANDIDATES))
    if len(values) <= max_axis_candidates:
        return values

    indexes = {0, len(values) - 1}
    slots = max(max_axis_candidates - len(indexes), 1)
    for index in range(slots):
        indexes.add(round(index * (len(values) - 1) / max(slots - 1, 1)))
    return [values[index] for index in sorted(indexes)[:max_axis_candidates]]


def _effort_search_limits(effort_percent, priority="preserve_shapes"):
    effort = _clamp(float(effort_percent or 0), 0.0, 100.0)
    high_usage = _normalize_priority(priority) == "maximize_usage"

    if high_usage:
        if effort >= 85:
            return MAX_AXIS_CANDIDATES_HIGH_EFFORT, MAX_CANDIDATE_POSITIONS_HIGH_EFFORT
        if effort >= 65:
            return 32, 120
        if effort >= 45:
            return MAX_AXIS_CANDIDATES, MAX_CANDIDATE_POSITIONS
        if effort >= 25:
            return 16, 48
        return 10, 24

    if effort >= 75:
        return MAX_AXIS_CANDIDATES, MAX_CANDIDATE_POSITIONS
    if effort >= 40:
        return 16, 48
    return 10, 24


def _interior_void_candidate_positions(width, height, sheet, margin, max_x, max_y):
    positions = []
    for placed in sheet:
        for poly in _polygon_parts_for_label_area(placed.get("geom")):
            for interior in poly.interiors:
                hole = sg.Polygon(interior)
                if hole.is_empty:
                    continue
                hminx, hminy, hmaxx, hmaxy = hole.bounds
                if width > hmaxx - hminx - 2 * margin + EPSILON:
                    continue
                if height > hmaxy - hminy - 2 * margin + EPSILON:
                    continue

                positions.extend([
                    (hminx + margin, hminy + margin),
                    (hmaxx - margin - width, hminy + margin),
                    (hminx + margin, hmaxy - margin - height),
                    (
                        hminx + (hmaxx - hminx - width) / 2.0,
                        hminy + (hmaxy - hminy - height) / 2.0,
                    ),
                ])

    return [
        (x, y)
        for x, y in positions
        if margin - EPSILON <= x <= max_x + EPSILON
        and margin - EPSILON <= y <= max_y + EPSILON
    ]


def _placement_is_valid(geom, sheet, bed_w, bed_h, margin):
    bounds = geom.bounds
    minx, miny, maxx, maxy = bounds
    if minx < margin - EPSILON or miny < margin - EPSILON:
        return False
    if maxx > bed_w - margin + EPSILON or maxy > bed_h - margin + EPSILON:
        return False

    for placed in sheet:
        if not _expanded_bounds_overlap(bounds, _piece_bounds(placed), margin):
            continue
        if geom.distance(placed["geom"]) < margin - EPSILON:
            return False

    return True


def _placement_is_valid_at_offset(geom, x, y, bounds, sheet, bed_w, bed_h, margin):
    minx, miny, maxx, maxy = bounds
    if minx < margin - EPSILON or miny < margin - EPSILON:
        return False
    if maxx > bed_w - margin + EPSILON or maxy > bed_h - margin + EPSILON:
        return False

    translated_geom = None
    for placed in sheet:
        if not _expanded_bounds_overlap(bounds, _piece_bounds(placed), margin):
            continue
        if translated_geom is None:
            translated_geom = translate(geom, xoff=x, yoff=y)
        if translated_geom.distance(placed["geom"]) < margin - EPSILON:
            return False

    return True


def _expanded_bounds_overlap(bounds_a, bounds_b, margin):
    aminx, aminy, amaxx, amaxy = bounds_a
    bminx, bminy, bmaxx, bmaxy = bounds_b
    return not (
        amaxx + margin < bminx
        or bmaxx + margin < aminx
        or amaxy + margin < bminy
        or bmaxy + margin < aminy
    )


def _placement_score(
    sheet_index,
    sheet_count,
    bounds,
    geom_area,
    sheet_bounds,
    sheet_area,
    bed_w,
    bed_h,
    priority="preserve_shapes",
):
    minx, miny, maxx, maxy = _combine_bounds(sheet_bounds, bounds)
    used_w = maxx - minx
    used_h = maxy - miny
    bbox_area = used_w * used_h
    is_new_sheet = 1 if sheet_index == sheet_count else 0
    total_area = sheet_area + geom_area
    fill_ratio = total_area / max(bbox_area, 1)

    if _normalize_priority(priority) == "maximize_usage":
        sheet_capacity = max(bed_w * bed_h, 1)
        sheet_fill = total_area / sheet_capacity
        # Prefer compact "corner blocks" before flat spreading:
        # when two placements have similar area usage, keep X span tighter first.
        # This reproduces denser layouts users expect from polygon nesting tools.
        return (
            is_new_sheet,
            bbox_area,
            used_w,
            used_h,
            -sheet_fill,
            sheet_index,
            bounds[0],
            bounds[1],
        )

    return (
        is_new_sheet,
        bbox_area,
        used_h,
        used_w,
        -fill_ratio,
        sheet_index,
        bounds[1],
        bounds[0],
    )


def _validate_sheets(sheets, bed_w, bed_h, margin):
    for sheet_index, sheet in enumerate(sheets):
        for piece_index, piece in enumerate(sheet):
            if not _placement_is_valid(piece["geom"], sheet[:piece_index] + sheet[piece_index + 1:], bed_w, bed_h, margin):
                raise ValueError(
                    f"Nesting invalido na folha {sheet_index + 1}, peca {piece['id']}."
                )


def _fits_any_rotation(geom, bed_w, bed_h, angles):
    for angle in angles:
        rotated = rotate(geom, angle, origin=geom.centroid.coords[0], use_radians=False)
        if _bounds_can_fit(rotated.bounds, bed_w, bed_h, 0):
            return True
    return False


def _bounds_can_fit(bounds, bed_w, bed_h, margin):
    minx, miny, maxx, maxy = bounds
    return (
        maxx - minx <= bed_w - 2 * margin + EPSILON
        and maxy - miny <= bed_h - 2 * margin + EPSILON
    )


def _rotation_angles_for_geom(geom, step):
    return _merge_rotation_angles(_rotation_angles(step), _fit_rotation_angles(geom))


def _fit_rotation_angles(geom):
    if geom is None or geom.is_empty:
        return []

    try:
        oriented = geom.minimum_rotated_rectangle
    except Exception:
        return []

    if oriented.is_empty or oriented.geom_type != "Polygon":
        return []

    coords = list(oriented.exterior.coords)
    angles = []
    for current, next_point in zip(coords, coords[1:]):
        dx = next_point[0] - current[0]
        dy = next_point[1] - current[1]
        if math.hypot(dx, dy) <= EPSILON:
            continue
        edge_angle = math.degrees(math.atan2(dy, dx))
        base = -edge_angle
        angles.extend([base, base + 90, base + 180, base + 270])
    return angles


def _merge_rotation_angles(*angle_groups):
    angles = set()
    for group in angle_groups:
        for angle in group:
            normalized = round(float(angle) % 360.0, 6)
            if abs(normalized - 360.0) <= EPSILON:
                normalized = 0.0
            angles.add(normalized)
    return sorted(angles)


def _rotation_angles(step):
    if step <= 0:
        return [0]

    count = int(math.ceil(360 / step))
    return sorted({round((index * step) % 360, 6) for index in range(count)})
