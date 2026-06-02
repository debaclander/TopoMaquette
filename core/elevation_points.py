def estimate_polygon_elevations_from_points(
    G,
    points,
    nearest_count=12,
    max_distance=None,
    exclude_layers=None,
):
    """
    Estimate each polygon elevation from 3D points that belong to each closed
    contour.

    The estimate uses the representative Z of the points detected on the
    contour boundary.
    Legacy args ``nearest_count`` and ``max_distance`` are kept for compatibility
    but no longer affect the calculation.
    """
    _ = nearest_count
    _ = max_distance

    if exclude_layers is None:
        exclude_layers = {"camadas"}
    exclude_layers = {layer.lower() for layer in exclude_layers}

    if not points:
        return G

    for node in G.nodes():
        data = G.nodes[node]["data"]
        layer = data.get("layer", "").lower()

        data["elevation"] = None
        data["is_descending"] = False
        data["reachable_from_base"] = False
        data["estimated_from_points"] = False

        if layer in exclude_layers:
            continue

        geom = data["geometry"]
        boundary = geom.exterior
        distances = sorted(
            (boundary.distance(point["geometry"]), point["z"])
            for point in points
        )
        zs = _select_boundary_point_zs(distances, geom)
        if not zs:
            continue

        data["elevation"] = _representative_z(zs)
        data["estimated_from_points"] = True
        data["reachable_from_base"] = True

    _mark_descending_edges(G)
    return G


def _select_boundary_point_zs(distance_z_pairs, geom):
    if not distance_z_pairs:
        return []

    minx, miny, maxx, maxy = geom.bounds
    span = max(maxx - minx, maxy - miny, 1.0)
    base_tol = max(span * 1e-6, 1e-6)

    min_distance = distance_z_pairs[0][0]
    tolerance = max(base_tol, (min_distance * 2.5) + base_tol)
    zs = [z for distance, z in distance_z_pairs if distance <= tolerance]

    if not zs:
        zs = [distance_z_pairs[0][1]]
    return zs


def _representative_z(zs):
    counts = {}
    for z in zs:
        key = round(float(z), 6)
        counts[key] = counts.get(key, 0) + 1

    return min(counts.items(), key=lambda item: (-item[1], item[0]))[0]


def _mark_descending_edges(G):
    for node in G.nodes():
        data = G.nodes[node]["data"]
        elev = data.get("elevation")
        parent_elevations = [
            G.nodes[parent]["data"].get("elevation")
            for parent in G.predecessors(node)
            if G.nodes[parent]["data"].get("elevation") is not None
        ]

        if elev is None or not parent_elevations:
            data["is_descending"] = False
            continue

        data["is_descending"] = elev < max(parent_elevations)
