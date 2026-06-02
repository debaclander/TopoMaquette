from collections import defaultdict, deque

import networkx as nx


def build_topology_tree(polygons_data):
    """
    Constroi uma arvore (ou floresta) topologica indicando qual poligono contem qual.
    polygons_data e uma lista de dicionarios com 'id' e 'geometry' (Shapely Polygon).
    Retorna um DiGraph do NetworkX onde uma aresta A -> B significa que A contem B diretamente.
    """
    # Work on a copy so callers keep their original polygon order.
    polygons_by_area = sorted(polygons_data, key=lambda p: p["geometry"].area, reverse=True)

    G = nx.DiGraph()
    for p in polygons_by_area:
        G.add_node(p["id"], data=p)

    for child in polygons_by_area:
        child_geom = child["geometry"]
        child_area = child_geom.area
        possible_parents = []

        for potential_parent in polygons_by_area:
            if potential_parent["id"] == child["id"]:
                continue

            parent_geom = potential_parent["geometry"]
            parent_area = parent_geom.area
            if parent_area <= child_area:
                continue

            # Use covers instead of contains because contour polygons can share edges.
            if parent_geom.covers(child_geom):
                possible_parents.append(potential_parent)

        if possible_parents:
            parent = min(possible_parents, key=lambda p: p["geometry"].area)
            G.add_edge(parent["id"], child["id"])

    return G


def assign_elevations(G, base_id, direction_changes, interval=1.0):
    """
    Atribui elevacoes aos nos ligados a base escolhida.

    Importante: curvas desligadas da base ficam com elevation=None. Isto e intencional:
    a geometria de contencao so permite inferir niveis dentro da mesma componente ligada.
    """
    if base_id not in G:
        return G

    for node in G.nodes():
        G.nodes[node]["data"]["elevation"] = None
        G.nodes[node]["data"]["is_descending"] = False
        G.nodes[node]["data"]["reachable_from_base"] = False

    G.nodes[base_id]["data"]["elevation"] = 0.0

    U = G.to_undirected()
    queue = deque([(base_id, 0.0, interval)])
    visited = {base_id}

    while queue:
        current, elev, delta = queue.popleft()

        if current in direction_changes and current != base_id:
            delta = -delta

        G.nodes[current]["data"]["elevation"] = elev
        G.nodes[current]["data"]["is_descending"] = delta < 0
        G.nodes[current]["data"]["reachable_from_base"] = True

        for neighbor in U.neighbors(current):
            if neighbor in visited:
                continue

            visited.add(neighbor)

            if G.has_edge(current, neighbor):
                next_elev = elev + delta
                queue.append((neighbor, next_elev, delta))
            elif G.has_edge(neighbor, current):
                next_elev = elev - delta
                queue.append((neighbor, next_elev, delta))

    return G


def assign_elevations_from_arrows(G, arrows, interval=1.0):
    """
    Atribui cotas a partir de setas manuais A->B com sentido subir/descer.

    Cada seta define uma progressao por passos de ``interval`` ao longo do caminho
    topologico mais curto entre os IDs indicados.
    """
    for node in G.nodes():
        G.nodes[node]["data"]["elevation"] = None
        G.nodes[node]["data"]["is_descending"] = False
        G.nodes[node]["data"]["reachable_from_base"] = False

    if not arrows:
        return G

    U = G.to_undirected()
    relative_graph = defaultdict(list)
    descending_votes = defaultdict(list)
    valid_paths = 0

    for arrow in arrows:
        start_raw = arrow.get("from_id")
        end_raw = arrow.get("to_id")
        direction = str(arrow.get("direction", "up")).strip().lower()
        start_id = _resolve_node_id(G, start_raw)
        end_id = _resolve_node_id(G, end_raw)
        if start_id is None or end_id is None or start_id == end_id:
            continue

        if direction not in {"up", "down"}:
            continue
        step = interval if direction == "up" else -interval

        try:
            path = nx.shortest_path(U, source=start_id, target=end_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        if len(path) < 2:
            continue

        valid_paths += 1
        for current, nxt in zip(path, path[1:]):
            relative_graph[current].append((nxt, step))
            relative_graph[nxt].append((current, -step))
            descending_votes[nxt].append(step < 0)

    if valid_paths == 0:
        return G

    potentials = {}
    seed_values = {}
    for node in relative_graph.keys():
        if node in seed_values:
            continue
        seed_values[node] = 0.0

    for seed, base_value in seed_values.items():
        if seed in potentials:
            continue

        queue = deque([seed])
        potentials[seed] = base_value
        while queue:
            current = queue.popleft()
            current_elev = potentials[current]
            for neighbor, delta in relative_graph[current]:
                candidate = current_elev + delta
                known = potentials.get(neighbor)
                if known is None:
                    potentials[neighbor] = candidate
                    queue.append(neighbor)
                    continue

                if abs(known - candidate) > 1e-6:
                    potentials[neighbor] = (known + candidate) / 2.0

    if not potentials:
        return G

    min_value = min(potentials.values())
    for node, value in potentials.items():
        normalized = value - min_value
        G.nodes[node]["data"]["elevation"] = round(normalized, 6)
        G.nodes[node]["data"]["is_descending"] = any(descending_votes.get(node, []))
        G.nodes[node]["data"]["reachable_from_base"] = True

    # Expand to neighbors not directly traversed by arrows to avoid gaps.
    _propagate_remaining_elevations(G, interval=interval)
    return G


def _resolve_node_id(G, raw_id):
    if raw_id in G:
        return raw_id

    raw_text = str(raw_id).strip()
    if raw_text == "":
        return None

    for node in G.nodes():
        if str(node).strip() == raw_text:
            return node
    return None


def _propagate_remaining_elevations(G, interval=1.0):
    U = G.to_undirected()
    queue = deque(
        node
        for node in G.nodes()
        if G.nodes[node]["data"].get("elevation") is not None
    )
    visited = set(queue)

    while queue:
        current = queue.popleft()
        current_elev = G.nodes[current]["data"].get("elevation")
        for neighbor in U.neighbors(current):
            if G.nodes[neighbor]["data"].get("elevation") is not None:
                continue

            if G.has_edge(current, neighbor):
                next_elev = current_elev + interval
                is_desc = False
            elif G.has_edge(neighbor, current):
                next_elev = current_elev - interval
                is_desc = True
            else:
                next_elev = current_elev
                is_desc = False

            G.nodes[neighbor]["data"]["elevation"] = round(next_elev, 6)
            G.nodes[neighbor]["data"]["is_descending"] = is_desc
            G.nodes[neighbor]["data"]["reachable_from_base"] = True
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
