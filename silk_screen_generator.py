import argparse
import heapq
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime

import numpy as np
import shapely
from bokeh.events import DocumentReady
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.plotting import figure, output_file, save, show
from shapely import affinity
from shapely.geometry import LineString, MultiLineString, Point, box
from shapely.ops import unary_union
from shapely.strtree import STRtree


GRID_MM = 0.05
SILK_TEXT_HEIGHT_MM = 0.60
SILK_CHAR_ASPECT = 0.6
SILK_RENDERED_CHAR_WIDTH_FACTOR = 1.05
SILK_RENDERED_HEIGHT_FACTOR = 1.60
SILK_PADDING_MM = 0.10
SILK_OFFSET_MM = 0.0
SILK_CLEARANCE_MM = 0.0
SILK_FOOTPRINT_CLEARANCE_MM = 0.20
SILK_MAX_DISTANCE_MM = 1.0
SILK_FAR_MAX_DISTANCE_MM = 30.0
SILK_FAR_STEP_MM = 0.5
SILK_ARROW_THRESHOLD_MM = 1.0
SILK_ARROW_MARGIN_MM = 0.15
SILK_ARROW_WIDTH_MM = 0.025
SILK_ARROW_NEARBY_RADIUS_MM = 3.0  # for a normal (non-cluster, non-escape)
# placement that ends up diagonal to its component but still within
# SILK_ARROW_THRESHOLD_MM, this is the radius checked for other nearby
# components before deciding a pointer arrow is actually needed -- an
# isolated diagonal label needs no arrow (nothing to confuse it with), a
# crowded one does
SILK_ARROW_NEARBY_COUNT = 1  # this many other components within
# SILK_ARROW_NEARBY_RADIUS_MM is enough to call a spot "crowded"
SILK_OFFBOARD_GAP_MM = 5.0  # fixed clearance between the board's own edge
# and the start of the guaranteed off-board overflow strip (last-resort
# placement for anything that found no legal on-board spot anywhere)
SILK_OFFBOARD_SPACING_MM = 1.0  # gap between consecutive items packed
# along the same off-board edge strip
SILK_ESCAPE_USE_PROCESSES = False  # the escape rescue's worker-process pool
# (see the driver in _auto_place_silkscreen_geometry) is off: the nearest-
# first ring search made the stage cheap, and every worker process needs
# its own half-gigabyte copy of the obstacle trees. The sequential path
# gives the identical result. Flip to True to allow the pool again.
SILK_ESCAPE_PARALLEL_MIN_ITEMS = 4  # escape rescue only spins up worker
# processes when at least this many failed items need a whole-board scan;
# below that, the process start-up + per-worker obstacle-tree rebuild
# (~seconds) would cost more than the scans it saves, so it runs the plain
# sequential path instead -- which is exactly the same code the parallel
# path falls back to, so the output is identical either way
SILK_MAX_GRID_CELLS = 5_000_000
SILK_GEOMETRY_WORKERS = max(1, min(8, os.cpu_count() or 1))
SILK_GEOMETRY_WINDOW_MULTIPLIER = 2
SILK_DYNAMIC_INDEX_CELL_MM = 2.0
SILK_MAX_STATIC_CANDIDATES_PER_MODE = 64
SILK_FAMILY_CLUSTER_GAP_MM = 3.0  # loose gap for CLUSTER FORMATION only --
# shared by pairs and rows alike (one union-find pass forms every cluster
# before size decides pair vs row), so this must stay generous or pairs stop
# clustering. Strict adjacency for rows specifically lives in
# _classify_family_pattern's row branch instead, not here -- see
# SILK_ROW_ADJACENT_GAP_MM
SILK_PARALLEL_ROTATION_TOLERANCE_DEG = 3.0  # how close two footprint rotations
# must be (or how close to 180 degrees apart) to count as "aligned" for the
# pair/row family-pattern classification -- observational only for now, does
# not change any placement decision
SILK_ROW_MAJORITY_THRESHOLD = 0.8  # fraction of a row's members that must
# share a common axis (rotation compared mod 180 -- a part flipped 180
# degrees occupies the same footprint outline, so it doesn't break the row)
# for the cluster to count as "row"; the rest are reported as outliers
# instead of voiding the whole cluster
SILK_ROW_ADJACENT_GAP_MM = 0.5  # consecutive members whose edge-to-edge gap
# is at or below this count as "strictly adjacent" -- spacing consistency is
# automatically satisfied and the pitch check below is skipped
SILK_LABEL_BUFFER_RATIO = 0.20  # post-placement reinforcement pass: a placed
# label needs at least this fraction of its own size clear directly above it
# (screen-up, i.e. smaller Y) and to each side (left and right) or it gets a
# chance to relocate -- see _reinforce_label_buffers_grid/_geometry
SILK_FAMILY_SHAPE_TOLERANCE_PCT = 0.08  # relative tolerance (8%) on each
# bounding-box side length for two footprints to count as the "same shape"
# for family membership -- user asked for something in the 5-10% range
# rather than an exact match; picked the middle of that range, tune freely
SILK_ORIENTATION_SCORE_CAP = 8  # stop counting free near-tier slots for an
# orientation once this many are found -- plenty to tell "roomy" from
# "cramped" without exhaustively scoring every dense candidate
SILK_SLIDE_STEP_MM = 0.25  # dense-ring slide granularity along an edge (ported from
# silkscreen_generator_revamp.py); the ring-to-ring gap step stays SILK_FAR_STEP_MM
SILK_ESCAPE_SCAN_STEP_MM = 1.0  # ring spacing (and on-ring lattice spacing)
# for the final "escape rescue" search in the geometry backend: still-failed
# components/blocks search outward from their own position in expanding
# square rings and stop at the first ring holding a legal spot, which is
# then refined at SILK_SLIDE_STEP_MM. The grid backend still uses it as its
# coarse whole-board lattice spacing.
SILK_DENSE_CLUSTER_GAP_MM = SILK_ROW_ADJACENT_GAP_MM  # proximity-only
# flood-fill threshold for grouping still-failed components into one
# "densely packed" cluster before escape rescue -- separate from
# SILK_FAMILY_CLUSTER_GAP_MM, which also requires same-family/same-shape;
# here only physical closeness matters. Reuses the same "strictly
# adjacent" edge-to-edge gap already established for row detection, so
# only components that are actually touching/near-touching count as one
# cluster -- a looser gap (5mm, the original first guess) let components
# that merely happened to be somewhat nearby merge into the same block
# even when there was no real crowding forcing them together.
SILK_CLUSTER_COMPACT_STEP_MM = 0.1  # per-round inward step used by
# _compact_cluster_no_overlap to squeeze a dense cluster's real member
# positions together as tightly as possible: every member steps this far
# toward the cluster centroid each round, and any overlap that step
# creates gets corrected by pushing just that pair apart -- so a member
# stuck behind another can be shoved sideways onto a clearer path rather
# than freezing in place.
SILK_ESCAPE_MARGIN_MM = 2.0 * SILK_TEXT_HEIGHT_MM  # buffer wanted on every
# side (left, right, top and bottom) of an escape-rescued block/label
# (geometry backend): two text-heights of empty space, so the label is
# comfortably visible as its own thing rather than wedged against whatever
# it landed next to. A preference, not a requirement -- see
# SILK_ESCAPE_PREFERENCE_REACH_MM. Once a label/block has been placed with
# its buffer, the buffer zone is reserved in the placed-label index so no
# later escape placement can move into it.
SILK_ESCAPE_PREFERENCE_REACH_MM = 10.0  # the escape search places as close
# as possible: the nearest legal spot fixes the distance, and the search
# looks only this much further for a better-quality spot (0 degrees first,
# then buffer clear, then trace-free). The nearest spot is
# usually right against an obstacle, so some reach is needed for "get the
# buffer if the space allows" to ever succeed; on the Apalis board 10 mm
# gave 179 of 264 rescued labels their buffer for about 3 mm of extra
# average distance (1.2 mm: 28 labels; 5 mm: 137). A label never travels
# further than this for it.
SILK_CONNECTOR_ROUTE_STEP_MM = 0.5  # lattice spacing for routing an escape /
# off-board connector line around obstacles (geometry backend); a lattice
# node is usable when a step-sized square there is clear, so a routed path
# keeps half a step off every obstacle edge.
SILK_CONNECTOR_ROUTE_MARGINS_MM = (5.0, 15.0)  # routing windows tried around
# the straight line's bounding box: the small one first, the wide one only
# if the small one holds no path. If neither does, the connector falls back
# to the clipped straight line (the previous behaviour).
SILK_CONNECTOR_PROBE_MM = 0.2  # clearance probe width for a routed segment
# (the drawn line itself is SILK_ARROW_WIDTH_MM); keeps the line visibly off
# obstacle edges after string-pulling.
SILK_CONNECTOR_MAX_EXPANSIONS = 60000  # A* safety cap per routing window
SILK_CONNECTOR_EMERGE_MM = 3.0  # within this distance of either end of a
# connector the router ignores obstacles: a rescued component is hemmed in
# by definition (that is why its label had to move), and a label may sit in
# a tight gap, so without this the endpoint is often unreachable and no
# route exists at all. The route is clean everywhere else; the pieces inside
# these two zones are then clipped around pads and footprints exactly as the
# straight line used to be, so nothing is ever drawn over an obstacle.
SILK_ESCAPE_BUFFER_RATIO = 0.75  # preferred clearance (top/left/right, as
# a fraction of the label's/block's own size -- same metric as the normal
# pipeline's SILK_LABEL_BUFFER_RATIO, just a much roomier bar) for an
# escape-rescue landing spot. Not a hard requirement: among the legal
# candidates a whole-board scan finds, ones meeting this are tried first,
# but the search still falls back to the plain nearest-legal spot if none
# qualify -- never loses a placement chasing a spacious one that doesn't
# exist. Deliberately higher than the normal 0.20 bar, since the point of
# an escaped label is to land in genuinely open space, not just any legal
# gap, so the connector line reads clearly against its surroundings.

DEFAULT_KICAD_PYTHON = r"C:\Program Files\KiCad\9.0\bin\python.exe"
DEFAULT_SILKSCREEN_UPDATER = (
    r"E:\Autocuro\Kicad_File_Parsing\Placement\kicad_brd_update\Silkscreen_Script.py"
)
FAB_LAYER_IDS = (33, 35)  # B.Fab and F.Fab in KiCad 9/10 layer numbering
# (pcbnew.B_Fab / pcbnew.F_Fab); cleared from the output board's
# project-local visible_layers mask so KiCad opens it with Fab hidden.
# (KiCad 8 numbered them 48 / 49.) No text item in the board is touched:
# the Fab text stays exactly as the designer left it and reappears when
# the layer is switched back on.
DEFAULT_BOARD_TIMEOUT_SECONDS = 200.0
DEFAULT_BOARD_READ_RETRY_SECONDS = 30.0
BOARD_READ_RETRY_INTERVAL_SECONDS = 1.0
SOURCE_BOARD_FILENAME = "updated_pcb_file.kicad_pcb"
OUTPUT_BOARD_FILENAME = "updated_pcb_file_w_silk.kicad_pcb"
BATCH_REPORT_FILENAME = "silkscreen_batch_report.json"


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def rotate_point(point, angle_deg):
    x, y = point
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return (x * cos_t - y * sin_t, x * sin_t + y * cos_t)


def _tokenize_sexp(text):
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch == ";":
            while i < n and text[i] not in "\r\n":
                i += 1
            continue
        if ch in "()":
            yield ch
            i += 1
            continue
        if ch == '"':
            i += 1
            chars = []
            while i < n:
                ch = text[i]
                if ch == "\\" and i + 1 < n:
                    chars.append(text[i + 1])
                    i += 2
                    continue
                if ch == '"':
                    i += 1
                    break
                chars.append(ch)
                i += 1
            yield "".join(chars)
            continue
        start = i
        while i < n and (not text[i].isspace()) and text[i] not in "()":
            i += 1
        yield text[start:i]


def parse_sexp(text):
    root = []
    stack = [root]
    for token in _tokenize_sexp(text):
        if token == "(":
            node = []
            stack[-1].append(node)
            stack.append(node)
        elif token == ")":
            if len(stack) <= 1:
                raise ValueError("Unexpected ')' in KiCad file")
            stack.pop()
        else:
            stack[-1].append(token)
    if len(stack) != 1:
        raise ValueError("Unclosed '(' in KiCad file")
    if len(root) == 1 and isinstance(root[0], list):
        return root[0]
    return root


def _is_node(value, name=None):
    if not isinstance(value, list) or not value:
        return False
    if name is None:
        return True
    return value[0] == name


def _children(node, name=None):
    return [item for item in node if _is_node(item, name)]


def _first_child(node, name):
    for item in node:
        if _is_node(item, name):
            return item
    return None


def _child_atom(node, name, index=1, default=None):
    child = _first_child(node, name)
    if child is None or len(child) <= index:
        return default
    return child[index]


def _child_floats(node, name, default=()):
    child = _first_child(node, name)
    if child is None:
        return tuple(default)
    return tuple(_safe_float(value) for value in child[1:] if not isinstance(value, list))


def _child_strings(node, name):
    child = _first_child(node, name)
    if child is None:
        return []
    return [str(value) for value in child[1:] if not isinstance(value, list)]


def _layer_sort_key(layer):
    if layer == "F.Cu":
        return (0, 0, layer)
    if layer.startswith("In") and layer.endswith(".Cu"):
        digits = "".join(ch for ch in layer[2:-3] if ch.isdigit())
        return (1, _safe_int(digits, 999), layer)
    if layer == "B.Cu":
        return (2, 0, layer)
    return (3, 0, layer)


def _pad_side_from_layers(layers, default_layer):
    layers = [str(layer) for layer in (layers or [])]
    has_wild = any(layer == "*.Cu" for layer in layers)
    has_top = has_wild or any(layer.startswith("F.") or layer == "F.Cu" for layer in layers)
    has_bottom = has_wild or any(layer.startswith("B.") or layer == "B.Cu" for layer in layers)
    if has_top and not has_bottom:
        return "F.Cu"
    if has_bottom and not has_top:
        return "B.Cu"
    if default_layer in ("F.Cu", "B.Cu"):
        return default_layer
    return "F.Cu"


def _net_name_from_node(node, net_map):
    net_node = _first_child(node, "net")
    if net_node is None or len(net_node) < 2:
        return None
    net_id = _safe_int(net_node[1], None)
    if len(net_node) >= 3:
        return str(net_node[2])
    return net_map.get(net_id, str(net_id) if net_id is not None else None)


def _stroke_width(node, default=0.15):
    stroke = _first_child(node, "stroke")
    if stroke is not None:
        width = _child_atom(stroke, "width", default=None)
        if width is not None:
            return _safe_float(width, default)
    width = _child_atom(node, "width", default=None)
    if width is not None:
        return _safe_float(width, default)
    return default


def _at_xy_angle(node):
    values = _child_floats(node, "at", default=(0.0, 0.0, 0.0))
    x = values[0] if len(values) >= 1 else 0.0
    y = values[1] if len(values) >= 2 else 0.0
    angle = values[2] if len(values) >= 3 else 0.0
    return x, y, angle


def _transform_local(fp_x, fp_y, fp_rot, local_x, local_y, mirrored=False):
    # KiCad uses a downward-positive board Y axis, so its footprint rotation
    # is the negative of the conventional Cartesian rotation used by
    # rotate_point(). Bottom footprints additionally reflect local X.
    if mirrored:
        local_x = -local_x
    rx, ry = rotate_point((local_x, local_y), -fp_rot)
    return fp_x + rx, fp_y + ry


def _pad_bounds(x, y, width, height, angle_rad, is_circle):
    if is_circle:
        radius = max(width, height) / 2.0
        return x - radius, y - radius, x + radius, y + radius
    half_w = width / 2.0
    half_h = height / 2.0
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    xs = []
    ys = []
    for lx, ly in (
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    ):
        xs.append(x + lx * cos_a - ly * sin_a)
        ys.append(y + lx * sin_a + ly * cos_a)
    return min(xs), min(ys), max(xs), max(ys)


def _circle_from_points(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    temp = x2 * x2 + y2 * y2
    bc = (x1 * x1 + y1 * y1 - temp) / 2.0
    cd = (temp - x3 * x3 - y3 * y3) / 2.0
    det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2)
    if abs(det) < 1e-9:
        return None
    cx = (bc * (y2 - y3) - cd * (y1 - y2)) / det
    cy = ((x1 - x2) * cd - (x2 - x3) * bc) / det
    radius = math.hypot(cx - x1, cy - y1)
    return cx, cy, radius


def _angle_between_ccw(a, b, c):
    if a <= b:
        return a <= c <= b
    return c >= a or c <= b


def _arc_to_segments(start, mid, end, width=0.0):
    circle = _circle_from_points(start, mid, end)
    if circle is None:
        return [
            (start[0], start[1], mid[0], mid[1], width),
            (mid[0], mid[1], end[0], end[1], width),
        ]

    cx, cy, radius = circle
    a0 = math.atan2(start[1] - cy, start[0] - cx)
    a1 = math.atan2(end[1] - cy, end[0] - cx)
    am = math.atan2(mid[1] - cy, mid[0] - cx)
    a0 = (a0 + 2.0 * math.pi) % (2.0 * math.pi)
    a1 = (a1 + 2.0 * math.pi) % (2.0 * math.pi)
    am = (am + 2.0 * math.pi) % (2.0 * math.pi)

    ccw = _angle_between_ccw(a0, a1, am)
    if ccw:
        if a1 < a0:
            a1 += 2.0 * math.pi
        sweep = a1 - a0
    else:
        if a1 > a0:
            a1 -= 2.0 * math.pi
        sweep = a1 - a0

    steps = max(8, int(abs(sweep) / (math.pi / 18.0)))
    steps = min(steps, 96)
    points = []
    for idx in range(steps + 1):
        theta = a0 + sweep * (idx / steps)
        points.append((cx + radius * math.cos(theta), cy + radius * math.sin(theta)))

    return [
        (points[idx][0], points[idx][1], points[idx + 1][0], points[idx + 1][1], width)
        for idx in range(len(points) - 1)
    ]


def _circle_to_segments(center, radius, steps=64):
    cx, cy = center
    steps = max(12, steps)
    points = []
    for idx in range(steps + 1):
        theta = 2.0 * math.pi * (idx / steps)
        points.append((cx + radius * math.cos(theta), cy + radius * math.sin(theta)))
    return [
        (points[idx][0], points[idx][1], points[idx + 1][0], points[idx + 1][1], 0.0)
        for idx in range(len(points) - 1)
    ]


def _points_from_pts_node(node):
    pts = _first_child(node, "pts")
    if pts is None:
        return []
    points = []
    for xy in _children(pts, "xy"):
        if len(xy) >= 3:
            points.append((_safe_float(xy[1]), _safe_float(xy[2])))
    return points


def _update_bounds(bounds, x, y):
    if bounds[0] is None:
        bounds[:] = [x, y, x, y]
        return
    bounds[0] = min(bounds[0], x)
    bounds[1] = min(bounds[1], y)
    bounds[2] = max(bounds[2], x)
    bounds[3] = max(bounds[3], y)


def _update_bounds_from_bbox(bounds, bbox):
    min_x, min_y, max_x, max_y = bbox
    _update_bounds(bounds, min_x, min_y)
    _update_bounds(bounds, max_x, max_y)


def _parse_layers(root):
    layer_node = _first_child(root, "layers")
    layers = []
    if layer_node is None:
        return ["F.Cu", "B.Cu"]
    for item in _children(layer_node):
        if len(item) >= 3:
            layer_name = str(item[1])
            layer_kind = str(item[2]).lower()
            if layer_name.endswith(".Cu") and layer_kind in {"signal", "power", "mixed"}:
                layers.append(layer_name)
    return sorted(set(layers or ["F.Cu", "B.Cu"]), key=_layer_sort_key)


def _parse_pad(node, fp, net_map, component_bounds):
    pad_no = str(node[1]) if len(node) >= 2 else ""
    pad_type = str(node[2]).lower() if len(node) >= 3 else ""
    pad_shape = str(node[3]).lower() if len(node) >= 4 else "rect"
    local_x, local_y, pad_rot = _at_xy_angle(node)
    mirrored = fp["layer"] == "B.Cu"
    x, y = _transform_local(
        fp["x"], fp["y"], fp["rotation"], local_x, local_y, mirrored
    )
    size = _child_floats(node, "size", default=(1.0, 1.0))
    width = size[0] if len(size) >= 1 and size[0] > 0 else 1.0
    height = size[1] if len(size) >= 2 and size[1] > 0 else width
    drill_node = _first_child(node, "drill")
    drill_values = []
    if drill_node is not None:
        for item in drill_node[1:]:
            if isinstance(item, list):
                continue
            value = _safe_float(item, None)
            if value is not None:
                drill_values.append(value)
    drill = max(drill_values) if drill_values else 0.0
    layers = _child_strings(node, "layers")
    side = _pad_side_from_layers(layers, fp["layer"])
    is_th = pad_type == "thru_hole" or drill > 0.0 or "*.Cu" in layers
    is_circle = pad_shape == "circle" and math.isclose(width, height, rel_tol=1e-6, abs_tol=1e-9)
    if pad_shape == "oval" and math.isclose(width, height, rel_tol=1e-6, abs_tol=1e-9):
        is_circle = True
    board_angle_deg = (fp["rotation"] + pad_rot) % 360.0
    angle_rad = (
        math.radians(pad_rot - fp["rotation"])
        if mirrored
        else -math.radians(board_angle_deg)
    )
    bbox = _pad_bounds(x, y, width, height, angle_rad, is_circle)
    _update_bounds_from_bbox(component_bounds, bbox)
    net_name = _net_name_from_node(node, net_map)
    return {
        "fp_id": fp["id"],
        "fp_ref": fp["reference"],
        "pad_no": pad_no,
        "x": x,
        "y": y,
        "w": width,
        "h": height,
        "angle": angle_rad,
        "angle_deg": board_angle_deg,
        "shape": pad_shape,
        "type": pad_type,
        "drill": drill,
        "is_th": bool(is_th),
        "is_circle": bool(is_circle),
        "side": side,
        "layers": layers,
        "net": net_name,
        "bbox": bbox,
        "extent": max(width, height) / 2.0,
    }


def _parse_footprint_graphic(node, fp, component_bounds):
    kind = node[0]
    layer = _child_atom(node, "layer", default=fp["layer"])
    width = _stroke_width(node, default=0.12)
    graphics = []

    def transform_point(values):
        if len(values) < 2:
            return None
        return _transform_local(
            fp["x"],
            fp["y"],
            fp["rotation"],
            values[0],
            values[1],
            fp["layer"] == "B.Cu",
        )

    if kind == "fp_line":
        start = transform_point(_child_floats(node, "start"))
        end = transform_point(_child_floats(node, "end"))
        if start and end:
            _update_bounds(component_bounds, start[0], start[1])
            _update_bounds(component_bounds, end[0], end[1])
            graphics.append({"kind": "line", "layer": layer, "width": width, "points": [start, end]})
    elif kind == "fp_rect":
        start_vals = _child_floats(node, "start")
        end_vals = _child_floats(node, "end")
        if len(start_vals) >= 2 and len(end_vals) >= 2:
            x0, y0 = start_vals[:2]
            x1, y1 = end_vals[:2]
            local_points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
            points = [
                _transform_local(
                    fp["x"],
                    fp["y"],
                    fp["rotation"],
                    x,
                    y,
                    fp["layer"] == "B.Cu",
                )
                for x, y in local_points
            ]
            for x, y in points:
                _update_bounds(component_bounds, x, y)
            graphics.append({"kind": "polyline", "layer": layer, "width": width, "points": points})
    elif kind == "fp_poly":
        points = [
            _transform_local(
                fp["x"],
                fp["y"],
                fp["rotation"],
                x,
                y,
                fp["layer"] == "B.Cu",
            )
            for x, y in _points_from_pts_node(node)
        ]
        if len(points) >= 3:
            for x, y in points:
                _update_bounds(component_bounds, x, y)
            graphics.append({"kind": "polygon", "layer": layer, "width": width, "points": points})
    elif kind == "fp_arc":
        start = transform_point(_child_floats(node, "start"))
        mid = transform_point(_child_floats(node, "mid"))
        end = transform_point(_child_floats(node, "end"))
        if start and mid and end:
            segments = _arc_to_segments(start, mid, end, width)
            points = [(segments[0][0], segments[0][1])] if segments else []
            points.extend((seg[2], seg[3]) for seg in segments)
            for x, y in points:
                _update_bounds(component_bounds, x, y)
            graphics.append({"kind": "polyline", "layer": layer, "width": width, "points": points})
        elif start and end:
            _update_bounds(component_bounds, start[0], start[1])
            _update_bounds(component_bounds, end[0], end[1])
            graphics.append({"kind": "line", "layer": layer, "width": width, "points": [start, end]})
    elif kind == "fp_circle":
        center = transform_point(_child_floats(node, "center"))
        end = transform_point(_child_floats(node, "end"))
        if center and end:
            radius = math.hypot(end[0] - center[0], end[1] - center[1])
            _update_bounds_from_bbox(
                component_bounds,
                (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius),
            )
            graphics.append(
                {"kind": "circle", "layer": layer, "width": width, "x": center[0], "y": center[1], "radius": radius}
            )
    # Bounds include the rendered stroke, not only each fp primitive's
    # centerline. The placement clearance is then measured from the visible
    # footprint boundary itself.
    for graphic in graphics:
        half_stroke = max(0.0, float(graphic.get("width") or 0.0)) / 2.0
        if graphic["kind"] == "circle":
            extent = graphic["radius"] + half_stroke
            _update_bounds_from_bbox(
                component_bounds,
                (
                    graphic["x"] - extent,
                    graphic["y"] - extent,
                    graphic["x"] + extent,
                    graphic["y"] + extent,
                ),
            )
            continue
        points = graphic.get("points") or []
        if points:
            _update_bounds_from_bbox(
                component_bounds,
                (
                    min(x for x, _y in points) - half_stroke,
                    min(y for _x, y in points) - half_stroke,
                    max(x for x, _y in points) + half_stroke,
                    max(y for _x, y in points) + half_stroke,
                ),
            )
    return graphics


def _footprint_reference(node, fallback):
    for prop in _children(node, "property"):
        if len(prop) >= 3 and prop[1] == "Reference":
            return str(prop[2])
    for text in _children(node, "fp_text"):
        if len(text) >= 3 and text[1] == "reference":
            return str(text[2])
    return fallback


def _parse_footprint(node, fp_id, net_map):
    fp_name = str(node[1]) if len(node) >= 2 else f"footprint_{fp_id}"
    fp_layer = _child_atom(node, "layer", default="F.Cu")
    fp_x, fp_y, fp_rot = _at_xy_angle(node)
    reference = _footprint_reference(node, fp_name)
    fp = {
        "id": fp_id,
        "name": fp_name,
        "reference": reference,
        "layer": fp_layer if fp_layer in ("F.Cu", "B.Cu") else "F.Cu",
        "x": fp_x,
        "y": fp_y,
        "rotation": fp_rot,
    }
    component_bounds = [None, None, None, None]
    pads = []
    graphics = []
    has_through_hole = False

    for child in _children(node):
        if child[0] == "pad":
            pad = _parse_pad(child, fp, net_map, component_bounds)
            pads.append(pad)
            has_through_hole = has_through_hole or pad["is_th"]
        elif child[0] in {"fp_line", "fp_rect", "fp_poly", "fp_arc", "fp_circle"}:
            graphics.extend(_parse_footprint_graphic(child, fp, component_bounds))

    if component_bounds[0] is None:
        component_bounds = [fp_x - 0.5, fp_y - 0.5, fp_x + 0.5, fp_y + 0.5]

    side = "F.Cu" if has_through_hole else fp["layer"]
    component = {
        "id": fp_id,
        "ref": reference,
        "name": fp_name,
        "side": side if side in ("F.Cu", "B.Cu") else "F.Cu",
        "layer": fp["layer"],
        "x": fp_x,
        "y": fp_y,
        "rotation": fp_rot,
        "min_x": component_bounds[0],
        "min_y": component_bounds[1],
        "max_x": component_bounds[2],
        "max_y": component_bounds[3],
        "cx": (component_bounds[0] + component_bounds[2]) / 2.0,
        "cy": (component_bounds[1] + component_bounds[3]) / 2.0,
        "has_through_hole": has_through_hole,
    }
    return component, pads, graphics


def _parse_segment(node, net_map):
    start = _child_floats(node, "start")
    end = _child_floats(node, "end")
    layer = _child_atom(node, "layer", default=None)
    width = _child_atom(node, "width", default=None)
    if len(start) < 2 or len(end) < 2 or not layer or width is None:
        return None
    return {
        "x0": start[0],
        "y0": start[1],
        "x1": end[0],
        "y1": end[1],
        "width": _safe_float(width, 0.15),
        "layer": layer,
        "net": _net_name_from_node(node, net_map),
    }


def _parse_arc_segments(node, net_map):
    start = _child_floats(node, "start")
    mid = _child_floats(node, "mid")
    end = _child_floats(node, "end")
    layer = _child_atom(node, "layer", default=None)
    width = _child_atom(node, "width", default=None)
    if len(start) < 2 or len(end) < 2 or not layer or width is None:
        return []
    width = _safe_float(width, 0.15)
    net = _net_name_from_node(node, net_map)
    if len(mid) >= 2:
        raw_segments = _arc_to_segments(start[:2], mid[:2], end[:2], width)
    else:
        raw_segments = [(start[0], start[1], end[0], end[1], width)]
    return [
        {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "width": sw, "layer": layer, "net": net}
        for x0, y0, x1, y1, sw in raw_segments
    ]


def _parse_via(node, net_map):
    at = _child_floats(node, "at")
    size = _child_atom(node, "size", default=None)
    if len(at) < 2 or size is None:
        return None
    return {
        "x": at[0],
        "y": at[1],
        "size": _safe_float(size, 0.5),
        "layers": _child_strings(node, "layers"),
        "net": _net_name_from_node(node, net_map),
    }


def _parse_edge_segments(node):
    kind = node[0]
    if _child_atom(node, "layer") != "Edge.Cuts":
        return []
    if kind == "gr_line":
        start = _child_floats(node, "start")
        end = _child_floats(node, "end")
        if len(start) >= 2 and len(end) >= 2:
            return [(start[0], start[1], end[0], end[1])]
    if kind == "gr_rect":
        start = _child_floats(node, "start")
        end = _child_floats(node, "end")
        if len(start) >= 2 and len(end) >= 2:
            x0, y0 = start[:2]
            x1, y1 = end[:2]
            min_x, max_x = min(x0, x1), max(x0, x1)
            min_y, max_y = min(y0, y1), max(y0, y1)
            return [
                (min_x, min_y, max_x, min_y),
                (max_x, min_y, max_x, max_y),
                (max_x, max_y, min_x, max_y),
                (min_x, max_y, min_x, min_y),
            ]
    if kind == "gr_poly":
        points = _points_from_pts_node(node)
        return _polyline_segments(points, closed=True)
    if kind == "gr_arc":
        start = _child_floats(node, "start")
        mid = _child_floats(node, "mid")
        end = _child_floats(node, "end")
        if len(start) >= 2 and len(mid) >= 2 and len(end) >= 2:
            return [(x0, y0, x1, y1) for x0, y0, x1, y1, _w in _arc_to_segments(start[:2], mid[:2], end[:2], 0.0)]
        if len(start) >= 2 and len(end) >= 2:
            return [(start[0], start[1], end[0], end[1])]
    if kind == "gr_circle":
        center = _child_floats(node, "center")
        end = _child_floats(node, "end")
        if len(center) >= 2 and len(end) >= 2:
            radius = math.hypot(end[0] - center[0], end[1] - center[1])
            return [(x0, y0, x1, y1) for x0, y0, x1, y1, _w in _circle_to_segments(center[:2], radius)]
    return []


def _polyline_segments(points, closed=False):
    if len(points) < 2:
        return []
    segments = []
    for idx in range(len(points) - 1):
        x0, y0 = points[idx]
        x1, y1 = points[idx + 1]
        segments.append((x0, y0, x1, y1))
    if closed and points[0] != points[-1]:
        x0, y0 = points[-1]
        x1, y1 = points[0]
        segments.append((x0, y0, x1, y1))
    return segments


def _parse_zone(node, net_map):
    layer = _child_atom(node, "layer", default=None)
    if not layer:
        return []
    net_name = _child_atom(node, "net_name", default=None) or _net_name_from_node(node, net_map)
    records = []
    filled = _children(node, "filled_polygon")
    source_polygons = filled if filled else _children(node, "polygon")
    for poly in source_polygons:
        poly_layer = _child_atom(poly, "layer", default=layer)
        points = _points_from_pts_node(poly)
        if len(points) >= 3:
            records.append({"layer": poly_layer, "net": net_name, "points": points})
    return records


def parse_kicad_board(kicad_path):
    with open(kicad_path, "r", encoding="utf-8", errors="ignore") as in_file:
        root = parse_sexp(in_file.read())
    if not _is_node(root, "kicad_pcb"):
        raise ValueError(f"Not a KiCad PCB file: {kicad_path}")

    net_map = {}
    for net_node in _children(root, "net"):
        if len(net_node) >= 3:
            net_map[_safe_int(net_node[1])] = str(net_node[2])

    copper_layers = _parse_layers(root)
    trace_segments = defaultdict(list)
    vias = []
    edge_segments = []
    components = []
    pads = []
    footprint_graphics = []
    zones = defaultdict(list)
    fp_id = 0

    for child in _children(root):
        tag = child[0]
        if tag == "footprint":
            component, fp_pads, graphics = _parse_footprint(child, fp_id, net_map)
            fp_id += 1
            components.append(component)
            pads.extend(fp_pads)
            footprint_graphics.extend(graphics)
        elif tag == "segment":
            seg = _parse_segment(child, net_map)
            if seg and str(seg["layer"]).endswith(".Cu"):
                trace_segments[seg["layer"]].append(seg)
        elif tag == "arc":
            for seg in _parse_arc_segments(child, net_map):
                if str(seg["layer"]).endswith(".Cu"):
                    trace_segments[seg["layer"]].append(seg)
        elif tag == "via":
            via = _parse_via(child, net_map)
            if via:
                vias.append(via)
        elif tag in {"gr_line", "gr_rect", "gr_poly", "gr_arc", "gr_circle"}:
            edge_segments.extend(_parse_edge_segments(child))
        elif tag == "zone":
            for zone in _parse_zone(child, net_map):
                zones[zone["layer"]].append(zone)

    return {
        "path": kicad_path,
        "net_map": net_map,
        "copper_layers": copper_layers,
        "trace_segments": dict(trace_segments),
        "vias": vias,
        "edge_segments": edge_segments,
        "components": components,
        "pads": pads,
        "footprint_graphics": footprint_graphics,
        "zones": dict(zones),
    }


def parse_kicad_board_with_retry(
    kicad_path,
    retry_seconds=DEFAULT_BOARD_READ_RETRY_SECONDS,
):
    """Read a board through eventually-consistent/shared mounts.

    The KiCad service and router service can access a shared board through
    separate processes. Immediately after pcbnew replaces a board, a reader
    can briefly observe an empty, partial, or stale file.
    """
    deadline = time.monotonic() + max(0.0, float(retry_seconds))
    last_error = None
    attempts = 0
    while True:
        attempts += 1
        try:
            board = parse_kicad_board(kicad_path)
            if attempts > 1:
                print(
                    f"KiCad board became readable after {attempts} attempts: "
                    f"{kicad_path}"
                )
            return board
        except (OSError, ValueError) as exc:
            last_error = exc
            if time.monotonic() >= deadline:
                raise last_error
            time.sleep(
                min(
                    BOARD_READ_RETRY_INTERVAL_SECONDS,
                    max(0.0, deadline - time.monotonic()),
                )
            )


def _distance_point_to_segment(px, py, x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    if math.isclose(dx, 0.0, abs_tol=1e-9) and math.isclose(dy, 0.0, abs_tol=1e-9):
        return math.hypot(px - x0, py - y0)
    t = ((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = x0 + t * dx
    proj_y = y0 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


class GridSpec:
    def __init__(self, min_x, min_y, max_x, max_y, cell):
        self.min_x = min_x
        self.min_y = min_y
        self.cell = cell
        self.cols = int(math.ceil((max_x - min_x) / cell)) + 1
        self.rows = int(math.ceil((max_y - min_y) / cell)) + 1

    def in_bounds(self, ix, iy):
        return 0 <= ix < self.cols and 0 <= iy < self.rows

    def bounds_to_indices(self, x_min, x_max, y_min, y_max):
        ix0 = int(math.floor((x_min - self.min_x) / self.cell))
        ix1 = int(math.ceil((x_max - self.min_x) / self.cell))
        iy0 = int(math.floor((y_min - self.min_y) / self.cell))
        iy1 = int(math.ceil((y_max - self.min_y) / self.cell))
        return (
            max(0, ix0),
            min(self.cols - 1, ix1),
            max(0, iy0),
            min(self.rows - 1, iy1),
        )


def _mark_circle_cells(grid, target_set, cx, cy, radius):
    ix0, ix1, iy0, iy1 = grid.bounds_to_indices(cx - radius, cx + radius, cy - radius, cy + radius)
    r2 = radius * radius
    for ix in range(ix0, ix1 + 1):
        x = grid.min_x + ix * grid.cell
        dx = x - cx
        for iy in range(iy0, iy1 + 1):
            y = grid.min_y + iy * grid.cell
            dy = y - cy
            if dx * dx + dy * dy <= r2:
                target_set.add((ix, iy))


def _mark_rect_cells(grid, target_set, cx, cy, width, height, angle_rad):
    half_w = width / 2.0
    half_h = height / 2.0
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    dx = abs(half_w * cos_a) + abs(half_h * sin_a)
    dy = abs(half_w * sin_a) + abs(half_h * cos_a)
    ix0, ix1, iy0, iy1 = grid.bounds_to_indices(cx - dx, cx + dx, cy - dy, cy + dy)
    for ix in range(ix0, ix1 + 1):
        x = grid.min_x + ix * grid.cell
        px = x - cx
        for iy in range(iy0, iy1 + 1):
            y = grid.min_y + iy * grid.cell
            py = y - cy
            local_x = px * cos_a + py * sin_a
            local_y = -px * sin_a + py * cos_a
            if abs(local_x) <= half_w and abs(local_y) <= half_h:
                target_set.add((ix, iy))


def _collect_rect_cells(grid, cx, cy, width, height, angle_rad):
    cells = set()
    _mark_rect_cells(grid, cells, cx, cy, width, height, angle_rad)
    return cells


def _mark_segment_cells(grid, target_set, x0, y0, x1, y1, width):
    half_w = width / 2.0
    ix0, ix1, iy0, iy1 = grid.bounds_to_indices(
        min(x0, x1) - half_w,
        max(x0, x1) + half_w,
        min(y0, y1) - half_w,
        max(y0, y1) + half_w,
    )
    for ix in range(ix0, ix1 + 1):
        x = grid.min_x + ix * grid.cell
        for iy in range(iy0, iy1 + 1):
            y = grid.min_y + iy * grid.cell
            if _distance_point_to_segment(x, y, x0, y0, x1, y1) <= half_w:
                target_set.add((ix, iy))


def _collect_segment_cells(grid, x0, y0, x1, y1, width):
    cells = set()
    _mark_segment_cells(grid, cells, x0, y0, x1, y1, width)
    return cells


def _board_bounds(edge_segments):
    if not edge_segments:
        return None
    min_x = min(min(x0, x1) for x0, _y0, x1, _y1 in edge_segments)
    max_x = max(max(x0, x1) for x0, _y0, x1, _y1 in edge_segments)
    min_y = min(min(y0, y1) for _x0, y0, _x1, y1 in edge_segments)
    max_y = max(max(y0, y1) for _x0, y0, _x1, y1 in edge_segments)
    return min_x, min_y, max_x, max_y


def _compute_grid_spec(pads, trace_segments, vias, edge_segments, components, cell, margin):
    xs = []
    ys = []
    for pad in pads:
        min_x, min_y, max_x, max_y = pad["bbox"]
        xs.extend((min_x, max_x))
        ys.extend((min_y, max_y))
    for segments in (trace_segments or {}).values():
        for seg in segments:
            half_w = seg["width"] / 2.0
            xs.extend((seg["x0"] - half_w, seg["x0"] + half_w, seg["x1"] - half_w, seg["x1"] + half_w))
            ys.extend((seg["y0"] - half_w, seg["y0"] + half_w, seg["y1"] - half_w, seg["y1"] + half_w))
    for via in vias:
        radius = via["size"] / 2.0
        xs.extend((via["x"] - radius, via["x"] + radius))
        ys.extend((via["y"] - radius, via["y"] + radius))
    for x0, y0, x1, y1 in edge_segments or []:
        xs.extend((x0, x1))
        ys.extend((y0, y1))
    for comp in components or []:
        xs.extend((comp["min_x"], comp["max_x"]))
        ys.extend((comp["min_y"], comp["max_y"]))
    if not xs or not ys:
        return None
    min_x = math.floor((min(xs) - margin) / cell) * cell
    min_y = math.floor((min(ys) - margin) / cell) * cell
    max_x = math.ceil((max(xs) + margin) / cell) * cell
    max_y = math.ceil((max(ys) + margin) / cell) * cell
    return GridSpec(min_x, min_y, max_x, max_y, cell)


def _compute_board_inside_cells(grid, edge_segments, clearance):
    if not edge_segments:
        return None
    barrier = set()
    barrier_width = max(grid.cell * 0.9, 0.05)
    for x0, y0, x1, y1 in edge_segments:
        _mark_segment_cells(grid, barrier, x0, y0, x1, y1, barrier_width)

    outside = set()
    queue = deque()

    def seed(ix, iy):
        if (ix, iy) in barrier or (ix, iy) in outside:
            return
        outside.add((ix, iy))
        queue.append((ix, iy))

    for ix in range(grid.cols):
        seed(ix, 0)
        seed(ix, grid.rows - 1)
    for iy in range(grid.rows):
        seed(0, iy)
        seed(grid.cols - 1, iy)

    while queue:
        ix, iy = queue.popleft()
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx = ix + dx
            ny = iy + dy
            if not grid.in_bounds(nx, ny):
                continue
            if (nx, ny) in barrier or (nx, ny) in outside:
                continue
            outside.add((nx, ny))
            queue.append((nx, ny))

    inside = {
        (ix, iy)
        for ix in range(grid.cols)
        for iy in range(grid.rows)
        if (ix, iy) not in barrier and (ix, iy) not in outside
    }
    if clearance > 0 and inside:
        original = inside
        edge_clearance = set()
        for x0, y0, x1, y1 in edge_segments:
            _mark_segment_cells(grid, edge_clearance, x0, y0, x1, y1, 2.0 * clearance)
        inside = inside - edge_clearance
        if not inside:
            inside = original
    return inside or None


def _build_silkscreen_obstacles(pads, trace_segments, vias, grid, clearance, include_traces=True):
    obstacles = set()
    for pad in pads:
        if pad["is_circle"]:
            _mark_circle_cells(grid, obstacles, pad["x"], pad["y"], pad["w"] / 2.0 + clearance)
        else:
            _mark_rect_cells(
                grid,
                obstacles,
                pad["x"],
                pad["y"],
                pad["w"] + 2.0 * clearance,
                pad["h"] + 2.0 * clearance,
                pad["angle"],
            )
    if include_traces:
        for segments in (trace_segments or {}).values():
            for seg in segments:
                _mark_segment_cells(
                    grid,
                    obstacles,
                    seg["x0"],
                    seg["y0"],
                    seg["x1"],
                    seg["y1"],
                    seg["width"] + 2.0 * clearance,
                )
    for via in vias:
        _mark_circle_cells(grid, obstacles, via["x"], via["y"], via["size"] / 2.0 + clearance)
    return obstacles


def _build_pad_obstacles_by_side(pads, grid, clearance):
    obstacles_by_side = {"F.Cu": set(), "B.Cu": set()}

    def mark(target, pad):
        if pad["is_circle"]:
            _mark_circle_cells(grid, target, pad["x"], pad["y"], pad["w"] / 2.0 + clearance)
        else:
            _mark_rect_cells(
                grid,
                target,
                pad["x"],
                pad["y"],
                pad["w"] + 2.0 * clearance,
                pad["h"] + 2.0 * clearance,
                pad["angle"],
            )

    for pad in pads:
        if pad.get("is_th"):
            sides = ("F.Cu", "B.Cu")
        else:
            side = pad.get("side")
            sides = (side,) if side in obstacles_by_side else ("F.Cu",)
        for side in sides:
            mark(obstacles_by_side[side], pad)
    return obstacles_by_side


def _build_footprint_obstacles_by_side(components, grid, clearance):
    obstacles_by_side = {"F.Cu": set(), "B.Cu": set()}
    obstacles_by_component = {}
    for component in components or []:
        width = max(0.0, component["max_x"] - component["min_x"])
        height = max(0.0, component["max_y"] - component["min_y"])
        cells = _collect_rect_cells(
            grid,
            component["cx"],
            component["cy"],
            width + 2.0 * clearance,
            height + 2.0 * clearance,
            0.0,
        )
        obstacles_by_component[component.get("id")] = cells
        if component.get("has_through_hole"):
            sides = ("F.Cu", "B.Cu")
        else:
            side = component.get("side")
            sides = (side,) if side in obstacles_by_side else ("F.Cu",)
        for side in sides:
            obstacles_by_side[side].update(cells)
    return obstacles_by_side, obstacles_by_component


def _estimate_text_box(text, height_mm):
    # KiCad's stroke-font bounding box is materially larger than its nominal
    # text size. These factors are calibrated against pcbnew.GetBoundingBox()
    # so placement reserves the rendered glyph area rather than the nominal
    # size written to the board.
    return (
        max(1, len(text)) * height_mm * SILK_RENDERED_CHAR_WIDTH_FACTOR,
        height_mm * SILK_RENDERED_HEIGHT_FACTOR,
    )


def _angle_deviation_from_axes(cx, cy, x, y):
    dx = x - cx
    dy = y - cy
    if math.isclose(dx, 0.0, abs_tol=1e-9) and math.isclose(dy, 0.0, abs_tol=1e-9):
        return 0.0
    angle = math.degrees(math.atan2(dy, dx)) % 90.0
    return min(angle, 90.0 - angle)


def _slide_offsets(reach, step):
    """Center-outward offsets covering [-reach, +reach] at `step` spacing: 0,
    then -step/+step, -2*step/+2*step, ... nearest-to-centerline first.
    Ported from silkscreen_generator_revamp.py."""
    offsets = [0.0]
    j = step
    while j <= reach + 1e-9:
        offsets.append(-j)
        offsets.append(j)
        j += step
    return offsets


_CANDIDATE_CACHE = {}


def _generate_label_candidates(
    bounds,
    width,
    height,
    max_distance_mm,
    prefer_vertical=False,
    preferred_side=None,
    dense=False,
):
    """Memoized front for _generate_label_candidates_uncached: the same
    component is asked for the same candidate ring several times per run
    (orientation scoring, the main loop, the rescues), and the answer
    depends only on the arguments below. Callers never mutate the returned
    list. The cache is cleared at the start of every auto_place_silkscreen
    call."""
    key = (
        bounds["cx"], bounds["cy"], bounds["min_x"], bounds["max_x"], bounds["min_y"], bounds["max_y"],
        width, height, max_distance_mm, bool(prefer_vertical), preferred_side, bool(dense),
    )
    cached = _CANDIDATE_CACHE.get(key)
    if cached is None:
        cached = _generate_label_candidates_uncached(
            bounds, width, height, max_distance_mm,
            prefer_vertical=prefer_vertical, preferred_side=preferred_side, dense=dense,
        )
        _CANDIDATE_CACHE[key] = cached
    return cached


def _generate_label_candidates_uncached(
    bounds,
    width,
    height,
    max_distance_mm,
    prefer_vertical=False,
    preferred_side=None,
    dense=False,
):
    cx = bounds["cx"]
    cy = bounds["cy"]
    min_x = bounds["min_x"]
    max_x = bounds["max_x"]
    min_y = bounds["min_y"]
    max_y = bounds["max_y"]
    max_offset = max(SILK_OFFSET_MM, max_distance_mm)
    offsets = []
    offset = SILK_OFFSET_MM
    step = max(GRID_MM, SILK_FAR_STEP_MM)
    while offset <= max_offset + 1e-9:
        offsets.append(round(offset, 6))
        offset += step
    offsets.append(round(max_offset, 6))
    offsets = sorted(set(offsets))
    comp_half_w = (max_x - min_x) / 2.0
    comp_half_h = (max_y - min_y) / 2.0
    candidates = []
    seen = set()

    def emit(pos, side):
        if pos not in seen:
            seen.add(pos)
            candidates.append((side, pos))

    side_priority = ["top", "bottom", "right", "left"]
    if preferred_side in side_priority:
        side_priority = [preferred_side] + [
            side for side in side_priority if side != preferred_side
        ]

    for offset in offsets:
        top_y = max_y + offset + height / 2.0
        bottom_y = min_y - offset - height / 2.0
        right_x = max_x + offset + width / 2.0
        left_x = min_x - offset - width / 2.0
        if not dense:
            side_positions = {
                "top": (cx, top_y),
                "bottom": (cx, bottom_y),
                "right": (right_x, cy),
                "left": (left_x, cy),
            }
            if preferred_side in side_positions:
                cardinal_sides = [preferred_side] + [
                    side for side in ("top", "bottom", "right", "left")
                    if side != preferred_side
                ]
            else:
                cardinal_sides = (
                    ["right", "left", "top", "bottom"]
                    if prefer_vertical
                    else ["top", "bottom", "right", "left"]
                )
            for side in cardinal_sides:
                emit(side_positions[side], side)
            corner_options = [
                (right_x, top_y),
                (left_x, top_y),
                (right_x, bottom_y),
                (left_x, bottom_y),
            ]
            for pos in corner_options:
                emit(pos, "corner")
            continue
        # Dense ring (ported from silkscreen_generator_revamp.py): slide the
        # label's center along all four edges of the ring at
        # SILK_SLIDE_STEP_MM steps instead of only the 8 sparse anchors, so
        # any free pocket on the ring's perimeter can be found, not just the
        # edge-midpoints and corners. Slide reach matches the old diagonal
        # corner extent, so the ring still closes exactly as before.
        h_slides = _slide_offsets(comp_half_w + offset + width / 2.0, SILK_SLIDE_STEP_MM)
        v_slides = _slide_offsets(comp_half_h + offset + height / 2.0, SILK_SLIDE_STEP_MM)
        for idx in range(max(len(h_slides), len(v_slides))):
            for side in side_priority:
                if side in ("top", "bottom") and idx < len(h_slides):
                    dx = h_slides[idx]
                    emit((cx + dx, top_y if side == "top" else bottom_y), side)
                elif side in ("right", "left") and idx < len(v_slides):
                    dy = v_slides[idx]
                    emit((right_x if side == "right" else left_x, cy + dy), side)
    # preferred_side (if any) now dominates the sort -- every candidate on
    # that side is tried, at any offset/slide index, before any candidate
    # on a different side. This used to be effectively reversed: sorting
    # purely by angle-deviation let a *closer* non-preferred-side candidate
    # outrank a *slightly farther* preferred-side one, silently overriding
    # the family's preference the moment its very nearest positions were
    # blocked. Angle-deviation is now only the secondary/tie-break key
    # within whichever side-priority tier a candidate falls into (still
    # prefers cardinal-aligned, near-ring positions within that tier,
    # exactly as before). With no preferred_side, every item falls into
    # the same tier and this reduces to the original pure-deviation order.
    candidates.sort(
        key=lambda item: (
            0 if item[0] == preferred_side else 1,
            _angle_deviation_from_axes(cx, cy, item[1][0], item[1][1]),
        )
    )
    return [pos for _side, pos in candidates]


def _label_orientation_specs(bounds, width_mm, height_mm):
    footprint_width = max(0.0, bounds["max_x"] - bounds["min_x"])
    footprint_height = max(0.0, bounds["max_y"] - bounds["min_y"])
    horizontal = (0.0, width_mm, height_mm, False)
    vertical = (90.0, height_mm, width_mm, True)
    return [vertical, horizontal] if footprint_height > footprint_width else [horizontal, vertical]


def _orientation_space_score_grid(
    bounds, width_mm, height_mm, grid, inside_cells, obstacles, footprint_obstacles_by_side,
    preferred_orientation=None,
):
    """Reorder the two orientation specs by how much free near-tier room
    each one actually has, instead of relying only on the footprint's own
    aspect ratio. If preferred_orientation ("horizontal"/"vertical") is
    given -- a family/row's own consensus, see
    _cluster_preferred_orientation_grid -- it's tried first, as a starting
    point only: the other orientation is still tried after if the
    preferred one doesn't work out, exactly as before. After that, plain
    horizontal (0 degrees, reads left-to-right) is preferred over vertical
    -- people read a board in one direction, so this sorts toward that
    without forcing it: vertical is still tried right after if horizontal
    finds no legal spot at all. With no family preference, behavior is
    otherwise unchanged: free-space count still breaks the final tie."""
    side = bounds.get("side")
    if side not in footprint_obstacles_by_side:
        side = "F.Cu"
    scored = []
    for spec in _label_orientation_specs(bounds, width_mm, height_mm):
        _rotation_deg, collision_width, collision_height, is_vertical = spec
        near = _generate_label_candidates(
            bounds, collision_width, collision_height, SILK_MAX_DISTANCE_MM, dense=True,
        )
        free = 0
        for x, y in near:
            cells = _collect_rect_cells(grid, x, y, collision_width, collision_height, 0.0)
            if not cells:
                continue
            if inside_cells is not None and not cells.issubset(inside_cells):
                continue
            if cells & obstacles:
                continue
            if cells & footprint_obstacles_by_side.get(side, set()):
                continue
            free += 1
            if free >= SILK_ORIENTATION_SCORE_CAP:
                break
        orientation_name = "vertical" if is_vertical else "horizontal"
        is_preferred = 0 if orientation_name == preferred_orientation else 1
        # A family member (marked "_family" after family clustering) has no
        # 0-degree tier: for it the two orientations are judged on family
        # consensus and free space alone, exactly as before the 0-degree rule.
        prefers_horizontal = 0 if bounds.get("_family") else (1 if is_vertical else 0)
        scored.append((is_preferred, prefers_horizontal, -free, spec))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    return [spec for _pref, _horiz, _neg_free, spec in scored]


def _orientation_space_score_geometry(
    bounds, width_mm, height_mm, footprint_trees, strict_tree, inside_board_fn, side,
    preferred_orientation=None, inside_board_mask=None,
):
    """Geometry-backend counterpart of _orientation_space_score_grid, using
    the Shapely trees instead of the grid's cell sets. Same preference
    order: family consensus first, then plain horizontal (0 degrees) over
    vertical, then free-space count as the final tie-break. With an
    inside_board_mask (vectorized inside_board over an array of
    geometries) the free-space count is taken in bulk; the count is the
    same, since it was capped at SILK_ORIENTATION_SCORE_CAP either way."""
    if side not in footprint_trees:
        side = "F.Cu"
    scored = []
    for spec in _label_orientation_specs(bounds, width_mm, height_mm):
        _rotation_deg, collision_width, collision_height, is_vertical = spec
        near = _generate_label_candidates(
            bounds, collision_width, collision_height, SILK_MAX_DISTANCE_MM, dense=True,
        )
        free = 0
        if inside_board_mask is not None:
            boxes = _boxes_at(near, collision_width, collision_height)
            if len(boxes):
                usable = inside_board_mask(boxes)
                usable &= ~_tree_hit_mask(footprint_trees.get(side), boxes)
                usable &= ~_tree_hit_mask(strict_tree, boxes)
                free = min(int(usable.sum()), SILK_ORIENTATION_SCORE_CAP)
        else:
            for x, y in near:
                geometry = _rotated_box(x, y, collision_width, collision_height)
                if not inside_board_fn(geometry):
                    continue
                if _tree_intersects(footprint_trees.get(side), geometry):
                    continue
                if _tree_intersects(strict_tree, geometry):
                    continue
                free += 1
                if free >= SILK_ORIENTATION_SCORE_CAP:
                    break
        orientation_name = "vertical" if is_vertical else "horizontal"
        is_preferred = 0 if orientation_name == preferred_orientation else 1
        # A family member (marked "_family" after family clustering) has no
        # 0-degree tier: for it the two orientations are judged on family
        # consensus and free space alone, exactly as before the 0-degree rule.
        prefers_horizontal = 0 if bounds.get("_family") else (1 if is_vertical else 0)
        scored.append((is_preferred, prefers_horizontal, -free, spec))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    return [spec for _pref, _horiz, _neg_free, spec in scored]


def _reference_family(ref):
    # Groups refs by their type prefix (e.g. "IN407" -> "IN", "R101" -> "R")
    # so label-side consistency is only encouraged within the same part type.
    match = re.match(r"[A-Za-z]+", ref or "")
    return match.group(0).upper() if match else (ref or "")


def _bbox_gap_distance(a, b):
    dx = max(0.0, a["min_x"] - b["max_x"], b["min_x"] - a["max_x"])
    dy = max(0.0, a["min_y"] - b["max_y"], b["min_y"] - a["max_y"])
    return math.hypot(dx, dy)


def _compact_cluster_no_overlap(
    members, specs_by_id, positions, centroid_x, centroid_y,
    step_mm=SILK_CLUSTER_COMPACT_STEP_MM, max_rounds=1000,
):
    """Mutates `positions` in place. Each round, every member takes one
    step_mm step straight toward the shared centroid; any overlap that
    step creates is then corrected by pushing just that pair apart by
    exactly half the overlap amount, along whichever axis needs the
    smaller push (the standard minimum-translation collision fix).
    Repeats until a full round produces no shrink step and no correction
    (converged) or max_rounds is hit.

    Pushing a colliding pair apart -- rather than simply refusing to move
    either of them, as an earlier version of this did -- is the important
    part: a member stuck directly behind an unrelated neighbor on its
    straight line to the centroid gets shoved sideways by the correction
    instead of freezing in place, so it can keep working its way further
    in on later rounds. The only hard rule enforced is "never overlap" --
    there's no attempt to preserve the exact original spacing ratios
    between members, only however each one keeps landing relative to the
    others as the whole group is squeezed inward."""

    detect_eps = 1e-9  # standard AABB overlap test needs only ox>0 and
    # oy>0 -- this is purely a float-noise guard, not a human-scale
    # tolerance. Using anything larger here (tried 1e-3 first) is a real
    # bug: it lets a pair with near-total overlap on one axis slip through
    # completely undetected whenever the *other* axis happens to have a
    # tiny gap, since detection was wrongly requiring *both* axes to clear
    # the threshold instead of the standard "both axes strictly positive."
    push_margin = 1e-6  # small buffer added past exact touching so a
    # resolved pair doesn't immediately re-trigger detection next round
    # due to float rounding landing it back at (or a hair past) zero.

    def overlap_amount(id_a, id_b):
        ax, ay = positions[id_a]
        bx, by = positions[id_b]
        half_w = specs_by_id[id_a][4] / 2.0 + specs_by_id[id_b][4] / 2.0
        half_h = specs_by_id[id_a][5] / 2.0 + specs_by_id[id_b][5] / 2.0
        overlap_x = half_w - abs(ax - bx)
        overlap_y = half_h - abs(ay - by)
        return (overlap_x, overlap_y) if overlap_x > detect_eps and overlap_y > detect_eps else None

    def resolve(id_a, id_b, overlap_x, overlap_y):
        ax, ay = positions[id_a]
        bx, by = positions[id_b]
        if overlap_x < overlap_y:
            push = overlap_x / 2.0 + push_margin
            sign = 1.0 if ax >= bx else -1.0
            positions[id_a] = (ax + sign * push, ay)
            positions[id_b] = (bx - sign * push, by)
        else:
            push = overlap_y / 2.0 + push_margin
            sign = 1.0 if ay >= by else -1.0
            positions[id_a] = (ax, ay + sign * push)
            positions[id_b] = (bx, by - sign * push)

    ids = [m["id"] for m in members]
    for _ in range(max_rounds):
        shrank = False
        for member_id in ids:
            x, y = positions[member_id]
            dx, dy = centroid_x - x, centroid_y - y
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                continue
            step = min(step_mm, dist)
            positions[member_id] = (x + dx / dist * step, y + dy / dist * step)
            shrank = True

        corrected = False
        for _resolve_pass in range(50):
            any_overlap = False
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    amounts = overlap_amount(ids[i], ids[j])
                    if amounts is not None:
                        resolve(ids[i], ids[j], *amounts)
                        any_overlap = True
                        corrected = True
            if not any_overlap:
                break

        if not shrank and not corrected:
            break

    # Final guarantee, decoupled from the shrink dynamics above: keep
    # resolving until a full pass finds nothing left to fix, so the
    # returned positions are certain to be overlap-free regardless of how
    # the shrink/resolve interplay happened to leave things.
    for _final_pass in range(1000):
        any_overlap = False
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                amounts = overlap_amount(ids[i], ids[j])
                if amounts is not None:
                    resolve(ids[i], ids[j], *amounts)
                    any_overlap = True
        if not any_overlap:
            break


def _pack_cluster_grid(members):
    """Compacts a dense cluster by starting every member at its real
    center position and iteratively squeezing the whole group inward
    (_compact_cluster_no_overlap) until nothing can move any closer
    without overlapping. Replaces an earlier analytic approach (scale
    every center toward the centroid by whichever single factor keeps the
    tightest-fitting pair just clear) that was too tied to the exact
    original center pattern: a lone outlier member's real distance from
    the rest of the cluster could keep the uniform scale small everywhere
    else too, and a member stuck directly behind another on its straight
    path to the centroid would simply stop rather than route around it,
    leaving real gaps far bigger than the labels themselves in the result.
    Iterative squeeze-with-collision-correction fixes both: the only hard
    rule is "never overlap," not "preserve the original spacing ratio."

    Every member is laid out reading left-to-right (0 degrees): the block
    is going to a spot chosen for it anyway, so the pipeline-wide "0
    degrees unless it does not fit" preference applies as-is. (An earlier
    version used each footprint's own longer side, which left most block
    members turned 90 degrees for no spatial reason.)

    Returns (specs, block_width_mm, block_height_mm) where specs is a list
    of (member, width_mm, height_mm, rotation_deg, collision_width,
    collision_height, local_x, local_y) -- local_x/local_y are the offsets
    from the block's own top-left corner.
    """
    specs_by_id = {}
    for m in members:
        w_mm, h_mm = _estimate_text_box(m["ref"], SILK_TEXT_HEIGHT_MM)
        specs = _label_orientation_specs(m, w_mm, h_mm)
        rotation_deg, cw, ch, _is_v = next((s for s in specs if abs(s[0]) < 1e-9), specs[0])
        specs_by_id[m["id"]] = (m, w_mm, h_mm, rotation_deg, cw, ch)

    centroid_x = sum(m["cx"] for m in members) / len(members)
    centroid_y = sum(m["cy"] for m in members) / len(members)
    positions = {m["id"]: (m["cx"], m["cy"]) for m in members}

    _compact_cluster_no_overlap(members, specs_by_id, positions, centroid_x, centroid_y)

    min_x = min(positions[m["id"]][0] - specs_by_id[m["id"]][4] / 2.0 for m in members)
    min_y = min(positions[m["id"]][1] - specs_by_id[m["id"]][5] / 2.0 for m in members)
    max_x = max(positions[m["id"]][0] + specs_by_id[m["id"]][4] / 2.0 for m in members)
    max_y = max(positions[m["id"]][1] + specs_by_id[m["id"]][5] / 2.0 for m in members)

    specs = []
    for m in members:
        member, w_mm, h_mm, rotation_deg, cw, ch = specs_by_id[m["id"]]
        px, py = positions[m["id"]]
        specs.append((member, w_mm, h_mm, rotation_deg, cw, ch, px - min_x, py - min_y))

    return specs, max_x - min_x, max_y - min_y


def _shape_key(comp):
    """(width, height) of a component's bounding box, in board-space axes
    exactly as placed -- NOT rotation-normalized. A footprint rendered as a
    tall vertical rectangle and the same-dimensioned footprint rendered as a
    wide horizontal rectangle look different on the actual board and should
    not be treated as a shape match. (A genuine 180-degree flip doesn't
    swap width/height, so this doesn't break same-shape matching for the
    alternating-rotation rows the axis check already allows -- only a truly
    90-degree-different rendering is excluded, which the separate rotation
    check would reject anyway.)"""
    width = comp["max_x"] - comp["min_x"]
    height = comp["max_y"] - comp["min_y"]
    return (width, height)


def _dimension_matches(a, b, tolerance_pct=SILK_FAMILY_SHAPE_TOLERANCE_PCT):
    """True if two lengths are within tolerance_pct of the larger one."""
    largest = max(a, b)
    if largest <= 1e-9:
        return True
    return abs(a - b) <= tolerance_pct * largest


def _shape_matches(a, b, tolerance_pct=SILK_FAMILY_SHAPE_TOLERANCE_PCT):
    """True if two components' footprint bounding boxes are the same shape
    as actually rendered on the board (width matches width, height matches
    height, each within tolerance_pct)."""
    a_w, a_h = _shape_key(a)
    b_w, b_h = _shape_key(b)
    return (
        _dimension_matches(a_w, b_w, tolerance_pct)
        and _dimension_matches(a_h, b_h, tolerance_pct)
    )


def _cluster_by_family_shape_and_gap(candidates, max_gap_mm):
    """Shared clustering core: chains same-family, same-shape components
    within max_gap_mm of each other (transitively). Two independent callers
    use this with two different gaps -- _build_family_clusters (loose, for
    pairs and side-preference voting) and _build_row_clusters (strict, for
    row detection) -- so tuning one gap can never change the other's
    results; they don't share a clustering pass, only this algorithm."""
    parent = {comp["id"]: comp["id"] for comp in candidates}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    by_family = defaultdict(list)
    for comp in candidates:
        by_family[_reference_family(comp.get("ref"))].append(comp)

    for members in by_family.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if _bbox_gap_distance(members[i], members[j]) <= max_gap_mm and _shape_matches(
                    members[i], members[j]
                ):
                    union(members[i]["id"], members[j]["id"])

    clusters = defaultdict(list)
    for comp in candidates:
        clusters[find(comp["id"])].append(comp)
    return [members for members in clusters.values() if len(members) >= 2]


def _build_family_clusters(candidates, max_gap_mm=SILK_FAMILY_CLUSTER_GAP_MM):
    """Chain same-family, same-shape components into clusters using a loose
    gap (SILK_FAMILY_CLUSTER_GAP_MM) -- feeds pair (twin/mirror)
    classification and the side-preference voting that actually influences
    placement. Deliberately independent from row detection, see
    _build_row_clusters."""
    return _cluster_by_family_shape_and_gap(candidates, max_gap_mm)


def _build_row_clusters(candidates, max_gap_mm=SILK_ROW_ADJACENT_GAP_MM):
    """Chain same-family, same-shape components into clusters using strict
    adjacency (SILK_ROW_ADJACENT_GAP_MM) -- feeds row detection only.
    Because the union criterion itself requires strict adjacency, two
    separate physical rows can never merge into one cluster here, unlike
    _build_family_clusters's loose gap. Independent on purpose: tuning this
    can never change pair detection or side-preference voting, and vice
    versa."""
    return _cluster_by_family_shape_and_gap(candidates, max_gap_mm)


def _rotation_diff_deg(a, b):
    """Smallest angular difference between two rotations, folded into [0, 180]."""
    diff = abs((a - b) % 360.0)
    return min(diff, 360.0 - diff)


def _axis_diff_deg(a, b):
    """Smallest angular difference between two rotations when facing
    direction doesn't matter -- 0 and 180 degrees are the same axis (e.g. a
    resistor flipped for routing occupies the same footprint outline either
    way). Folded into [0, 90]."""
    diff = abs((a - b) % 180.0)
    return min(diff, 180.0 - diff)


def _best_axis_agreement(rotations, tolerance_deg):
    """Given a list of rotations, finds the axis (mod 180) that the most of
    them agree with. Returns (best_axis, indices_that_agree)."""
    best_axis = rotations[0]
    best_indices = []
    for candidate in rotations:
        indices = [
            i for i, rot in enumerate(rotations)
            if _axis_diff_deg(candidate, rot) <= tolerance_deg
        ]
        if len(indices) > len(best_indices):
            best_indices = indices
            best_axis = candidate
    return best_axis, best_indices


def _spatial_order(members):
    """Sorts members along whichever axis their centers spread out over more
    -- the dominant direction the row actually runs in, independent of each
    member's own rotation."""
    xs = [m["cx"] for m in members]
    ys = [m["cy"] for m in members]
    if (max(xs) - min(xs)) >= (max(ys) - min(ys)):
        return sorted(members, key=lambda m: m["cx"])
    return sorted(members, key=lambda m: m["cy"])


def _consecutive_gaps(ordered_members):
    return [
        _bbox_gap_distance(ordered_members[i], ordered_members[i + 1])
        for i in range(len(ordered_members) - 1)
    ]


def _longest_adjacent_run(members, adjacent_gap_mm=SILK_ROW_ADJACENT_GAP_MM):
    """Sorts members along the row's dominant axis and returns the longest
    run of consecutive members each strictly adjacent (gap at or below
    adjacent_gap_mm) to the next. This is what keeps a cluster that
    accidentally spans two separate physical rows -- chained together only
    through the loose clustering gap, e.g. one close pair bridging them --
    from being treated as one row: the real row is whichever run is
    longest, not the whole merged group."""
    if len(members) < 2:
        return list(members)
    ordered = _spatial_order(members)
    gaps = _consecutive_gaps(ordered)
    best_run = [ordered[0]]
    current_run = [ordered[0]]
    for member, gap in zip(ordered[1:], gaps):
        current_run = current_run + [member] if gap <= adjacent_gap_mm else [member]
        if len(current_run) > len(best_run):
            best_run = current_run
    return best_run


def _classify_family_pattern(
    members,
    tolerance_deg=SILK_PARALLEL_ROTATION_TOLERANCE_DEG,
    row_majority_threshold=SILK_ROW_MAJORITY_THRESHOLD,
):
    """Classifies a family cluster (as returned by _build_family_clusters):
    - "twin_pair": exactly 2 members, near-identical rotation.
    - "mirror_pair": exactly 2 members, rotations ~180 degrees apart.
    - "row": 3+ members. Among those sharing one common rotation axis (mod
      180 -- facing direction doesn't matter), find the longest strictly
      adjacent run (_longest_adjacent_run); if that run covers at least
      row_majority_threshold of the whole cluster, it's a row. Everyone else
      -- rotation outliers and any spatially separate sub-group, e.g. a
      second physical row chained in only via the loose clustering gap --
      is reported as an outlier (see _row_outliers), not a reason to reject
      the classification.
    - None: no consistent relationship found.
    """
    if len(members) < 2:
        return None
    rotations = [m.get("rotation", 0.0) or 0.0 for m in members]
    if len(members) == 2:
        diff = _rotation_diff_deg(rotations[0], rotations[1])
        if diff <= tolerance_deg:
            return "twin_pair"
        if abs(diff - 180.0) <= tolerance_deg:
            return "mirror_pair"
        return None
    _best_axis, agreeing = _best_axis_agreement(rotations, tolerance_deg)
    aligned_members = [members[i] for i in agreeing]
    row_run = _longest_adjacent_run(aligned_members)
    if len(row_run) / len(members) < row_majority_threshold:
        return None
    return "row"


def _row_outliers(members, tolerance_deg=SILK_PARALLEL_ROTATION_TOLERANCE_DEG):
    """For a cluster classified as "row", returns the refs of members not in
    the winning run: rotation outliers, plus any spatially separate
    sub-group (e.g. a second physical row chained in only via the loose
    clustering gap) even if its rotation matched."""
    rotations = [m.get("rotation", 0.0) or 0.0 for m in members]
    _best_axis, agreeing = _best_axis_agreement(rotations, tolerance_deg)
    aligned_members = [members[i] for i in agreeing]
    row_run = _longest_adjacent_run(aligned_members)
    row_run_ids = {id(m) for m in row_run}
    return [m.get("ref") for m in members if id(m) not in row_run_ids]


def _classify_family_clusters(clusters):
    """Row logic runs first: for every loose cluster (_build_family_clusters),
    strict re-clustering (_build_row_clusters), scoped to just that
    cluster's own members, extracts every valid row (3+, strictly adjacent,
    rotation-aligned) -- so a loose cluster that actually spans multiple
    separate physical rows is reported as multiple rows, not one
    incoherent blob. Pair logic runs second, only on whatever wasn't
    captured by a row -- including loose clusters that were only ever 2
    members to begin with, since those can never contain a row at all. The
    pair check uses the original loose grouping (not the strict gap), so a
    genuine pair spaced beyond SILK_ROW_ADJACENT_GAP_MM still gets found.
    Purely observational/reporting -- does not feed into or change any
    placement decision yet."""
    summary = {"row": 0, "mirror_pair": 0, "twin_pair": 0, "unclassified": 0}
    detail = []

    def record(members, pattern):
        key = pattern or "unclassified"
        summary[key] += 1
        entry = {"pattern": key, "members": [m.get("ref") for m in members]}
        if pattern == "row":
            entry["outliers"] = _row_outliers(members)
        detail.append(entry)

    for members in clusters:
        covered_ids = set()
        for strict_members in _build_row_clusters(members):
            if len(strict_members) < 3:
                continue
            pattern = _classify_family_pattern(strict_members)
            if pattern == "row":
                record(strict_members, pattern)
                covered_ids.update(m["id"] for m in strict_members)

        remaining = [m for m in members if m["id"] not in covered_ids]
        if len(remaining) == 2:
            record(remaining, _classify_family_pattern(remaining))
        elif remaining:
            record(remaining, None)

    return summary, detail


def _cluster_preferred_side_votes(members, preferred_orientation=None):
    """Per-member side -> (x, y, collision_width, collision_height) at zero
    offset, used to vote on the cluster's majority-feasible label side.
    Uses the cluster's own orientation preference (see
    _cluster_preferred_orientation_grid/_geometry) for each member's
    collision box when one exists, instead of always defaulting to that
    member's own footprint-aspect-ratio orientation -- so the side vote is
    computed against the same shape placement will actually try, not a
    stale assumption that may not match once orientation is decided."""
    votes = []
    for comp in members:
        width_mm, height_mm = _estimate_text_box(comp["ref"], SILK_TEXT_HEIGHT_MM)
        specs = _label_orientation_specs(comp, width_mm, height_mm)
        if preferred_orientation == "vertical":
            spec = next((s for s in specs if s[3]), specs[0])
        elif preferred_orientation == "horizontal":
            spec = next((s for s in specs if not s[3]), specs[0])
        else:
            spec = specs[0]
        _rotation_deg, collision_width, collision_height, _is_vertical = spec
        cx, cy = comp["cx"], comp["cy"]
        # A zero offset places the label's near edge exactly on the
        # component's own bounding box edge -- but SILK_FOOTPRINT_CLEARANCE_MM
        # expands every component's own footprint into an obstacle too (no
        # self-exclusion), so a zero-gap position always overlaps its own
        # halo and would never pass the feasibility check below, for any
        # component, on any side. Push past that halo so the vote can
        # actually pass when a side genuinely has room.
        clear = SILK_FOOTPRINT_CLEARANCE_MM + SILK_SLIDE_STEP_MM
        positions = {
            "top": (cx, comp["max_y"] + clear + collision_height / 2.0),
            "bottom": (cx, comp["min_y"] - clear - collision_height / 2.0),
            "right": (comp["max_x"] + clear + collision_width / 2.0, cy),
            "left": (comp["min_x"] - clear - collision_width / 2.0, cy),
        }
        votes.append((comp, collision_width, collision_height, positions))
    return votes


def _label_gap_distance(bounds, x, y, width, height):
    label_min_x = x - width / 2.0
    label_max_x = x + width / 2.0
    label_min_y = y - height / 2.0
    label_max_y = y + height / 2.0
    dx = max(bounds["min_x"] - label_max_x, label_min_x - bounds["max_x"], 0.0)
    dy = max(bounds["min_y"] - label_max_y, label_min_y - bounds["max_y"], 0.0)
    return math.hypot(dx, dy)


def _count_nearby_components(x, y, radius_mm, components, exclude_id):
    """How many components (besides exclude_id) have any part of their
    bounding box within radius_mm of (x, y) -- stops counting the moment
    it reaches SILK_ARROW_NEARBY_COUNT, since callers only care whether
    it's "crowded enough," not the exact count."""
    count = 0
    r2 = radius_mm * radius_mm
    for c in components:
        if c.get("id") == exclude_id:
            continue
        dx = max(0.0, c["min_x"] - x, x - c["max_x"])
        dy = max(0.0, c["min_y"] - y, y - c["max_y"])
        if dx * dx + dy * dy <= r2:
            count += 1
            if count >= SILK_ARROW_NEARBY_COUNT:
                break
    return count


def _label_needs_pointer(bounds, x, y, width, height, gap_mm, components=None, exclude_id=None):
    """A genuinely far placement always needs a pointer. A diagonal
    (off-axis) placement only needs one if it risks being confused for a
    different component -- checked via how many other components are
    actually nearby (components=None skips this check entirely and
    treats every diagonal placement as unambiguous, e.g. for callers that
    already unconditionally want a pointer for other reasons)."""
    if gap_mm > SILK_ARROW_THRESHOLD_MM:
        return True
    label_min_x = x - width / 2.0
    label_max_x = x + width / 2.0
    label_min_y = y - height / 2.0
    label_max_y = y + height / 2.0
    aligned_x = label_max_x >= bounds["min_x"] and label_min_x <= bounds["max_x"]
    aligned_y = label_max_y >= bounds["min_y"] and label_min_y <= bounds["max_y"]
    diagonal = not aligned_x and not aligned_y
    if not diagonal or not components:
        return False
    return _count_nearby_components(x, y, SILK_ARROW_NEARBY_RADIUS_MM, components, exclude_id) >= SILK_ARROW_NEARBY_COUNT


def _arrow_points(bounds, x, y, width, height):
    cx = bounds["cx"]
    cy = bounds["cy"]
    dx_center = x - cx
    dy_center = y - cy
    if math.isclose(dx_center, 0.0, abs_tol=1e-9) and math.isclose(dy_center, 0.0, abs_tol=1e-9):
        return None

    scales = []
    if not math.isclose(dx_center, 0.0, abs_tol=1e-9):
        edge_x = bounds["max_x"] if dx_center > 0 else bounds["min_x"]
        scales.append((edge_x - cx) / dx_center)
    if not math.isclose(dy_center, 0.0, abs_tol=1e-9):
        edge_y = bounds["max_y"] if dy_center > 0 else bounds["min_y"]
        scales.append((edge_y - cy) / dy_center)
    positive_scales = [scale for scale in scales if scale >= 0.0]
    scale = min(positive_scales) if positive_scales else 0.0
    unit_len = math.hypot(dx_center, dy_center)
    unit_x = dx_center / unit_len
    unit_y = dy_center / unit_len
    start_x = cx + dx_center * scale + unit_x * SILK_ARROW_MARGIN_MM
    start_y = cy + dy_center * scale + unit_y * SILK_ARROW_MARGIN_MM

    dx = x - start_x
    dy = y - start_y
    if dx > 0:
        end_x = x - width / 2.0 - SILK_ARROW_MARGIN_MM
    elif dx < 0:
        end_x = x + width / 2.0 + SILK_ARROW_MARGIN_MM
    else:
        end_x = x
    if dy > 0:
        end_y = y - height / 2.0 - SILK_ARROW_MARGIN_MM
    elif dy < 0:
        end_y = y + height / 2.0 + SILK_ARROW_MARGIN_MM
    else:
        end_y = y
    if math.hypot(end_x - start_x, end_y - start_y) <= GRID_MM:
        return None
    return start_x, start_y, end_x, end_y


def _rect_to_rect_line(from_cx, from_cy, from_half_w, from_half_h, to_cx, to_cy, to_half_w, to_half_h):
    """Same idea as _arrow_points -- a straight line from one rectangle's
    edge to the other's, each trimmed back by SILK_ARROW_MARGIN_MM -- but
    generalized to two arbitrary rectangles instead of a single component
    and its label. Used for the one shared connector a dense-cluster block
    gets back to the cluster's own location (as opposed to a per-component
    arrow, which _arrow_points already handles)."""
    dx_center = to_cx - from_cx
    dy_center = to_cy - from_cy
    length = math.hypot(dx_center, dy_center)
    if length <= 1e-9:
        return None
    ux, uy = dx_center / length, dy_center / length

    def edge_point(cx, cy, half_w, half_h, sign):
        scales = []
        if not math.isclose(ux, 0.0, abs_tol=1e-9):
            scales.append(half_w / abs(ux))
        if not math.isclose(uy, 0.0, abs_tol=1e-9):
            scales.append(half_h / abs(uy))
        scale = min(scales) if scales else 0.0
        return cx + sign * ux * scale, cy + sign * uy * scale

    start_x, start_y = edge_point(from_cx, from_cy, from_half_w, from_half_h, 1.0)
    start_x += ux * SILK_ARROW_MARGIN_MM
    start_y += uy * SILK_ARROW_MARGIN_MM
    end_x, end_y = edge_point(to_cx, to_cy, to_half_w, to_half_h, -1.0)
    end_x -= ux * SILK_ARROW_MARGIN_MM
    end_y -= uy * SILK_ARROW_MARGIN_MM
    if math.hypot(end_x - start_x, end_y - start_y) <= GRID_MM:
        return None
    return start_x, start_y, end_x, end_y


def _clip_arrow_to_visible_segments(arrow, obstacle_geometries, min_segment_mm=GRID_MM):
    """Cuts the parts of an arrow line that cross any obstacle, keeping
    whatever's left as one or more separate visible pieces (a long escape
    connector will almost always cross something on a populated board; this
    keeps the line instead of discarding it outright). Returns an ordered
    list of (x0, y0, x1, y1) segments, nearest-the-component first, or []
    if the whole line is obstructed. Fragments shorter than min_segment_mm
    are dropped as noise."""
    x0, y0, x1, y1 = arrow
    full_line = LineString([(x0, y0), (x1, y1)])
    obstacles = [g for g in obstacle_geometries if g is not None]
    if not obstacles:
        return [arrow]
    visible = full_line.difference(unary_union(obstacles))
    if visible.is_empty:
        return []
    if isinstance(visible, LineString):
        pieces = [visible]
    elif isinstance(visible, MultiLineString):
        pieces = list(visible.geoms)
    else:
        pieces = [g for g in getattr(visible, "geoms", []) if isinstance(g, LineString)]

    def dist2_from_origin(px, py):
        return (px - x0) ** 2 + (py - y0) ** 2

    segments = []
    for piece in pieces:
        if piece.length < min_segment_mm:
            continue
        (sx, sy), (ex, ey) = piece.coords[0], piece.coords[-1]
        if dist2_from_origin(ex, ey) < dist2_from_origin(sx, sy):
            sx, sy, ex, ey = ex, ey, sx, sy
        segments.append((sx, sy, ex, ey))
    segments.sort(key=lambda seg: dist2_from_origin(seg[0], seg[1]))
    return segments


_SQRT2 = math.sqrt(2.0)


def _astar_lattice(start, goal, usable, window, max_expansions):
    """A* over integer lattice nodes with 8-way moves inside `window`
    (i_lo, i_hi, j_lo, j_hi). usable(i, j) says whether a node may be
    stepped on; a diagonal move also needs both orthogonal neighbours
    usable, so the path never cuts an obstacle's corner. Returns the node
    list from start to goal, or None."""
    i_lo, i_hi, j_lo, j_hi = window

    def heuristic(node):
        dx, dy = abs(node[0] - goal[0]), abs(node[1] - goal[1])
        return dx + dy + (_SQRT2 - 2.0) * min(dx, dy)

    moves = (
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, _SQRT2), (1, -1, _SQRT2), (-1, 1, _SQRT2), (-1, -1, _SQRT2),
    )
    open_heap = [(heuristic(start), 0.0, start)]
    best_cost = {start: 0.0}
    came_from = {}
    closed = set()
    expansions = 0
    while open_heap:
        _f, cost, node = heapq.heappop(open_heap)
        if node in closed:
            continue
        if node == goal:
            path = [node]
            while node in came_from:
                node = came_from[node]
                path.append(node)
            path.reverse()
            return path
        closed.add(node)
        expansions += 1
        if expansions > max_expansions:
            return None
        for di, dj, move_cost in moves:
            nb = (node[0] + di, node[1] + dj)
            if not (i_lo <= nb[0] <= i_hi and j_lo <= nb[1] <= j_hi) or nb in closed:
                continue
            if not usable(*nb):
                continue
            if di and dj and not (usable(node[0] + di, node[1]) and usable(node[0], node[1] + dj)):
                continue
            new_cost = cost + move_cost
            if new_cost < best_cost.get(nb, float("inf")):
                best_cost[nb] = new_cost
                came_from[nb] = node
                heapq.heappush(open_heap, (new_cost + heuristic(nb), new_cost, nb))
    return None


def _route_connector(
    start, goal, point_clear, segment_clear,
    step=SILK_CONNECTOR_ROUTE_STEP_MM, margins=SILK_CONNECTOR_ROUTE_MARGINS_MM,
    max_expansions=SILK_CONNECTOR_MAX_EXPANSIONS, emerge=SILK_CONNECTOR_EMERGE_MM,
):
    """Routes a connector line from start to goal around obstacles.
    point_clear(x, y) says whether a lattice node (a step-sized square
    around the point) is clear; segment_clear(x0, y0, x1, y1) whether a
    straight piece is. Returns the route as a list of (x, y) waypoints,
    start first and goal last, or None when no route exists inside the
    widest window.

    The straight line is tried first and, when clear, is the route -- the
    common case, costing one check. Otherwise A* runs on a lattice
    anchored at start (so start is a node; goal snaps to the nearest node,
    at most half a step off) inside a window around the straight line's
    bounding box, the smaller margin first. Nodes within `emerge` of
    either end are always usable -- both ends sit in crowded space, and
    the caller clips whatever the route draws there (see
    SILK_CONNECTOR_EMERGE_MM). The node path is then string-pulled: from
    each waypoint, the run of following waypoints is merged for as long
    as the straight piece to them is clear (or lies wholly inside one
    emergence zone), so the result is a few clean segments. Outside the
    emergence zones every merged segment has passed segment_clear and the
    pieces between neighbouring nodes are covered by their clear
    step-sized squares."""
    sx, sy = start
    gx, gy = goal
    if segment_clear(sx, sy, gx, gy):
        return [start, goal]
    goal_node = (int(round((gx - sx) / step)), int(round((gy - sy) / step)))
    if goal_node == (0, 0):
        return None
    emerge2 = emerge * emerge

    def near_start(x, y):
        return (x - sx) ** 2 + (y - sy) ** 2 <= emerge2

    def near_goal(x, y):
        return (x - gx) ** 2 + (y - gy) ** 2 <= emerge2

    cache = {}

    def usable(i, j):
        node = (i, j)
        known = cache.get(node)
        if known is None:
            x, y = sx + i * step, sy + j * step
            known = (
                node == (0, 0) or node == goal_node
                or near_start(x, y) or near_goal(x, y)
                or point_clear(x, y)
            )
            cache[node] = known
        return known

    def piece_ok(a, b):
        if segment_clear(a[0], a[1], b[0], b[1]):
            return True
        if near_start(*a) and near_start(*b):
            return True
        return near_goal(*a) and near_goal(*b)

    for margin in margins:
        m = int(math.ceil(margin / step))
        window = (
            min(0, goal_node[0]) - m, max(0, goal_node[0]) + m,
            min(0, goal_node[1]) - m, max(0, goal_node[1]) + m,
        )
        path = _astar_lattice((0, 0), goal_node, usable, window, max_expansions)
        if path is None:
            continue
        waypoints = [(sx + i * step, sy + j * step) for i, j in path]
        waypoints[0] = start
        waypoints[-1] = goal
        pulled = [waypoints[0]]
        idx = 0
        while idx < len(waypoints) - 1:
            far = idx + 1
            while far + 1 < len(waypoints) and piece_ok(waypoints[idx], waypoints[far + 1]):
                far += 1
            pulled.append(waypoints[far])
            idx = far
        return pulled
    return None


def _waypoints_to_segments(waypoints):
    segments = []
    for (x0, y0), (x1, y1) in zip(waypoints, waypoints[1:]):
        if math.hypot(x1 - x0, y1 - y0) > 1e-9:
            segments.append((x0, y0, x1, y1))
    return segments


def _arrow_segments(arrow):
    if not arrow:
        return []
    x0, y0, x1, y1 = arrow
    if math.hypot(x1 - x0, y1 - y0) <= 1e-9:
        return []
    return [(x0, y0, x1, y1)]


def _arrow_cells(grid, arrow):
    cells = set()
    for x0, y0, x1, y1 in _arrow_segments(arrow):
        cells.update(_collect_segment_cells(grid, x0, y0, x1, y1, SILK_ARROW_WIDTH_MM))
    return cells


def _geometry_tree(geometries):
    geometries = list(geometries)
    return STRtree(geometries) if geometries else None


def _tree_intersects(tree, geometry):
    return tree is not None and len(tree.query(geometry, predicate="intersects")) > 0


# --- Bulk geometry helpers ------------------------------------------------
# Almost all of the placer's time used to be Python overhead around tiny
# geometry calls made one candidate at a time: constructing a box, one
# STRtree query per tree, a representative-point test. These helpers do the
# same work for a whole batch of candidates in single C calls. They are
# exact equivalents: boxes are built from the same float expressions as
# _rotated_box, and the tree predicate is the same "intersects" that
# _tree_intersects uses, so every per-candidate answer is unchanged -- only
# the time to get it.
def _boxes_at(positions, width, height):
    """Axis-aligned width x height boxes centred on every (x, y) in
    positions, as an object array; entry i equals
    _rotated_box(positions[i][0], positions[i][1], width, height)."""
    if not positions:
        return np.empty(0, dtype=object)
    xs = np.fromiter((p[0] for p in positions), dtype=float, count=len(positions))
    ys = np.fromiter((p[1] for p in positions), dtype=float, count=len(positions))
    return shapely.box(xs - width / 2.0, ys - height / 2.0, xs + width / 2.0, ys + height / 2.0)


def _tree_hit_mask(tree, geometries):
    """Boolean array: which of `geometries` intersect anything in `tree` --
    one bulk query, same predicate as _tree_intersects."""
    mask = np.zeros(len(geometries), dtype=bool)
    if tree is not None and len(geometries):
        mask[tree.query(geometries, predicate="intersects")[0]] = True
    return mask


def _edge_arrays(edge_segments):
    """The board outline's segments as four float64 arrays (x0, y0, x1, y1)
    for _points_inside_edge_segments; None when there is no outline."""
    if not edge_segments:
        return None
    arr = np.asarray(list(edge_segments), dtype=float).reshape(-1, 4)
    return arr[:, 0].copy(), arr[:, 1].copy(), arr[:, 2].copy(), arr[:, 3].copy()


def _points_inside_edge_segments(xs, ys, edge_arrays, chunk=512):
    """Vectorized _point_inside_edge_segments for arrays of points: the same
    ray-casting parity test with the same float expressions evaluated in
    the same order, so every answer matches the scalar version. On a board
    whose outline has hundreds of segments the scalar loop was the single
    largest cost of a run. Points are processed in chunks to bound the
    points x edges intermediate."""
    x0, y0, x1, y1 = edge_arrays
    inside = np.zeros(len(xs), dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for start in range(0, len(xs), chunk):
            px = xs[start:start + chunk, None]
            py = ys[start:start + chunk, None]
            crosses = (y0 > py) != (y1 > py)
            intersect_x = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
            hits = crosses & (px < intersect_x)
            inside[start:start + chunk] = (hits.sum(axis=1) % 2) == 1
    return inside


def _rotated_box(cx, cy, width, height, angle_rad=0.0):
    geometry = box(
        cx - width / 2.0,
        cy - height / 2.0,
        cx + width / 2.0,
        cy + height / 2.0,
    )
    if not math.isclose(angle_rad, 0.0, abs_tol=1e-12):
        geometry = affinity.rotate(
            geometry,
            angle_rad,
            origin=(cx, cy),
            use_radians=True,
        )
    return geometry


def _pad_geometry(pad, clearance=0.0):
    if pad["is_circle"]:
        return Point(pad["x"], pad["y"]).buffer(
            pad["w"] / 2.0 + clearance,
            quad_segs=8,
        )
    return _rotated_box(
        pad["x"],
        pad["y"],
        pad["w"] + 2.0 * clearance,
        pad["h"] + 2.0 * clearance,
        pad["angle"],
    )


def _segment_geometry(x0, y0, x1, y1, width):
    line = LineString(((x0, y0), (x1, y1)))
    return line.buffer(max(width / 2.0, 1e-6), cap_style="round")


def _point_inside_edge_segments(x, y, edge_segments):
    inside = False
    for x0, y0, x1, y1 in edge_segments or ():
        if (y0 > y) == (y1 > y):
            continue
        intersect_x = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        if x < intersect_x:
            inside = not inside
    return inside


class _DynamicGeometryIndex:
    """Small mutable spatial hash for already accepted labels and arrows."""

    def __init__(self, cell_mm=SILK_DYNAMIC_INDEX_CELL_MM):
        self.cell_mm = max(0.25, float(cell_mm))
        self.geometries = []
        self.buckets = defaultdict(list)

    def _keys(self, geometry):
        min_x, min_y, max_x, max_y = geometry.bounds
        ix0 = math.floor(min_x / self.cell_mm)
        ix1 = math.floor(max_x / self.cell_mm)
        iy0 = math.floor(min_y / self.cell_mm)
        iy1 = math.floor(max_y / self.cell_mm)
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                yield ix, iy

    def add(self, geometry):
        index = len(self.geometries)
        self.geometries.append(geometry)
        for key in self._keys(geometry):
            self.buckets[key].append(index)
        return index

    def remove(self, index):
        """Un-adds a previously added geometry by its add() index -- used to
        free a label's own contribution before testing whether relocating it
        elsewhere would be an improvement. Leaves stale bucket entries in
        place (harmless: intersects() skips nulled-out slots)."""
        if 0 <= index < len(self.geometries):
            self.geometries[index] = None

    def intersects(self, geometry):
        candidate_indices = set()
        for key in self._keys(geometry):
            candidate_indices.update(self.buckets.get(key, ()))
        return any(
            self.geometries[index] is not None and self.geometries[index].intersects(geometry)
            for index in candidate_indices
        )

    def hits(self, geometry):
        """add() indices of every live geometry intersecting `geometry` --
        for callers that must exclude their own entries from the test."""
        candidate_indices = set()
        for key in self._keys(geometry):
            candidate_indices.update(self.buckets.get(key, ()))
        return [
            index for index in sorted(candidate_indices)
            if self.geometries[index] is not None and self.geometries[index].intersects(geometry)
        ]


# --- Escape-rescue worker processes ------------------------------------
# The whole-board scan escape rescue runs per failed item is the single
# most expensive thing in the pipeline, and most of each legality check is
# plain Python (dynamic-index lookups, the point-in-outline edge test), so
# threads gain nothing under the GIL. Instead, a batch of items is scanned
# concurrently in worker *processes*, each against the same frozen
# snapshot of already-placed labels, and results are committed back in the
# original sequential order with a conflict check (see the driver in
# _auto_place_silkscreen_geometry) that only accepts a speculative result
# when it is provably what the sequential run would have produced -- any
# item that might have been affected by an earlier commit is simply
# re-scanned sequentially. So the output is identical to the sequential
# version in every case; only wall-clock time changes.
_ESCAPE_WORKER_CONTEXT = None


def _build_escape_static_context(candidates, all_components, pads, trace_segments, vias, edge_segments):
    """Rebuilds, inside a worker process, exactly the static obstacle
    structures _auto_place_silkscreen_geometry builds for itself -- same
    construction in the same order, so every legality answer matches the
    main process. Deliberately duplicates that setup rather than routing
    the main function through a shared helper: the main placement path is
    left untouched."""
    pad_geometries = [_pad_geometry(pad, SILK_CLEARANCE_MM) for pad in pads]
    via_geometries = [
        Point(via["x"], via["y"]).buffer(
            via["size"] / 2.0 + SILK_CLEARANCE_MM,
            quad_segs=8,
        )
        for via in vias
    ]
    trace_geometries = [
        _segment_geometry(
            segment["x0"],
            segment["y0"],
            segment["x1"],
            segment["y1"],
            segment["width"] + 2.0 * SILK_CLEARANCE_MM,
        )
        for segments in (trace_segments or {}).values()
        for segment in segments
    ]
    relaxed_tree = _geometry_tree(pad_geometries + via_geometries)
    strict_tree = _geometry_tree(pad_geometries + via_geometries + trace_geometries)

    footprint_geometries = {"F.Cu": [], "B.Cu": []}
    for component in all_components or candidates:
        reference = str(component.get("ref") or "").strip().upper()
        if not reference or "***" in reference:
            continue
        geometry = box(
            component["min_x"] - SILK_FOOTPRINT_CLEARANCE_MM,
            component["min_y"] - SILK_FOOTPRINT_CLEARANCE_MM,
            component["max_x"] + SILK_FOOTPRINT_CLEARANCE_MM,
            component["max_y"] + SILK_FOOTPRINT_CLEARANCE_MM,
        )
        if component.get("has_through_hole"):
            sides = ("F.Cu", "B.Cu")
        else:
            side = component.get("side")
            sides = (side,) if side in footprint_geometries else ("F.Cu",)
        for side in sides:
            footprint_geometries[side].append(geometry)
    footprint_trees = {
        side: _geometry_tree(items) for side, items in footprint_geometries.items()
    }

    edge_lines = [LineString(((x0, y0), (x1, y1))) for x0, y0, x1, y1 in edge_segments or ()]
    edge_tree = _geometry_tree(edge_lines)

    def inside_board(geometry):
        if not edge_segments:
            return True
        if _tree_intersects(edge_tree, geometry):
            return False
        probe = geometry.representative_point()
        return _point_inside_edge_segments(probe.x, probe.y, edge_segments)

    board_scan_xs = []
    board_scan_ys = []
    for pad in pads:
        min_x, min_y, max_x, max_y = pad["bbox"]
        board_scan_xs.extend((min_x, max_x))
        board_scan_ys.extend((min_y, max_y))
    for segments in (trace_segments or {}).values():
        for seg in segments:
            half_w = seg["width"] / 2.0
            board_scan_xs.extend((seg["x0"] - half_w, seg["x0"] + half_w, seg["x1"] - half_w, seg["x1"] + half_w))
            board_scan_ys.extend((seg["y0"] - half_w, seg["y0"] + half_w, seg["y1"] - half_w, seg["y1"] + half_w))
    for via in vias:
        r = via["size"] / 2.0
        board_scan_xs.extend((via["x"] - r, via["x"] + r))
        board_scan_ys.extend((via["y"] - r, via["y"] + r))
    for x0, y0, x1, y1 in edge_segments or []:
        board_scan_xs.extend((x0, x1))
        board_scan_ys.extend((y0, y1))
    for comp in (all_components or candidates):
        board_scan_xs.extend((comp["min_x"], comp["max_x"]))
        board_scan_ys.extend((comp["min_y"], comp["max_y"]))

    return {
        "strict_tree": strict_tree,
        "relaxed_tree": relaxed_tree,
        "footprint_trees": footprint_trees,
        "inside_board": inside_board,
        "board_scan_min_x": min(board_scan_xs) if board_scan_xs else 0.0,
        "board_scan_max_x": max(board_scan_xs) if board_scan_xs else 0.0,
        "board_scan_min_y": min(board_scan_ys) if board_scan_ys else 0.0,
        "board_scan_max_y": max(board_scan_ys) if board_scan_ys else 0.0,
    }


def _escape_worker_init(static_inputs):
    global _ESCAPE_WORKER_CONTEXT
    _ESCAPE_WORKER_CONTEXT = _build_escape_static_context(*static_inputs)


def _escape_ring_points(anchor_cx, anchor_cy, radius, step):
    """Lattice points on the square ring max(|dx|, |dy|) == radius around
    the anchor (just the anchor itself for radius 0), so every ring covers
    all directions at once and each point is visited exactly once."""
    if radius <= 0.0:
        return [(round(anchor_cx, 6), round(anchor_cy, 6))]
    n = int(round(radius / step))
    points = []
    for i in range(-n, n + 1):
        dx = i * step
        points.append((round(anchor_cx + dx, 6), round(anchor_cy - radius, 6)))
        points.append((round(anchor_cx + dx, 6), round(anchor_cy + radius, 6)))
    for j in range(-n + 1, n):
        dy = j * step
        points.append((round(anchor_cx - radius, 6), round(anchor_cy + dy, 6)))
        points.append((round(anchor_cx + radius, 6), round(anchor_cy + dy, 6)))
    return points


def _escape_ring_scan(
    anchor_cx, anchor_cy, dims, margin, bounds, inside_board, relaxed_blocked, strict_blocked,
    reach=SILK_ESCAPE_PREFERENCE_REACH_MM, step=SILK_ESCAPE_SCAN_STEP_MM, bulk=None,
):
    """Nearest-first expanding-ring search for a label/block of any of the
    (width, height) pairs in `dims` (preference order, e.g. 0 degrees
    first). Rings step outward from the anchor one lattice step at a time,
    so all directions are explored together and the direction is simply
    whichever side opens up first; the cost is proportional to the area
    explored before the first hit, not to the whole board.

    A spot is legal when its box is inside the board and clear of
    everything relaxed_blocked knows about (pads, vias, footprints, placed
    labels, connectors -- not traces). The nearest legal spot (straight-
    line distance, not ring index -- a ring's corners are further out than
    its sides) fixes the distance; rings keep expanding only while they
    can still hold a point within `reach` of it, and among every legal
    spot inside that window the best quality wins: the earliest dims
    entry (0 degrees first -- the pipeline-wide preference), then a buffer
    of `margin` clear on every side (judged with relaxed_blocked), then
    trace-free (strict_blocked clear) -- nearest on ties. A +/-step fine
    snap at SILK_SLIDE_STEP_MM then pulls the winner as close as it
    legally gets without losing that quality. So the label lands as close
    as possible and gets its preferred orientation / buffer / trace-free
    spot whenever the space right there allows it, never by travelling
    further for it.

    Returns None, or a dict with spec_index, strict (trace-free), buffered,
    position, coarse_hit, ring_radius and nearest_hits (every legal spot
    at the nearest distance -- the parallel driver's conflict check needs
    them, since they are what fixed the search window)."""
    min_x, min_y, max_x, max_y = bounds
    diag = math.hypot(max_x - min_x, max_y - min_y)

    def evaluate(x, y):
        """Best (key, spec_index, strict, buffered) at this point, or None;
        key = ((-spec_index, buffered, strict), -distance^2)."""
        best = None
        for spec_index, (bw, bh) in enumerate(dims):
            if not (min_x + bw / 2.0 - 1e-9 <= x <= max_x - bw / 2.0 + 1e-9
                    and min_y + bh / 2.0 - 1e-9 <= y <= max_y - bh / 2.0 + 1e-9):
                continue
            box_geom = _rotated_box(x, y, bw, bh)
            if not inside_board(box_geom) or relaxed_blocked(box_geom):
                continue
            strict_ok = not strict_blocked(box_geom)
            buffered = margin > 0.0 and not relaxed_blocked(_rotated_box(x, y, bw + 2.0 * margin, bh + 2.0 * margin))
            key = ((-spec_index, 1 if buffered else 0, 1 if strict_ok else 0), -((x - anchor_cx) ** 2 + (y - anchor_cy) ** 2))
            if best is None or key > best[0]:
                best = (key, spec_index, strict_ok, buffered)
        return best

    def evaluate_ring(points):
        """evaluate() for every point of a ring at once: `bulk` is
        (inside_mask, relaxed_mask, strict_mask), each a callable over an
        array of geometries returning a boolean array, so each ring costs
        one box build and one query per tree per dims entry instead of a
        handful of geometry calls per point. Same boxes (same float
        expressions), same predicates, same per-point best -- returns a
        list aligned with `points` of evaluate()'s results."""
        inside_mask, relaxed_mask, strict_mask = bulk
        results = [None] * len(points)
        xs = np.fromiter((p[0] for p in points), dtype=float, count=len(points))
        ys = np.fromiter((p[1] for p in points), dtype=float, count=len(points))
        for spec_index, (bw, bh) in enumerate(dims):
            in_bounds = (
                (xs >= min_x + bw / 2.0 - 1e-9) & (xs <= max_x - bw / 2.0 + 1e-9)
                & (ys >= min_y + bh / 2.0 - 1e-9) & (ys <= max_y - bh / 2.0 + 1e-9)
            )
            idx = np.flatnonzero(in_bounds)
            if not len(idx):
                continue
            px, py = xs[idx], ys[idx]
            boxes = shapely.box(px - bw / 2.0, py - bh / 2.0, px + bw / 2.0, py + bh / 2.0)
            legal = inside_mask(boxes) & ~relaxed_mask(boxes)
            legal_idx = idx[legal]
            if not len(legal_idx):
                continue
            legal_boxes = boxes[legal]
            strict_ok = ~strict_mask(legal_boxes)
            if margin > 0.0:
                probe_w, probe_h = bw + 2.0 * margin, bh + 2.0 * margin
                lx, ly = xs[legal_idx], ys[legal_idx]
                buffered = ~relaxed_mask(shapely.box(
                    lx - probe_w / 2.0, ly - probe_h / 2.0, lx + probe_w / 2.0, ly + probe_h / 2.0,
                ))
            else:
                buffered = np.zeros(len(legal_idx), dtype=bool)
            for k_pt, i in enumerate(legal_idx):
                x, y = points[i]
                is_strict = bool(strict_ok[k_pt])
                is_buffered = bool(buffered[k_pt])
                key = ((-spec_index, 1 if is_buffered else 0, 1 if is_strict else 0), -((x - anchor_cx) ** 2 + (y - anchor_cy) ** 2))
                if results[i] is None or key > results[i][0]:
                    results[i] = (key, spec_index, is_strict, is_buffered)
        return results

    candidates = []  # (key, spec_index, strict, buffered, (x, y), ring_radius) for every legal spot seen
    nearest_d = None
    k = 0
    while True:
        radius = k * step
        if nearest_d is not None and radius > nearest_d + reach + 1e-9:
            break  # every point from here on is at least `radius` away: outside the window
        if radius > diag + step:
            break  # every ring from here on lies wholly outside the area
        points = _escape_ring_points(anchor_cx, anchor_cy, radius, step)
        found_per_point = evaluate_ring(points) if bulk is not None else [evaluate(x, y) for x, y in points]
        for (x, y), found in zip(points, found_per_point):
            if found is None:
                continue
            d = math.sqrt(-found[0][1])
            if nearest_d is None or d < nearest_d:
                nearest_d = d
            candidates.append(found + ((x, y), radius))
        k += 1
    if not candidates:
        return None
    window = [c for c in candidates if math.sqrt(-c[0][1]) <= nearest_d + reach + 1e-9]
    best = max(window, key=lambda c: c[0])
    nearest_hits = [c[4] for c in candidates if math.sqrt(-c[0][1]) <= nearest_d + 1e-6]
    (quality, _negd), spec_index, strict_ok, buffered, (cx, cy), radius = best

    refined = (-((cx - anchor_cx) ** 2 + (cy - anchor_cy) ** 2), cx, cy)
    dx = -step
    while dx <= step + 1e-9:
        dy = -step
        while dy <= step + 1e-9:
            x, y = round(cx + dx, 6), round(cy + dy, 6)
            found = evaluate(x, y)
            if (
                found is not None and found[1] == spec_index
                and found[0][0] >= quality and found[0][1] > refined[0]
            ):
                refined = (found[0][1], x, y)
            dy += SILK_SLIDE_STEP_MM
        dx += SILK_SLIDE_STEP_MM
    return {
        "spec_index": spec_index,
        "strict": strict_ok,
        "buffered": buffered,
        "position": (refined[1], refined[2]),
        "coarse_hit": (cx, cy),
        "ring_radius": radius,
        "nearest_hits": nearest_hits,
    }


def _escape_blocked_check(footprint_tree, obstacle_tree, global_index, arrows_index):
    """The escape search's notion of 'blocked': any footprint on the
    label's side, the pass's obstacle tree (pads/vias, plus traces for the
    strict tree), any placed label, any placed connector line."""
    def blocked(geometry):
        if _tree_intersects(footprint_tree, geometry):
            return True
        if _tree_intersects(obstacle_tree, geometry):
            return True
        if global_index.intersects(geometry):
            return True
        if arrows_index.intersects(geometry):
            return True
        return False
    return blocked


def _escape_scan_worker(task):
    """Runs one failed item's full search sequence -- every (dimensions,
    pass) combination in the same order the sequential code tries them --
    against the snapshot of placed labels carried in the task, and returns
    the first hit (or that nothing was found)."""
    ctx = _ESCAPE_WORKER_CONTEXT
    global_index = _DynamicGeometryIndex()
    for geometry in task["global_geometries"]:
        global_index.add(geometry)
    arrows_index = _DynamicGeometryIndex()
    for geometry in task["arrow_geometries"]:
        arrows_index.add(geometry)
    footprint_tree = ctx["footprint_trees"].get(task["side"])
    strict_blocked = _escape_blocked_check(footprint_tree, ctx["strict_tree"], global_index, arrows_index)
    relaxed_blocked = (
        _escape_blocked_check(footprint_tree, ctx["relaxed_tree"], global_index, arrows_index)
        if task["allow_relaxed"] else strict_blocked
    )
    bounds = (ctx["board_scan_min_x"], ctx["board_scan_min_y"], ctx["board_scan_max_x"], ctx["board_scan_max_y"])
    result = _escape_ring_scan(
        task["anchor_cx"], task["anchor_cy"], task["dims"], SILK_ESCAPE_MARGIN_MM, bounds,
        ctx["inside_board"], relaxed_blocked, strict_blocked,
    )
    if result is None:
        return {"found": False}
    result["found"] = True
    return result


def _escape_influence_region(nearest_hits, coarse_hit, max_extent, margin):
    """Bounding box of everything a speculative ring search's result can
    still depend on once the search is over: every legal spot at the
    nearest distance (they fixed the search window -- lose them all and
    the sequential search would look further, at spots this one never
    saw), plus the fine-snap window around the winning coarse hit; each
    grown by the largest probe box's half-size (block plus margin) and
    padding.
    Earlier commits can only remove legal space, never add it, so every
    ring the search rejected stays rejected no matter where a commit
    lands, and a spot the search saw but did not choose can only get
    worse; only a commit reaching into this box could change which spot
    wins (or whether it is still legal). If none does, the sequential run
    would have produced the identical result."""
    pad = SILK_ESCAPE_SCAN_STEP_MM + 0.5 * max_extent + margin + SILK_PADDING_MM + 1.0
    xs = [pos[0] for pos in nearest_hits] + [coarse_hit[0]]
    ys = [pos[1] for pos in nearest_hits] + [coarse_hit[1]]
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def _bbox_overlaps(a, b):
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _cluster_preferred_sides_geometry(
    clusters, footprint_trees, strict_tree, inside_board_fn, cluster_preferred_orientation=None
):
    preferred = {}
    side_order = ("top", "bottom", "right", "left")
    for members in clusters:
        cluster_orientation = (cluster_preferred_orientation or {}).get(members[0]["id"])
        votes = {side: 0 for side in side_order}
        for comp, collision_width, collision_height, positions in _cluster_preferred_side_votes(
            members, cluster_orientation
        ):
            side = comp.get("side")
            if side not in footprint_trees:
                side = "F.Cu"
            for side_name, (x, y) in positions.items():
                geometry = _rotated_box(x, y, collision_width, collision_height)
                if not inside_board_fn(geometry):
                    continue
                if _tree_intersects(footprint_trees.get(side), geometry):
                    continue
                if _tree_intersects(strict_tree, geometry):
                    continue
                votes[side_name] += 1
        best_side = max(side_order, key=lambda side: votes[side])
        if votes[best_side] > 0:
            for comp in members:
                preferred[comp["id"]] = best_side
    return preferred


def _cluster_preferred_orientation_geometry(clusters, footprint_trees, strict_tree, inside_board_fn, inside_board_mask=None):
    """Family-level orientation preference: for each cluster, tallies which
    orientation each member's own space-score favors (see
    _orientation_space_score_geometry) and uses the majority as a
    starting-point preference for every member -- soft bias only, the
    normal per-component search still runs (and can still land on the
    other orientation) if the preference doesn't work out for a specific
    member."""
    preferred = {}
    for members in clusters:
        votes = {"horizontal": 0, "vertical": 0}
        for comp in members:
            width_mm, height_mm = _estimate_text_box(comp["ref"], SILK_TEXT_HEIGHT_MM)
            side = comp.get("side")
            if side not in footprint_trees:
                side = "F.Cu"
            ordered = _orientation_space_score_geometry(
                comp, width_mm, height_mm, footprint_trees, strict_tree, inside_board_fn, side,
                inside_board_mask=inside_board_mask,
            )
            _rotation_deg, _cw, _ch, is_vertical = ordered[0]
            votes["vertical" if is_vertical else "horizontal"] += 1
        best = "vertical" if votes["vertical"] > votes["horizontal"] else "horizontal"
        if votes[best] > 0:
            for comp in members:
                preferred[comp["id"]] = best
    return preferred


# Post-placement buffer reinforcement (Stage: shift labels that are too
# tight on top or on either side). "top"/"left"/"right" here are fixed
# screen directions (top = smaller Y, i.e. visually up), NOT the
# top/bottom/left/right side-of-component convention _generate_label_candidates
# uses -- the two are unrelated on purpose: buffer direction is about how the
# label reads on screen, side-of-component is about where it sits relative
# to its part.
_SIDE_ALTERNATES = {
    # For relocation: try the opposite side of the component first (180
    # degrees from wherever the label currently sits), then the two
    # perpendicular sides (90/270 degrees).
    "top": ("bottom", "right", "left"),
    "bottom": ("top", "right", "left"),
    "left": ("right", "top", "bottom"),
    "right": ("left", "top", "bottom"),
}


def _infer_label_side_from_component(x, y, comp):
    """Which of top/bottom/left/right (component-relative, matching
    _generate_label_candidates's convention) a placed label sits on."""
    dx = x - comp["cx"]
    dy = y - comp["cy"]
    if abs(dy) >= abs(dx):
        return "top" if dy > 0 else "bottom"
    return "right" if dx > 0 else "left"


def _label_buffer_ratio_grid(grid, x, y, width_mm, height_mm, direction, inside_cells, blocked_sets, steps=10):
    """Fraction (0..1, `steps` increments) of the label's own size that is
    free immediately beyond its edge in `direction` ('top' = screen-up,
    smaller Y; 'left'/'right' = the label's own sides)."""
    free = 0
    for k in range(1, steps + 1):
        if direction == "top":
            strip_h = height_mm * k / steps
            probe_x, probe_y = x, y - height_mm / 2.0 - strip_h / 2.0
            probe_w, probe_h = width_mm, strip_h
        elif direction == "left":
            strip_w = width_mm * k / steps
            probe_x, probe_y = x - width_mm / 2.0 - strip_w / 2.0, y
            probe_w, probe_h = strip_w, height_mm
        else:
            strip_w = width_mm * k / steps
            probe_x, probe_y = x + width_mm / 2.0 + strip_w / 2.0, y
            probe_w, probe_h = strip_w, height_mm
        cells = _collect_rect_cells(grid, probe_x, probe_y, probe_w, probe_h, 0.0)
        if not cells:
            break
        if inside_cells is not None and not cells.issubset(inside_cells):
            break
        if any(cells & blocked for blocked in blocked_sets):
            break
        free = k
    return free / steps


def _label_meets_buffer_grid(
    grid, x, y, width_mm, height_mm, inside_cells, blocked_sets, min_ratio=SILK_LABEL_BUFFER_RATIO
):
    worst = 1.0
    for direction in ("top", "left", "right"):
        ratio = _label_buffer_ratio_grid(grid, x, y, width_mm, height_mm, direction, inside_cells, blocked_sets)
        worst = min(worst, ratio)
        if ratio < min_ratio:
            return False, worst
    return True, worst


def _buffer_probe_boxes(x, y, width_mm, height_mm, direction, steps):
    """The `steps` probe strips _label_buffer_ratio_geometry tests in one
    direction, as (minx, miny, maxx, maxy) tuples in test order -- the same
    float expressions as before, so the boxes are identical."""
    probes = []
    for k in range(1, steps + 1):
        if direction == "top":
            strip_h = height_mm * k / steps
            probe_x, probe_y = x, y - height_mm / 2.0 - strip_h / 2.0
            probe_w, probe_h = width_mm, strip_h
        elif direction == "left":
            strip_w = width_mm * k / steps
            probe_x, probe_y = x - width_mm / 2.0 - strip_w / 2.0, y
            probe_w, probe_h = strip_w, height_mm
        else:
            strip_w = width_mm * k / steps
            probe_x, probe_y = x + width_mm / 2.0 + strip_w / 2.0, y
            probe_w, probe_h = strip_w, height_mm
        probes.append((
            probe_x - probe_w / 2.0, probe_y - probe_h / 2.0,
            probe_x + probe_w / 2.0, probe_y + probe_h / 2.0,
        ))
    return probes


def _label_buffer_ratio_geometry(
    x, y, width_mm, height_mm, direction, inside_board_fn, blocked_check, steps=10, bulk_blocked=None,
):
    """Geometry-backend counterpart of _label_buffer_ratio_grid. blocked_check
    is a callable geometry -> bool (True = blocked). With bulk_blocked -- a
    callable over an array of geometries returning a boolean array that is
    True where a probe is outside the board or blocked -- all probes are
    judged in one pass; the ratio is the same, being the run of clear
    probes before the first blocked one either way."""
    probes = _buffer_probe_boxes(x, y, width_mm, height_mm, direction, steps)
    if bulk_blocked is not None:
        boxes = shapely.box(
            np.array([p[0] for p in probes]), np.array([p[1] for p in probes]),
            np.array([p[2] for p in probes]), np.array([p[3] for p in probes]),
        )
        blocked = bulk_blocked(boxes)
        free = 0
        for k in range(steps):
            if blocked[k]:
                break
            free = k + 1
        return free / steps
    free = 0
    for k, (minx, miny, maxx, maxy) in enumerate(probes, start=1):
        geometry = box(minx, miny, maxx, maxy)
        if not inside_board_fn(geometry):
            break
        if blocked_check(geometry):
            break
        free = k
    return free / steps


def _label_meets_buffer_geometry(
    x, y, width_mm, height_mm, inside_board_fn, blocked_check, min_ratio=SILK_LABEL_BUFFER_RATIO, bulk_blocked=None,
):
    worst = 1.0
    for direction in ("top", "left", "right"):
        ratio = _label_buffer_ratio_geometry(
            x, y, width_mm, height_mm, direction, inside_board_fn, blocked_check, bulk_blocked=bulk_blocked,
        )
        worst = min(worst, ratio)
        if ratio < min_ratio:
            return False, worst
    return True, worst


def _auto_place_silkscreen_geometry(
    candidates,
    all_components,
    pads,
    trace_segments,
    vias,
    edge_segments,
    allow_trace_overlap_fallback,
):
    """Memory-bounded placement backend for large/high-resolution boards."""

    pad_geometries = [_pad_geometry(pad, SILK_CLEARANCE_MM) for pad in pads]
    via_geometries = [
        Point(via["x"], via["y"]).buffer(
            via["size"] / 2.0 + SILK_CLEARANCE_MM,
            quad_segs=8,
        )
        for via in vias
    ]
    trace_geometries = [
        _segment_geometry(
            segment["x0"],
            segment["y0"],
            segment["x1"],
            segment["y1"],
            segment["width"] + 2.0 * SILK_CLEARANCE_MM,
        )
        for segments in (trace_segments or {}).values()
        for segment in segments
    ]
    relaxed_tree = _geometry_tree(pad_geometries + via_geometries)
    strict_tree = _geometry_tree(pad_geometries + via_geometries + trace_geometries)

    pad_by_side = {"F.Cu": [], "B.Cu": []}
    for pad, geometry in zip(pads, pad_geometries):
        if pad.get("is_th"):
            sides = ("F.Cu", "B.Cu")
        else:
            side = pad.get("side")
            sides = (side,) if side in pad_by_side else ("F.Cu",)
        for side in sides:
            pad_by_side[side].append(geometry)
    pad_trees = {side: _geometry_tree(items) for side, items in pad_by_side.items()}

    footprint_geometries = {"F.Cu": [], "B.Cu": []}
    footprint_ids = {"F.Cu": [], "B.Cu": []}
    for component in all_components or candidates:
        reference = str(component.get("ref") or "").strip().upper()
        if not reference or "***" in reference:
            # Placeholder/graphics-only footprints are removed by the silk
            # updater and must not reserve the whole board as an obstacle.
            continue
        geometry = box(
            component["min_x"] - SILK_FOOTPRINT_CLEARANCE_MM,
            component["min_y"] - SILK_FOOTPRINT_CLEARANCE_MM,
            component["max_x"] + SILK_FOOTPRINT_CLEARANCE_MM,
            component["max_y"] + SILK_FOOTPRINT_CLEARANCE_MM,
        )
        if component.get("has_through_hole"):
            sides = ("F.Cu", "B.Cu")
        else:
            side = component.get("side")
            sides = (side,) if side in footprint_geometries else ("F.Cu",)
        for side in sides:
            footprint_geometries[side].append(geometry)
            footprint_ids[side].append(component.get("id"))
    footprint_trees = {
        side: _geometry_tree(items) for side, items in footprint_geometries.items()
    }

    edge_lines = [LineString(((x0, y0), (x1, y1))) for x0, y0, x1, y1 in edge_segments or ()]
    edge_tree = _geometry_tree(edge_lines)
    edge_arrays = _edge_arrays(edge_segments)

    def inside_board(geometry):
        if not edge_segments:
            return True
        if _tree_intersects(edge_tree, geometry):
            return False
        probe = geometry.representative_point()
        return _point_inside_edge_segments(probe.x, probe.y, edge_segments)

    def inside_board_mask(geometries):
        """Vectorized inside_board over an array of geometries: the same
        two tests (no board edge crossed; representative point inside the
        outline), so entry i equals inside_board(geometries[i])."""
        count = len(geometries)
        if not edge_segments:
            return np.ones(count, dtype=bool)
        mask = ~_tree_hit_mask(edge_tree, geometries)
        hit_free = np.flatnonzero(mask)
        if len(hit_free):
            probes = shapely.get_coordinates(shapely.point_on_surface(geometries[hit_free]))
            mask[hit_free] = _points_inside_edge_segments(probes[:, 0], probes[:, 1], edge_arrays)
        return mask

    def intersects_other_footprint(side, geometry, component_id):
        tree = footprint_trees.get(side)
        if tree is None:
            return False
        indices = tree.query(geometry, predicate="intersects")
        ids = footprint_ids[side]
        return any(ids[int(index)] != component_id for index in indices)

    def arrow_clear(arrow, side, component_id):
        """A pointer arrow is usable when its buffered line is inside the
        board and clear of pads and of other components' footprints --
        judged on the buffered geometry exactly as before. The bare centre
        line is tested first: whatever the line hits, its buffer hits too,
        so a crossing arrow is rejected without building the (costly)
        buffer at all; only a line that passes gets the full test."""
        line = LineString(((arrow[0], arrow[1]), (arrow[2], arrow[3])))
        if edge_segments and _tree_intersects(edge_tree, line):
            return False
        if _tree_intersects(pad_trees.get(side), line):
            return False
        if intersects_other_footprint(side, line, component_id):
            return False
        geometry = _segment_geometry(*arrow, SILK_ARROW_WIDTH_MM)
        if not inside_board(geometry):
            return False
        if _tree_intersects(pad_trees.get(side), geometry):
            return False
        return not intersects_other_footprint(side, geometry, component_id)

    def connector_clearance_checks(side, excluded_component_ids, on_board, footprints_block=True):
        """(point_clear, segment_clear) for routing a connector on `side`:
        pads, every placed label (a line must never run through text),
        for an on-board connector the board outline, and -- when
        footprints_block -- other components' footprint boxes, all via the
        same STRtrees/indexes the placer uses."""
        pad_tree = pad_trees.get(side)
        footprint_tree = footprint_trees.get(side) if footprints_block else None
        ids = footprint_ids[side]
        excluded = set(excluded_component_ids or ())
        half = SILK_CONNECTOR_ROUTE_STEP_MM / 2.0

        def blocked(geometry):
            if on_board and not inside_board(geometry):
                return True
            if _tree_intersects(pad_tree, geometry):
                return True
            if footprint_tree is not None:
                for idx in footprint_tree.query(geometry, predicate="intersects"):
                    if ids[int(idx)] not in excluded:
                        return True
            return global_labels.intersects(geometry)

        def point_clear(x, y):
            return not blocked(box(x - half, y - half, x + half, y + half))

        def segment_clear(x0, y0, x1, y1):
            return not blocked(_segment_geometry(x0, y0, x1, y1, SILK_CONNECTOR_PROBE_MM))

        return point_clear, segment_clear

    def clip_straight_connector(full_arrow, side, excluded_component_ids, on_board, footprints_block=True):
        """Cut the parts of a straight piece that cross any pad (and, when
        footprints_block, another component's footprint) and keep the
        visible pieces, nearest-the-origin first; an on-board piece that
        would leave the board is vetoed outright. Only queries obstacles
        near the piece. With footprints_block this is exactly the previous
        connector behaviour, kept as the last-resort fallback."""
        line_geom = LineString([(full_arrow[0], full_arrow[1]), (full_arrow[2], full_arrow[3])])
        if on_board and not inside_board(line_geom):
            return []
        excluded = set(excluded_component_ids or ())
        obstacle_geoms = []
        pad_tree = pad_trees.get(side)
        if pad_tree is not None:
            obstacle_geoms.extend(
                pad_by_side[side][int(idx)] for idx in pad_tree.query(line_geom, predicate="intersects")
            )
        footprint_tree = footprint_trees.get(side) if footprints_block else None
        if footprint_tree is not None:
            ids = footprint_ids[side]
            geoms = footprint_geometries[side]
            obstacle_geoms.extend(
                geoms[int(idx)]
                for idx in footprint_tree.query(line_geom, predicate="intersects")
                if ids[int(idx)] not in excluded
            )
        return _clip_arrow_to_visible_segments(full_arrow, obstacle_geoms)

    def route_connector(full_arrow, side, excluded_component_ids=(), on_board=True):
        """Connector line from full_arrow's start to its end as an ordered
        list of (x0, y0, x1, y1) segments. Two routing tiers, then the old
        fallback: (1) a path clear of pads, other footprints and placed
        labels (the straight line itself when that is already clear);
        (2) if no such corridor exists -- on a crowded board the footprint
        boxes often form an unbroken wall -- a path clear of pads and
        labels only, crossing footprint outlines where unavoidable;
        (3) the clipped straight line. A pad is never drawn over: every
        routed piece is still clipped around pads (that only ever bites
        inside the emergence zones at the two ends, where the router
        deliberately ignores obstacles)."""
        if full_arrow is None:
            return []
        start = (full_arrow[0], full_arrow[1])
        goal = (full_arrow[2], full_arrow[3])
        waypoints = None
        for footprints_block in (True, False):
            point_clear, segment_clear = connector_clearance_checks(
                side, excluded_component_ids, on_board, footprints_block=footprints_block,
            )
            waypoints = _route_connector(start, goal, point_clear, segment_clear)
            if waypoints is not None:
                break
        if waypoints is None:
            return clip_straight_connector(full_arrow, side, excluded_component_ids, on_board)
        segments = []
        for piece in _waypoints_to_segments(waypoints):
            segments.extend(
                clip_straight_connector(piece, side, excluded_component_ids, on_board, footprints_block=False)
            )
        return segments

    def clip_arrow_against_obstacles(full_arrow, side, exclude_component_id=None, exclude_component_ids=()):
        """On-board connector for an escape-rescued label or block: routed
        around obstacles and kept inside the board; see route_connector.
        The component(s) the line points at are not obstacles for it."""
        excluded = set(exclude_component_ids)
        if exclude_component_id is not None:
            excluded.add(exclude_component_id)
        return route_connector(full_arrow, side, excluded, on_board=True)

    def clipped_escape_arrow(comp, x, y, cw, ch, gap_mm, side):
        """Per-component pointer line from comp to (x, y), clipped around
        obstacles via clip_arrow_against_obstacles. Used for lone (not
        clustered) escape rescues, where each placed label is its own
        distinct component and needs its own connector."""
        if not _label_needs_pointer(comp, x, y, cw, ch, gap_mm):
            return []
        full_arrow = _arrow_points(comp, x, y, cw, ch)
        return clip_arrow_against_obstacles(full_arrow, side, exclude_component_id=comp.get("id"))

    family_clusters = _build_family_clusters(candidates)
    # Members of a real family (two or more parts) are marked: for them 0
    # and 90 degrees are equally welcome and the orientation with more room
    # wins (see _orientation_space_score_*); everything else keeps 0 first.
    for members in family_clusters:
        if len(members) >= 2:
            for member in members:
                member["_family"] = True
    family_pattern_summary, family_pattern_detail = _classify_family_clusters(family_clusters)
    print(
        "Family patterns: {} clusters ({} rows, {} mirror pairs, {} twin pairs, "
        "{} unclassified)".format(
            len(family_clusters),
            family_pattern_summary["row"],
            family_pattern_summary["mirror_pair"],
            family_pattern_summary["twin_pair"],
            family_pattern_summary["unclassified"],
        )
    )
    cluster_preferred_orientation = _cluster_preferred_orientation_geometry(
        family_clusters, footprint_trees, strict_tree, inside_board, inside_board_mask=inside_board_mask,
    )
    cluster_preferred_side = _cluster_preferred_sides_geometry(
        family_clusters, footprint_trees, strict_tree, inside_board, cluster_preferred_orientation
    )

    def static_candidates(component):
        width_mm, height_mm = _estimate_text_box(
            component["ref"], SILK_TEXT_HEIGHT_MM
        )
        side = component.get("side")
        if side not in footprint_trees:
            side = "F.Cu"
        preferred_side = cluster_preferred_side.get(component.get("id"))
        preferred_orientation = cluster_preferred_orientation.get(component.get("id"))

        strict_results = []
        relaxed_results = []
        for rotation_deg, collision_width, collision_height, is_vertical in (
            _orientation_space_score_geometry(
                component, width_mm, height_mm, footprint_trees, strict_tree, inside_board, side,
                preferred_orientation=preferred_orientation, inside_board_mask=inside_board_mask,
            )
        ):
            near = _generate_label_candidates(
                component,
                collision_width,
                collision_height,
                SILK_MAX_DISTANCE_MM,
                prefer_vertical=is_vertical,
                preferred_side=preferred_side,
                dense=True,
            )
            near_seen = set(near)
            far = [
                position
                for position in _generate_label_candidates(
                    component,
                    collision_width,
                    collision_height,
                    SILK_FAR_MAX_DISTANCE_MM,
                    prefer_vertical=is_vertical,
                    preferred_side=preferred_side,
                )
                if position not in near_seen
            ]
            positions = near + far
            if not positions:
                continue
            # Every candidate's static legality in bulk -- one box build and
            # one query per tree for the whole batch -- then the candidates
            # are walked in their original order, so what gets kept, and in
            # what order, is exactly what the one-at-a-time loop produced.
            boxes = _boxes_at(positions, collision_width, collision_height)
            usable = inside_board_mask(boxes)
            usable &= ~_tree_hit_mask(footprint_trees.get(side), boxes)
            strict_clear = ~_tree_hit_mask(strict_tree, boxes)
            relaxed_clear = ~_tree_hit_mask(relaxed_tree, boxes) if allow_trace_overlap_fallback else None
            for i, (x, y) in enumerate(positions):
                if not usable[i]:
                    continue
                is_strict = bool(strict_clear[i])
                if not is_strict and not (
                    allow_trace_overlap_fallback
                    and relaxed_clear[i]
                    and len(relaxed_results) < SILK_MAX_STATIC_CANDIDATES_PER_MODE
                ):
                    continue  # would have been dropped after the arrow work anyway
                gap_mm = _label_gap_distance(
                    component, x, y, collision_width, collision_height
                )
                arrow = None
                if _label_needs_pointer(
                    component,
                    x,
                    y,
                    collision_width,
                    collision_height,
                    gap_mm,
                    components=candidates,
                    exclude_id=component.get("id"),
                ):
                    arrow = _arrow_points(
                        component,
                        x,
                        y,
                        collision_width,
                        collision_height,
                    )
                    if arrow is None:
                        continue
                    if not arrow_clear(arrow, side, component.get("id")):
                        continue

                result = (
                    x,
                    y,
                    width_mm,
                    height_mm,
                    collision_width,
                    collision_height,
                    rotation_deg,
                    gap_mm,
                    arrow,
                )
                if is_strict:
                    strict_results.append(result + ("strict",))
                else:
                    relaxed_results.append(result + ("trace_relaxed",))
                if len(strict_results) >= SILK_MAX_STATIC_CANDIDATES_PER_MODE:
                    break
            if len(strict_results) >= SILK_MAX_STATIC_CANDIDATES_PER_MODE:
                break
        return strict_results + relaxed_results

    global_labels = _DynamicGeometryIndex()
    labels_by_side = {
        "F.Cu": _DynamicGeometryIndex(),
        "B.Cu": _DynamicGeometryIndex(),
    }
    arrows_by_side = {
        "F.Cu": _DynamicGeometryIndex(),
        "B.Cu": _DynamicGeometryIndex(),
    }
    placed = []
    failures = []
    worker_count = min(SILK_GEOMETRY_WORKERS, len(candidates))
    window = max(worker_count, worker_count * SILK_GEOMETRY_WINDOW_MULTIPLIER)

    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="silk-place",
    ) as executor:
        futures = {}
        for index in range(min(window, len(candidates))):
            futures[index] = executor.submit(static_candidates, candidates[index])

        for index, component in enumerate(candidates):
            candidate_results = futures.pop(index).result()
            next_index = index + window
            if next_index < len(candidates):
                futures[next_index] = executor.submit(
                    static_candidates,
                    candidates[next_index],
                )

            accepted = None
            for (
                x,
                y,
                width_mm,
                height_mm,
                collision_width,
                collision_height,
                rotation_deg,
                gap_mm,
                arrow,
                mode,
            ) in candidate_results:
                side = component.get("side")
                if side not in labels_by_side:
                    side = "F.Cu"
                label_geometry = _rotated_box(
                    x, y, collision_width, collision_height
                )
                if global_labels.intersects(label_geometry):
                    continue
                if arrows_by_side[side].intersects(label_geometry):
                    continue

                arrow_geometry = None
                if arrow is not None:
                    arrow_geometry = _segment_geometry(
                        *arrow,
                        SILK_ARROW_WIDTH_MM,
                    )
                    if labels_by_side[side].intersects(arrow_geometry):
                        continue
                    if arrows_by_side[side].intersects(arrow_geometry):
                        continue

                padded_geometry = _rotated_box(
                    x,
                    y,
                    collision_width + 2.0 * SILK_PADDING_MM,
                    collision_height + 2.0 * SILK_PADDING_MM,
                )
                accepted = {
                    "component_id": component.get("id"),
                    "component_name": component.get("name"),
                    "text": component["ref"],
                    "x": x,
                    "y": y,
                    "width_mm": width_mm,
                    "height_mm": SILK_TEXT_HEIGHT_MM,
                    "collision_width_mm": collision_width,
                    "collision_height_mm": collision_height,
                    "rotation_deg": rotation_deg,
                    "font_size": f"{SILK_TEXT_HEIGHT_MM * 2.8346:.2f}pt",
                    "side": side,
                    "gap_mm": gap_mm,
                    "arrow_cells": set(),
                    "placement_mode": mode,
                }
                if arrow is not None:
                    accepted.update(
                        {
                            "arrow_start_x": arrow[0],
                            "arrow_start_y": arrow[1],
                            "arrow_end_x": arrow[2],
                            "arrow_end_y": arrow[3],
                        }
                    )
                accepted["_padded_geometry"] = padded_geometry
                accepted["_global_index"] = global_labels.add(padded_geometry)
                accepted["_side_index"] = labels_by_side[side].add(padded_geometry)
                if arrow_geometry is not None:
                    accepted["_arrow_index"] = arrows_by_side[side].add(arrow_geometry)
                break

            if accepted is None:
                failures.append(component["ref"])
            else:
                placed.append(accepted)
            if (index + 1) % 25 == 0 or index + 1 == len(candidates):
                print(
                    f"Silkscreen placement: {index + 1}/{len(candidates)} "
                    f"components, {len(placed)} placed, {len(failures)} failed"
                )

    # Backtracking rescue pass (post-placement, greedy, single-swap only):
    # for each failed component, check whether its best candidate spot is
    # blocked by exactly one already-placed label (not a static obstacle,
    # which can't move). If that one label can find an alternate valid spot
    # of its own, relocate it and place the failure in the freed spot --
    # net +1 placed. If not, leave both exactly as they were.
    comp_by_id = {c["id"]: c for c in candidates}
    # Placed labels by their live entry in global_labels: the blocker test
    # below asks the index which placed boxes a candidate hits instead of
    # rebuilding every placed label's box for every candidate. At this stage
    # each index entry is exactly one placed label's padded box -- the same
    # _rotated_box(x, y, cw + 2*pad, ch + 2*pad) the old scan rebuilt -- so
    # the blockers found are identical.
    label_by_global_index = {label["_global_index"]: label for label in placed}
    still_failed = []
    rescued_count = 0
    for ref in failures:
        comp = next((c for c in candidates if c["ref"] == ref), None)
        if comp is None:
            still_failed.append(ref)
            continue
        f_side = comp.get("side")
        if f_side not in footprint_trees:
            f_side = "F.Cu"
        width_mm, height_mm = _estimate_text_box(comp["ref"], SILK_TEXT_HEIGHT_MM)
        orientation_specs = _orientation_space_score_geometry(
            comp, width_mm, height_mm, footprint_trees, strict_tree, inside_board, f_side,
            preferred_orientation=cluster_preferred_orientation.get(comp.get("id")),
            inside_board_mask=inside_board_mask,
        )
        preferred_side = cluster_preferred_side.get(comp.get("id"))
        rescued = False
        for rotation_deg, collision_width, collision_height, is_vertical in orientation_specs:
            near = _generate_label_candidates(
                comp, collision_width, collision_height, SILK_MAX_DISTANCE_MM,
                prefer_vertical=is_vertical, preferred_side=preferred_side, dense=True,
            )
            if not near:
                continue
            near_boxes = _boxes_at(near, collision_width, collision_height)
            near_usable = inside_board_mask(near_boxes)
            near_usable &= ~_tree_hit_mask(footprint_trees.get(f_side), near_boxes)
            near_usable &= ~_tree_hit_mask(strict_tree, near_boxes)
            for i, (x, y) in enumerate(near):
                if not near_usable[i]:
                    continue
                label_geometry = near_boxes[i]
                hit_indices = global_labels.hits(label_geometry)
                if not hit_indices:
                    continue  # would already have succeeded; nothing to rescue here
                blockers = [
                    label_by_global_index[index] for index in hit_indices if index in label_by_global_index
                ]
                if len(blockers) != 1:
                    continue
                blocker_label = blockers[0]
                blocker_comp = comp_by_id.get(blocker_label["component_id"])
                if blocker_comp is None:
                    continue
                blocker_side = blocker_label["side"] if blocker_label["side"] in labels_by_side else "F.Cu"
                blocker_mode = blocker_label.get("placement_mode") or "strict"
                blocker_tree = strict_tree if blocker_mode != "trace_relaxed" else relaxed_tree
                blocker_backup = dict(blocker_label)

                global_labels.remove(blocker_label["_global_index"])
                labels_by_side[blocker_side].remove(blocker_label["_side_index"])
                if "_arrow_index" in blocker_label:
                    arrows_by_side[blocker_side].remove(blocker_label["_arrow_index"])
                reserved_index = global_labels.add(label_geometry)  # reserve F's spot

                b_width_mm, b_height_mm = _estimate_text_box(blocker_comp["ref"], SILK_TEXT_HEIGHT_MM)
                b_orientation_specs = _orientation_space_score_geometry(
                    blocker_comp, b_width_mm, b_height_mm, footprint_trees, strict_tree, inside_board, blocker_side,
                    preferred_orientation=cluster_preferred_orientation.get(blocker_comp.get("id")),
                )
                b_preferred_side = cluster_preferred_side.get(blocker_comp.get("id"))
                new_x = new_y = None
                for b_rot, b_cw, b_ch, b_is_v in b_orientation_specs:
                    b_near = _generate_label_candidates(
                        blocker_comp, b_cw, b_ch, SILK_MAX_DISTANCE_MM,
                        prefer_vertical=b_is_v, preferred_side=b_preferred_side, dense=True,
                    )
                    if not b_near:
                        continue
                    b_boxes = _boxes_at(b_near, b_cw, b_ch)
                    b_usable = inside_board_mask(b_boxes)
                    b_usable &= ~_tree_hit_mask(footprint_trees.get(blocker_side), b_boxes)
                    b_usable &= ~_tree_hit_mask(blocker_tree, b_boxes)
                    for j, (bx, by) in enumerate(b_near):
                        if not b_usable[j]:
                            continue
                        b_geom = b_boxes[j]
                        if global_labels.intersects(b_geom):
                            continue
                        if arrows_by_side[blocker_side].intersects(b_geom):
                            continue
                        new_x, new_y, new_rot, new_cw, new_ch = bx, by, b_rot, b_cw, b_ch
                        break
                    if new_x is not None:
                        break
                global_labels.remove(reserved_index)

                if new_x is None:
                    restore_padded = _rotated_box(
                        blocker_backup["x"], blocker_backup["y"],
                        blocker_backup["collision_width_mm"] + 2.0 * SILK_PADDING_MM,
                        blocker_backup["collision_height_mm"] + 2.0 * SILK_PADDING_MM,
                    )
                    blocker_label["_padded_geometry"] = restore_padded
                    blocker_label["_global_index"] = global_labels.add(restore_padded)
                    label_by_global_index[blocker_label["_global_index"]] = blocker_label
                    blocker_label["_side_index"] = labels_by_side[blocker_side].add(restore_padded)
                    if blocker_backup.get("arrow_start_x") is not None:
                        blocker_label["_arrow_index"] = arrows_by_side[blocker_side].add(
                            _segment_geometry(
                                blocker_backup["arrow_start_x"], blocker_backup["arrow_start_y"],
                                blocker_backup["arrow_end_x"], blocker_backup["arrow_end_y"], SILK_ARROW_WIDTH_MM,
                            )
                        )
                    continue

                blocker_label.clear()
                blocker_label.update(blocker_backup)
                blocker_label["x"], blocker_label["y"] = new_x, new_y
                blocker_label["rotation_deg"] = new_rot
                blocker_label["collision_width_mm"], blocker_label["collision_height_mm"] = new_cw, new_ch
                blocker_label["gap_mm"] = _label_gap_distance(blocker_comp, new_x, new_y, new_cw, new_ch)
                for key in ("arrow_start_x", "arrow_start_y", "arrow_end_x", "arrow_end_y"):
                    blocker_label.pop(key, None)
                new_padded = _rotated_box(
                    new_x, new_y, new_cw + 2.0 * SILK_PADDING_MM, new_ch + 2.0 * SILK_PADDING_MM,
                )
                blocker_label["_padded_geometry"] = new_padded
                blocker_label["_global_index"] = global_labels.add(new_padded)
                label_by_global_index[blocker_label["_global_index"]] = blocker_label
                blocker_label["_side_index"] = labels_by_side[blocker_side].add(new_padded)

                f_gap = _label_gap_distance(comp, x, y, collision_width, collision_height)
                f_arrow = None
                if _label_needs_pointer(
                    comp, x, y, collision_width, collision_height, f_gap,
                    components=candidates, exclude_id=comp.get("id"),
                ):
                    f_arrow = _arrow_points(comp, x, y, collision_width, collision_height)
                f_label = {
                    "component_id": comp.get("id"),
                    "component_name": comp.get("name"),
                    "text": comp["ref"],
                    "x": x, "y": y,
                    "width_mm": width_mm, "height_mm": SILK_TEXT_HEIGHT_MM,
                    "collision_width_mm": collision_width, "collision_height_mm": collision_height,
                    "rotation_deg": rotation_deg,
                    "font_size": f"{SILK_TEXT_HEIGHT_MM * 2.8346:.2f}pt",
                    "side": f_side, "gap_mm": f_gap, "arrow_cells": set(),
                    "placement_mode": "strict",
                }
                if f_arrow is not None:
                    f_label.update({
                        "arrow_start_x": f_arrow[0], "arrow_start_y": f_arrow[1],
                        "arrow_end_x": f_arrow[2], "arrow_end_y": f_arrow[3],
                    })
                f_padded = _rotated_box(
                    x, y, collision_width + 2.0 * SILK_PADDING_MM, collision_height + 2.0 * SILK_PADDING_MM,
                )
                f_label["_padded_geometry"] = f_padded
                f_label["_global_index"] = global_labels.add(f_padded)
                label_by_global_index[f_label["_global_index"]] = f_label
                f_label["_side_index"] = labels_by_side[f_side].add(f_padded)
                if f_arrow is not None:
                    f_label["_arrow_index"] = arrows_by_side[f_side].add(
                        _segment_geometry(*f_arrow, SILK_ARROW_WIDTH_MM)
                    )
                placed.append(f_label)
                rescued = True
                rescued_count += 1
                break
            if rescued:
                break
        if not rescued:
            still_failed.append(ref)
    failures = still_failed
    if rescued_count:
        print(f"Backtracking rescue: {rescued_count} previously-failed component(s) placed")

    comp_by_id = {c["id"]: c for c in candidates}
    relocated_count = 0
    for label in placed:
        comp = comp_by_id.get(label["component_id"])
        if comp is None:
            continue
        side = label["side"] if label["side"] in labels_by_side else "F.Cu"
        mode = label.get("placement_mode") or "strict"
        obstacle_tree = strict_tree if mode != "trace_relaxed" else relaxed_tree
        # Free this label's own contribution first so the buffer/relocation
        # checks below don't collide with themselves.
        global_labels.remove(label["_global_index"])
        labels_by_side[side].remove(label["_side_index"])
        if "_arrow_index" in label:
            arrows_by_side[side].remove(label["_arrow_index"])

        def blocked_check(geometry, side=side, obstacle_tree=obstacle_tree):
            if _tree_intersects(footprint_trees.get(side), geometry):
                return True
            if _tree_intersects(obstacle_tree, geometry):
                return True
            if global_labels.intersects(geometry):
                return True
            if arrows_by_side[side].intersects(geometry):
                return True
            return False

        def bulk_blocked(geometries, side=side, obstacle_tree=obstacle_tree):
            """Vectorized 'outside the board or blocked_check' over an array
            of geometries: the trees in bulk, the dynamic indexes per
            geometry. Entry i equals (not inside_board(g) or blocked_check(g))."""
            blocked = ~inside_board_mask(geometries)
            blocked |= _tree_hit_mask(footprint_trees.get(side), geometries)
            blocked |= _tree_hit_mask(obstacle_tree, geometries)
            for i in np.flatnonzero(~blocked):
                geometry = geometries[i]
                if global_labels.intersects(geometry) or arrows_by_side[side].intersects(geometry):
                    blocked[i] = True
            return blocked

        width_mm = label["collision_width_mm"]
        height_mm = label["collision_height_mm"]
        ok, _ratio = _label_meets_buffer_geometry(
            label["x"], label["y"], width_mm, height_mm, inside_board, blocked_check, bulk_blocked=bulk_blocked,
        )
        relocated = False
        if not ok:
            current_side = _infer_label_side_from_component(label["x"], label["y"], comp)
            for alt_side in _SIDE_ALTERNATES[current_side]:
                near = _generate_label_candidates(
                    comp, width_mm, height_mm, SILK_MAX_DISTANCE_MM,
                    preferred_side=alt_side, dense=True,
                )
                if not near:
                    continue
                near_blocked = bulk_blocked(_boxes_at(near, width_mm, height_mm))
                for i, (x, y) in enumerate(near):
                    if near_blocked[i]:
                        continue
                    gap_mm = _label_gap_distance(comp, x, y, width_mm, height_mm)
                    arrow = None
                    arrow_geometry = None
                    if _label_needs_pointer(
                        comp, x, y, width_mm, height_mm, gap_mm,
                        components=candidates, exclude_id=comp.get("id"),
                    ):
                        arrow = _arrow_points(comp, x, y, width_mm, height_mm)
                        if arrow is None:
                            continue
                        if not arrow_clear(arrow, side, comp.get("id")):
                            continue
                        arrow_geometry = _segment_geometry(*arrow, SILK_ARROW_WIDTH_MM)
                    buffer_ok, ratio = _label_meets_buffer_geometry(
                        x, y, width_mm, height_mm, inside_board, blocked_check, bulk_blocked=bulk_blocked,
                    )
                    if not buffer_ok:
                        continue
                    # Commit: this candidate both fits and has better buffers.
                    label["x"], label["y"] = x, y
                    label["gap_mm"] = gap_mm
                    for key in ("arrow_start_x", "arrow_start_y", "arrow_end_x", "arrow_end_y"):
                        label.pop(key, None)
                    if arrow is not None:
                        label.update({
                            "arrow_start_x": arrow[0], "arrow_start_y": arrow[1],
                            "arrow_end_x": arrow[2], "arrow_end_y": arrow[3],
                        })
                    padded_geometry = _rotated_box(
                        x, y, width_mm + 2.0 * SILK_PADDING_MM, height_mm + 2.0 * SILK_PADDING_MM,
                    )
                    label["_padded_geometry"] = padded_geometry
                    label["_global_index"] = global_labels.add(padded_geometry)
                    label["_side_index"] = labels_by_side[side].add(padded_geometry)
                    if arrow_geometry is not None:
                        label["_arrow_index"] = arrows_by_side[side].add(arrow_geometry)
                    else:
                        label.pop("_arrow_index", None)
                    relocated = True
                    relocated_count += 1
                    break
                if relocated:
                    break
        if not relocated:
            # Restore exactly as it was.
            label["_global_index"] = global_labels.add(label["_padded_geometry"])
            label["_side_index"] = labels_by_side[side].add(label["_padded_geometry"])
            if label.get("arrow_start_x") is not None:
                arrow_geometry = _segment_geometry(
                    label["arrow_start_x"], label["arrow_start_y"],
                    label["arrow_end_x"], label["arrow_end_y"], SILK_ARROW_WIDTH_MM,
                )
                label["_arrow_index"] = arrows_by_side[side].add(arrow_geometry)
    if relocated_count:
        print(f"Buffer reinforcement: {relocated_count} label(s) relocated for better clearance")

    # Final polish: a label that's slightly off-axis (dense-slide drift) gets
    # one shot at snapping to the exact on-axis point on its own side, same
    # distance -- never switches sides, never forced, never loses a
    # placement. Skips labels with a pointer arrow (recomputing arrow
    # validity is a separate concern, not handled here).
    snapped_count = 0
    for label in placed:
        if label.get("arrow_start_x") is not None:
            continue
        comp = comp_by_id.get(label["component_id"])
        if comp is None:
            continue
        x, y = label["x"], label["y"]
        if _angle_deviation_from_axes(comp["cx"], comp["cy"], x, y) <= 1e-6:
            continue
        current_side = _infer_label_side_from_component(x, y, comp)
        snap_x, snap_y = (comp["cx"], y) if current_side in ("top", "bottom") else (x, comp["cy"])
        width_mm = label["collision_width_mm"]
        height_mm = label["collision_height_mm"]
        side = label["side"] if label["side"] in labels_by_side else "F.Cu"
        mode = label.get("placement_mode") or "strict"
        obstacle_tree = strict_tree if mode != "trace_relaxed" else relaxed_tree

        global_labels.remove(label["_global_index"])
        labels_by_side[side].remove(label["_side_index"])

        snap_geom = _rotated_box(snap_x, snap_y, width_mm, height_mm)
        ok = (
            inside_board(snap_geom)
            and not _tree_intersects(footprint_trees.get(side), snap_geom)
            and not _tree_intersects(obstacle_tree, snap_geom)
            and not global_labels.intersects(snap_geom)
            and not arrows_by_side[side].intersects(snap_geom)
        )
        if ok:
            label["x"], label["y"] = snap_x, snap_y
            label["gap_mm"] = _label_gap_distance(comp, snap_x, snap_y, width_mm, height_mm)
            padded = _rotated_box(
                snap_x, snap_y, width_mm + 2.0 * SILK_PADDING_MM, height_mm + 2.0 * SILK_PADDING_MM,
            )
            label["_padded_geometry"] = padded
            label["_global_index"] = global_labels.add(padded)
            label["_side_index"] = labels_by_side[side].add(padded)
            snapped_count += 1
        else:
            label["_global_index"] = global_labels.add(label["_padded_geometry"])
            label["_side_index"] = labels_by_side[side].add(label["_padded_geometry"])
    if snapped_count:
        print(f"Axis snap: {snapped_count} label(s) tightened to exact on-axis position")

    # Escape rescue (final pass): geometry-backend counterpart of
    # auto_place_silkscreen's dense-cluster block rescue -- same idea, same
    # SILK_ESCAPE_SCAN_STEP_MM/SILK_DENSE_CLUSTER_GAP_MM constants, but
    # obstacle legality here uses this backend's own Shapely trees and
    # dynamic indexes instead of grid cells. See that function's comment
    # for the full rationale. Must run before the bookkeeping cleanup below
    # -- it still needs the live dynamic indexes.
    board_scan_xs = []
    board_scan_ys = []
    for pad in pads:
        min_x, min_y, max_x, max_y = pad["bbox"]
        board_scan_xs.extend((min_x, max_x))
        board_scan_ys.extend((min_y, max_y))
    for segments in (trace_segments or {}).values():
        for seg in segments:
            half_w = seg["width"] / 2.0
            board_scan_xs.extend((seg["x0"] - half_w, seg["x0"] + half_w, seg["x1"] - half_w, seg["x1"] + half_w))
            board_scan_ys.extend((seg["y0"] - half_w, seg["y0"] + half_w, seg["y1"] - half_w, seg["y1"] + half_w))
    for via in vias:
        r = via["size"] / 2.0
        board_scan_xs.extend((via["x"] - r, via["x"] + r))
        board_scan_ys.extend((via["y"] - r, via["y"] + r))
    for x0, y0, x1, y1 in edge_segments or []:
        board_scan_xs.extend((x0, x1))
        board_scan_ys.extend((y0, y1))
    for comp in (all_components or candidates):
        board_scan_xs.extend((comp["min_x"], comp["max_x"]))
        board_scan_ys.extend((comp["min_y"], comp["max_y"]))
    board_scan_min_x = min(board_scan_xs) if board_scan_xs else 0.0
    board_scan_max_x = max(board_scan_xs) if board_scan_xs else 0.0
    board_scan_min_y = min(board_scan_ys) if board_scan_ys else 0.0
    board_scan_max_y = max(board_scan_ys) if board_scan_ys else 0.0

    # Escape search state shared by the sequential path and the parallel
    # driver's conflict fallback: the same _escape_ring_scan the worker
    # processes run, fed the live indexes instead of a snapshot.
    escape_scan_bounds = (board_scan_min_x, board_scan_min_y, board_scan_max_x, board_scan_max_y)

    def escape_blocked_checks(side):
        """(relaxed_blocked, strict_blocked) against the live indexes; with
        trace overlap disallowed both are the strict check."""
        footprint_tree = footprint_trees.get(side)
        strict_blocked = _escape_blocked_check(footprint_tree, strict_tree, global_labels, arrows_by_side[side])
        if not allow_trace_overlap_fallback:
            return strict_blocked, strict_blocked
        return _escape_blocked_check(footprint_tree, relaxed_tree, global_labels, arrows_by_side[side]), strict_blocked

    def escape_bulk_checks(side):
        """(inside_mask, relaxed_mask, strict_mask) for _escape_ring_scan's
        bulk path: the same footprint / obstacle-tree / placed-label /
        connector tests as _escape_blocked_check, trees in bulk, dynamic
        indexes per geometry."""
        footprint_tree = footprint_trees.get(side)
        arrows_index = arrows_by_side[side]

        def mask_for(obstacle_tree):
            def blocked_mask(geometries):
                blocked = _tree_hit_mask(footprint_tree, geometries) | _tree_hit_mask(obstacle_tree, geometries)
                for i in np.flatnonzero(~blocked):
                    geometry = geometries[i]
                    if global_labels.intersects(geometry) or arrows_index.intersects(geometry):
                        blocked[i] = True
                return blocked
            return blocked_mask

        strict_mask = mask_for(strict_tree)
        relaxed_mask = mask_for(relaxed_tree) if allow_trace_overlap_fallback else strict_mask
        return inside_board_mask, relaxed_mask, strict_mask

    comp_by_ref = {c["ref"]: c for c in candidates}
    failed_comps = [comp_by_ref[ref] for ref in failures if ref in comp_by_ref]

    parent = {c["id"]: c["id"] for c in failed_comps}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    for i in range(len(failed_comps)):
        for j in range(i + 1, len(failed_comps)):
            a, b = failed_comps[i], failed_comps[j]
            if a["side"] == b["side"] and _bbox_gap_distance(a, b) <= SILK_DENSE_CLUSTER_GAP_MM:
                union(a["id"], b["id"])

    cluster_groups = defaultdict(list)
    for c in failed_comps:
        cluster_groups[find(c["id"])].append(c)
    dense_groups = [members for members in cluster_groups.values() if len(members) >= 2]
    clustered_refs = {m["ref"] for members in dense_groups for m in members}

    escaped_count = 0
    still_failed = []
    failed_cluster_blocks = []  # (members, specs, block_w, block_h) for
    # clusters whose on-board scan found nowhere to land -- handled by the
    # guaranteed off-board fallback below as one combined block, not
    # scattered back into individual failures.

    # Work list in exactly the order the sequential version processed
    # these: every dense cluster (one combined block each), then every lone
    # failure. The per-item preparation here (_pack_cluster_grid, the
    # orientation scoring) depends only on static data, so doing it up
    # front is identical to doing it the moment each item is handled.
    work_items = []
    for members in dense_groups:
        specs, block_w, block_h = _pack_cluster_grid(members)
        side = members[0].get("side")
        if side not in footprint_trees:
            side = "F.Cu"
        cluster_min_x = min(m["min_x"] for m in members)
        cluster_min_y = min(m["min_y"] for m in members)
        cluster_max_x = max(m["max_x"] for m in members)
        cluster_max_y = max(m["max_y"] for m in members)
        work_items.append(
            {
                "kind": "cluster",
                "members": members,
                "specs": specs,
                "block_w": block_w,
                "block_h": block_h,
                "side": side,
                "cluster_bbox": (cluster_min_x, cluster_min_y, cluster_max_x, cluster_max_y),
                "anchor_cx": (cluster_min_x + cluster_max_x) / 2.0,
                "anchor_cy": (cluster_min_y + cluster_max_y) / 2.0,
                "dims": [(block_w, block_h)],
            }
        )
    for ref in failures:
        if ref in clustered_refs:
            continue
        comp = comp_by_ref.get(ref)
        if comp is None:
            still_failed.append(ref)
            continue
        f_side = comp.get("side")
        if f_side not in footprint_trees:
            f_side = "F.Cu"
        width_mm, height_mm = _estimate_text_box(comp["ref"], SILK_TEXT_HEIGHT_MM)
        orientation_specs = _orientation_space_score_geometry(
            comp, width_mm, height_mm, footprint_trees, strict_tree, inside_board, f_side,
            preferred_orientation=cluster_preferred_orientation.get(comp.get("id")),
            inside_board_mask=inside_board_mask,
        )
        work_items.append(
            {
                "kind": "individual",
                "ref": ref,
                "comp": comp,
                "side": f_side,
                "width_mm": width_mm,
                "height_mm": height_mm,
                "orientation_specs": orientation_specs,
                "anchor_cx": comp["cx"],
                "anchor_cy": comp["cy"],
                "dims": [(cw, ch) for _rot, cw, ch, _is_v in orientation_specs],
            }
        )

    def commit_cluster(item, block_cx, block_cy, mode, buffered=False):
        """The sequential version's cluster commit. Returns the geometries
        it added to the dynamic indexes (for the batch conflict check in
        the parallel driver). A block placed with its buffer reserves the
        buffer zone too, so later escape placements keep out of it."""
        nonlocal escaped_count
        members, specs = item["members"], item["specs"]
        block_w, block_h, side = item["block_w"], item["block_h"], item["side"]
        cluster_min_x, cluster_min_y, cluster_max_x, cluster_max_y = item["cluster_bbox"]
        cluster_cx, cluster_cy = item["anchor_cx"], item["anchor_cy"]
        block_min_x = block_cx - block_w / 2.0
        block_min_y = block_cy - block_h / 2.0

        new_labels = []
        for m, w_mm, h_mm, rotation_deg, cw, ch, local_x, local_y in specs:
            x = block_min_x + local_x
            y = block_min_y + local_y
            new_labels.append(
                {
                    "component_id": m.get("id"),
                    "component_name": m.get("name"),
                    "text": m["ref"],
                    "x": x,
                    "y": y,
                    "width_mm": w_mm,
                    "height_mm": SILK_TEXT_HEIGHT_MM,
                    "collision_width_mm": cw,
                    "collision_height_mm": ch,
                    "rotation_deg": rotation_deg,
                    "font_size": f"{SILK_TEXT_HEIGHT_MM * 2.8346:.2f}pt",
                    "side": side,
                    "gap_mm": _label_gap_distance(m, x, y, cw, ch),
                    "arrow_cells": set(),
                    "placement_mode": mode,
                }
            )

        # One shared connector for the whole cluster -- from whichever
        # label ends up closest to the cluster's own location, clipped
        # around any obstacle it crosses -- not one line per member, since
        # they're all part of the same crowded area.
        cluster_half_w = (cluster_max_x - cluster_min_x) / 2.0
        cluster_half_h = (cluster_max_y - cluster_min_y) / 2.0
        best_arrow = None
        best_length = None
        best_label = None
        for lbl in new_labels:
            candidate = _rect_to_rect_line(
                lbl["x"], lbl["y"], lbl["collision_width_mm"] / 2.0, lbl["collision_height_mm"] / 2.0,
                cluster_cx, cluster_cy, cluster_half_w, cluster_half_h,
            )
            if candidate is None:
                continue
            length = math.hypot(candidate[2] - candidate[0], candidate[3] - candidate[1])
            if best_length is None or length < best_length:
                best_arrow, best_length, best_label = candidate, length, lbl

        arrow_geometries = []
        if best_arrow is not None:
            segments = clip_arrow_against_obstacles(
                best_arrow, side, exclude_component_ids={m.get("id") for m in members},
            )
            if segments:
                best_label["arrow_start_x"], best_label["arrow_start_y"], best_label["arrow_end_x"], best_label["arrow_end_y"] = segments[0]
                if len(segments) > 1:
                    best_label["extra_arrow_segments"] = segments[1:]
                arrow_geometries = [_segment_geometry(*seg, SILK_ARROW_WIDTH_MM) for seg in segments]

        reserve = SILK_PADDING_MM + (SILK_ESCAPE_MARGIN_MM if buffered else 0.0)
        padded_block_geometry = _rotated_box(
            block_cx, block_cy, block_w + 2.0 * reserve, block_h + 2.0 * reserve,
        )
        block_global_index = global_labels.add(padded_block_geometry)
        block_side_index = labels_by_side[side].add(padded_block_geometry)

        # Every member shares the block's combined dynamic-index footprint
        # for its own label box, but only the anchor carries the single
        # shared connector's arrow registration.
        for lbl in new_labels:
            lbl["_padded_geometry"] = padded_block_geometry
            lbl["_global_index"] = block_global_index
            lbl["_side_index"] = block_side_index
            lbl["_cluster_bbox"] = item["cluster_bbox"]  # for the final upright pass's connector re-route
            lbl["_cluster_member_ids"] = {m.get("id") for m in members}
        if arrow_geometries:
            best_label["_arrow_indices"] = [arrows_by_side[side].add(g) for g in arrow_geometries]

        placed.extend(new_labels)
        escaped_count += len(new_labels)
        return [padded_block_geometry] + arrow_geometries

    def commit_individual(item, spec_index, x, y, mode, buffered=False):
        """The sequential version's lone-failure commit. A label placed with
        its buffer reserves the buffer zone too."""
        nonlocal escaped_count
        comp, f_side = item["comp"], item["side"]
        width_mm, height_mm = item["width_mm"], item["height_mm"]
        rotation_deg, collision_width, collision_height, _is_vertical = item["orientation_specs"][spec_index]
        gap_mm = _label_gap_distance(comp, x, y, collision_width, collision_height)
        segments = clipped_escape_arrow(comp, x, y, collision_width, collision_height, gap_mm, f_side)
        arrow_geometries = [_segment_geometry(*seg, SILK_ARROW_WIDTH_MM) for seg in segments]
        label = {
            "component_id": comp.get("id"),
            "component_name": comp.get("name"),
            "text": comp["ref"],
            "x": x,
            "y": y,
            "width_mm": width_mm,
            "height_mm": SILK_TEXT_HEIGHT_MM,
            "collision_width_mm": collision_width,
            "collision_height_mm": collision_height,
            "rotation_deg": rotation_deg,
            "font_size": f"{SILK_TEXT_HEIGHT_MM * 2.8346:.2f}pt",
            "side": f_side,
            "gap_mm": gap_mm,
            "arrow_cells": set(),
            "placement_mode": mode,
        }
        if segments:
            label["arrow_start_x"], label["arrow_start_y"], label["arrow_end_x"], label["arrow_end_y"] = segments[0]
            if len(segments) > 1:
                label["extra_arrow_segments"] = segments[1:]
        reserve = SILK_PADDING_MM + (SILK_ESCAPE_MARGIN_MM if buffered else 0.0)
        padded = _rotated_box(
            x, y, collision_width + 2.0 * reserve, collision_height + 2.0 * reserve,
        )
        label["_padded_geometry"] = padded
        label["_global_index"] = global_labels.add(padded)
        label["_side_index"] = labels_by_side[f_side].add(padded)
        if arrow_geometries:
            label["_arrow_indices"] = [arrows_by_side[f_side].add(g) for g in arrow_geometries]
        placed.append(label)
        escaped_count += 1
        return [padded] + arrow_geometries

    def scan_item_sequential(item):
        """The sequential version's own search for one item, run against
        the live indexes -- the same _escape_ring_scan and pass order the
        worker processes use, so both paths agree. Returns (spec_index,
        mode, (x, y), buffered) or None. This is the fallback the parallel driver
        uses whenever a speculative result can't be proven safe, and the
        whole path when the parallel driver isn't used at all."""
        relaxed_blocked, strict_blocked = escape_blocked_checks(item["side"])
        result = _escape_ring_scan(
            item["anchor_cx"], item["anchor_cy"], item["dims"], SILK_ESCAPE_MARGIN_MM, escape_scan_bounds,
            inside_board, relaxed_blocked, strict_blocked, bulk=escape_bulk_checks(item["side"]),
        )
        if result is None:
            return None
        return (
            result["spec_index"], mode_for(item, "strict" if result["strict"] else "relaxed"),
            result["position"], result["buffered"],
        )

    def record_failure(item):
        if item["kind"] == "cluster":
            failed_cluster_blocks.append((item["members"], item["specs"], item["block_w"], item["block_h"], item["side"]))
        else:
            still_failed.append(item["ref"])

    def apply_sequential(item):
        result = scan_item_sequential(item)
        if result is None:
            record_failure(item)
            return []
        spec_index, mode, (x, y), buffered = result
        if item["kind"] == "cluster":
            return commit_cluster(item, x, y, mode, buffered)
        return commit_individual(item, spec_index, x, y, mode, buffered)

    def mode_for(item, tree_name):
        if item["kind"] == "cluster":
            return "escape_cluster" if tree_name == "strict" else "escape_cluster_trace_relaxed"
        return "escape" if tree_name == "strict" else "escape_trace_relaxed"

    # Parallel driver: speculate, validate, commit. Each batch of items is
    # scanned concurrently in worker processes against the same frozen
    # snapshot of already-placed labels, then committed in the original
    # order. A speculative result is accepted only if no item committed
    # earlier in the batch landed inside the region its scan depended on
    # (_escape_influence_region) -- earlier commits can only remove legal
    # space, never add it, so an untouched region means the sequential run
    # would have produced the identical result. Anything else is re-scanned
    # sequentially. A worker finding nothing anywhere is also final: the
    # live state only has *more* obstacles than the snapshot it scanned.
    processed = 0
    speculative_commits = 0
    sequential_fallbacks = 0
    use_pool = (
        SILK_ESCAPE_USE_PROCESSES
        and len(work_items) >= SILK_ESCAPE_PARALLEL_MIN_ITEMS
        and SILK_GEOMETRY_WORKERS > 1
    )
    if use_pool:
        worker_count = min(SILK_GEOMETRY_WORKERS, len(work_items))
        static_inputs = (candidates, all_components, pads, trace_segments, vias, edge_segments)
        try:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_escape_worker_init,
                initargs=(static_inputs,),
            ) as executor:
                while processed < len(work_items):
                    batch = work_items[processed:processed + worker_count]
                    global_snapshot = [g for g in global_labels.geometries if g is not None]
                    arrow_snapshots = {
                        s: [g for g in index.geometries if g is not None] for s, index in arrows_by_side.items()
                    }
                    tasks = [
                        {
                            "anchor_cx": item["anchor_cx"],
                            "anchor_cy": item["anchor_cy"],
                            "side": item["side"],
                            "dims": item["dims"],
                            "allow_relaxed": allow_trace_overlap_fallback,
                            "global_geometries": global_snapshot,
                            "arrow_geometries": arrow_snapshots.get(item["side"], []),
                        }
                        for item in batch
                    ]
                    results = list(executor.map(_escape_scan_worker, tasks))
                    committed_regions = []
                    for item, result in zip(batch, results):
                        if not result["found"]:
                            record_failure(item)
                            speculative_commits += 1
                            processed += 1
                            continue
                        spec_index = result["spec_index"]
                        x, y = result["position"]
                        max_extent = max(max(w, h) for w, h in item["dims"])
                        region = _escape_influence_region(
                            result["nearest_hits"], result["coarse_hit"], max_extent, SILK_ESCAPE_MARGIN_MM,
                        )
                        conflict = any(_bbox_overlaps(region, r) for r in committed_regions)
                        if conflict:
                            geoms = apply_sequential(item)
                            sequential_fallbacks += 1
                        else:
                            mode = mode_for(item, "strict" if result["strict"] else "relaxed")
                            if item["kind"] == "cluster":
                                geoms = commit_cluster(item, x, y, mode, result["buffered"])
                            else:
                                geoms = commit_individual(item, spec_index, x, y, mode, result["buffered"])
                            speculative_commits += 1
                        committed_regions.extend(g.bounds for g in geoms)
                        processed += 1
        except Exception as exc:
            print(f"Escape rescue: parallel scan unavailable ({exc!r}); finishing sequentially")
    for item in work_items[processed:]:
        apply_sequential(item)
    if use_pool:
        print(
            f"Escape rescue parallel: {speculative_commits} item(s) committed from concurrent scans, "
            f"{sequential_fallbacks} re-scanned sequentially (possible conflict)"
        )

    if escaped_count:
        print(f"Escape rescue: {escaped_count} previously-failed component(s) placed via combined blocks or long-range rescue")

    # Guaranteed off-board fallback: anything still in still_failed (a lone
    # component) or failed_cluster_blocks (a whole cluster that found no
    # on-board spot) gets placed just outside the board instead, on
    # whichever edge is nearest to it, ordered along that edge to match
    # its real relative position (so "X sits left of Y on the board" is
    # still true in the off-board strip). This is unconditional and never
    # fails -- off-board space is unbounded -- so `failed` is empty after
    # this point; off_board_refs tracks what actually needed it, since
    # these still need a human to move them onto the real board. No
    # connector line is drawn for them: the text alone sits in the strip.
    off_board_refs = []
    if still_failed or failed_cluster_blocks:
        def edge_side_for(cx, cy):
            d_top = cy - board_scan_min_y
            d_bottom = board_scan_max_y - cy
            d_left = cx - board_scan_min_x
            d_right = board_scan_max_x - cx
            closest = min(d_top, d_bottom, d_left, d_right)
            if closest == d_top:
                return "top"
            if closest == d_bottom:
                return "bottom"
            if closest == d_left:
                return "left"
            return "right"

        off_board_items = []
        for ref in still_failed:
            comp = comp_by_ref.get(ref)
            if comp is None:
                continue
            edge = edge_side_for(comp["cx"], comp["cy"])
            sort_key = comp["cx"] if edge in ("top", "bottom") else comp["cy"]
            off_board_items.append((edge, sort_key, "individual", comp))
        for members, specs, block_w, block_h, side in failed_cluster_blocks:
            cl_min_x = min(m["min_x"] for m in members)
            cl_min_y = min(m["min_y"] for m in members)
            cl_max_x = max(m["max_x"] for m in members)
            cl_max_y = max(m["max_y"] for m in members)
            cl_cx = (cl_min_x + cl_max_x) / 2.0
            cl_cy = (cl_min_y + cl_max_y) / 2.0
            edge = edge_side_for(cl_cx, cl_cy)
            sort_key = cl_cx if edge in ("top", "bottom") else cl_cy
            payload = (members, specs, block_w, block_h, side, cl_cx, cl_cy, cl_min_x, cl_min_y, cl_max_x, cl_max_y)
            off_board_items.append((edge, sort_key, "cluster", payload))

        for edge in ("top", "bottom", "left", "right"):
            group = sorted((item for item in off_board_items if item[0] == edge), key=lambda item: item[1])
            cursor = 0.0
            for _edge, _sort_key, kind, payload in group:
                if kind == "individual":
                    comp = payload
                    f_side = comp.get("side")
                    if f_side not in footprint_trees:
                        f_side = "F.Cu"
                    width_mm, height_mm = _estimate_text_box(comp["ref"], SILK_TEXT_HEIGHT_MM)
                    orientation_specs = _orientation_space_score_geometry(
                        comp, width_mm, height_mm, footprint_trees, strict_tree, inside_board, f_side,
                        preferred_orientation=cluster_preferred_orientation.get(comp.get("id")),
                    )
                    rotation_deg, item_w, item_h, _is_v = orientation_specs[0]
                else:
                    members, specs, block_w, block_h, f_side, cl_cx, cl_cy, cl_min_x, cl_min_y, cl_max_x, cl_max_y = payload
                    item_w, item_h = block_w, block_h

                if edge in ("top", "bottom"):
                    item_cx = board_scan_min_x + cursor + item_w / 2.0
                    item_cy = (
                        board_scan_min_y - SILK_OFFBOARD_GAP_MM - item_h / 2.0
                        if edge == "top"
                        else board_scan_max_y + SILK_OFFBOARD_GAP_MM + item_h / 2.0
                    )
                    cursor += item_w + SILK_OFFBOARD_SPACING_MM
                else:
                    item_cy = board_scan_min_y + cursor + item_h / 2.0
                    item_cx = (
                        board_scan_min_x - SILK_OFFBOARD_GAP_MM - item_w / 2.0
                        if edge == "left"
                        else board_scan_max_x + SILK_OFFBOARD_GAP_MM + item_w / 2.0
                    )
                    cursor += item_h + SILK_OFFBOARD_SPACING_MM

                if kind == "individual":
                    gap_mm = _label_gap_distance(comp, item_cx, item_cy, item_w, item_h)
                    label = {
                        "component_id": comp.get("id"),
                        "component_name": comp.get("name"),
                        "text": comp["ref"],
                        "x": item_cx,
                        "y": item_cy,
                        "width_mm": width_mm,
                        "height_mm": SILK_TEXT_HEIGHT_MM,
                        "collision_width_mm": item_w,
                        "collision_height_mm": item_h,
                        "rotation_deg": rotation_deg,
                        "font_size": f"{SILK_TEXT_HEIGHT_MM * 2.8346:.2f}pt",
                        "side": f_side,
                        "gap_mm": gap_mm,
                        "arrow_cells": set(),
                        "placement_mode": "off_board",
                    }
                    placed.append(label)
                    off_board_refs.append(comp["ref"])
                else:
                    block_min_x = item_cx - block_w / 2.0
                    block_min_y = item_cy - block_h / 2.0
                    new_labels = []
                    for m, w_mm, h_mm, rotation_deg2, cw, ch, local_x, local_y in specs:
                        x = block_min_x + local_x
                        y = block_min_y + local_y
                        new_labels.append(
                            {
                                "component_id": m.get("id"),
                                "component_name": m.get("name"),
                                "text": m["ref"],
                                "x": x,
                                "y": y,
                                "width_mm": w_mm,
                                "height_mm": SILK_TEXT_HEIGHT_MM,
                                "collision_width_mm": cw,
                                "collision_height_mm": ch,
                                "rotation_deg": rotation_deg2,
                                "font_size": f"{SILK_TEXT_HEIGHT_MM * 2.8346:.2f}pt",
                                "side": f_side,
                                "gap_mm": _label_gap_distance(m, x, y, cw, ch),
                                "arrow_cells": set(),
                                "placement_mode": "off_board_cluster",
                            }
                        )
                    placed.extend(new_labels)
                    off_board_refs.extend(m["ref"] for m in members)
    if off_board_refs:
        print(f"Off-board fallback: {len(off_board_refs)} label(s) placed outside the board -- still need manual placement")

    # Final upright pass: the pipeline-wide preference is 0 degrees unless
    # it does not fit, and this is where that is guaranteed. Every on-board
    # label still at 90 degrees is turned to 0 at the very same centre when
    # the 0-degree box fits there under the rules it was placed with: inside
    # the board, clear of its obstacle tree (traces included unless it was
    # a trace-relaxed placement), of other footprints, of every other label
    # (its own block-mates checked one by one) and of every connector but
    # its own -- and if it had reserved a buffer, the buffer stays reserved
    # around the new box. Escape labels get their connector re-routed to
    # the new box. Left alone: off-board labels (packed along the edge by
    # their dimensions), a family whose members deliberately chose vertical,
    # and normal far placements with a pointer (arrow validity is the main
    # loop's concern, not this pass's).
    mates_by_index = defaultdict(list)
    for label in placed:
        if label.get("_global_index") is not None:
            mates_by_index[label["_global_index"]].append(label)
    upright_count = 0
    for label in placed:
        mode = label.get("placement_mode") or ""
        if mode.startswith("off_board") or abs(_safe_float(label.get("rotation_deg"))) < 1e-6:
            continue
        comp = comp_by_id.get(label["component_id"])
        if comp is None:
            continue
        is_escape = mode.startswith("escape")
        has_arrow = label.get("arrow_start_x") is not None
        if comp.get("_family"):
            continue  # a family's orientation is decided on space alone; no 0-degree pull
        if has_arrow and not is_escape:
            continue
        side = label["side"] if label.get("side") in labels_by_side else "F.Cu"
        w_mm, h_mm = _estimate_text_box(comp["ref"], SILK_TEXT_HEIGHT_MM)
        specs = _label_orientation_specs(comp, w_mm, h_mm)
        _rot0, cw0, ch0, _v = next((s for s in specs if abs(s[0]) < 1e-9), specs[-1])
        x, y = label["x"], label["y"]
        old_padded = label.get("_padded_geometry")
        own_index = label.get("_global_index")
        block_mates = [m for m in mates_by_index.get(own_index, []) if m is not label] if own_index is not None else []
        if old_padded is not None and not block_mates:
            reserve = max(SILK_PADDING_MM, (old_padded.bounds[2] - old_padded.bounds[0] - label["collision_width_mm"]) / 2.0)
        else:
            reserve = SILK_PADDING_MM
        core = _rotated_box(x, y, cw0, ch0)
        if not inside_board(core):
            continue
        obstacle_tree = relaxed_tree if "trace_relaxed" in mode else strict_tree
        if _tree_intersects(obstacle_tree, core):
            continue
        # Every footprint, the label's own component included -- the same
        # test every placement path uses. (An earlier version excluded the
        # own footprint here, which let a label turned to 0 degrees widen
        # onto the very component it names.)
        if _tree_intersects(footprint_trees.get(side), core):
            continue
        reserved = _rotated_box(x, y, cw0 + 2.0 * reserve, ch0 + 2.0 * reserve)
        if any(index != own_index for index in global_labels.hits(reserved)):
            continue
        if any(
            reserved.intersects(_rotated_box(
                mate["x"], mate["y"],
                mate["collision_width_mm"] + 2.0 * SILK_PADDING_MM, mate["collision_height_mm"] + 2.0 * SILK_PADDING_MM,
            ))
            for mate in block_mates
        ):
            continue
        own_arrows = set(label.get("_arrow_indices") or ())
        if label.get("_arrow_index") is not None:
            own_arrows.add(label["_arrow_index"])
        if any(index not in own_arrows for index in arrows_by_side[side].hits(reserved)):
            continue

        # Commit the turn.
        label["rotation_deg"] = 0.0
        label["collision_width_mm"], label["collision_height_mm"] = cw0, ch0
        label["gap_mm"] = _label_gap_distance(comp, x, y, cw0, ch0)
        if own_index is not None and not block_mates:
            global_labels.remove(own_index)
            if label.get("_side_index") is not None:
                labels_by_side[side].remove(label["_side_index"])
        label["_padded_geometry"] = reserved
        label["_global_index"] = global_labels.add(reserved)
        label["_side_index"] = labels_by_side[side].add(reserved)
        if is_escape and has_arrow:
            for index in own_arrows:
                arrows_by_side[side].remove(index)
            for key in ("arrow_start_x", "arrow_start_y", "arrow_end_x", "arrow_end_y", "extra_arrow_segments", "_arrow_index", "_arrow_indices"):
                label.pop(key, None)
            if label.get("_cluster_bbox") is not None:
                cl_min_x, cl_min_y, cl_max_x, cl_max_y = label["_cluster_bbox"]
                full_arrow = _rect_to_rect_line(
                    x, y, cw0 / 2.0, ch0 / 2.0,
                    (cl_min_x + cl_max_x) / 2.0, (cl_min_y + cl_max_y) / 2.0,
                    (cl_max_x - cl_min_x) / 2.0, (cl_max_y - cl_min_y) / 2.0,
                )
                segments = clip_arrow_against_obstacles(
                    full_arrow, side, exclude_component_ids=label.get("_cluster_member_ids") or (),
                ) if full_arrow is not None else []
            else:
                segments = clipped_escape_arrow(comp, x, y, cw0, ch0, label["gap_mm"], side)
            if segments:
                label["arrow_start_x"], label["arrow_start_y"], label["arrow_end_x"], label["arrow_end_y"] = segments[0]
                if len(segments) > 1:
                    label["extra_arrow_segments"] = segments[1:]
                label["_arrow_indices"] = [
                    arrows_by_side[side].add(_segment_geometry(*seg, SILK_ARROW_WIDTH_MM)) for seg in segments
                ]
        upright_count += 1
    if upright_count:
        print(f"Upright pass: {upright_count} label(s) turned from 90 to 0 degrees where the space allowed")

    # Bookkeeping cleanup belongs here, at the true end of the pipeline, not
    # at the end of whichever post-placement stage happens to be last --
    # two stage-ordering crashes in a row came from that assumption being
    # wrong once a new stage got appended after the previous "last" one.
    for label in placed:
        for key in (
            "_global_index", "_side_index", "_arrow_index", "_arrow_indices", "_padded_geometry",
            "_cluster_bbox", "_cluster_member_ids",
        ):
            label.pop(key, None)

    return placed, {
        "placed": len(placed),
        "total": len(candidates),
        "failed": [],
        "trace_relaxed": sum(
            1
            for label in placed
            if label.get("placement_mode") == "trace_relaxed"
        ),
        "escaped": escaped_count,
        "off_board": off_board_refs,
        "family_patterns": family_pattern_summary,
        "family_pattern_detail": family_pattern_detail,
    }


def _cluster_preferred_sides_grid(
    clusters, grid, inside_cells, obstacles, footprint_obstacles_by_side, cluster_preferred_orientation=None
):
    preferred = {}
    side_order = ("top", "bottom", "right", "left")
    for members in clusters:
        cluster_orientation = (cluster_preferred_orientation or {}).get(members[0]["id"])
        votes = {side: 0 for side in side_order}
        for comp, collision_width, collision_height, positions in _cluster_preferred_side_votes(
            members, cluster_orientation
        ):
            side = comp.get("side")
            if side not in footprint_obstacles_by_side:
                side = "F.Cu"
            for side_name, (x, y) in positions.items():
                cells = _collect_rect_cells(grid, x, y, collision_width, collision_height, 0.0)
                if not cells:
                    continue
                if inside_cells is not None and not cells.issubset(inside_cells):
                    continue
                if cells & obstacles:
                    continue
                if cells & footprint_obstacles_by_side.get(side, set()):
                    continue
                votes[side_name] += 1
        best_side = max(side_order, key=lambda side: votes[side])
        if votes[best_side] > 0:
            for comp in members:
                preferred[comp["id"]] = best_side
    return preferred


def _cluster_preferred_orientation_grid(clusters, grid, inside_cells, obstacles, footprint_obstacles_by_side):
    """Grid-backend counterpart of _cluster_preferred_orientation_geometry."""
    preferred = {}
    for members in clusters:
        votes = {"horizontal": 0, "vertical": 0}
        for comp in members:
            width_mm, height_mm = _estimate_text_box(comp["ref"], SILK_TEXT_HEIGHT_MM)
            ordered = _orientation_space_score_grid(
                comp, width_mm, height_mm, grid, inside_cells, obstacles, footprint_obstacles_by_side
            )
            _rotation_deg, _cw, _ch, is_vertical = ordered[0]
            votes["vertical" if is_vertical else "horizontal"] += 1
        best = "vertical" if votes["vertical"] > votes["horizontal"] else "horizontal"
        if votes[best] > 0:
            for comp in members:
                preferred[comp["id"]] = best
    return preferred


def auto_place_silkscreen(
    components,
    pads,
    trace_segments,
    vias,
    edge_segments,
    allow_trace_overlap_fallback=True,
):
    _CANDIDATE_CACHE.clear()
    candidates = []
    for comp in components or []:
        ref = (comp.get("ref") or "").strip()
        if not ref or "***" in ref.upper():
            continue
        candidates.append(dict(comp))
    total = len(candidates)
    if not candidates:
        return [], {"placed": 0, "total": 0, "failed": []}

    grid = _compute_grid_spec(
        pads,
        trace_segments,
        vias,
        edge_segments,
        candidates,
        GRID_MM,
        margin=SILK_TEXT_HEIGHT_MM + SILK_FAR_MAX_DISTANCE_MM + SILK_PADDING_MM,
    )
    if grid is None:
        return [], {"placed": 0, "total": total, "failed": [c["ref"] for c in candidates]}

    estimated_grid_cells = grid.cols * grid.rows
    if estimated_grid_cells > SILK_MAX_GRID_CELLS:
        candidates.sort(
            key=lambda comp: (comp["max_x"] - comp["min_x"])
            * (comp["max_y"] - comp["min_y"]),
            reverse=True,
        )
        print(
            "Silkscreen board requires "
            f"{estimated_grid_cells:,} cells at {GRID_MM:g} mm; using the "
            f"memory-bounded geometry backend with "
            f"{min(SILK_GEOMETRY_WORKERS, len(candidates))} shared-memory "
            "worker threads."
        )
        return _auto_place_silkscreen_geometry(
            candidates,
            components,
            pads,
            trace_segments,
            vias,
            edge_segments,
            allow_trace_overlap_fallback,
        )

    inside_cells = _compute_board_inside_cells(grid, edge_segments, 0.0)
    strict_obstacles = _build_silkscreen_obstacles(
        pads,
        trace_segments,
        vias,
        grid,
        SILK_CLEARANCE_MM,
        include_traces=True,
    )
    relaxed_obstacles = strict_obstacles
    if allow_trace_overlap_fallback:
        relaxed_obstacles = _build_silkscreen_obstacles(
            pads,
            trace_segments,
            vias,
            grid,
            SILK_CLEARANCE_MM,
            include_traces=False,
        )
    pad_obstacles_by_side = _build_pad_obstacles_by_side(pads, grid, SILK_CLEARANCE_MM)
    (
        footprint_obstacles_by_side,
        footprint_obstacles_by_component,
    ) = _build_footprint_obstacles_by_side(
        components,
        grid,
        SILK_FOOTPRINT_CLEARANCE_MM,
    )
    family_clusters = _build_family_clusters(candidates)
    # Members of a real family (two or more parts) are marked: for them 0
    # and 90 degrees are equally welcome and the orientation with more room
    # wins (see _orientation_space_score_*); everything else keeps 0 first.
    for members in family_clusters:
        if len(members) >= 2:
            for member in members:
                member["_family"] = True
    family_pattern_summary, family_pattern_detail = _classify_family_clusters(family_clusters)
    print(
        "Family patterns: {} clusters ({} rows, {} mirror pairs, {} twin pairs, "
        "{} unclassified)".format(
            len(family_clusters),
            family_pattern_summary["row"],
            family_pattern_summary["mirror_pair"],
            family_pattern_summary["twin_pair"],
            family_pattern_summary["unclassified"],
        )
    )
    cluster_preferred_orientation = _cluster_preferred_orientation_grid(
        family_clusters, grid, inside_cells, strict_obstacles, footprint_obstacles_by_side
    )
    cluster_preferred_side = _cluster_preferred_sides_grid(
        family_clusters, grid, inside_cells, strict_obstacles, footprint_obstacles_by_side, cluster_preferred_orientation
    )
    label_blocked = set()
    label_blocked_by_side = {"F.Cu": set(), "B.Cu": set()}
    arrow_blocked_by_side = {"F.Cu": set(), "B.Cu": set()}

    def label_fits(cells, side, obstacles):
        if inside_cells is not None and not cells.issubset(inside_cells):
            return False
        if cells & obstacles:
            return False
        if cells & footprint_obstacles_by_side.get(side, set()):
            return False
        if cells & label_blocked:
            return False
        if cells & arrow_blocked_by_side.get(side, set()):
            return False
        return True

    candidates.sort(
        key=lambda comp: (comp["max_x"] - comp["min_x"]) * (comp["max_y"] - comp["min_y"]),
        reverse=True,
    )
    placed = []
    failures = []

    def try_place(
        bounds,
        positions,
        width_mm,
        height_mm,
        collision_width,
        collision_height,
        rotation_deg,
        obstacles,
        placement_mode,
    ):
        for x, y in positions:
            cells = _collect_rect_cells(
                grid, x, y, collision_width, collision_height, 0.0
            )
            if not cells:
                continue
            side = bounds["side"] if bounds["side"] in label_blocked_by_side else "F.Cu"
            if not label_fits(cells, side, obstacles):
                continue
            gap_mm = _label_gap_distance(
                bounds, x, y, collision_width, collision_height
            )
            arrow = None
            arrow_cells = set()
            if _label_needs_pointer(
                bounds,
                x,
                y,
                collision_width,
                collision_height,
                gap_mm,
                components=candidates,
                exclude_id=bounds.get("id"),
            ):
                arrow = _arrow_points(
                    bounds, x, y, collision_width, collision_height
                )
                if arrow is None:
                    continue
                arrow_cells = _arrow_cells(grid, arrow)
                if not arrow_cells:
                    continue
                if inside_cells is not None and not arrow_cells.issubset(inside_cells):
                    continue
                other_footprints = (
                    footprint_obstacles_by_side.get(side, set())
                    - footprint_obstacles_by_component.get(bounds.get("id"), set())
                )
                arrow_obstacles = (
                    pad_obstacles_by_side.get(side, set())
                    | label_blocked_by_side.get(side, set())
                    | other_footprints
                )
                if arrow_cells & arrow_obstacles:
                    continue
            label = {
                "component_id": bounds.get("id"),
                "component_name": bounds.get("name"),
                "text": bounds["ref"],
                "x": x,
                "y": y,
                "width_mm": width_mm,
                "height_mm": SILK_TEXT_HEIGHT_MM,
                "collision_width_mm": collision_width,
                "collision_height_mm": collision_height,
                "rotation_deg": rotation_deg,
                "font_size": f"{SILK_TEXT_HEIGHT_MM * 2.8346:.2f}pt",
                "side": side,
                "gap_mm": gap_mm,
                "arrow_cells": arrow_cells,
                "placement_mode": placement_mode,
            }
            if arrow:
                label.update(
                    {
                        "arrow_start_x": arrow[0],
                        "arrow_start_y": arrow[1],
                        "arrow_end_x": arrow[2],
                        "arrow_end_y": arrow[3],
                    }
                )
            padded_cells = _collect_rect_cells(
                grid,
                x,
                y,
                collision_width + 2.0 * SILK_PADDING_MM,
                collision_height + 2.0 * SILK_PADDING_MM,
                0.0,
            )
            return label, padded_cells or cells
        return None, None

    for comp in candidates:
        width_mm, height_mm = _estimate_text_box(comp["ref"], SILK_TEXT_HEIGHT_MM)
        preferred_orientation = cluster_preferred_orientation.get(comp.get("id"))
        orientation_specs = _orientation_space_score_grid(
            comp, width_mm, height_mm, grid, inside_cells, strict_obstacles, footprint_obstacles_by_side,
            preferred_orientation=preferred_orientation,
        )
        preferred_side = cluster_preferred_side.get(comp.get("id"))

        def try_orientations(obstacles, placement_mode):
            for rotation_deg, collision_width, collision_height, is_vertical in orientation_specs:
                near = _generate_label_candidates(
                    comp,
                    collision_width,
                    collision_height,
                    SILK_MAX_DISTANCE_MM,
                    prefer_vertical=is_vertical,
                    preferred_side=preferred_side,
                    dense=True,
                )
                label_result, cells_result = try_place(
                    comp,
                    near,
                    width_mm,
                    height_mm,
                    collision_width,
                    collision_height,
                    rotation_deg,
                    obstacles,
                    placement_mode,
                )
                if label_result is not None:
                    return label_result, cells_result
                if SILK_FAR_MAX_DISTANCE_MM > SILK_MAX_DISTANCE_MM:
                    near_seen = set(near)
                    far = [
                        pos
                        for pos in _generate_label_candidates(
                            comp,
                            collision_width,
                            collision_height,
                            SILK_FAR_MAX_DISTANCE_MM,
                            prefer_vertical=is_vertical,
                            preferred_side=preferred_side,
                        )
                        if pos not in near_seen
                    ]
                    label_result, cells_result = try_place(
                        comp,
                        far,
                        width_mm,
                        height_mm,
                        collision_width,
                        collision_height,
                        rotation_deg,
                        obstacles,
                        placement_mode,
                    )
                    if label_result is not None:
                        return label_result, cells_result
            return None, None

        label, blocked_cells = try_orientations(strict_obstacles, "strict")
        if label is None and allow_trace_overlap_fallback:
            label, blocked_cells = try_orientations(
                relaxed_obstacles, "trace_relaxed"
            )
        if label is None:
            failures.append(comp["ref"])
            continue
        placed.append(label)
        label_blocked.update(blocked_cells)
        side = label["side"] if label["side"] in label_blocked_by_side else "F.Cu"
        label_blocked_by_side[side].update(blocked_cells)
        arrow_blocked_by_side[side].update(label.get("arrow_cells") or set())

    # Backtracking rescue pass (post-placement, greedy, single-swap only):
    # for each failed component, check whether its best candidate spot is
    # blocked by exactly one already-placed label (not a static obstacle,
    # which can't move). If that one label can find an alternate valid spot
    # of its own, relocate it and place the failure in the freed spot --
    # net +1 placed. If not, leave both exactly as they were.
    comp_by_id = {c["id"]: c for c in candidates}
    still_failed = []
    rescued_count = 0
    for ref in failures:
        comp = next((c for c in candidates if c["ref"] == ref), None)
        if comp is None:
            still_failed.append(ref)
            continue
        width_mm, height_mm = _estimate_text_box(comp["ref"], SILK_TEXT_HEIGHT_MM)
        orientation_specs = _orientation_space_score_grid(
            comp, width_mm, height_mm, grid, inside_cells, strict_obstacles, footprint_obstacles_by_side,
            preferred_orientation=cluster_preferred_orientation.get(comp.get("id")),
        )
        preferred_side = cluster_preferred_side.get(comp.get("id"))
        rescued = False
        for rotation_deg, collision_width, collision_height, is_vertical in orientation_specs:
            near = _generate_label_candidates(
                comp, collision_width, collision_height, SILK_MAX_DISTANCE_MM,
                prefer_vertical=is_vertical, preferred_side=preferred_side, dense=True,
            )
            for x, y in near:
                cells = _collect_rect_cells(grid, x, y, collision_width, collision_height, 0.0)
                if not cells:
                    continue
                if inside_cells is not None and not cells.issubset(inside_cells):
                    continue
                if cells & strict_obstacles:
                    continue
                f_side = comp.get("side") if comp.get("side") in label_blocked_by_side else "F.Cu"
                if cells & footprint_obstacles_by_side.get(f_side, set()):
                    continue
                if cells & arrow_blocked_by_side.get(f_side, set()):
                    continue
                if not (cells & label_blocked):
                    continue  # would already have succeeded; nothing to rescue here
                blockers = []
                for other in placed:
                    other_cells = _collect_rect_cells(
                        grid, other["x"], other["y"],
                        other["collision_width_mm"] + 2.0 * SILK_PADDING_MM,
                        other["collision_height_mm"] + 2.0 * SILK_PADDING_MM, 0.0,
                    )
                    if other_cells & cells:
                        blockers.append((other, other_cells))
                        if len(blockers) > 1:
                            break
                if len(blockers) != 1:
                    continue
                blocker_label, blocker_cells = blockers[0]
                blocker_comp = comp_by_id.get(blocker_label["component_id"])
                if blocker_comp is None:
                    continue
                blocker_side = blocker_label["side"] if blocker_label["side"] in label_blocked_by_side else "F.Cu"
                blocker_mode = blocker_label.get("placement_mode") or "strict"
                blocker_obstacles = strict_obstacles if blocker_mode != "trace_relaxed" else relaxed_obstacles
                blocker_backup = dict(blocker_label)

                label_blocked.difference_update(blocker_cells)
                label_blocked_by_side[blocker_side].difference_update(blocker_cells)
                arrow_blocked_by_side[blocker_side].difference_update(blocker_label.get("arrow_cells") or set())
                label_blocked.update(cells)  # reserve F's spot so the blocker can't just re-claim it

                b_width_mm, b_height_mm = _estimate_text_box(blocker_comp["ref"], SILK_TEXT_HEIGHT_MM)
                b_orientation_specs = _orientation_space_score_grid(
                    blocker_comp, b_width_mm, b_height_mm, grid, inside_cells, strict_obstacles, footprint_obstacles_by_side,
                    preferred_orientation=cluster_preferred_orientation.get(blocker_comp.get("id")),
                )
                b_preferred_side = cluster_preferred_side.get(blocker_comp.get("id"))
                new_blocker_label, new_blocker_cells = None, None
                for b_rot, b_cw, b_ch, b_is_v in b_orientation_specs:
                    b_near = _generate_label_candidates(
                        blocker_comp, b_cw, b_ch, SILK_MAX_DISTANCE_MM,
                        prefer_vertical=b_is_v, preferred_side=b_preferred_side, dense=True,
                    )
                    new_blocker_label, new_blocker_cells = try_place(
                        blocker_comp, b_near, b_width_mm, b_height_mm, b_cw, b_ch, b_rot, blocker_obstacles, blocker_mode,
                    )
                    if new_blocker_label is not None:
                        break
                label_blocked.difference_update(cells)  # release the temporary reservation

                if new_blocker_label is None:
                    label_blocked.update(blocker_cells)
                    label_blocked_by_side[blocker_side].update(blocker_cells)
                    arrow_blocked_by_side[blocker_side].update(blocker_label.get("arrow_cells") or set())
                    continue

                f_label, f_cells = try_place(
                    comp, [(x, y)], width_mm, height_mm, collision_width, collision_height,
                    rotation_deg, strict_obstacles, "strict",
                )
                if f_label is None:
                    label_blocked.update(blocker_cells)
                    label_blocked_by_side[blocker_side].update(blocker_cells)
                    arrow_blocked_by_side[blocker_side].update(blocker_backup.get("arrow_cells") or set())
                    continue

                blocker_label.clear()
                blocker_label.update(new_blocker_label)
                label_blocked.update(new_blocker_cells)
                label_blocked_by_side[blocker_side].update(new_blocker_cells)
                arrow_blocked_by_side[blocker_side].update(blocker_label.get("arrow_cells") or set())

                placed.append(f_label)
                label_blocked.update(f_cells)
                label_blocked_by_side[f_side].update(f_cells)
                arrow_blocked_by_side[f_side].update(f_label.get("arrow_cells") or set())
                rescued = True
                rescued_count += 1
                break
            if rescued:
                break
        if not rescued:
            still_failed.append(ref)
    failures = still_failed
    if rescued_count:
        print(f"Backtracking rescue: {rescued_count} previously-failed component(s) placed")

    comp_by_id = {c["id"]: c for c in candidates}
    relocated_count = 0
    for label in placed:
        comp = comp_by_id.get(label["component_id"])
        if comp is None:
            continue
        side = label["side"] if label["side"] in label_blocked_by_side else "F.Cu"
        mode = label.get("placement_mode") or "strict"
        obstacles = strict_obstacles if mode != "trace_relaxed" else relaxed_obstacles
        buf_width_mm = label["collision_width_mm"]
        buf_height_mm = label["collision_height_mm"]

        own_cells = _collect_rect_cells(
            grid, label["x"], label["y"],
            buf_width_mm + 2.0 * SILK_PADDING_MM, buf_height_mm + 2.0 * SILK_PADDING_MM, 0.0,
        )
        own_arrow_cells = label.get("arrow_cells") or set()
        # Free this label's own contribution first so the buffer/relocation
        # checks below don't collide with themselves.
        label_blocked.difference_update(own_cells)
        label_blocked_by_side[side].difference_update(own_cells)
        arrow_blocked_by_side[side].difference_update(own_arrow_cells)

        blocked_sets = (
            obstacles,
            footprint_obstacles_by_side.get(side, set()),
            label_blocked,
            arrow_blocked_by_side.get(side, set()),
        )
        ok, _ratio = _label_meets_buffer_grid(
            grid, label["x"], label["y"], buf_width_mm, buf_height_mm, inside_cells, blocked_sets
        )
        relocated = False
        if not ok:
            current_side = _infer_label_side_from_component(label["x"], label["y"], comp)
            for alt_side in _SIDE_ALTERNATES[current_side]:
                near = _generate_label_candidates(
                    comp, buf_width_mm, buf_height_mm, SILK_MAX_DISTANCE_MM,
                    preferred_side=alt_side, dense=True,
                )
                new_label, new_cells = try_place(
                    comp, near, label["width_mm"], label["height_mm"],
                    buf_width_mm, buf_height_mm, label["rotation_deg"], obstacles, mode,
                )
                if new_label is None:
                    continue
                buffer_ok, _ratio2 = _label_meets_buffer_grid(
                    grid, new_label["x"], new_label["y"], buf_width_mm, buf_height_mm, inside_cells, blocked_sets
                )
                if not buffer_ok:
                    continue
                label.clear()
                label.update(new_label)
                label_blocked.update(new_cells)
                label_blocked_by_side[side].update(new_cells)
                arrow_blocked_by_side[side].update(label.get("arrow_cells") or set())
                relocated = True
                relocated_count += 1
                break
        if not relocated:
            label_blocked.update(own_cells)
            label_blocked_by_side[side].update(own_cells)
            arrow_blocked_by_side[side].update(own_arrow_cells)
    if relocated_count:
        print(f"Buffer reinforcement: {relocated_count} label(s) relocated for better clearance")

    # Final polish: a label that's slightly off-axis (dense-slide drift) gets
    # one shot at snapping to the exact on-axis point on its own side, same
    # distance -- never switches sides, never forced, never loses a
    # placement. Skips labels with a pointer arrow (recomputing arrow
    # validity is a separate concern, not handled here).
    snapped_count = 0
    for label in placed:
        if label.get("arrow_start_x") is not None:
            continue
        comp = comp_by_id.get(label["component_id"])
        if comp is None:
            continue
        x, y = label["x"], label["y"]
        if _angle_deviation_from_axes(comp["cx"], comp["cy"], x, y) <= 1e-6:
            continue
        current_side = _infer_label_side_from_component(x, y, comp)
        snap_x, snap_y = (comp["cx"], y) if current_side in ("top", "bottom") else (x, comp["cy"])
        width_mm = label["collision_width_mm"]
        height_mm = label["collision_height_mm"]
        side = label["side"] if label["side"] in label_blocked_by_side else "F.Cu"
        mode = label.get("placement_mode") or "strict"
        obstacles = strict_obstacles if mode != "trace_relaxed" else relaxed_obstacles

        own_cells = _collect_rect_cells(
            grid, x, y, width_mm + 2.0 * SILK_PADDING_MM, height_mm + 2.0 * SILK_PADDING_MM, 0.0,
        )
        label_blocked.difference_update(own_cells)
        label_blocked_by_side[side].difference_update(own_cells)

        new_label, new_cells = try_place(
            comp, [(snap_x, snap_y)], label["width_mm"], label["height_mm"],
            width_mm, height_mm, label["rotation_deg"], obstacles, mode,
        )
        if new_label is not None and new_label.get("arrow_start_x") is None:
            label.clear()
            label.update(new_label)
            label_blocked.update(new_cells)
            label_blocked_by_side[side].update(new_cells)
            snapped_count += 1
        else:
            label_blocked.update(own_cells)
            label_blocked_by_side[side].update(own_cells)
    if snapped_count:
        print(f"Axis snap: {snapped_count} label(s) tightened to exact on-axis position")

    # Escape rescue (final pass): components still in `failures` here have
    # no legal spot anywhere within the normal near/far search radius --
    # typically a densely packed area with no room left nearby. Rather than
    # lose them, they're first grouped by pure physical proximity (a
    # flood-fill among the failures themselves, not the family/shape
    # clustering used earlier in the pipeline). A group of 2+ nearby
    # failures becomes one combined label block: each member keeps its own
    # natural orientation (its own footprint's longer side -- the same rule
    # used everywhere else, but decided per component, not once for the
    # whole cluster, so each ref's own text orientation still tells you
    # which component it is) and its real relative left-to-right/top-to-
    # bottom order, packed edge-to-edge with the real-world gaps removed.
    # That whole block is then treated as a single rigid shape and placed
    # at the closest legal spot anywhere on the board, with one connector
    # line back to the cluster's own location (reusing the same pointer-
    # line idea used elsewhere, just anchored to the cluster's bounding box
    # instead of a single component's). A lone failure with nothing else
    # nearby falls back to the simpler single-component whole-board search
    # instead. Nothing here is ever forced -- whatever finds no room stays
    # a failure, exactly as before.
    def scan_whole_board_rect(anchor_cx, anchor_cy, side, block_width, block_height, obstacles, top_k=8):
        def legal(x, y):
            cells = _collect_rect_cells(grid, x, y, block_width, block_height, 0.0)
            return cells if cells and label_fits(cells, side, obstacles) else None

        # Coarse pass: one lattice point every SILK_ESCAPE_SCAN_STEP_MM
        # across the whole board -- cheap enough to cover everything, but
        # too coarse on its own to find the tightest fit.
        coarse_hits = []
        y = grid.min_y + block_height / 2.0
        y_max = grid.min_y + grid.rows * grid.cell - block_height / 2.0
        while y <= y_max:
            x = grid.min_x + block_width / 2.0
            x_max = grid.min_x + grid.cols * grid.cell - block_width / 2.0
            while x <= x_max:
                if legal(x, y):
                    coarse_hits.append((x, y))
                x += SILK_ESCAPE_SCAN_STEP_MM
            y += SILK_ESCAPE_SCAN_STEP_MM
        if not coarse_hits:
            return []
        coarse_hits.sort(key=lambda pos: (pos[0] - anchor_cx) ** 2 + (pos[1] - anchor_cy) ** 2)

        # Refine pass: around each of the closest few coarse hits, slide at
        # the same fine step the rest of the algorithm already uses for
        # dense candidate search, to close the gap between lattice
        # resolution and the true nearest legal point in that same pocket.
        refined = []
        seen = set()
        for anchor_x, anchor_y in coarse_hits[:top_k]:
            dx = -SILK_ESCAPE_SCAN_STEP_MM
            while dx <= SILK_ESCAPE_SCAN_STEP_MM + 1e-9:
                dy = -SILK_ESCAPE_SCAN_STEP_MM
                while dy <= SILK_ESCAPE_SCAN_STEP_MM + 1e-9:
                    x, y = round(anchor_x + dx, 6), round(anchor_y + dy, 6)
                    if (x, y) not in seen and legal(x, y):
                        seen.add((x, y))
                        refined.append((x, y))
                    dy += SILK_SLIDE_STEP_MM
                dx += SILK_SLIDE_STEP_MM
        pool = list(dict.fromkeys(refined or coarse_hits))
        pool.sort(key=lambda pos: (pos[0] - anchor_cx) ** 2 + (pos[1] - anchor_cy) ** 2)

        # Prefer genuinely spacious spots over just the nearest legal one,
        # so escape-rescued labels land somewhere their connector line
        # reads clearly, not wedged into the first gap that happened to
        # fit. Never a hard requirement -- if nothing in the pool is roomy
        # enough, the plain nearest-legal order (already computed above)
        # is used as-is, so a placement is never lost chasing spaciousness
        # that doesn't exist nearby.
        blocked_sets = (
            obstacles,
            footprint_obstacles_by_side.get(side, set()),
            label_blocked,
            arrow_blocked_by_side.get(side, set()),
        )
        spacious = [
            pos for pos in pool
            if _label_meets_buffer_grid(
                grid, pos[0], pos[1], block_width, block_height, inside_cells, blocked_sets,
                min_ratio=SILK_ESCAPE_BUFFER_RATIO,
            )[0]
        ]
        if spacious:
            spacious_set = set(spacious)
            pool = spacious + [pos for pos in pool if pos not in spacious_set]
        return pool

    def scan_whole_board_component(comp, collision_width, collision_height, obstacles, top_k=8):
        side = comp["side"] if comp["side"] in label_blocked_by_side else "F.Cu"
        return scan_whole_board_rect(comp["cx"], comp["cy"], side, collision_width, collision_height, obstacles, top_k)

    def clip_arrow_against_obstacles_grid(full_arrow, side, exclude_component_id=None):
        """Grid-backend counterpart of the geometry backend's
        clip_arrow_against_obstacles: cuts the parts of a candidate line
        that fall on a blocked cell (pad/other footprint/other label),
        keeping whatever's left as one or more visible pieces, instead of
        discarding the whole line the moment it touches anything -- the
        old behavior a long connector on a populated board would trigger
        almost every time. Samples the line every SILK_SLIDE_STEP_MM and
        merges consecutive clear samples into segments."""
        if full_arrow is None:
            return []
        other_footprints = footprint_obstacles_by_side.get(side, set())
        if exclude_component_id is not None:
            other_footprints = other_footprints - footprint_obstacles_by_component.get(exclude_component_id, set())
        arrow_obstacles = (
            pad_obstacles_by_side.get(side, set())
            | label_blocked_by_side.get(side, set())
            | other_footprints
        )

        def is_blocked(px, py):
            cells = _collect_rect_cells(grid, px, py, SILK_ARROW_WIDTH_MM, SILK_ARROW_WIDTH_MM, 0.0)
            if not cells:
                return True
            if inside_cells is not None and not cells.issubset(inside_cells):
                return True
            return bool(cells & arrow_obstacles)

        x0, y0, x1, y1 = full_arrow
        length = math.hypot(x1 - x0, y1 - y0)
        if length <= 1e-9:
            return []
        steps = max(1, int(math.ceil(length / SILK_SLIDE_STEP_MM)))
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        segments = []
        seg_start = None
        prev_point = (x0, y0)
        for i in range(steps + 1):
            t = min(length, i * SILK_SLIDE_STEP_MM)
            point = (x0 + ux * t, y0 + uy * t)
            blocked = is_blocked(*point)
            if not blocked and seg_start is None:
                seg_start = point
            elif blocked and seg_start is not None:
                if math.hypot(prev_point[0] - seg_start[0], prev_point[1] - seg_start[1]) >= GRID_MM:
                    segments.append((seg_start[0], seg_start[1], prev_point[0], prev_point[1]))
                seg_start = None
            prev_point = point
        if seg_start is not None and math.hypot(prev_point[0] - seg_start[0], prev_point[1] - seg_start[1]) >= GRID_MM:
            segments.append((seg_start[0], seg_start[1], prev_point[0], prev_point[1]))
        return segments

    comp_by_ref = {c["ref"]: c for c in candidates}
    failed_comps = [comp_by_ref[ref] for ref in failures if ref in comp_by_ref]

    parent = {c["id"]: c["id"] for c in failed_comps}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    for i in range(len(failed_comps)):
        for j in range(i + 1, len(failed_comps)):
            a, b = failed_comps[i], failed_comps[j]
            if a["side"] == b["side"] and _bbox_gap_distance(a, b) <= SILK_DENSE_CLUSTER_GAP_MM:
                union(a["id"], b["id"])

    cluster_groups = defaultdict(list)
    for c in failed_comps:
        cluster_groups[find(c["id"])].append(c)
    dense_groups = [members for members in cluster_groups.values() if len(members) >= 2]
    clustered_refs = {m["ref"] for members in dense_groups for m in members}

    escaped_count = 0
    still_failed = []
    failed_cluster_blocks = []  # (members, specs, block_w, block_h, side)
    # for clusters whose on-board scan found nowhere to land -- handled by
    # the guaranteed off-board fallback below as one combined block.

    # One combined block per dense cluster.
    for members in dense_groups:
        specs, block_w, block_h = _pack_cluster_grid(members)

        side = members[0]["side"] if members[0]["side"] in label_blocked_by_side else "F.Cu"
        cluster_min_x = min(m["min_x"] for m in members)
        cluster_min_y = min(m["min_y"] for m in members)
        cluster_max_x = max(m["max_x"] for m in members)
        cluster_max_y = max(m["max_y"] for m in members)
        cluster_cx = (cluster_min_x + cluster_max_x) / 2.0
        cluster_cy = (cluster_min_y + cluster_max_y) / 2.0

        positions = scan_whole_board_rect(cluster_cx, cluster_cy, side, block_w, block_h, strict_obstacles)
        mode = "escape_cluster"
        if not positions and allow_trace_overlap_fallback:
            positions = scan_whole_board_rect(cluster_cx, cluster_cy, side, block_w, block_h, relaxed_obstacles)
            mode = "escape_cluster_trace_relaxed"
        if not positions:
            failed_cluster_blocks.append((members, specs, block_w, block_h, side))
            continue

        block_cx, block_cy = positions[0]
        block_min_x = block_cx - block_w / 2.0
        block_min_y = block_cy - block_h / 2.0

        new_labels = []
        for m, w_mm, h_mm, rotation_deg, cw, ch, local_x, local_y in specs:
            x = block_min_x + local_x
            y = block_min_y + local_y
            new_labels.append(
                {
                    "component_id": m.get("id"),
                    "component_name": m.get("name"),
                    "text": m["ref"],
                    "x": x,
                    "y": y,
                    "width_mm": w_mm,
                    "height_mm": SILK_TEXT_HEIGHT_MM,
                    "collision_width_mm": cw,
                    "collision_height_mm": ch,
                    "rotation_deg": rotation_deg,
                    "font_size": f"{SILK_TEXT_HEIGHT_MM * 2.8346:.2f}pt",
                    "side": side,
                    "gap_mm": _label_gap_distance(m, x, y, cw, ch),
                    "arrow_cells": set(),
                    "placement_mode": mode,
                }
            )

        # One shared connector for the whole cluster -- from whichever
        # label ends up closest to the cluster's own location, clipped
        # around any obstacle it crosses -- not one line per member, since
        # they're all part of the same crowded area.
        cluster_half_w = (cluster_max_x - cluster_min_x) / 2.0
        cluster_half_h = (cluster_max_y - cluster_min_y) / 2.0
        best_arrow = None
        best_length = None
        best_label = None
        for lbl in new_labels:
            candidate = _rect_to_rect_line(
                lbl["x"], lbl["y"], lbl["collision_width_mm"] / 2.0, lbl["collision_height_mm"] / 2.0,
                cluster_cx, cluster_cy, cluster_half_w, cluster_half_h,
            )
            if candidate is None:
                continue
            length = math.hypot(candidate[2] - candidate[0], candidate[3] - candidate[1])
            if best_length is None or length < best_length:
                best_arrow, best_length, best_label = candidate, length, lbl

        combined_arrow_cells = set()
        if best_arrow is not None:
            segments = clip_arrow_against_obstacles_grid(best_arrow, side)
            if segments:
                best_label["arrow_start_x"], best_label["arrow_start_y"], best_label["arrow_end_x"], best_label["arrow_end_y"] = segments[0]
                if len(segments) > 1:
                    best_label["extra_arrow_segments"] = segments[1:]
                for seg in segments:
                    combined_arrow_cells |= _arrow_cells(grid, seg)
                best_label["arrow_cells"] = combined_arrow_cells

        block_cells = _collect_rect_cells(grid, block_cx, block_cy, block_w, block_h, 0.0)
        placed.extend(new_labels)
        label_blocked.update(block_cells)
        label_blocked_by_side[side].update(block_cells)
        arrow_blocked_by_side[side].update(combined_arrow_cells)
        escaped_count += len(new_labels)

    # Lone failures (no other failure nearby) keep the simpler
    # single-component whole-board search from before.
    for ref in failures:
        if ref in clustered_refs:
            continue
        comp = comp_by_ref.get(ref)
        if comp is None:
            still_failed.append(ref)
            continue
        width_mm, height_mm = _estimate_text_box(comp["ref"], SILK_TEXT_HEIGHT_MM)
        orientation_specs = _orientation_space_score_grid(
            comp, width_mm, height_mm, grid, inside_cells, strict_obstacles, footprint_obstacles_by_side,
            preferred_orientation=cluster_preferred_orientation.get(comp.get("id")),
        )
        rescued = False
        for rotation_deg, collision_width, collision_height, _is_vertical in orientation_specs:
            board_wide = scan_whole_board_component(comp, collision_width, collision_height, strict_obstacles)
            label, cells = try_place(
                comp, board_wide, width_mm, height_mm, collision_width, collision_height,
                rotation_deg, strict_obstacles, "escape",
            )
            if label is None and allow_trace_overlap_fallback:
                board_wide_relaxed = scan_whole_board_component(comp, collision_width, collision_height, relaxed_obstacles)
                label, cells = try_place(
                    comp, board_wide_relaxed, width_mm, height_mm, collision_width, collision_height,
                    rotation_deg, relaxed_obstacles, "escape_trace_relaxed",
                )
            if label is not None:
                placed.append(label)
                label_blocked.update(cells)
                side = label["side"] if label["side"] in label_blocked_by_side else "F.Cu"
                label_blocked_by_side[side].update(cells)
                arrow_blocked_by_side[side].update(label.get("arrow_cells") or set())
                rescued = True
                escaped_count += 1
                break
        if not rescued:
            still_failed.append(ref)

    if escaped_count:
        print(f"Escape rescue: {escaped_count} previously-failed component(s) placed via combined blocks or long-range rescue")

    # Guaranteed off-board fallback: anything still in still_failed (a lone
    # component) or failed_cluster_blocks (a whole cluster that found no
    # on-board spot) gets placed just outside the board instead, on
    # whichever edge is nearest to it, ordered along that edge to match
    # its real relative position. Unconditional and never fails -- off-
    # board space is unbounded -- so `failed` is empty after this point;
    # off_board_refs tracks what actually needed it, since these still
    # need a human to move them onto the real board. clip_arrow_against_
    # obstacles_grid already treats off-board points as blocked at each
    # sampled step (outside inside_cells), so the returned visible
    # segments are naturally just the on-board portion of the line --
    # no special-casing needed here the way the geometry backend needed
    # one to bypass its whole-line inside-board veto.
    board_min_x = min(
        [pad["bbox"][0] for pad in pads]
        + [comp["min_x"] for comp in candidates]
        + [x0 for x0, _y0, x1, _y1 in edge_segments or [] for x0 in (x0, x1)]
        or [grid.min_x]
    )
    board_max_x = max(
        [pad["bbox"][2] for pad in pads]
        + [comp["max_x"] for comp in candidates]
        + [x1 for x0, _y0, x1, _y1 in edge_segments or [] for x1 in (x0, x1)]
        or [grid.min_x + grid.cols * grid.cell]
    )
    board_min_y = min(
        [pad["bbox"][1] for pad in pads]
        + [comp["min_y"] for comp in candidates]
        + [y0 for _x0, y0, _x1, y1 in edge_segments or [] for y0 in (y0, y1)]
        or [grid.min_y]
    )
    board_max_y = max(
        [pad["bbox"][3] for pad in pads]
        + [comp["max_y"] for comp in candidates]
        + [y1 for _x0, y0, _x1, y1 in edge_segments or [] for y1 in (y0, y1)]
        or [grid.min_y + grid.rows * grid.cell]
    )

    off_board_refs = []
    if still_failed or failed_cluster_blocks:
        def edge_side_for(cx, cy):
            d_top = cy - board_min_y
            d_bottom = board_max_y - cy
            d_left = cx - board_min_x
            d_right = board_max_x - cx
            closest = min(d_top, d_bottom, d_left, d_right)
            if closest == d_top:
                return "top"
            if closest == d_bottom:
                return "bottom"
            if closest == d_left:
                return "left"
            return "right"

        off_board_items = []
        for ref in still_failed:
            comp = comp_by_ref.get(ref)
            if comp is None:
                continue
            edge = edge_side_for(comp["cx"], comp["cy"])
            sort_key = comp["cx"] if edge in ("top", "bottom") else comp["cy"]
            off_board_items.append((edge, sort_key, "individual", comp))
        for members, specs, block_w, block_h, side in failed_cluster_blocks:
            cl_min_x = min(m["min_x"] for m in members)
            cl_min_y = min(m["min_y"] for m in members)
            cl_max_x = max(m["max_x"] for m in members)
            cl_max_y = max(m["max_y"] for m in members)
            cl_cx = (cl_min_x + cl_max_x) / 2.0
            cl_cy = (cl_min_y + cl_max_y) / 2.0
            edge = edge_side_for(cl_cx, cl_cy)
            sort_key = cl_cx if edge in ("top", "bottom") else cl_cy
            payload = (members, specs, block_w, block_h, side, cl_cx, cl_cy, cl_min_x, cl_min_y, cl_max_x, cl_max_y)
            off_board_items.append((edge, sort_key, "cluster", payload))

        for edge in ("top", "bottom", "left", "right"):
            group = sorted((item for item in off_board_items if item[0] == edge), key=lambda item: item[1])
            cursor = 0.0
            for _edge, _sort_key, kind, payload in group:
                if kind == "individual":
                    comp = payload
                    width_mm, height_mm = _estimate_text_box(comp["ref"], SILK_TEXT_HEIGHT_MM)
                    orientation_specs = _orientation_space_score_grid(
                        comp, width_mm, height_mm, grid, inside_cells, strict_obstacles, footprint_obstacles_by_side,
                        preferred_orientation=cluster_preferred_orientation.get(comp.get("id")),
                    )
                    rotation_deg, item_w, item_h, _is_v = orientation_specs[0]
                    f_side = comp["side"] if comp["side"] in label_blocked_by_side else "F.Cu"
                else:
                    members, specs, block_w, block_h, f_side, cl_cx, cl_cy, cl_min_x, cl_min_y, cl_max_x, cl_max_y = payload
                    item_w, item_h = block_w, block_h

                if edge in ("top", "bottom"):
                    item_cx = board_min_x + cursor + item_w / 2.0
                    item_cy = (
                        board_min_y - SILK_OFFBOARD_GAP_MM - item_h / 2.0
                        if edge == "top"
                        else board_max_y + SILK_OFFBOARD_GAP_MM + item_h / 2.0
                    )
                    cursor += item_w + SILK_OFFBOARD_SPACING_MM
                else:
                    item_cy = board_min_y + cursor + item_h / 2.0
                    item_cx = (
                        board_min_x - SILK_OFFBOARD_GAP_MM - item_w / 2.0
                        if edge == "left"
                        else board_max_x + SILK_OFFBOARD_GAP_MM + item_w / 2.0
                    )
                    cursor += item_h + SILK_OFFBOARD_SPACING_MM

                if kind == "individual":
                    gap_mm = _label_gap_distance(comp, item_cx, item_cy, item_w, item_h)
                    label = {
                        "component_id": comp.get("id"),
                        "component_name": comp.get("name"),
                        "text": comp["ref"],
                        "x": item_cx,
                        "y": item_cy,
                        "width_mm": width_mm,
                        "height_mm": SILK_TEXT_HEIGHT_MM,
                        "collision_width_mm": item_w,
                        "collision_height_mm": item_h,
                        "rotation_deg": rotation_deg,
                        "font_size": f"{SILK_TEXT_HEIGHT_MM * 2.8346:.2f}pt",
                        "side": f_side,
                        "gap_mm": gap_mm,
                        "arrow_cells": set(),
                        "placement_mode": "off_board",
                    }
                    placed.append(label)
                    off_board_refs.append(comp["ref"])
                else:
                    block_min_x = item_cx - block_w / 2.0
                    block_min_y = item_cy - block_h / 2.0
                    new_labels = []
                    for m, w_mm, h_mm, rotation_deg2, cw, ch, local_x, local_y in specs:
                        x = block_min_x + local_x
                        y = block_min_y + local_y
                        new_labels.append(
                            {
                                "component_id": m.get("id"),
                                "component_name": m.get("name"),
                                "text": m["ref"],
                                "x": x,
                                "y": y,
                                "width_mm": w_mm,
                                "height_mm": SILK_TEXT_HEIGHT_MM,
                                "collision_width_mm": cw,
                                "collision_height_mm": ch,
                                "rotation_deg": rotation_deg2,
                                "font_size": f"{SILK_TEXT_HEIGHT_MM * 2.8346:.2f}pt",
                                "side": f_side,
                                "gap_mm": _label_gap_distance(m, x, y, cw, ch),
                                "arrow_cells": set(),
                                "placement_mode": "off_board_cluster",
                            }
                        )
                    placed.extend(new_labels)
                    off_board_refs.extend(m["ref"] for m in members)
    if off_board_refs:
        print(f"Off-board fallback: {len(off_board_refs)} label(s) placed outside the board -- still need manual placement")

    return placed, {
        "placed": len(placed),
        "total": total,
        "failed": [],
        "trace_relaxed": sum(1 for label in placed if label.get("placement_mode") == "trace_relaxed"),
        "escaped": escaped_count,
        "off_board": off_board_refs,
        "family_patterns": family_pattern_summary,
        "family_pattern_detail": family_pattern_detail,
    }


def _segment_to_rect(x0, y0, x1, y1, width):
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length <= 0:
        return None
    half_w = width / 2.0
    scale = half_w / length
    ox = -dy * scale
    oy = dx * scale
    return [x0 + ox, x1 + ox, x1 - ox, x0 - ox], [y0 + oy, y1 + oy, y1 - oy, y0 - oy]


def _layer_color(layer):
    if layer == "F.Cu":
        return "#d95f02"
    if layer == "B.Cu":
        return "#1b9e77"
    palette = ["#66c2a5", "#8da0cb", "#fc8d62", "#e78ac3", "#a6d854", "#ffd92f", "#b3b3b3"]
    return palette[abs(hash(layer)) % len(palette)]


def _silk_layer_label(side):
    return "BOTTOM" if side == "B.Cu" else "TOP"


def _transform_xy(x, y, origin_offset=None, board_bounds=None, mirror_y=False, translate_origin=False):
    x_val = _safe_float(x)
    y_val = _safe_float(y)
    if mirror_y and board_bounds is not None:
        y_val = board_bounds[1] + board_bounds[3] - y_val
    if translate_origin and origin_offset is not None:
        x_val -= origin_offset[0]
        y_val -= origin_offset[1]
    return x_val, y_val


def _transformed_label(label, origin_offset=None, board_bounds=None, mirror_y=False, translate_origin=False):
    out = dict(label)
    out["x"], out["y"] = _transform_xy(
        label.get("x"),
        label.get("y"),
        origin_offset=origin_offset,
        board_bounds=board_bounds,
        mirror_y=mirror_y,
        translate_origin=translate_origin,
    )
    if all(label.get(key) is not None for key in ("arrow_start_x", "arrow_start_y", "arrow_end_x", "arrow_end_y")):
        out["arrow_start_x"], out["arrow_start_y"] = _transform_xy(
            label.get("arrow_start_x"),
            label.get("arrow_start_y"),
            origin_offset=origin_offset,
            board_bounds=board_bounds,
            mirror_y=mirror_y,
            translate_origin=translate_origin,
        )
        out["arrow_end_x"], out["arrow_end_y"] = _transform_xy(
            label.get("arrow_end_x"),
            label.get("arrow_end_y"),
            origin_offset=origin_offset,
            board_bounds=board_bounds,
            mirror_y=mirror_y,
            translate_origin=translate_origin,
        )
    if label.get("extra_arrow_segments"):
        out["extra_arrow_segments"] = [
            (
                *_transform_xy(x0, y0, origin_offset=origin_offset, board_bounds=board_bounds,
                               mirror_y=mirror_y, translate_origin=translate_origin),
                *_transform_xy(x1, y1, origin_offset=origin_offset, board_bounds=board_bounds,
                               mirror_y=mirror_y, translate_origin=translate_origin),
            )
            for x0, y0, x1, y1 in label["extra_arrow_segments"]
        ]
    return out


def write_silkscreen_labels(labels, output_path, origin_offset=None, board_bounds=None, mirror_y=False, translate_origin=False):
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_file:
        # ExtraArrowSegments_mm carries the rest of a routed (multi-segment)
        # connector after its first segment, as "x0 y0 x1 y1;x0 y0 x1 y1"
        # in the same coordinates as the arrow columns. A reader that only
        # knows the original eleven columns (csv.DictReader by name) simply
        # ignores it and still draws the first segment.
        out_file.write(
            "RefDes,X_mm,Y_mm,Width_mm,Height_mm,Layer,Rotation_deg,"
            "ArrowStartX_mm,ArrowStartY_mm,ArrowEndX_mm,ArrowEndY_mm,ExtraArrowSegments_mm\n"
        )
        for label in labels or []:
            transformed = _transformed_label(
                label,
                origin_offset=origin_offset,
                board_bounds=board_bounds,
                mirror_y=mirror_y,
                translate_origin=translate_origin,
            )
            text = (transformed.get("text") or "").replace(",", " ")
            arrow_values = ["", "", "", ""]
            if all(
                transformed.get(key) is not None
                for key in ("arrow_start_x", "arrow_start_y", "arrow_end_x", "arrow_end_y")
            ):
                arrow_values = [
                    f"{transformed['arrow_start_x']:.4f}",
                    f"{transformed['arrow_start_y']:.4f}",
                    f"{transformed['arrow_end_x']:.4f}",
                    f"{transformed['arrow_end_y']:.4f}",
                ]
            extra_segments = ";".join(
                f"{x0:.4f} {y0:.4f} {x1:.4f} {y1:.4f}"
                for x0, y0, x1, y1 in transformed.get("extra_arrow_segments") or []
            )
            out_file.write(
                f"{text},{transformed['x']:.4f},{transformed['y']:.4f},"
                f"{_safe_float(label.get('width_mm')):.4f},{_safe_float(label.get('height_mm')):.4f},"
                f"{_silk_layer_label(label.get('side'))},"
                f"{_safe_float(label.get('rotation_deg')):.1f},{','.join(arrow_values)},{extra_segments}\n"
            )


def write_silkscreen_json(labels, output_path):
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    serializable = []
    for label in labels or []:
        item = {key: value for key, value in label.items() if not key.endswith("_cells")}
        serializable.append(item)
    with open(output_path, "w", encoding="utf-8") as out_file:
        json.dump(serializable, out_file, indent=2)


def write_silkscreen_report(board, labels, stats, output_path):
    """Two-section HTML summary for whoever runs this placement: which
    labels still need to be placed by hand (the off-board fallback list --
    the tool admits it couldn't legally fit them rather than forcing
    something wrong), and run analytics (placement breakdown, family
    patterns, how many dense clusters were formed) so the results can be
    trusted at a glance instead of re-deriving them from the raw output."""
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    off_board_refs = list(stats.get("off_board") or [])
    total = int(stats.get("total") or 0)
    placed = int(stats.get("placed") or 0)
    escaped = int(stats.get("escaped") or 0)
    trace_relaxed = int(stats.get("trace_relaxed") or 0)

    mode_counts = defaultdict(int)
    for label in labels or []:
        mode_counts[label.get("placement_mode") or "unknown"] += 1

    # Each dense cluster gets exactly one connector arrow, carried by
    # whichever member ended up as its anchor -- counting those is a
    # cheap, accurate stand-in for "how many clusters were formed" without
    # needing the pipeline to separately report cluster membership.
    cluster_count = sum(
        1
        for label in labels or []
        if (label.get("placement_mode") or "").endswith("cluster") and label.get("arrow_start_x") is not None
    )

    family = stats.get("family_patterns") or {}

    def esc(text):
        return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    if not off_board_refs:
        needs_placement_html = '<p class="ok">All labels placed on the board -- nothing needs manual placement.</p>'
    else:
        chips = "".join(f'<span class="ref">{esc(ref)}</span>' for ref in sorted(off_board_refs))
        needs_placement_html = (
            f'<p class="warn">{len(off_board_refs)} label(s) found no legal spot anywhere on the board and '
            "were placed just outside its edge instead, without a connector line. "
            "These still need to be moved onto the board by hand.</p>"
            f'<div class="reflist">{chips}</div>'
        )

    mode_rows = "".join(
        f"<tr><td>{esc(mode)}</td><td>{count}</td></tr>"
        for mode, count in sorted(mode_counts.items(), key=lambda kv: -kv[1])
    )

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Silkscreen placement report</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", sans-serif; margin: 0; padding: 0; background: #f6f6f6; color: #222; }}
header {{ background: #1f2933; color: #fff; padding: 20px 30px; }}
header h1 {{ margin: 0 0 4px 0; font-size: 20px; }}
header p {{ margin: 0; color: #b8c2cc; font-size: 13px; }}
main {{ max-width: 900px; margin: 0 auto; padding: 20px 30px 60px; }}
section {{ background: #fff; border-radius: 8px; padding: 20px 24px; margin: 20px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
section h2 {{ margin-top: 0; font-size: 16px; border-bottom: 1px solid #eee; padding-bottom: 8px; }}
section h3 {{ font-size: 14px; margin: 20px 0 4px; }}
.ok {{ color: #1a7f37; font-weight: 600; }}
.warn {{ color: #b45309; font-weight: 600; }}
.reflist {{ margin-top: 10px; }}
.ref {{ display: inline-block; background: #fef3c7; color: #92400e; border-radius: 4px; padding: 3px 8px; margin: 3px; font-family: monospace; font-size: 12px; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 6px; }}
th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #eee; font-size: 13px; }}
th {{ color: #666; font-weight: 600; }}
.stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-top: 10px; }}
.stat {{ background: #f8f9fa; border-radius: 6px; padding: 12px; }}
.stat .value {{ font-size: 22px; font-weight: 700; }}
.stat .label {{ font-size: 12px; color: #666; }}
</style>
</head>
<body>
<header>
<h1>Silkscreen placement report</h1>
<p>{esc(os.path.basename(str(board.get("path", "") or "")))} -- generated {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
</header>
<main>

<section id="needs-placement">
<h2>Needs manual placement</h2>
{needs_placement_html}
</section>

<section id="analytics">
<h2>Analytics</h2>
<div class="stat-grid">
<div class="stat"><div class="value">{placed}/{total}</div><div class="label">labels placed</div></div>
<div class="stat"><div class="value">{escaped}</div><div class="label">rescued via escape logic</div></div>
<div class="stat"><div class="value">{cluster_count}</div><div class="label">dense clusters formed</div></div>
<div class="stat"><div class="value">{len(off_board_refs)}</div><div class="label">need manual placement</div></div>
<div class="stat"><div class="value">{trace_relaxed}</div><div class="label">trace-relaxed placements</div></div>
</div>

<h3>Placement breakdown</h3>
<table>
<tr><th>Mode</th><th>Count</th></tr>
{mode_rows}
</table>

<h3>Family patterns identified</h3>
<div class="stat-grid">
<div class="stat"><div class="value">{family.get("row", 0)}</div><div class="label">rows</div></div>
<div class="stat"><div class="value">{family.get("mirror_pair", 0)}</div><div class="label">mirror pairs</div></div>
<div class="stat"><div class="value">{family.get("twin_pair", 0)}</div><div class="label">twin pairs</div></div>
<div class="stat"><div class="value">{family.get("unclassified", 0)}</div><div class="label">unclassified clusters</div></div>
</div>
</section>

</main>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as out_file:
        out_file.write(html)


def plot_silkscreen_bokeh(
    board, labels, output_html, show_plot=False, failed_refs=None,
    dense_clusters=None, rescued_refs=None,
):
    p_fig = figure(
        width=1500,
        height=800,
        title=f"Silkscreen placement - {os.path.basename(board['path'])}",
        match_aspect=True,
        tools="pan,wheel_zoom,reset,save",
        active_scroll="wheel_zoom",
        active_drag="pan",
    )

    edge_xs = []
    edge_ys = []
    for x0, y0, x1, y1 in board["edge_segments"]:
        edge_xs.append([x0, x1])
        edge_ys.append([y0, y1])
    if edge_xs:
        p_fig.multi_line(edge_xs, edge_ys, line_color="black", line_width=2, legend_label="Board outline")

    zones = board.get("zones") or {}
    for layer in sorted(zones, key=_layer_sort_key):
        xs = []
        ys = []
        labels_for_layer = []
        for zone in zones[layer]:
            points = zone.get("points") or []
            if len(points) >= 3:
                xs.append([x for x, _y in points])
                ys.append([y for _x, y in points])
                labels_for_layer.append(zone.get("net") or "")
        if xs:
            color = _layer_color(layer)
            p_fig.patches(
                xs,
                ys,
                fill_color=color,
                fill_alpha=0.10,
                line_color=color,
                line_alpha=0.25,
                legend_label=f"Zones ({layer})",
            )

    comp_src = defaultdict(list)
    for comp in board["components"]:
        comp_src["x"].append((comp["min_x"] + comp["max_x"]) / 2.0)
        comp_src["y"].append((comp["min_y"] + comp["max_y"]) / 2.0)
        comp_src["w"].append(max(comp["max_x"] - comp["min_x"], 0.2))
        comp_src["h"].append(max(comp["max_y"] - comp["min_y"], 0.2))
        comp_src["ref"].append(comp["ref"])
    if comp_src["x"]:
        p_fig.rect(
            x="x",
            y="y",
            width="w",
            height="h",
            source=ColumnDataSource(comp_src),
            fill_alpha=0.04,
            fill_color="#9a9a9a",
            line_alpha=0.45,
            line_color="#4a4a4a",
            legend_label="Component bounds",
        )

    if failed_refs:
        failed_ref_set = set(failed_refs)
        failed_src = defaultdict(list)
        for comp in board["components"]:
            if comp["ref"] not in failed_ref_set:
                continue
            failed_src["x"].append((comp["min_x"] + comp["max_x"]) / 2.0)
            failed_src["y"].append((comp["min_y"] + comp["max_y"]) / 2.0)
            failed_src["w"].append(max(comp["max_x"] - comp["min_x"], 0.2))
            failed_src["h"].append(max(comp["max_y"] - comp["min_y"], 0.2))
            failed_src["ref"].append(comp["ref"])
        if failed_src["x"]:
            failed_source = ColumnDataSource(failed_src)
            p_fig.rect(
                x="x",
                y="y",
                width="w",
                height="h",
                source=failed_source,
                fill_alpha=0.25,
                fill_color="#e41a1c",
                line_alpha=0.9,
                line_color="#e41a1c",
                line_width=2,
                legend_label="Failed / unplaced components",
            )
            p_fig.text(
                x="x",
                y="y",
                text="ref",
                text_font_size="9px",
                text_align="center",
                text_baseline="middle",
                text_color="#e41a1c",
                source=failed_source,
                legend_label="Failed / unplaced components",
            )

    if dense_clusters:
        cluster_xs = []
        cluster_ys = []
        for min_x, min_y, max_x, max_y in dense_clusters:
            cluster_xs.append([min_x, max_x, max_x, min_x])
            cluster_ys.append([min_y, min_y, max_y, max_y])
        p_fig.patches(
            cluster_xs,
            cluster_ys,
            fill_alpha=0.06,
            fill_color="#ff7f00",
            line_alpha=0.9,
            line_color="#ff7f00",
            line_width=2,
            line_dash="dashed",
            legend_label="Densely packed / unplaceable area",
        )

    if rescued_refs:
        rescued_ref_set = set(rescued_refs)
        comp_by_ref = {comp["ref"]: comp for comp in board["components"]}
        before_src = defaultdict(list)
        after_src = defaultdict(list)
        for label in labels or []:
            if label["text"] not in rescued_ref_set:
                continue
            comp = comp_by_ref.get(label["text"])
            if comp is None:
                continue
            before_src["x"].append(comp["cx"])
            before_src["y"].append(comp["cy"])
            before_src["ref"].append(label["text"])
            after_src["x"].append(label["x"])
            after_src["y"].append(label["y"])
            after_src["ref"].append(label["text"])
        if before_src["x"]:
            p_fig.circle(
                x="x", y="y", size=6,
                source=ColumnDataSource(before_src),
                fill_color="#377eb8", fill_alpha=0.9, line_color="#377eb8",
                legend_label="Escape-rescued: original component position",
            )
        if after_src["x"]:
            p_fig.circle(
                x="x", y="y", size=9,
                source=ColumnDataSource(after_src),
                fill_color=None, line_color="#377eb8", line_width=2,
                legend_label="Escape-rescued: final label position",
            )

    graphic_lines = defaultdict(lambda: defaultdict(list))
    graphic_patches = defaultdict(lambda: defaultdict(list))
    graphic_circles = defaultdict(lambda: defaultdict(list))
    for graphic in board.get("footprint_graphics") or []:
        layer = graphic.get("layer") or ""
        if graphic["kind"] in {"line", "polyline"}:
            points = graphic.get("points") or []
            if len(points) >= 2:
                graphic_lines[layer]["xs"].append([x for x, _y in points])
                graphic_lines[layer]["ys"].append([y for _x, y in points])
        elif graphic["kind"] == "polygon":
            points = graphic.get("points") or []
            if len(points) >= 3:
                graphic_patches[layer]["xs"].append([x for x, _y in points])
                graphic_patches[layer]["ys"].append([y for _x, y in points])
        elif graphic["kind"] == "circle":
            graphic_circles[layer]["x"].append(graphic["x"])
            graphic_circles[layer]["y"].append(graphic["y"])
            graphic_circles[layer]["radius"].append(graphic["radius"])
    for layer, data in graphic_patches.items():
        if data["xs"]:
            p_fig.patches(
                data["xs"],
                data["ys"],
                fill_alpha=0.0,
                line_color="#555555",
                line_alpha=0.55,
                legend_label=f"Footprint polygons ({layer})",
            )
    for layer, data in graphic_lines.items():
        if data["xs"]:
            p_fig.multi_line(
                data["xs"],
                data["ys"],
                line_color="#555555",
                line_alpha=0.55,
                line_width=1,
                legend_label=f"Footprint drawing ({layer})",
            )
    for layer, data in graphic_circles.items():
        if data["x"]:
            p_fig.circle(
                x="x",
                y="y",
                radius="radius",
                source=ColumnDataSource(data),
                fill_alpha=0.0,
                line_color="#555555",
                line_alpha=0.55,
                legend_label=f"Footprint circles ({layer})",
            )

    for layer in sorted(board["trace_segments"], key=_layer_sort_key):
        polys_x = []
        polys_y = []
        for seg in board["trace_segments"].get(layer) or []:
            rect = _segment_to_rect(seg["x0"], seg["y0"], seg["x1"], seg["y1"], seg["width"])
            if rect:
                xs, ys = rect
                polys_x.append(xs)
                polys_y.append(ys)
        if polys_x:
            color = _layer_color(layer)
            p_fig.patches(
                polys_x,
                polys_y,
                fill_color=color,
                line_color=color,
                fill_alpha=0.75,
                line_alpha=0.75,
                legend_label=f"Traces ({layer})",
            )

    pad_rect = {"F.Cu": defaultdict(list), "B.Cu": defaultdict(list)}
    pad_circle = {"F.Cu": defaultdict(list), "B.Cu": defaultdict(list)}
    th_rect = defaultdict(list)
    th_circle = defaultdict(list)
    for pad in board["pads"]:
        target_rect = th_rect if pad["is_th"] else pad_rect.get(pad["side"], pad_rect["F.Cu"])
        target_circle = th_circle if pad["is_th"] else pad_circle.get(pad["side"], pad_circle["F.Cu"])
        if pad["is_circle"]:
            target_circle["x"].append(pad["x"])
            target_circle["y"].append(pad["y"])
            target_circle["radius"].append(pad["w"] / 2.0)
        else:
            target_rect["x"].append(pad["x"])
            target_rect["y"].append(pad["y"])
            target_rect["w"].append(pad["w"])
            target_rect["h"].append(pad["h"])
            target_rect["angle"].append(pad["angle"])

    for side, color, label in (("F.Cu", "#d95f02", "SMD pads (top)"), ("B.Cu", "#1b9e77", "SMD pads (bottom)")):
        if pad_rect[side]["x"]:
            p_fig.rect(
                x="x",
                y="y",
                width="w",
                height="h",
                angle="angle",
                source=ColumnDataSource(pad_rect[side]),
                color=color,
                alpha=0.88,
                legend_label=label,
            )
        if pad_circle[side]["x"]:
            p_fig.circle(
                x="x",
                y="y",
                radius="radius",
                source=ColumnDataSource(pad_circle[side]),
                color=color,
                alpha=0.88,
                legend_label=f"{label} circular",
            )
    if th_rect["x"]:
        p_fig.rect(
            x="x",
            y="y",
            width="w",
            height="h",
            angle="angle",
            source=ColumnDataSource(th_rect),
            color="#4daf4a",
            alpha=0.75,
            legend_label="Through-hole pads",
        )
    if th_circle["x"]:
        p_fig.circle(
            x="x",
            y="y",
            radius="radius",
            source=ColumnDataSource(th_circle),
            color="#4daf4a",
            alpha=0.75,
            legend_label="Through-hole pads circular",
        )

    via_src = defaultdict(list)
    for via in board["vias"]:
        via_src["x"].append(via["x"])
        via_src["y"].append(via["y"])
        via_src["radius"].append(via["size"] / 2.0)
    if via_src["x"]:
        p_fig.circle(
            x="x",
            y="y",
            radius="radius",
            source=ColumnDataSource(via_src),
            color="#6a3d9a",
            alpha=0.85,
            legend_label="Vias",
        )

    silk_sources = []
    for side, text_color, arrow_color, legend_suffix in (
        ("F.Cu", "#111111", "#111111", "top"),
        ("B.Cu", "#575757", "#575757", "bottom"),
    ):
        text_src = defaultdict(list)
        arrow_src = defaultdict(list)
        for label in labels or []:
            if (label.get("side") or "F.Cu") != side:
                continue
            text_src["x"].append(label["x"])
            text_src["y"].append(label["y"])
            text_src["text"].append(label["text"])
            text_src["font_size"].append(label["font_size"])
            text_src["height_mm"].append(label.get("height_mm", SILK_TEXT_HEIGHT_MM))
            text_src["angle_rad"].append(
                math.radians(_safe_float(label.get("rotation_deg")))
            )
            if all(label.get(key) is not None for key in ("arrow_start_x", "arrow_start_y", "arrow_end_x", "arrow_end_y")):
                for x0, y0, x1, y1 in _arrow_segments(
                    (label["arrow_start_x"], label["arrow_start_y"], label["arrow_end_x"], label["arrow_end_y"])
                ) + list(label.get("extra_arrow_segments") or []):
                    arrow_src["x0"].append(x0)
                    arrow_src["y0"].append(y0)
                    arrow_src["x1"].append(x1)
                    arrow_src["y1"].append(y1)
        if arrow_src["x0"]:
            p_fig.segment(
                x0="x0",
                y0="y0",
                x1="x1",
                y1="y1",
                source=ColumnDataSource(arrow_src),
                line_color=arrow_color,
                line_width=1.5,
                legend_label=f"Silk pointers ({legend_suffix})",
            )
        if text_src["x"]:
            source = ColumnDataSource(text_src)
            silk_sources.append(source)
            p_fig.text(
                x="x",
                y="y",
                text="text",
                text_font_size="font_size",
                text_align="center",
                text_baseline="middle",
                angle="angle_rad",
                text_color=text_color,
                source=source,
                legend_label=f"Silk labels ({legend_suffix})",
            )

    if silk_sources:
        callback = CustomJS(
            args={"plot": p_fig, "sources": silk_sources},
            code="""
            function update_sizes() {
                const width = plot.frame_width || plot.inner_width || plot.plot_width;
                const height = plot.frame_height || plot.inner_height || plot.plot_height;
                const xspan = Math.abs(plot.x_range.end - plot.x_range.start);
                const yspan = Math.abs(plot.y_range.end - plot.y_range.start);
                if (!width || !xspan) {
                    return;
                }
                const px_per_unit_x = width / xspan;
                const px_per_unit_y = height && yspan ? height / yspan : px_per_unit_x;
                const px_per_unit = Math.min(px_per_unit_x, px_per_unit_y);
                for (const src of sources) {
                    const heights = src.data.height_mm || [];
                    const sizes = [];
                    for (let i = 0; i < heights.length; i++) {
                        sizes.push(Math.max(2.0, heights[i] * px_per_unit).toFixed(1) + "px");
                    }
                    src.data.font_size = sizes;
                    src.change.emit();
                }
            }
            update_sizes();
            """,
        )
        p_fig.js_on_event(DocumentReady, callback)
        p_fig.x_range.js_on_change("start", callback)
        p_fig.x_range.js_on_change("end", callback)
        p_fig.y_range.js_on_change("start", callback)
        p_fig.y_range.js_on_change("end", callback)

    p_fig.legend.location = "top_left"
    p_fig.legend.click_policy = "hide"
    p_fig.background_fill_color = "#f3f3f3"
    p_fig.grid.visible = False
    output_file(output_html, title="Silkscreen generator")
    save(p_fig)
    if show_plot:
        show(p_fig)


def _resolve_kicad_path(board_folder, kicad_path):
    if kicad_path:
        return os.path.abspath(kicad_path)
    board_folder = os.path.abspath(board_folder or os.getcwd())
    return os.path.join(board_folder, SOURCE_BOARD_FILENAME)


def _find_board_folders(root_folder):
    root_folder = os.path.abspath(root_folder)
    matches = []
    for current_folder, _directories, filenames in os.walk(root_folder):
        if SOURCE_BOARD_FILENAME in filenames:
            matches.append(current_folder)
    return sorted(matches, key=lambda path: path.casefold())


def _batch_child_command(args, board_folder):
    command = [
        sys.executable,
        os.path.abspath(__file__),
        board_folder,
        "--single-board",
        "--txt-coordinates",
        args.txt_coordinates,
        "--updater-script",
        args.updater_script,
        "--kicad-python",
        args.kicad_python,
    ]
    if args.strict_trace_clearance:
        command.append("--strict-trace-clearance")
    if args.skip_board_update:
        command.append("--skip-board-update")
    if args.show_fab_layers:
        command.append("--show-fab-layers")
    return command


def _timeout_text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _run_child_with_timeout(command, timeout_seconds):
    creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creation_flags,
        start_new_session=os.name != "nt",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            process.kill()
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            command,
            timeout_seconds,
            output=stdout,
            stderr=stderr,
        )
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def run_recursive_batch(args):
    root_folder = os.path.abspath(args.board_folder)
    if not os.path.isdir(root_folder):
        raise NotADirectoryError(f"Boards directory not found: {root_folder}")

    board_folders = _find_board_folders(root_folder)
    if not board_folders:
        raise FileNotFoundError(
            f"No {SOURCE_BOARD_FILENAME} files found under: {root_folder}"
        )

    custom_outputs = [
        args.html_output,
        args.txt_output,
        args.json_output,
        args.pcb_output,
    ]
    if any(custom_outputs):
        raise ValueError(
            "Per-file output overrides cannot be used in recursive batch mode. "
            "Each board uses the default output names in its own folder."
        )
    if args.show:
        raise ValueError("--show cannot be used in recursive batch mode.")

    started_at = datetime.now().astimezone()
    results = []
    print(
        f"Found {len(board_folders)} board(s) under {root_folder}. "
        f"Timeout: {args.timeout:g} seconds per board."
    )

    for index, board_folder in enumerate(board_folders, start=1):
        print(f"[{index}/{len(board_folders)}] Processing: {board_folder}")
        started = time.monotonic()
        result = {
            "board_folder": board_folder,
            "source_board": os.path.join(board_folder, SOURCE_BOARD_FILENAME),
            "output_board": os.path.join(board_folder, OUTPUT_BOARD_FILENAME),
        }
        try:
            completed = _run_child_with_timeout(
                _batch_child_command(args, board_folder),
                args.timeout,
            )
            result["return_code"] = completed.returncode
            result["stdout"] = completed.stdout
            result["stderr"] = completed.stderr
            if completed.returncode == 0:
                result["status"] = "success"
                print("  SUCCESS")
            else:
                result["status"] = "failed"
                result["error"] = (
                    completed.stderr.strip()
                    or completed.stdout.strip()
                    or f"Exited with code {completed.returncode}"
                )
                print(f"  FAILED (exit code {completed.returncode})")
        except subprocess.TimeoutExpired as exc:
            result["status"] = "timeout"
            result["return_code"] = None
            result["stdout"] = _timeout_text(exc.stdout)
            result["stderr"] = _timeout_text(exc.stderr)
            result["error"] = f"Exceeded {args.timeout:g}-second timeout"
            print(f"  TIMEOUT after {args.timeout:g} seconds")
        except Exception as exc:
            result["status"] = "failed"
            result["return_code"] = None
            result["stdout"] = ""
            result["stderr"] = ""
            result["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {exc}")

        result["duration_seconds"] = round(time.monotonic() - started, 3)
        results.append(result)

    success_count = sum(item["status"] == "success" for item in results)
    timeout_count = sum(item["status"] == "timeout" for item in results)
    failed_count = len(results) - success_count
    report_path = os.path.abspath(
        args.report_output or os.path.join(root_folder, BATCH_REPORT_FILENAME)
    )
    report = {
        "root_folder": root_folder,
        "source_board_filename": SOURCE_BOARD_FILENAME,
        "output_board_filename": OUTPUT_BOARD_FILENAME,
        "timeout_seconds": args.timeout,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "total_boards": len(results),
        "success_count": success_count,
        "failed_count": failed_count,
        "timeout_count": timeout_count,
        "boards": results,
    }
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)

    print(
        f"Batch complete: {success_count} succeeded, {failed_count} failed "
        f"({timeout_count} timed out)."
    )
    print(f"Saved batch report: {report_path}")
    return 0 if failed_count == 0 else 1


def generate_silkscreen_outputs(
    kicad_path,
    output_folder=None,
    *,
    strict_trace_clearance=False,
    write_html=True,
):
    """Generate label artifacts for one routed KiCad board.

    Board mutation is intentionally left to the pcbnew-based KiCad service so
    callers can preserve the non-silkscreen routed board as a fallback.
    """
    kicad_path = os.path.abspath(kicad_path)
    if not os.path.isfile(kicad_path):
        raise FileNotFoundError(f"KiCad board not found: {kicad_path}")

    output_folder = os.path.abspath(
        output_folder or os.path.dirname(kicad_path)
    )
    os.makedirs(output_folder, exist_ok=True)
    txt_output = os.path.join(output_folder, "labels_output.txt")
    json_output = os.path.join(output_folder, "silkscreen_data.json")
    html_output = os.path.join(output_folder, "silkscreen_bokeh.html")
    report_output = os.path.join(output_folder, "silkscreen_report.html")

    board = parse_kicad_board_with_retry(kicad_path)
    labels, stats = auto_place_silkscreen(
        board["components"],
        board["pads"],
        board["trace_segments"],
        board["vias"],
        board["edge_segments"],
        allow_trace_overlap_fallback=not strict_trace_clearance,
    )
    bounds = _board_bounds(board["edge_segments"])
    write_silkscreen_labels(labels, txt_output)
    write_silkscreen_json(labels, json_output)
    write_silkscreen_report(board, labels, stats, report_output)
    if write_html:
        plot_silkscreen_bokeh(board, labels, html_output, show_plot=False)

    return {
        "board_file": kicad_path,
        "labels_file": txt_output,
        "json_file": json_output,
        "html_file": html_output if write_html else None,
        "report_file": report_output,
        "placed": int(stats.get("placed", 0)),
        "total": int(stats.get("total", 0)),
        "failed": list(stats.get("failed", ())),
        "trace_relaxed": int(stats.get("trace_relaxed", 0)),
        "escaped": int(stats.get("escaped", 0)),
        "off_board": list(stats.get("off_board", ())),
        "family_patterns": stats.get("family_patterns", {}),
        "family_pattern_detail": stats.get("family_pattern_detail", []),
        "board_bounds": bounds,
    }


def update_kicad_board(
    source_board,
    output_board,
    labels_path,
    coordinate_mode,
    updater_script=DEFAULT_SILKSCREEN_UPDATER,
    kicad_python=DEFAULT_KICAD_PYTHON,
):
    source_board = os.path.abspath(source_board)
    output_board = os.path.abspath(output_board)
    updater_script = os.path.abspath(updater_script)
    kicad_python = os.path.abspath(kicad_python)

    if source_board == output_board:
        raise ValueError("The silk output board must differ from the source board.")
    if not os.path.isfile(updater_script):
        raise FileNotFoundError(f"Silkscreen updater not found: {updater_script}")
    if not os.path.isfile(kicad_python):
        raise FileNotFoundError(f"KiCad Python not found: {kicad_python}")

    output_dir = os.path.dirname(output_board)
    os.makedirs(output_dir, exist_ok=True)
    handle, temporary_board = tempfile.mkstemp(
        prefix=f".{os.path.splitext(os.path.basename(output_board))[0]}_",
        suffix=".kicad_pcb",
        dir=output_dir,
    )
    os.close(handle)

    try:
        shutil.copy2(source_board, temporary_board)
        subprocess.run(
            [
                kicad_python,
                updater_script,
                temporary_board,
                "--labels",
                os.path.abspath(labels_path),
                "--coordinates",
                coordinate_mode,
            ],
            check=True,
        )
        os.replace(temporary_board, output_board)
        _adopt_project_sidecars(temporary_board, output_board)
    except Exception:
        if os.path.exists(temporary_board):
            os.remove(temporary_board)
        raise

    return output_board


def _adopt_project_sidecars(temporary_board, output_board):
    """pcbnew.SaveBoard writes a .kicad_pro and a .kicad_prl next to the
    board it saves, named after it. The board itself is renamed to
    output_board; this renames those two sidecars to the output name too
    (replacing stale ones) instead of leaving orphans named after the
    temporary file."""
    temp_base = os.path.splitext(temporary_board)[0]
    out_base = os.path.splitext(output_board)[0]
    for extension in (".kicad_pro", ".kicad_prl"):
        source = temp_base + extension
        if os.path.isfile(source):
            os.replace(source, out_base + extension)


def _visible_layers_without(mask_text, layer_ids):
    """KiCad's LSET hex mask ('ffffffff_ffffffff_ffffffff_ffffffff', most
    significant group first) with the given layer bits cleared."""
    groups = mask_text.split("_")
    bits = int("".join(groups), 16)
    width = 4 * sum(len(group) for group in groups)
    for layer_id in layer_ids:
        if 0 <= layer_id < width:
            bits &= ~(1 << layer_id)
    text = f"{bits:0{width // 4}x}"
    out, position = [], 0
    for group in groups:
        out.append(text[position:position + len(group)])
        position += len(group)
    return "_".join(out)


def hide_fab_layers_in_project(board_path):
    """Switch F.Fab and B.Fab off in the board's project-local settings
    (<board>.kicad_prl) so KiCad opens the output with the Fab layers
    hidden. Layer visibility is not part of the .kicad_pcb; with no such
    file KiCad shows every layer. The file pcbnew wrote for this board is
    edited in place; if there is none, a minimal one is created. Returns
    the .kicad_prl path."""
    prl_path = os.path.splitext(os.path.abspath(board_path))[0] + ".kicad_prl"
    settings = {}
    if os.path.isfile(prl_path):
        with open(prl_path, "r", encoding="utf-8") as handle:
            settings = json.load(handle)
    board_settings = settings.setdefault("board", {})
    mask = board_settings.get("visible_layers") or "ffffffff_ffffffff_ffffffff_ffffffff"
    board_settings["visible_layers"] = _visible_layers_without(mask, FAB_LAYER_IDS)
    meta = settings.setdefault("meta", {})
    meta["filename"] = os.path.basename(prl_path)
    meta.setdefault("version", 5)
    with open(prl_path, "w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2)
    return prl_path



def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate silkscreen outputs and create updated_pcb_file_w_silk.kicad_pcb "
            "from a board folder, or recursively process a directory of board folders."
        )
    )
    parser.add_argument(
        "board_folder",
        nargs="?",
        default=os.getcwd(),
        help=(
            "A board folder containing updated_pcb_file.kicad_pcb, or a parent "
            "directory whose board folders should be processed recursively."
        ),
    )
    parser.add_argument("--kicad", default=None, help="Optional explicit .kicad_pcb path.")
    parser.add_argument("--html-output", default=None, help="Bokeh HTML output path.")
    parser.add_argument("--txt-output", default=None, help="Silkscreen import txt output path.")
    parser.add_argument("--json-output", default=None, help="Optional JSON output path.")
    parser.add_argument("--placement-report-output", default=None, help="Placement summary HTML report output path.")
    parser.add_argument(
        "--pcb-output",
        default=None,
        help="Updated KiCad board output path.",
    )
    parser.add_argument(
        "--updater-script",
        default=DEFAULT_SILKSCREEN_UPDATER,
        help="Path to Silkscreen_Script.py.",
    )
    parser.add_argument(
        "--kicad-python",
        default=DEFAULT_KICAD_PYTHON,
        help="Path to the KiCad 8 Python executable.",
    )
    parser.add_argument(
        "--skip-board-update",
        action="store_true",
        help="Generate TXT/JSON/HTML only; do not create a silk-updated PCB.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively process all board folders, even if the root is itself a board folder.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_BOARD_TIMEOUT_SECONDS,
        help="Maximum seconds allowed per board in recursive mode (default: 120).",
    )
    parser.add_argument(
        "--report-output",
        default=None,
        help=(
            "Recursive batch report path. Defaults to silkscreen_batch_report.json "
            "in the supplied root directory."
        ),
    )
    parser.add_argument("--single-board", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--txt-coordinates",
        choices=("altium", "kicad"),
        default="kicad",
        help="Use translated/mirrored Altium-compatible output or native KiCad coordinates.",
    )
    parser.add_argument(
        "--strict-trace-clearance",
        action="store_true",
        help="Do not use the fallback pass that allows labels over solder-masked traces.",
    )
    parser.add_argument(
        "--show-fab-layers",
        action="store_true",
        help=(
            "Leave the Fab layers visible. By default the output board's project-local "
            "settings (.kicad_prl) are written with F.Fab and B.Fab switched off."
        ),
    )
    parser.add_argument("--show", action="store_true", help="Open the Bokeh plot after saving.")
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")

    direct_board = os.path.join(os.path.abspath(args.board_folder), SOURCE_BOARD_FILENAME)
    should_run_batch = (
        not args.single_board
        and args.kicad is None
        and (args.recursive or not os.path.isfile(direct_board))
    )
    if should_run_batch:
        return run_recursive_batch(args)

    kicad_path = _resolve_kicad_path(args.board_folder, args.kicad)
    if not os.path.exists(kicad_path):
        raise FileNotFoundError(f"KiCad board not found: {kicad_path}")
    board_folder = os.path.dirname(kicad_path)
    html_output = args.html_output or os.path.join(board_folder, "silkscreen_bokeh.html")
    txt_output = args.txt_output or os.path.join(board_folder, "labels_output.txt")
    json_output = args.json_output or os.path.join(board_folder, "silkscreen_data.json")
    placement_report_output = args.placement_report_output or os.path.join(board_folder, "silkscreen_report.html")
    pcb_output = args.pcb_output or os.path.join(
        board_folder, "updated_pcb_file_w_silk.kicad_pcb"
    )

    board = parse_kicad_board(kicad_path)
    labels, stats = auto_place_silkscreen(
        board["components"],
        board["pads"],
        board["trace_segments"],
        board["vias"],
        board["edge_segments"],
        allow_trace_overlap_fallback=not args.strict_trace_clearance,
    )
    bounds = _board_bounds(board["edge_segments"])
    origin = (bounds[0], bounds[1]) if bounds else None
    mirror_y = args.txt_coordinates == "altium"
    translate_origin = args.txt_coordinates == "altium"

    write_silkscreen_labels(
        labels,
        txt_output,
        origin_offset=origin,
        board_bounds=bounds,
        mirror_y=mirror_y,
        translate_origin=translate_origin,
    )
    write_silkscreen_json(labels, json_output)
    write_silkscreen_report(board, labels, stats, placement_report_output)
    plot_silkscreen_bokeh(board, labels, html_output, show_plot=args.show)

    if not args.skip_board_update:
        update_kicad_board(
            kicad_path,
            pcb_output,
            txt_output,
            args.txt_coordinates,
            updater_script=args.updater_script,
            kicad_python=args.kicad_python,
        )
        if not args.show_fab_layers:
            fab_prl_path = hide_fab_layers_in_project(pcb_output)

    total_traces = sum(len(items) for items in board["trace_segments"].values())
    print(f"Board: {kicad_path}")
    print(
        "Parsed:"
        f" {len(board['components'])} components,"
        f" {len(board['pads'])} pads,"
        f" {total_traces} trace segments,"
        f" {len(board['vias'])} vias,"
        f" {sum(len(v) for v in board.get('zones', {}).values())} zone polygons"
    )
    print(f"Silkscreen labels: {stats['placed']}/{stats['total']} placed")
    if stats.get("trace_relaxed"):
        print(f"Trace-relaxed placements: {stats['trace_relaxed']}")
    if stats["failed"]:
        print(f"Unplaced labels: {', '.join(stats['failed'][:20])}")
        if len(stats["failed"]) > 20:
            print(f"... plus {len(stats['failed']) - 20} more")
    off_board = stats.get("off_board") or []
    if off_board:
        print(f"Placed outside board (need manual placement): {', '.join(off_board[:20])}")
        if len(off_board) > 20:
            print(f"... plus {len(off_board) - 20} more")
    print(f"Saved txt: {txt_output}")
    print(f"Saved json: {json_output}")
    print(f"Saved Bokeh HTML: {html_output}")
    print(f"Saved placement report: {placement_report_output}")
    if not args.skip_board_update:
        print(f"Saved silk-updated PCB: {pcb_output}")
        if not args.show_fab_layers:
            print(f"Fab layers switched off in project settings (no text item changed): {fab_prl_path}")


if __name__ == "__main__":
    raise SystemExit(main())
