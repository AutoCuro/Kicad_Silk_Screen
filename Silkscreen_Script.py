import argparse
import csv
import os

import pcbnew


def _normalize_layer(value):
    normalized = str(value or "").strip().lower()
    if normalized in {"top", "f.cu", "f.silks", "f.silk"}:
        return pcbnew.F_SilkS
    if normalized in {"bottom", "bot", "b.cu", "b.silks", "b.silk"}:
        return pcbnew.B_SilkS
    return None


def _optional_float(row, key):
    value = str(row.get(key) or "").strip()
    return float(value) if value else None


def _read_rows(labels_path):
    with open(labels_path, "r", encoding="utf-8", newline="") as labels_file:
        rows = list(csv.DictReader(labels_file))
    if not rows:
        raise ValueError(f"No silkscreen rows found in {labels_path}")
    return rows


def _remove_previous_generated_items(board, refdes_values):
    silk_layers = {pcbnew.F_SilkS, pcbnew.B_SilkS}
    removed = 0
    hidden_references = 0

    # KiCad stores the normal reference designator as a footprint field, not
    # in board.GetDrawings(). Hide that field without removing the footprint
    # identity needed by connectivity, BOM, and future board updates.
    all_refdes_values = set(refdes_values)
    for footprint in board.GetFootprints():
        reference = str(footprint.GetReference() or "").strip()
        if reference:
            all_refdes_values.add(reference)
        try:
            reference_field = footprint.Reference()
            if (
                reference_field.GetLayer() in silk_layers
                and reference_field.IsVisible()
            ):
                reference_field.SetVisible(False)
                hidden_references += 1
        except (AttributeError, RuntimeError):
            continue

    for drawing in list(board.GetDrawings()):
        if drawing.GetLayer() not in silk_layers:
            continue
        if isinstance(drawing, pcbnew.PCB_TEXT):
            if drawing.GetText().strip() not in all_refdes_values:
                continue
        elif isinstance(drawing, pcbnew.PCB_SHAPE):
            try:
                is_generated_pointer = (
                    drawing.GetShape() == pcbnew.SHAPE_T_SEGMENT
                    and any(
                        abs(pcbnew.ToMM(drawing.GetWidth()) - width) <= 0.005
                        for width in (0.10, 0.025)
                    )
                )
                if not is_generated_pointer:
                    continue
            except Exception:
                continue
        else:
            continue
        board.Delete(drawing)
        removed += 1
    return removed, hidden_references


def _add_pointer(board, layer, row):
    values = [
        _optional_float(row, "ArrowStartX_mm"),
        _optional_float(row, "ArrowStartY_mm"),
        _optional_float(row, "ArrowEndX_mm"),
        _optional_float(row, "ArrowEndY_mm"),
    ]
    if any(value is None for value in values):
        return False

    x0, y0, x1, y1 = values
    pointer = pcbnew.PCB_SHAPE(board)
    pointer.SetShape(pcbnew.SHAPE_T_SEGMENT)
    pointer.SetStart(pcbnew.VECTOR2I(pcbnew.FromMM(x0), pcbnew.FromMM(y0)))
    pointer.SetEnd(pcbnew.VECTOR2I(pcbnew.FromMM(x1), pcbnew.FromMM(y1)))
    pointer.SetWidth(pcbnew.FromMM(0.025))
    pointer.SetLayer(layer)
    board.Add(pointer)
    return True


def add_refdes_to_silkscreen(brd_path, file_path):
    """Apply generated native-KiCad silkscreen coordinates to a board."""
    if not os.path.isfile(brd_path):
        raise FileNotFoundError(f"Board file not found: {brd_path}")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Silkscreen labels not found: {file_path}")

    board = pcbnew.LoadBoard(brd_path)
    rows = _read_rows(file_path)
    refdes_values = {
        str(row.get("RefDes") or "").strip()
        for row in rows
        if str(row.get("RefDes") or "").strip()
    }
    removed, hidden_references = _remove_previous_generated_items(
        board, refdes_values
    )
    added_text = 0
    added_pointers = 0
    skipped = 0

    for row in rows:
        refdes = str(row.get("RefDes") or "").strip()
        layer = _normalize_layer(row.get("Layer"))
        if not refdes or layer is None:
            skipped += 1
            continue

        try:
            x_mm = float(row["X_mm"])
            y_mm = float(row["Y_mm"])
            height_mm = max(0.15, float(row.get("Height_mm") or 0.30))
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue

        rotation_deg = _optional_float(row, "Rotation_deg") or 0.0

        text = pcbnew.PCB_TEXT(board)
        text.SetText(refdes)
        text.SetPosition(
            pcbnew.VECTOR2I(pcbnew.FromMM(x_mm), pcbnew.FromMM(y_mm))
        )
        text.SetLayer(layer)
        text.SetTextSize(
            pcbnew.VECTOR2I(
                pcbnew.FromMM(height_mm),
                pcbnew.FromMM(height_mm),
            )
        )
        text.SetTextThickness(pcbnew.FromMM(max(0.025, height_mm * 0.10)))
        text.SetTextAngle(pcbnew.EDA_ANGLE(rotation_deg, pcbnew.DEGREES_T))
        board.Add(text)
        added_text += 1
        if _add_pointer(board, layer, row):
            added_pointers += 1

    if added_text == 0:
        raise ValueError(
            f"No valid silkscreen labels could be applied from {file_path}"
        )

    pcbnew.SaveBoard(brd_path, board)
    result = {
        "removed": removed,
        "hidden_references": hidden_references,
        "added_text": added_text,
        "added_pointers": added_pointers,
        "skipped": skipped,
    }
    print(
        "Silkscreen update complete: "
        f"{added_text} labels, {added_pointers} pointers, "
        f"{skipped} skipped, {removed} previous items removed, "
        f"{hidden_references} footprint references hidden."
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Apply generated silkscreen labels to a KiCad PCB."
    )
    parser.add_argument("board", help="Input/output .kicad_pcb path")
    parser.add_argument(
        "--labels",
        default=None,
        help="labels_output.txt path; defaults beside the board",
    )
    parser.add_argument(
        "--coordinates",
        choices=("kicad",),
        default="kicad",
        help="The generator/server integration uses native KiCad coordinates.",
    )
    args = parser.parse_args()
    board_path = os.path.abspath(args.board)
    labels_path = args.labels or os.path.join(
        os.path.dirname(board_path), "labels_output.txt"
    )
    add_refdes_to_silkscreen(board_path, labels_path)


if __name__ == "__main__":
    main()
