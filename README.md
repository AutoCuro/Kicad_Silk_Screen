# KiCad Silkscreen Generator

Automatic reference-designator (RefDes) placement for KiCad PCB silkscreen.

Give it a routed KiCad board and it works out where each component's label can
go, keeping clear of pads, vias, traces, footprints and the board edge, and
drawing pointer lines where a label sits away from its part. The result is
written into a copy of the board, ready to open in KiCad.

This is an open-source project developed by **AutoCuro**. See [LICENSE](LICENSE).

---

## What it does

- **Parses the board directly.** A built-in S-expression parser reads the
  `.kicad_pcb` file: footprints, pads, tracks, arcs, vias, zones and the
  `Edge.Cuts` outline. Placement does not need KiCad installed.
- **Places labels near their parts.** Labels go next to their component where
  possible, clear of pads, vias, other footprints and the board edge. Text runs
  along the part: horizontal for a wide footprint, rotated 90° for a tall one,
  with the other orientation tried next.
- **Groups similar parts.** Parts sharing a reference prefix, a matching shape
  (within 8%) and close spacing are treated as one family, and its members
  favour the same label side and orientation. The patterns found — rows, mirror
  pairs and twin pairs — are listed in the placement report.
- **Leaves breathing room.** A second pass moves labels that are too cramped
  if a roomier spot exists.
- **Handles crowded parts.** Where there is no nearby space, the label moves to
  the closest open area, found by an outward ring search, and a connector line
  routed around obstacles links it back to its part.
- **Overflow strip.** Labels with no room left on the board are placed just
  outside the edge, where they can be moved in by hand.
- **Works on large boards.** A fast cell grid is used for normal boards. Very
  large or dense boards switch to a lower-memory geometry backend
  (shapely + STRtree).
- **Updates the board safely.** The original board is never modified. The
  labels are written into a new `updated_pcb_file_w_silk.kicad_pcb`, and the
  Fab layers are hidden in its project settings so the silkscreen is easy to
  review.

## Repository contents

| File | Runs under | Purpose |
|---|---|---|
| `silk_screen_generator.py` | Any Python 3 environment | Parses the board, computes label placement, writes outputs, and calls the updater |
| `Silkscreen_Script.py` | KiCad's bundled Python (`pcbnew`) | Applies the computed labels and pointer lines to a `.kicad_pcb` |

## Requirements

**Generator**
- Python 3.9+
- `numpy`
- `shapely` 2.x
- `bokeh`

```bash
pip install -r requirements.txt
```

**Board updater**
- KiCad 9 (developed against 9.0; KiCad 7+ `pcbnew` APIs are used). The
  script must be run with KiCad's own Python interpreter, e.g.
  `C:\Program Files\KiCad\9.0\bin\python.exe` on Windows.

## Usage

### Single board

Put the board in a folder as `updated_pcb_file.kicad_pcb`, then:

```bash
python silk_screen_generator.py path/to/board_folder \
    --updater-script path/to/Silkscreen_Script.py \
    --kicad-python "C:\Program Files\KiCad\9.0\bin\python.exe"
```

Or point at any board file directly:

```bash
python silk_screen_generator.py --kicad path/to/my_board.kicad_pcb \
    --updater-script path/to/Silkscreen_Script.py
```

> **Note:** the built-in default for `--updater-script` is a machine-specific
> path. Always pass `--updater-script` pointing at the `Silkscreen_Script.py` in
> this repository.

### Batch mode

Point the generator at a parent folder. It finds every
`updated_pcb_file.kicad_pcb` below it and processes each board in turn, each
with its own timeout:

```bash
python silk_screen_generator.py path/to/boards_root --recursive --timeout 200 \
    --updater-script path/to/Silkscreen_Script.py
```

A summary is written to `silkscreen_batch_report.json` in the root folder.

### Generate outputs only

```bash
python silk_screen_generator.py path/to/board_folder --skip-board-update
```

### Apply labels manually

```bash
"C:\Program Files\KiCad\9.0\bin\python.exe" Silkscreen_Script.py board.kicad_pcb \
    --labels labels_output.txt
```

This modifies the given board **in place**. Work on a copy.

### Command-line options

| Option | Description |
|---|---|
| `board_folder` | Board folder, or parent folder for batch mode (default: current directory) |
| `--kicad` | Explicit `.kicad_pcb` path |
| `--pcb-output` | Output board path (default `updated_pcb_file_w_silk.kicad_pcb`) |
| `--txt-output`, `--json-output`, `--html-output`, `--placement-report-output` | Override output paths (single-board mode only) |
| `--updater-script` | Path to `Silkscreen_Script.py` |
| `--kicad-python` | Path to KiCad's Python interpreter |
| `--skip-board-update` | Write label files only; do not create a new board |
| `--recursive` | Force batch mode |
| `--timeout` | Seconds allowed per board in batch mode (default 200) |
| `--report-output` | Batch report path |
| `--txt-coordinates {kicad,altium}` | Coordinates for the label file (default `kicad`) |
| `--strict-trace-clearance` | Never allow labels over solder-masked traces |
| `--show-fab-layers` | Keep F.Fab / B.Fab visible in the output project |
| `--show` | Open the interactive plot after saving |

## Outputs

Written next to the input board:

| File | Contents |
|---|---|
| `labels_output.txt` | Label placements (CSV, see below) |
| `silkscreen_data.json` | Full placement data for every label |
| `silkscreen_report.html` | Placement summary, including any labels put on the overflow strip |
| `silkscreen_bokeh.html` | Interactive board plot with labels and pointer lines |
| `updated_pcb_file_w_silk.kicad_pcb` | Copy of the board with the new silkscreen |
| `updated_pcb_file_w_silk.kicad_prl` | Project-local settings with Fab layers hidden |

### Label file format

`labels_output.txt` is comma-separated with a header row. Units are
millimetres in KiCad board coordinates (Y points down):

```
RefDes,X_mm,Y_mm,Width_mm,Height_mm,Layer,Rotation_deg,ArrowStartX_mm,ArrowStartY_mm,ArrowEndX_mm,ArrowEndY_mm,ExtraArrowSegments_mm
```

- `Layer` is `TOP` or `BOTTOM`.
- The arrow columns are empty when no pointer line is needed.
- `ExtraArrowSegments_mm` holds the remaining segments of a routed connector as
  `x0 y0 x1 y1;x0 y0 x1 y1;...`.

## Tuning

Placement behaviour is controlled by the constants at the top of
`silk_screen_generator.py`. Each one is documented inline. The most useful are:

| Constant | Default | Effect |
|---|---|---|
| `SILK_TEXT_HEIGHT_MM` | 0.60 | Label text height |
| `SILK_FOOTPRINT_CLEARANCE_MM` | 0.20 | Keep-out around each footprint |
| `SILK_MAX_DISTANCE_MM` | 1.0 | How far a label may sit from its part before counting as "far" |
| `SILK_FAR_MAX_DISTANCE_MM` | 30.0 | Maximum search distance |
| `SILK_ARROW_THRESHOLD_MM` | 1.0 | Distance beyond which a pointer line is drawn |
| `SILK_FAMILY_CLUSTER_GAP_MM` | 3.0 | Spacing used to group similar parts |
| `SILK_ESCAPE_PREFERENCE_REACH_MM` | 10.0 | Extra distance a rescued label may travel for a roomier spot |
| `SILK_MAX_GRID_CELLS` | 5,000,000 | Grid size above which the geometry backend is used |

## Known limitations

- **Routed connectors:** `Silkscreen_Script.py` currently draws only the
  first segment of a multi-segment connector. The `ExtraArrowSegments_mm`
  column is not yet applied to the board.
- **Altium coordinates:** `--txt-coordinates altium` is for exporting label
  files only. Combine it with `--skip-board-update`, because the board updater
  accepts native KiCad coordinates only.
- **Cleanup of previous runs:** when re-applied, the updater removes silk
  text that matches a reference designator, and silk lines exactly 0.10 mm or
  0.025 mm wide. Hand-drawn silkscreen with those exact properties would also
  be removed.

## Contributing

Issues and pull requests are welcome. Please describe the board scenario
(ideally with a sample `.kicad_pcb`) when reporting a placement problem.

## License

Released under the MIT License. See [LICENSE](LICENSE).

Copyright © 2026 AutoCuro.
