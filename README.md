# UFP

UFP is the project utility package for the OpenSeesPy frame-shear-wall models in
`E:\openseespy\KJ`. It keeps reusable modeling, dynamic-analysis, recorder,
and post-processing logic outside individual notebooks.

## Stable Entry Point

Use the package-level API in notebooks:

```python
import ufp

ufp.apply_model_table(show_sections=False)
summary = ufp.analyze_modal_participation(num_modes=15)
ufp.apply_uniform_damping(summary, zeta=0.05)
```

`ufp.user_function` remains the implementation/compatibility module. Existing
`import ufp` calls are preserved. New project code should prefer `ufp.<name>`
instead of importing implementation helpers directly.

## Package Layout

- `ufp/__init__.py`: grouped, explicit public compatibility API.
- `ufp/user_function.py`: current implementations for modeling and analysis.
- `ufp/model_table_generator.py`: Excel-to-Python model generator used by
  `ufp.apply_model_table()`.
- `tests/`: standard-library regression and OpenSees smoke tests.
- `docs/ARCHITECTURE.md`: responsibilities and the safe future split plan.

The `ufp/backups` directory and `*.rar` files are local archives, not package
source. They are ignored by Git but are not deleted by this cleanup.

## Main Capability Groups

- Section registries, fiber sections, and layered-shell sections.
- Beam, column, wall, vertical-rebar, and shared-node mesh generation.
- Model connectivity checks and component visualization.
- Story self-weight, rigid diaphragms, gravity loads, and mass assignment.
- Excel-driven model generation and replacement-table overlays.
- Modal participation, Rayleigh/Uniform damping, and ground-motion processing.
- Native OpenSees recorders, transient analysis, plotting-data extraction, and
  frame-wall shear-share post-processing.

## Unit Contract

The structural models use **N-mm-s** unless a function explicitly states
otherwise:

- displacement: `mm`
- force: `N`
- stress: `MPa = N/mm^2`
- model acceleration: `mm/s^2`
- plotted floor acceleration: `Gal`, using `1 Gal = 10 mm/s^2`

For `UniformExcitation`, the raw OpenSees/ODB nodal acceleration is relative to
the ground. Native post-processing adds the ground-motion acceleration before
producing absolute floor acceleration and PFA in Gal. Do not compare an ODB
relative-acceleration cloud directly with absolute PFA tables.

## Generated-File Contract

`ufp.apply_model_table()` reads the active model workbook and regenerates:

- `table_sections.py`
- `model_story_*.py`

Those files are generated artifacts. Edit the Excel management tables or UFP
generator logic, then regenerate; do not maintain generated model code by hand.

## Regression Check

Run with the OpenSees environment from the repository root:

```powershell
E:\Users\Yujie\miniconda3\envs\opensees\python.exe -B -m unittest discover -s tests -v
```

The tests check API uniqueness, duplicate top-level definitions, pure parsing
helpers, tag helpers, and a minimal OpenSees node-registry workflow.
