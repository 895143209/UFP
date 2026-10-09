# UFP Architecture

## Current Compatibility Design

The KJ notebooks and generated `model_story_*.py` files call the package-level
`ufp` API. Therefore `ufp/__init__.py` is the compatibility boundary. Public
names should not be removed or renamed without first updating every generated
model and adding a migration period.

`user_function.py` is intentionally retained during the first cleanup. It has
several strongly coupled concerns and uses OpenSees commands in the module
namespace. Moving functions before regression coverage would create more risk
than value.

## Responsibility Map

The implementation currently follows these consecutive groups:

1. tags and cyclic-analysis helpers
2. section registries and section builders
3. beam, column, wall, and reinforcement meshing
4. visualization and connectivity diagnostics
5. gravity, story weight, diaphragms, and mass
6. table-driven generation and replacement overlays
7. modal analysis, damping, and ground motions
8. native recorders, transient analysis, and post-processing

## Safe Second-Phase Split

Once representative Standard/Defective/Replacement smoke models are automated,
`user_function.py` can be split behind the unchanged `ufp` facade:

- `sections.py`
- `mesh.py`
- `model_checks.py`
- `gravity_mass.py`
- `table_model.py`
- `dynamics.py`
- `ground_motion.py`
- `native_recorders.py`
- `postprocess.py`

Dependencies should point from high-level workflows toward low-level helpers;
post-processing must not import model-generation modules. Shared registries and
unit constants should live in a small `core.py` module.

## Compatibility Rules

- Keep `import ufp` working for all existing notebooks.
- Preserve public function signatures unless a migration is documented.
- Keep N-mm-s as the default unit system.
- Label relative and absolute acceleration explicitly.
- Never silently change recorder column order or Plot Data schemas.
- Regenerate table-driven files after generator changes and diff the outputs.
- Run `python -m unittest discover -s tests -v` before syncing package changes.

## Repository Hygiene

Python caches, egg-info output, local backups, and RAR archives are not source.
The cleanup adds ignore rules but deliberately does not delete or untrack the
user's existing files. That can be handled in a separate reviewed Git cleanup.
