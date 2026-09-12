# Contributing

## Branches

Create focused branches from the current default branch. Use lowercase kebab-case
with a Conventional Commits-style prefix:

```text
<type>/<short-description>
```

Examples:

```text
feat/ostium-candidate-detection
fix/physical-coordinate-conversion
test/cropped-aorta-ends
docs/offline-installation
chore/update-wheelhouse
```

Allowed branch prefixes are:

- `feat/` for new functionality.
- `fix/` for bug fixes.
- `perf/` for runtime or memory improvements.
- `refactor/` for behavior-preserving code changes.
- `test/` for test-only changes.
- `docs/` for documentation-only changes.
- `build/` for dependency or packaging changes.
- `ci/` for automation changes.
- `chore/` for repository maintenance.

Keep each branch limited to one coherent change. Do not include issue numbers,
initials, or personal names unless the team explicitly adopts that convention.

## Commits

Use the Conventional Commits structure:

```text
<type>: <imperative summary>
```

Examples:

```text
feat: add ostium candidate extraction
fix: preserve SimpleITK physical direction
perf: limit vessel filtering to the aortic ROI
test: cover mismatched image geometry
docs: clarify offline wheel installation
build: refresh Windows CPython 3.14 wheelhouse
```

Use one of these commit types:

- `feat`: introduce functionality.
- `fix`: correct faulty behavior.
- `perf`: improve runtime or memory usage.
- `refactor`: restructure code without changing behavior.
- `test`: add or update tests.
- `docs`: change documentation only.
- `build`: change dependencies, packaging, or build tooling.
- `ci`: change continuous-integration configuration.
- `chore`: perform maintenance not covered above.
- `revert`: revert a previous commit.

Commit summaries must be imperative, lowercase after the colon, concise, and
without a trailing period. Add a body when the reason, tradeoff, or evaluation
impact is not obvious. Mark breaking changes with `!` before the colon and add a
`BREAKING CHANGE:` footer.

## Before opening a pull request

Run the checks appropriate to the change:

```text
python -m unittest discover -s tests -v
python -m pip check
git diff --check
```

For backend changes, also run the evaluator command against a local case. For
frontend changes, confirm the Streamlit application starts offline. See
`AGENTS.md` for the complete project constraints and verification commands.

Pull requests should explain what changed, why it changed, how it was verified,
and any known runtime, memory, or detection-quality impact.
