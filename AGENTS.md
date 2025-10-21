# Codex Agent Guidelines

## Project
- Language(s): Python, JavaScript/TypeScript.
- Package managers: pip/uv, npm/pnpm/yarn.
- Tests: pytest or vitest/jest; prefer adding/adjusting tests with any change.

## Style/Tools
- Python: black, ruff, pytest.
- JS/TS: eslint, prettier, jest/vitest.
- CI: GitHub Actions basic test workflow.

## Guardrails
- May READ any repo file.
- May PROPOSE diffs; WRITING files and RUNNING shell commands require approval.
- No destructive ops (rm -rf, drop DBs, force pushes).
- Prefer small, reviewable commits with clear messages.

## Preferences
- Aim for clear, self-documenting code; minimal dependencies.
- Add/adjust tests for new behavior; keep total coverage ≥ pre-change.
