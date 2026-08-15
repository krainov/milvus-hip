# Test feature notes

Scratch document used to validate the repository workflow (branching, commits,
pull requests) on the `v2.5.4` fork line. It contains **no functional changes**:
nothing here is compiled, packaged or read at runtime.

## Scope

- Branch: `test_feature`
- Base: `v2.5.4` (default branch of this fork)
- Change type: documentation only

## Checklist

- [x] Branch created from the fork default branch
- [x] Commits touch documentation only (`docs/*.md`)
- [x] No changes under `internal/`, `pkg/`, `cmd/`, `client/` or `build/`
- [ ] Pull request opened against `v2.5.4`
- [ ] Branch deleted after the workflow check is done
