# MYNAH Development Rules

Development rules:
- no backward compatibility unless explicitly required
- clean up unused code when touching a related area
- no fallback behavior unless explicitly required
- prefer direct, simple implementations over abstraction-heavy designs
- keep dependencies low and justify each one
- keep security boundaries explicit
- keep behavior inspectable and auditable
- commit after each meaningful increment

Working expectations:
- remove stale code and stale docs instead of layering contradictions
- prefer rewriting unclear structures over patching around them
- fail clearly instead of failing open
- keep the common path cheap and simple
- document major architectural changes in `SPEC.md`
- prefer tests and harnesses that clean up automatically and can be rerun without manual reset

## Closed-Loop Development

- prefer development loops where behavior can be tested immediately
- build the smallest runnable harness before expanding scope
- isolate one module or agent behavior at a time
- debug with real inputs whenever possible
- avoid coupling new work to unrelated infrastructure too early
- use `EVALS.md` for experiment and evaluation rules
