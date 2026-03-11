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
