# MYNAH Evaluation Rules

Evaluation is part of development.
Do not treat it as a final reporting step after implementation is already fixed.

## Evaluation Principles

- validate hypotheses, not only current outputs
- compare approaches against alternatives, not only against logs
- evaluate one harness or module at a time when possible
- start with the earliest high-impact module in the pipeline
- optimize one primary vector at a time:
  - accuracy
  - latency
  - cost
  - code quality
- keep the other vectors acceptable while optimizing the primary one

## Experiment Design

- isolate the target module as a clean black box with explicit inputs and outputs
- expose only a small number of experiment knobs at once
- use explicit parameters or feature flags for model, prompt, tool, or behavior changes
- keep experiments repeatable
- prefer evals and staged interaction tests that clean up after themselves
- test runs must be rerunnable without manual reset steps
- disable persistence, caching, and side effects unless they are part of what is being tested
- make sure one test case does not affect another

## Test Case Design

- prefer real production inputs when available
- for stateful agent tests, prefer staged human-like interaction sequences that exercise session creation, user identification, memory routing, and recall
- log inputs and outputs in structured form from the start
- if production data is missing, create deliberate hand-written or synthetic cases
- keep datasets deduplicated and diverse
- avoid overweighting one narrow subdomain
- prefer grounded expected outputs over vague scoring when possible

## Test Hygiene

- evals should leave no durable artifacts unless persistence is the explicit thing being tested
- if persistence is part of the test, isolate it in a temporary sandbox and remove it afterward
- temp files, temp databases, and temp agent state should be owned by the test harness and cleaned automatically
- one failed run should not poison the next run
- manual cleanup steps are a smell and should be avoided

## Live Model Checks

- keep live model smoke tests separate from deterministic contract tests
- live tests should be opt-in so the default suite stays fast, cheap, and rerunnable
- use broad behavior assertions for live tests rather than brittle exact phrasing checks
- live tests must still use temporary sandboxes and leave no durable local artifacts

## Metrics

- prefer deterministic metrics wherever possible
- use LLM-as-judge only where deterministic checks are not enough
- always record:
  - wall time
  - token usage
  - cost
  - error rate
- for retrieval-like systems, use retrieval metrics such as precision, recall, or overlap
- for structured outputs, validate format directly
- for task agents, inspect tool-use behavior and success in verifiable environments

## Interpreting Results

- compare new approaches against both the current baseline and realistic alternatives
- visualize results instead of relying only on raw logs
- prefer consistency and distribution quality, not only averages
- eliminate approaches that fail too often even if they are cheaper or faster
- keep reusable eval harnesses so new models or implementation changes can be tested quickly later
