# Source selection

Prepared from selected application source, with fresh public Git history.
The public snapshot was reviewed and hardened on September 24, 2026.

Included: team identity resolution, canonical schema utilities, feature builders,
LightGBM training, calibration, eligibility, sizing, dashboard grouping, and
selected regression tests. Original module paths are retained for readability.

Excluded: credentials, provider datasets, operational databases, trained models,
betting records, screenshots, source film, annotations, deployment jobs,
internal instructions, and prior Git history. The full application and live
data ingestion are not part of this repository.

Publication adaptations:

- Root discovery uses `README.md` instead of an internal context file.
- The canonical builder registry is empty because ingestion builders are outside
  this selection. Shared paths and entity-resolution behavior remain available.
- Internal manifest references are replaced with an explicit data requirement.
- Added a synthetic example, focused dependency pins, public documentation, and
  Linux CI. Existing selected tests use in-memory or temporary data.

Model calculations are retained. The offline example and tests establish the
documented code contracts; they do not reproduce historical research results.
Training and feature modules require user-supplied schemas, data, and manifests.

## Publication hardening

All financial fixtures are invented. Source comments retain technical rationale
without operational incident narratives. Public Git authorship uses the account
handle and a noreply address. The publication file allowlist, pinned CI actions,
and full-history secret scan enforce the boundary described in
[the publication policy](PUBLICATION_POLICY.md). Public CLAUDE.md and AGENTS.md
contain only general maintenance rules, not private session history.

Models now use LightGBM's native text format and calibrators use validated JSON
thresholds. Executable pickle loading is not supported in this public version.
The feature builder includes its required atomic-write helper. Dynamic SQL
identifiers are validated before query construction. These changes affect
persistence and input validation; scoring formulas remain unchanged.
