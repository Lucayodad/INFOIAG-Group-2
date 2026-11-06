# Ontology input

`group2_Ontology.owl` is Luca's updated RDF/XML export (commit `5a6c46a`).
Pipeline execution preserves this file byte-for-byte.
The namespace is `http://www.semanticweb.org/group2/ontology#`.
The current loader accepts RDF/XML. It reads a local file and does not fetch imports.
Its in-memory inference and temporary candidate checks do not change this file.
See `ontology_tools.py` for the exact supported reasoning scope.

## Source evidence extension

`evidence_schema.ttl` extends the base `Tweet`/`Topic` vocabulary and relationships
`aboutTopic`/`containsClaim` with literal properties for
IDs, text, quotations, labels, extraction status, target, stance and provenance.
It reuses `Claim`, `hasText`, and `tweetHasText` from the OWL file.
The old `tweetText` declaration is retained as an equivalent legacy property.
Readers explicitly query either text property; they do not depend on OWL
equivalence inference. New graphs write only `tweetHasText`.

`../populate_ontology.py` builds each run from a filtered copy of the asserted
base plus the extension, then adds validated tweets and extracted claims.
`pipeline_base_graph` excludes base `Tweet`, `Claim`, `ClaimAssessment`, and
`Summary` instances, including named subclasses such as `SportsClaim`, and
triples pointing to them. In this export that removes five tweets, two claims,
and one assessment (32 triples). It retains all remaining schema/domain
assertions and the `SuperBowl` topic and `DivSummDataset` source references.
Standalone `OntologyTools` still exposes the complete unfiltered file.

`TOPIC_IRIS` in `ontology_tools.py` is shared by population and revision retrieval:
`Obama` maps to `Topic_Obama`, `McDonalds` to `Topic_McDonalds`, and the dataset
key `Superbowl` to the existing `SuperBowl` topic. The existing domain entity
`SuperBowl1` remains distinct from that topic, as does `Obama` from `Topic_Obama`.
New topic labels use `hasTopicName`; claims also link through `claimAboutTopic`.
Revision retrieval additionally accepts old saved `Topic_Superbowl` links.
Dialect labels are literal
`dialectLabel` properties on tweets, following the team's architecture decision;
there is no dialect class, group individual or group-membership object property.
The agent works on a copy of the graph without them: they are evaluation data.
Tweet/claim IRIs use a unique run namespace; local IDs are also retained
as literals. Graph population does not assert that extracted claims are true.
The saved manifest records `base_evidence_policy` so the filtering policy is
visible alongside source paths and run counts. Counts describe run evidence,
not the source file's examples.

`ontology_tools.py` loads the complete supplied OWL file. See the repository
README for commands.

Existing Turtle files are not rewritten. Regenerate any graph built from the
unfiltered updated base before using it for run-wide evidence counts or checks.
Legacy text/topic spellings remain readable. Equal alias values do not inflate
support counts; conflicting text values are flagged and cannot support a
definitive factual judgment.
