# Ontology input

`group2_Ontology.owl` is the user's supplied RDF/XML export, preserved byte-for-byte.
The namespace is `http://www.semanticweb.org/group2/ontology#`.
The current loader accepts RDF/XML. It reads a local file and does not fetch imports.
Its in-memory inference and temporary candidate checks do not change this file.
See `../docs/SECTION_2.md` for the exact supported reasoning scope.

## Source evidence extension

`evidence_schema.ttl` declares `Tweet`, `Topic`, the relationships
`aboutTopic`, `containsClaim`, and literal properties for
IDs, text, quotations, labels, extraction status, target, stance and provenance.
It reuses `Claim` and `hasText` from the original OWL file.

`../populate_ontology.py` loads both files into a fresh graph for each run, then
adds validated tweets and extracted claims. The original RDF/XML export remains
unchanged. Topic individuals have separate names such as `Topic_Obama`, avoiding
collisions with domain individuals such as `Obama`. Dialect labels are literal
`dialectLabel` properties on tweets, following the team's architecture decision;
there is no dialect class, group individual or group-membership object property.
Tweet/claim IRIs use a unique run namespace; local IDs are also retained
as literals. Graph population does not assert that extracted claims are true.

The standalone ontology query command loads only the original OWL file. To
inspect populated evidence, use `../queries/source_evidence.rq` on the returned
graph or on its saved Turtle file. See the repository README for commands.
