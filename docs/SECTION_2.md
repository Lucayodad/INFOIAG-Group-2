# Section 2: ontology loading and query tools

## What is completed

- The supplied `group2_Ontology.owl` is loaded as an RDFLib graph.
- Asserted triples and a graph with limited inferences are held separately.
- Q1-Q5 from report section 4.1 are executable SPARQL templates. Fixed target
  identifiers are parameterized with `initBindings`; prefixes, DISTINCT, and
  ordering make them standalone and give predictable output.
- Q6 implements the report's Allergy/Injury example as a query for membership
  in declared disjoint named classes, including named subclass paths.
- Tools return JSON-serializable RDF terms, full IRIs, and status information.
- Candidate type assertions can be checked in a temporary graph.
- The supplied ontology is unchanged. There is no LLM or summary generation yet.

RDFLib executes the SPARQL, while the small `limited_closure` function implements
the inference rules listed below. RDFLib alone does not automatically perform
OWL reasoning just because an OWL file has been loaded.

## Actual ontology inventory

| Item | Count |
|---|---:|
| Asserted triples | 567 |
| Triples after limited inference | 591 |
| Added triples | 24 |
| Named classes | 81 |
| Object properties | 60 |
| Datatype properties | 26 |
| Named individuals | 20 |
| ClaimAssessment instances | 0 |

## Query results

| Query | Result on supplied ontology |
|---|---|
| Q1: employees | chef1, waitress1 |
| Q2: partners | OtherCompany |
| Q3: match result and ritual | team1, team2, HalftimeShow |
| Q4: occupation and friend | president, Person1 |
| Q5: claims assessed false | No matches; no assessments are populated |
| Q6: disjoint class conflict | No matches in the supplied graph |

Q1 and Q4 return no matches in asserted-only mode. Q1 needs the inverse
`employs`/`worksFor` relation; Q4 needs symmetric `friendsWith`.

Q6 returns a conflict when the report's hypothetical ExampleCondition is typed
as both Allergy and Injury in a separate temporary graph. No Q6 matches does
not establish overall ontology consistency or an unbiased summary.

## Tool usage

Run from the project directory after installing requirements:

```python
from ontology_tools import OntologyTools

tools = OntologyTools()

# Default report query, then a parameterized query:
employees = tools.run_query("Q1")
partners = tools.run_query("Q2", {"target": "OtherCompany"})

# Retrieve structured entity evidence and all source-specific assessments:
facts = tools.get_entity_facts("Obama")
assessments = tools.get_claim_assessments("SomeClaim")

# Hypothetical type assertions; nothing is written to the ontology:
check = tools.check_candidate_types([
    {"entity": "ExampleCondition", "class": "Allergy"},
    {"entity": "ExampleCondition", "class": "Injury"},
])
print(check["status"])  # conflicts_found
```

`run_query` only chooses among Q1-Q6. It does not accept arbitrary generated
SPARQL. Identifiers are passed as RDF terms, not substituted into query strings.
`get_entity_facts` marks each triple as asserted or inferred. Query outputs
specify whether the graph uses limited inference but do not include a full
logical proof for each row. `get_claim_assessments` retains disagreeing sources
and incomplete assessments instead of merging them into one truth value.

## Reasoning scope

Implemented rules:

1. If two properties are inverse, reverse matching assertions in either direction.
2. If a property is symmetric, add the reverse assertion.
3. If a property is transitive, add the implied two-hop relationship, repeating
   until the selected rules produce no new triples.
4. Propagate instance membership through named `rdfs:subClassOf` relationships.

This is deliberately a small subset, not a full OWL or OWL RL reasoner. It does
not reason over domain/range, datatype facets such as age constraints,
equivalent/intersection/union class definitions, cardinalities, or identity
inferences from functional/inverse-functional properties. It also does not
detect arbitrary natural-language contradictions between summary sentences.
Those require additional validation or a suitable complete reasoner.

## What the results mean for the agent

The current graph includes illustrative entities such as team1, team2,
OtherCompany, and Person1, plus undated assertions such as Obama's age of 55.
The tools expose these as ontology assertions, not verified present-day facts.
Do not automatically turn team1 into the Ravens, overwrite historical tweet
content from the age value, or turn the Rational trait assertion into an
objective claim about Obama's personality.

Q5 returning nothing does not mean every claim is true. An assessment's
`isValid` value belongs to its `assessedBy` source. The tools do not fabricate
ClaimAssessment instances or set `isValid=false` simply because evidence is
missing. Future claim extraction and checking must keep source attribution.

The supplied ontology has no tweet/dialect/summary-sentence coverage links.
Consequently, the report's group coverage check cannot yet run. A later section
needs a temporary evidence representation linking each tweet's dataset label,
extracted claim, target, and supporting summary sentence. The three dialect
categories are already present in the data; do not infer them from writing style.

The report mentions flagging every unrepresented claim. A five-sentence summary
cannot reproduce all 90 tweets. In the later coverage stage, distinguish normal
compression from omission of a salient distinct viewpoint before requesting a
rewrite. Group coverage is not simply matching equal sentence counts.

For target checking, note that `relatedTo` has a union range of Company, Event,
Person, and Sport. Advertisement and Competition are not explicitly in that
union, and Competition is not declared a subclass of Event in this export.
This warrants a modeling decision before encoding those targets under that
property; the code has not silently changed the ontology's range.

## Validation

Eight tests passed using Python 3.14, RDFLib 7.3.0, and pyparsing 3.3.3. Tests
check Q1-Q4 against the report, absent and conflicting assessments, isolated
Q6 conflicts, inherited class membership, transitivity, evidence provenance,
and preservation of the source file. The original section-1 command also
still loads all 270 tweets. Pyparsing may print deprecation warnings originating
from RDFLib during unittest; they do not change these test results.

Captured machine-readable results are in `section2_results.json`. The fixture
claims and conditions in tests are synthetic and never added to the deliverable
ontology. The packaged requirements pin the versions used for this check.

Sources: supplied `report_AI.pdf`, sections 3.3 and 4; supplied
`group2_Ontology.owl`; RDFLib's SPARQL documentation:
https://rdflib.readthedocs.io/en/7.3.0/intro_to_sparql.html

## Pause point

Section 2 stops here. Next, after the user's response, connect the chosen LLM
and generate the initial five-sentence summary. The later checker and revision
loop will call the tools implemented here.
