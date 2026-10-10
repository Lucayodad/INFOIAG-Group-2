"""Section 2: local ontology queries with a small, explicit inference subset.

This module does not perform complete OWL consistency checking. It never
rewrites the ontology file, creates factual claim assessments, or reads human
reference summaries. Query results are ontology assertions, not verified facts.
"""

from pathlib import Path
import re

from rdflib import BNode, Graph, Literal, Namespace, RDF, RDFS, OWL, URIRef


ROOT = Path(__file__).resolve().parent
DEFAULT_ONTOLOGY = ROOT / "ontology" / "group2_Ontology.owl"
NS = Namespace("http://www.semanticweb.org/group2/ontology#")
QUERY_DEFAULTS = {
    "Q1": {"company": "McDonalds"},
    "Q2": {"target": "McDonalds"},
    "Q3": {"match": "SuperBowlMatch1"},
    "Q4": {"person": "Obama"},
    "Q5": {},
    "Q6": {},
}
EVIDENCE_NOTE = (
    "Results describe the supplied ontology, not independently verified reality. "
    "It includes illustrative entities and undated assertions. No match means "
    "unknown or absent in this graph; it does not mean false."
)


class OntologyError(ValueError):
    """Invalid ontology input, query identifier, or tool argument."""


def copy_graph(source: Graph) -> Graph:
    result = Graph()
    for triple in source:
        result.add(triple)
    return result


def limited_closure(source: Graph) -> Graph:
    """Apply only inverse, symmetric, transitive, and named subclass rules.

    Deliberately excludes sameAs, functional/inverse-functional identity rules,
    domain/range typing, equivalent classes, datatype restrictions, and general
    cardinality reasoning. Use a complete OWL reasoner for those requirements.
    """
    graph = copy_graph(source)
    inverses = list(source.subject_objects(OWL.inverseOf))
    symmetric = list(source.subjects(RDF.type, OWL.SymmetricProperty))
    transitive = list(source.subjects(RDF.type, OWL.TransitiveProperty))
    subclasses = [
        (child, parent) for child, parent in source.subject_objects(RDFS.subClassOf)
        if isinstance(child, URIRef) and isinstance(parent, URIRef)
    ]
    while True:
        additions = set()
        for left, right in inverses:
            for subject, obj in graph.subject_objects(left):
                if not isinstance(obj, Literal):
                    additions.add((obj, right, subject))
            for subject, obj in graph.subject_objects(right):
                if not isinstance(obj, Literal):
                    additions.add((obj, left, subject))
        for prop in symmetric:
            for subject, obj in graph.subject_objects(prop):
                if not isinstance(obj, Literal):
                    additions.add((obj, prop, subject))
        for prop in transitive:
            for subject, middle in graph.subject_objects(prop):
                for obj in graph.objects(middle, prop):
                    additions.add((subject, prop, obj))
        for child, parent in subclasses:
            for instance in graph.subjects(RDF.type, child):
                additions.add((instance, RDF.type, parent))
        additions.difference_update(graph)
        if not additions:
            return graph
        for triple in additions:
            graph.add(triple)


def encode_term(term) -> dict:
    """JSON-safe RDF term preserving full identifiers and literal datatypes."""
    if isinstance(term, URIRef):
        value = str(term)
        return {"type": "uri", "value": value,
                "label": value[len(str(NS)):] if value.startswith(str(NS)) else value}
    if isinstance(term, Literal):
        return {"type": "literal", "value": str(term),
                "datatype": str(term.datatype) if term.datatype else None,
                "language": term.language}
    if isinstance(term, BNode):
        return {"type": "bnode", "value": str(term)}
    raise OntologyError(f"Unsupported RDF term: {term!r}")


class OntologyTools:
    def __init__(self, path: Path = DEFAULT_ONTOLOGY):
        self.path = Path(path)
        try:
            raw = self.path.read_bytes()
            graph = Graph().parse(data=raw, format="xml")
        except Exception as exc:
            raise OntologyError(f"Cannot load RDF/XML ontology {self.path}: {exc}") from exc
        if not (NS.Person, RDF.type, OWL.Class) in graph:
            raise OntologyError("Expected the group2 ontology namespace and Person class.")
        self.asserted = graph
        self.inferred = limited_closure(graph)

    @staticmethod
    def entity_iri(name: str) -> URIRef:
        """Accept a group2 local name or full group2 IRI; never interpolate SPARQL."""
        if not isinstance(name, str):
            raise OntologyError("Entity name must be a string.")
        local = name[len(str(NS)):] if name.startswith(str(NS)) else name
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", local):
            raise OntologyError(f"Invalid group2 entity identifier: {name!r}")
        return NS[local]

    def statistics(self) -> dict:
        return {
            "asserted_triples": len(self.asserted),
            "triples_after_limited_inference": len(self.inferred),
            "added_triples": len(self.inferred) - len(self.asserted),
            "named_classes": sum(isinstance(x, URIRef) for x in self.asserted.subjects(RDF.type, OWL.Class)),
            "object_properties": len(set(self.asserted.subjects(RDF.type, OWL.ObjectProperty))),
            "datatype_properties": len(set(self.asserted.subjects(RDF.type, OWL.DatatypeProperty))),
            "named_individuals": len(set(self.asserted.subjects(RDF.type, OWL.NamedIndividual))),
            "claim_assessments": len(set(self.inferred.subjects(RDF.type, NS.ClaimAssessment))),
        }

    def _select(self, graph: Graph, query: str, bindings: dict) -> list[dict]:
        return [
            {str(key): encode_term(value) for key, value in row.asdict().items()}
            for row in graph.query(query, initBindings=bindings)
        ]

    def run_query(self, query_id: str, bindings: dict | None = None,
                  use_inference: bool = True) -> dict:
        """Execute one of the six local templates, using RDF term bindings."""
        if query_id not in QUERY_DEFAULTS:
            raise OntologyError(f"Unknown query {query_id!r}; choose Q1 through Q6.")
        values = dict(QUERY_DEFAULTS[query_id])
        supplied = bindings or {}
        if set(supplied) - set(values):
            raise OntologyError(f"Allowed bindings for {query_id}: {sorted(values)}")
        values.update(supplied)
        terms = {key: self.entity_iri(value) for key, value in values.items()}
        query = (ROOT / "queries" / f"{query_id}.rq").read_text(encoding="utf-8")
        graph = self.inferred if use_inference else self.asserted
        rows = self._select(graph, query, terms)
        return {"query_id": query_id, "bindings": {k: str(v) for k, v in terms.items()},
                "reasoning": "limited" if use_inference else "asserted_only",
                "status": "matches" if rows else "no_matches", "rows": rows,
                "evidence_note": EVIDENCE_NOTE}

    def get_entity_facts(self, entity: str) -> dict:
        """Retrieve incoming/outgoing triples and distinguish inferred assertions."""
        iri = self.entity_iri(entity)
        triples = set(self.inferred.triples((iri, None, None)))
        triples.update(self.inferred.triples((None, None, iri)))
        return {"entity": str(iri), "status": "matches" if triples else "unknown_entity",
                "triples": [
                    {"subject": encode_term(s), "predicate": encode_term(p),
                     "object": encode_term(o),
                     "origin": "asserted" if (s, p, o) in self.asserted else "inferred"}
                    for s, p, o in sorted(triples, key=lambda t: tuple(map(str, t)))
                ], "evidence_note": EVIDENCE_NOTE}

    def get_claim_assessments(self, claim: str) -> dict:
        """Keep every source's assessment, including conflicting/incomplete ones."""
        query = """
        PREFIX : <http://www.semanticweb.org/group2/ontology#>
        SELECT ?assessment ?source ?valid WHERE {
            ?assessment :assessmentOf ?claim .
            OPTIONAL { ?assessment :assessedBy ?source . }
            OPTIONAL { ?assessment :isValid ?valid . }
        } ORDER BY ?assessment ?source ?valid
        """
        rows = self._select(self.inferred, query, {"claim": self.entity_iri(claim)})
        return {"claim": str(self.entity_iri(claim)),
                "status": "assessments_found" if rows else "no_assessment",
                "rows": rows, "evidence_note": EVIDENCE_NOTE}

    def check_candidate_types(self, candidate_types: list[dict]) -> dict:
        """Test hypothetical entity/class pairs in an isolated graph using Q6.

        Input: [{"entity": "ExampleCondition", "class": "Allergy"}, ...].
        No candidate triples are persisted. Their truth still needs source review.
        """
        temporary = copy_graph(self.asserted)
        for row in candidate_types:
            if not isinstance(row, dict) or set(row) != {"entity", "class"}:
                raise OntologyError("Each candidate needs exactly 'entity' and 'class'.")
            entity, cls = self.entity_iri(row["entity"]), self.entity_iri(row["class"])
            if (cls, RDF.type, OWL.Class) not in self.asserted:
                raise OntologyError(f"Unknown ontology class: {row['class']}")
            temporary.add((entity, RDF.type, cls))
        query = (ROOT / "queries" / "Q6.rq").read_text(encoding="utf-8")
        rows = self._select(limited_closure(temporary), query, {})
        return {"query_id": "Q6", "status": "conflicts_found" if rows else "no_disjoint_conflict_found",
                "rows": rows, "candidate_count": len(candidate_types),
                "scope": "Named disjoint classes only; not complete OWL consistency or bias checking."}
