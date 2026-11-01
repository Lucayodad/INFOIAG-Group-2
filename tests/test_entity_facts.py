"""Asserted entity retrieval checks; graph assertions are not truth judgments."""

import hashlib
import json
import unittest

from rdflib import BNode, Graph, Literal, RDF, URIRef, XSD

from fact_tools import lookup_entity_facts
from ontology_tools import DEFAULT_ONTOLOGY, NS, OntologyError, OntologyTools


class EntityFactTests(unittest.TestCase):
    def test_existing_obama_assertions_and_source_file_are_unchanged(self):
        before_file = hashlib.sha256(DEFAULT_ONTOLOGY.read_bytes()).hexdigest()
        graph = Graph().parse(DEFAULT_ONTOLOGY, format="xml")
        before_graph = set(graph)
        result = lookup_entity_facts(graph, "Obama")
        self.assertEqual(result["status"], "matches")
        self.assertEqual(result["entity"], str(NS.Obama))
        self.assertEqual(result["scope"], "asserted_only")
        expected = set(graph.triples((NS.Obama, None, None)))
        expected.update(graph.triples((None, None, NS.Obama)))
        self.assertEqual(len(result["triples"]), len(expected))
        age = next(row for row in result["triples"]
                   if row["predicate"]["value"] == str(NS.hasAge))
        self.assertEqual(age["object"], {
            "type": "literal", "value": "55", "datatype": str(XSD.integer), "language": None,
        })
        for row in result["triples"]:
            self.assertEqual(row["provenance"]["origin"], "asserted")
            self.assertIsNone(row["provenance"]["source_file"])
            self.assertEqual(row["provenance"]["graph_identifier"],
                             {"type": "bnode", "value": str(graph.identifier)})
        json.dumps(result)
        self.assertEqual(set(graph), before_graph)
        self.assertEqual(hashlib.sha256(DEFAULT_ONTOLOGY.read_bytes()).hexdigest(), before_file)

    def test_incoming_outgoing_and_self_loop_are_preserved(self):
        graph = Graph(identifier=URIRef("https://example.org/asserted-fixture"))
        graph.add((NS.Person1, NS.friendsWith, NS.Obama))
        graph.add((NS.Obama, NS.hasOccupation, NS.president))
        graph.add((NS.Obama, NS.friendsWith, NS.Obama))
        result = lookup_entity_facts(graph, "Obama")
        self.assertEqual(len(result["triples"]), 3)
        incoming = next(r for r in result["triples"] if r["subject"]["value"] == str(NS.Person1))
        outgoing = next(r for r in result["triples"] if r["object"]["value"] == str(NS.president))
        loop = next(r for r in result["triples"]
                    if r["subject"]["value"] == r["object"]["value"])
        self.assertEqual(incoming["directions"], ["incoming"])
        self.assertEqual(outgoing["directions"], ["outgoing"])
        self.assertEqual(loop["directions"], ["outgoing", "incoming"])
        self.assertEqual(loop["provenance"]["graph_identifier"]["value"], str(graph.identifier))

    def test_explicit_full_iri_and_local_name_are_equivalent(self):
        graph = Graph().parse(DEFAULT_ONTOLOGY, format="xml")
        self.assertEqual(lookup_entity_facts(graph, "Obama"),
                         lookup_entity_facts(graph, str(NS.Obama)))

    def test_unknown_entity_is_absent_not_false(self):
        graph = Graph().parse(DEFAULT_ONTOLOGY, format="xml")
        result = lookup_entity_facts(graph, "UnknownPerson")
        self.assertEqual(result["status"], "unknown_entity")
        self.assertEqual(result["triples"], [])
        self.assertNotIn("isValid", result)
        self.assertNotIn("verification_status", result)
        self.assertIn("does not mean false", result["evidence_note"])

    def test_only_reads_supplied_assertions_and_does_not_add_inferences(self):
        ontology = OntologyTools()
        self.assertIn((NS.chef1, NS.worksFor, NS.McDonalds), ontology.inferred)
        self.assertNotIn((NS.chef1, NS.worksFor, NS.McDonalds), ontology.asserted)
        result = lookup_entity_facts(ontology.asserted, "chef1")
        self.assertEqual(result["status"], "matches")
        self.assertFalse(any(row["predicate"]["value"] == str(NS.worksFor)
                             for row in result["triples"]))
        # Unrelated facts in the original OWL file are not loaded implicitly.
        self.assertEqual(lookup_entity_facts(Graph(), "Obama")["status"], "unknown_entity")

    def test_preserves_language_blank_nodes_and_full_identifiers(self):
        graph = Graph()
        graph.add((NS.Obama, NS.hasName, Literal("Obama", lang="en")))
        graph.add((NS.Obama, NS.friendsWith, BNode("anonymous-person")))
        result = lookup_entity_facts(graph, "Obama")
        name = next(r for r in result["triples"] if r["predicate"]["value"] == str(NS.hasName))
        friend = next(r for r in result["triples"] if r["predicate"]["value"] == str(NS.friendsWith))
        self.assertEqual(name["object"]["language"], "en")
        self.assertEqual(name["subject"]["value"], str(NS.Obama))
        self.assertEqual(friend["object"], {"type": "bnode", "value": "anonymous-person"})

    def test_rejects_arbitrary_text_foreign_namespaces_and_invalid_inputs(self):
        graph = Graph()
        for entity in ("President Obama", "Obama's speeches", "", None,
                       "https://example.org/Obama", "Obama } UNION { ?s ?p ?o"):
            with self.subTest(entity=entity), self.assertRaises(OntologyError):
                lookup_entity_facts(graph, entity)
        with self.assertRaises(TypeError):
            lookup_entity_facts(None, "Obama")


if __name__ == "__main__":
    unittest.main()
