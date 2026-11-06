"""RDF evidence/provenance checks; synthetic claims are fixtures, not evaluations."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from rdflib import Graph, Literal, OWL, RDF, RDFS
from rdflib.compare import isomorphic

from data_loader import TOPICS, Tweet, load_tweets
from fact_check import prepare_assessment
from fact_tools import lookup_fact
from ontology_tools import DEFAULT_ONTOLOGY, NS, TOPIC_IRIS, OntologyTools, limited_closure, pipeline_base_graph
from populate_ontology import EVIDENCE_SCHEMA, ROOT, EvidenceError, populate_ontology
from revision_retrieval import topic_tweets


def extraction(tweet, status="extracted"):
    return {
        "tweet_id": tweet.id, "dialect_label": tweet.dialect_label, "status": status,
        "claims": [{
            "claim_id": f"{tweet.id}_C01", "tweet_id": tweet.id,
            "dialect_label": tweet.dialect_label, "source_quote": tweet.text,
            "claim": "Synthetic interpretation for testing only.",
            "target": "Synthetic target", "stance": "neutral",
        }] if status == "extracted" else [],
    }


class PopulationTests(unittest.TestCase):
    def setUp(self):
        self.tweets = [
            Tweet("T1", "Obama", "AA", "A synthetic opinion.\nSecond line!", "test.txt", 1),
            Tweet("T2", "Obama", "Hisp", "Synthetic unclear text.", "test.txt", 2),
            Tweet("T3", "Obama", "White", "Synthetic non-claim.", "test.txt", 3),
        ]
        self.records = [extraction(t, s) for t, s in zip(
            self.tweets, ("extracted", "unclear", "no_claim"))]

    def test_schema_and_base_are_loaded_without_truth_assessments(self):
        graph = populate_ontology(self.tweets, self.records)
        self.assertIn((NS.Person, RDF.type, OWL.Class), graph)
        schema = Graph().parse(EVIDENCE_SCHEMA, format="turtle")
        self.assertTrue(set(schema).issubset(graph))
        for prop, target in ((NS.containsClaim, NS.Claim), (NS.aboutTopic, NS.Topic)):
            self.assertIn((prop, RDFS.domain, NS.Tweet), graph)
            self.assertIn((prop, RDFS.range, target), graph)
        self.assertIn((NS.dialectLabel, RDF.type, OWL.DatatypeProperty), graph)
        self.assertIn((NS.dialectLabel, RDFS.domain, NS.Tweet), graph)
        self.assertNotIn((NS.DialectGroup, RDF.type, OWL.Class), graph)
        self.assertFalse(list(graph.triples((NS.belongsToDialectGroup, None, None))))
        self.assertFalse(list(graph.triples((None, NS.belongsToDialectGroup, None))))
        self.assertFalse(list(graph.subjects(RDF.type, NS.DialectGroup)))
        self.assertEqual(set(graph.subjects(NS.dialectLabel, None)),
                         set(graph.subjects(RDF.type, NS.Tweet)))
        self.assertFalse(list(graph.subjects(RDF.type, NS.ClaimAssessment)))
        self.assertFalse(list(graph.triples((None, NS.isValid, None))))

    def test_quote_uses_actual_source_whitespace(self):
        self.records[0]["claims"][0]["source_quote"] = "opinion. Second line!"
        graph = populate_ontology(self.tweets, self.records)
        self.assertEqual(list(graph.objects(None, NS.sourceQuote)),
                         [Literal("opinion.\nSecond line!")])

    def test_full_topic_synthetic_extractions_keep_all_90_tweets_and_groups(self):
        for topic in TOPICS:
            with self.subTest(topic=topic):
                tweets = load_tweets(topic)
                records = [extraction(tweet, ("extracted", "unclear", "no_claim")[i % 3])
                           for i, tweet in enumerate(tweets)]
                graph = populate_ontology(tweets, records)
                self.assertEqual(len(set(graph.subjects(RDF.type, NS.Tweet))), 90)
                for label in ("AA", "Hisp", "White"):  # stored as data for the evaluation only
                    self.assertEqual(len(set(graph.subjects(NS.dialectLabel, Literal(label)))), 30)
                self.assertEqual(topic_tweets(graph, topic)[1], [])
                self.assertEqual(len(list(graph.triples((None, NS.containsClaim, None)))), 30)
                self.assertEqual(set(graph.objects(None, NS.claimAboutTopic)), {TOPIC_IRIS[topic]})
                self.assertFalse(list(graph.triples((None, NS.tweetText, None))))
                claim = {"claim_id": "S01_C01", "claim": "Synthetic summary.",
                         "cited_source_ids": [tweets[0].id]}
                retrieved = lookup_fact(graph, claim)["tweets"][0]
                self.assertEqual(retrieved["text"], tweets[0].text)
                self.assertEqual(retrieved["topic_uri"], str(TOPIC_IRIS[topic]))

    def test_base_filter_keeps_schema_domain_facts_and_reference_individuals(self):
        tools = OntologyTools()
        # Multi-step subclass membership must be excluded, including SportsClaim.
        tools.asserted.add((NS.ExampleSportsClaim, RDFS.subClassOf, NS.SportsClaim))
        tools.asserted.add((NS.ExtraClaim, RDF.type, NS.ExampleSportsClaim))
        tools.asserted.add((NS.ExtraSummary, RDF.type, NS.Summary))
        tools.asserted.add((NS.ExtraSummary, NS.coversClaim, NS.ExtraClaim))
        tools.asserted.add((NS.Obama, NS.relatedTo, NS.ExtraClaim))
        tools.inferred = limited_closure(tools.asserted)
        before = set(tools.asserted)
        graph = pipeline_base_graph(tools)
        for subject in (NS.Tweet1, NS.claim1, NS.claim2, NS.ClaimAssessment1,
                        NS.ExtraClaim, NS.ExtraSummary):
            self.assertFalse(list(graph.triples((subject, None, None))))
            self.assertFalse(list(graph.triples((None, None, subject))))
        schema_types = (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty)
        for kind in schema_types:
            self.assertEqual(set(graph.subjects(RDF.type, kind)),
                             set(tools.asserted.subjects(RDF.type, kind)))
        for triple in ((NS.Obama, NS.hasAge, Literal(55)),
                       (NS.SuperBowl, RDF.type, NS.Topic),
                       (NS.DivSummDataset, RDF.type, NS.Source)):
            self.assertIn(triple, graph)
        self.assertEqual(set(tools.asserted), before)

    def test_legacy_text_and_superbowl_topic_remain_readable_without_duplicate_rows(self):
        tweet = replace(self.tweets[0], topic="Superbowl")
        graph = populate_ontology([tweet], [extraction(tweet)])
        iri = next(graph.subjects(NS.tweetId, Literal(tweet.id)))
        claim = {"claim_id": "S01_C01", "claim": "Synthetic summary.",
                 "cited_source_ids": [tweet.id]}
        # Both spellings with equal values must not inflate tweets or claims.
        graph.add((iri, NS.tweetText, Literal(tweet.text)))
        graph.add((iri, NS.aboutTopic, NS.Topic_Superbowl))
        tweets, problems = topic_tweets(graph, "Superbowl")
        self.assertEqual((len(tweets), problems), (1, []))
        self.assertEqual(len(lookup_fact(graph, claim)["tweets"][0]["source_claims"]), 1)
        graph.remove((iri, NS.tweetHasText, None))
        graph.remove((iri, NS.aboutTopic, NS.SuperBowl))
        before = set(graph)
        self.assertEqual(len(topic_tweets(graph, "Superbowl")[0]), 1)
        self.assertEqual(lookup_fact(graph, claim)["tweets"][0]["text"], tweet.text)
        self.assertEqual(set(graph), before)

    def test_conflicting_legacy_and_canonical_text_cannot_produce_definitive_checks(self):
        tweet = self.tweets[0]
        graph = populate_ontology([tweet], [extraction(tweet)])
        iri = next(graph.subjects(NS.tweetId, Literal(tweet.id)))
        graph.add((iri, NS.tweetText, Literal("Different source text.")))
        self.assertTrue(any("Conflicting tweet text" in issue for issue in topic_tweets(graph, "Obama")[1]))
        claim = {"claim_id": "S01_C01", "claim": "Synthetic summary.",
                 "cited_source_ids": [tweet.id]}
        result = prepare_assessment(claim, lookup_fact(graph, claim))
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertFalse(result["evidence_complete"])
        self.assertEqual(set(result["retrieval"]["tweets"][0]["conflicting_text_values"]),
                         {tweet.text, "Different source text."})

    def test_runs_are_isolated_and_inputs_and_source_files_are_unchanged(self):
        before = deepcopy(self.records)
        checksums = {p: hashlib.sha256(p.read_bytes()).digest()
                     for p in (DEFAULT_ONTOLOGY, EVIDENCE_SCHEMA)}
        first = populate_ontology(self.tweets, self.records)
        second = populate_ontology(self.tweets, self.records)
        self.assertNotEqual(first.identifier, second.identifier)
        for cls in (NS.Tweet, NS.Claim):
            self.assertTrue(set(first.subjects(RDF.type, cls)).isdisjoint(
                second.subjects(RDF.type, cls)))
        first.add((NS.TestOnly, RDF.type, NS.Tweet))
        self.assertNotIn((NS.TestOnly, RDF.type, NS.Tweet), second)
        self.assertEqual(self.records, before)
        for path, digest in checksums.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), digest)

    def test_turtle_round_trip_and_reserved_identifier_characters(self):
        tweet = replace(self.tweets[0], id="T /?#%é")
        graph = populate_ontology([tweet], [extraction(tweet)], run_id="test_run")
        restored = Graph().parse(data=graph.serialize(format="turtle"), format="turtle")
        self.assertTrue(isomorphic(graph, restored))
        self.assertEqual(list(restored.objects(None, NS.tweetId)), [Literal(tweet.id)])
        with self.assertRaises(EvidenceError):
            populate_ontology([tweet], [extraction(tweet)], run_id="bad/run")

    def test_rejects_inconsistent_records(self):
        cases = []
        def case(name, edit):
            rows = deepcopy(self.records)
            edit(rows)
            cases.append((name, rows))

        case("missing tweet", lambda r: r.pop())
        case("duplicate record", lambda r: r.append(deepcopy(r[0])))
        case("unknown tweet", lambda r: r[0].update(tweet_id="unknown"))
        case("wrong record dialect", lambda r: r[0].update(dialect_label="White"))
        case("wrong claim tweet", lambda r: r[0]["claims"][0].update(tweet_id="T2"))
        case("wrong claim dialect", lambda r: r[0]["claims"][0].update(dialect_label="Hisp"))
        case("invented quote", lambda r: r[0]["claims"][0].update(source_quote="Not in source"))
        case("unknown status", lambda r: r[0].update(status="failed"))
        case("status type", lambda r: r[0].update(status=[]))
        case("empty extracted", lambda r: r[0].update(claims=[]))
        case("claims with unclear", lambda r: r[0].update(status="unclear"))
        case("claims with no_claim", lambda r: r[0].update(status="no_claim"))
        case("claims type", lambda r: r[0].update(claims={}))
        case("bad stance", lambda r: r[0]["claims"][0].update(stance="true"))
        case("empty field", lambda r: r[0]["claims"][0].update(target=" "))
        case("missing field", lambda r: r[0]["claims"][0].pop("claim_id"))
        case("duplicate claim", lambda r: r[0]["claims"].append(deepcopy(r[0]["claims"][0])))
        case("bad record", lambda r: r.append("invalid"))
        case("bad claim", lambda r: r[0].update(claims=["invalid"]))
        cases.append(("records type", {}))
        for name, rows in cases:
            with self.subTest(name=name), self.assertRaises(EvidenceError):
                populate_ontology(self.tweets, rows)
        for tweets in ([], self.tweets + [self.tweets[0]],
                       [replace(self.tweets[0], dialect_label="unknown")],
                       [self.tweets[0], replace(self.tweets[1], topic="McDonalds")]):
            with self.subTest(tweets=tweets), self.assertRaises(EvidenceError):
                populate_ontology(tweets, self.records)

    def test_cli_requires_the_complete_topic_and_saves_graph_and_manifest(self):
        # Fixture extractions using real source text; no model is invoked.
        tweets = load_tweets("Obama")
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            claims = folder / "claims.json"
            claims.write_text(json.dumps([extraction(tweets[0])]), encoding="utf-8")
            command = [sys.executable, str(ROOT / "populate_ontology.py"), "--topic", "Obama",
                       "--claims", str(claims), "--output-dir", str(folder / "output")]
            incomplete = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(incomplete.returncode, 0)
            self.assertIn("Missing extraction records", incomplete.stderr)
            self.assertFalse((folder / "output").exists())
            claims.write_text(json.dumps([extraction(tweet) for tweet in tweets]), encoding="utf-8")
            complete = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(complete.returncode, 0, complete.stderr)
            self.assertIn("Complete topic: 90 tweets", complete.stdout)
            manifest_file = next((folder / "output").glob("*/manifest.json"))
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertTrue(manifest["complete_topic"])
            self.assertEqual(manifest["tweet_count"], 90)
            self.assertEqual(manifest["base_evidence_policy"],
                             "exclude_tweets_claims_assessments_summaries")
            graph = Graph().parse(manifest_file.parent / "Obama.ttl", format="turtle")
            self.assertEqual(len(set(graph.subjects(RDF.type, NS.Tweet))), 90)


if __name__ == "__main__":
    unittest.main()
