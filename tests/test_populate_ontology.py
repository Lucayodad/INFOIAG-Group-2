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

from data_loader import Tweet, load_tweets
from ontology_tools import DEFAULT_ONTOLOGY, NS
from populate_ontology import EVIDENCE_SCHEMA, ROOT, EvidenceError, populate_ontology


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

    def test_source_evidence_query_keeps_claimless_tweets_and_supporting_ids(self):
        graph = populate_ontology(self.tweets, self.records)
        query = (ROOT / "queries" / "source_evidence.rq").read_text(encoding="utf-8")
        rows = list(graph.query(query, initBindings={"topic": NS.Topic_Obama}))
        self.assertEqual(len(rows), 3)
        by_id = {str(row.tweet_id): row for row in rows}
        first = by_id["T1"]
        self.assertEqual(str(first.text), self.tweets[0].text)
        self.assertEqual(str(first.quote), self.tweets[0].text)
        self.assertEqual(str(first.claim_text), self.records[0]["claims"][0]["claim"])
        self.assertEqual(str(first.claim_id), "T1_C01")
        self.assertEqual(str(first.dialect_label), "AA")
        self.assertEqual(graph.value(first.tweet, NS.dialectLabel), Literal("AA"))
        self.assertEqual(str(first.target), "Synthetic target")
        self.assertEqual(str(first.stance), "neutral")
        self.assertEqual(graph.value(first.tweet, NS.sourceFile), Literal("test.txt"))
        self.assertEqual(graph.value(first.tweet, NS.sourceLine), Literal(1))
        for tweet_id, status in (("T2", "unclear"), ("T3", "no_claim")):
            self.assertEqual(str(by_id[tweet_id].status), status)
            self.assertIsNone(by_id[tweet_id].claim)
        filtered = list(graph.query(query, initBindings={
            "topic": NS.Topic_Obama, "dialect_label": Literal("AA")}))
        self.assertEqual([str(row.tweet_id) for row in filtered], ["T1"])

    def test_quote_uses_actual_source_whitespace(self):
        self.records[0]["claims"][0]["source_quote"] = "opinion. Second line!"
        graph = populate_ontology(self.tweets, self.records)
        self.assertEqual(list(graph.objects(None, NS.sourceQuote)),
                         [Literal("opinion.\nSecond line!")])

    def test_full_topic_synthetic_extractions_keep_all_90_tweets_and_groups(self):
        tweets = load_tweets("Obama")
        records = [extraction(tweet, ("extracted", "unclear", "no_claim")[i % 3])
                   for i, tweet in enumerate(tweets)]
        graph = populate_ontology(tweets, records)
        self.assertEqual(len(set(graph.subjects(RDF.type, NS.Tweet))), 90)
        for label in ("AA", "Hisp", "White"):
            self.assertEqual(len(set(graph.subjects(
                NS.dialectLabel, Literal(label)))), 30)
        self.assertEqual(len(list(graph.triples((None, NS.containsClaim, None)))), 30)

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

    def test_cli_requires_complete_topic_and_saves_explicit_samples(self):
        # Fixture extractions using real source text; no model is invoked.
        tweet = load_tweets("Obama")[0]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            claims = folder / "claims.json"
            claims.write_text(json.dumps([extraction(tweet)]), encoding="utf-8")
            command = [sys.executable, str(ROOT / "populate_ontology.py"), "--topic", "Obama",
                       "--claims", str(claims), "--output-dir", str(folder / "output")]
            incomplete = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(incomplete.returncode, 0)
            self.assertIn("Missing extraction records", incomplete.stderr)
            self.assertFalse((folder / "output").exists())
            for _ in range(2):
                sample = subprocess.run(command + ["--sample"], capture_output=True, text=True)
                self.assertEqual(sample.returncode, 0, sample.stderr)
                self.assertIn("Sample: 1/90", sample.stdout)
            manifests = list((folder / "output").glob("*/manifest.json"))
            self.assertEqual(len(manifests), 2)
            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertTrue(manifest["sample"])
            self.assertFalse(manifest["complete_topic"])
            self.assertEqual(manifest["tweet_count"], 1)
            graph = Graph().parse(manifests[0].parent / "Obama.ttl", format="turtle")
            self.assertEqual(len(set(graph.subjects(RDF.type, NS.Tweet))), 1)


if __name__ == "__main__":
    unittest.main()
