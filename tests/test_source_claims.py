"""Quotation normalization preserves exact original spans; all LLM calls mocked."""

import json
from html import unescape
import unittest
from unittest.mock import Mock

from data_loader import TOPICS, load_tweets
from source_claims import SourceExtractionError, extract_one, extract_with_retries, find_source_quote
from summary_claims import extract_sentence


class SourceQuotationTests(unittest.TestCase):
    def test_html_entities_map_back_to_exact_original_substrings(self):
        examples = [
            ('Alpha & Beta', 'Before: Alpha &amp; Beta. After.', 'Alpha &amp; Beta'),
            ('Alpha &amp; Beta', 'Alpha & Beta', 'Alpha & Beta'),
            ('A & B', 'A &#38; B', 'A &#38; B'),
            ('A & B', 'A &#x26; B', 'A &#x26; B'),
            ('"Alpha & Beta"', 'Alpha &amp; Beta', 'Alpha &amp; Beta'),
            ('&quot;Alpha &amp; Beta&quot;', 'Alpha &amp; Beta', 'Alpha &amp; Beta'),
            ('A B', 'A&nbsp;B', 'A&nbsp;B'),
            ('"A B"', 'A\t&#32;\n B', 'A\t&#32;\n B'),
            ('"A & B"', '&quot;A &amp; B&quot;', '&quot;A &amp; B&quot;'),
            ('&', '&amp;', '&amp;'),
        ]
        for quote, original, expected in examples:
            with self.subTest(quote=quote, original=original):
                self.assertEqual(find_source_quote(quote, original), expected)

    def test_html_does_not_hide_changed_numbers_negation_or_partial_entities(self):
        original = 'It costs &#50;0 dollars and is not approved.'
        for quote in ('It costs 30 dollars and is not approved.', 'It costs 20 dollars and is approved.',
                      'It costs twenty dollars and is not approved.', 'This costs 20 dollars and is not approved.'):
            with self.subTest(quote=quote):
                self.assertIsNone(find_source_quote(quote, original))
        self.assertEqual(find_source_quote('It costs 20 dollars and is not approved.', original), original)
        self.assertIsNone(find_source_quote('≂', '&NotEqualTilde;'))  # Do not drop the combining negation mark.
        self.assertEqual(find_source_quote('≂̸', '&NotEqualTilde;'), '&NotEqualTilde;')
        self.assertIsNone(find_source_quote('amp', '&amp;'))
        self.assertIsNone(find_source_quote('<', '&amp;lt;'))  # One decoding layer only.
        self.assertIsNone(find_source_quote('AB', 'A&#11;B'))  # Invalid references must not disappear.
        for quote, source in [('20', '200'), ('2', '&#50;0'), ('20', '-20'), ('2', '2.5'),
                              ('supported', 'unsupported')]:
            with self.subTest(quote=quote, source=source):
                self.assertIsNone(find_source_quote(quote, source))

    def test_obama_030_html_alias_and_all_source_tweets_preserve_original_text(self):
        for topic in TOPICS:
            for tweet in load_tweets(topic):
                with self.subTest(tweet=tweet.id):
                    found = find_source_quote('"' + unescape(tweet.text) + '"', tweet.text)
                    self.assertIsNotNone(found)
                    self.assertIn(found, tweet.text)
                    self.assertEqual(' '.join(unescape(found).split()), ' '.join(unescape(tweet.text).split()))
        tweet = next(t for t in load_tweets('Obama') if t.id == 'Obama_030')
        self.assertIn('&amp;', tweet.text)
        self.assertEqual(find_source_quote(unescape(tweet.text), tweet.text), tweet.text)

    def test_exact_quotes_after_literal_whitespace_markers_remain_valid(self):
        for marker in (r'\n', r'\r', r'\t'):
            with self.subTest(marker=marker):
                self.assertEqual(find_source_quote('Next words.', 'Before.' + marker + 'Next words.'), 'Next words.')
        self.assertIsNone(find_source_quote('words', 'longerwords'))

    def test_exact_quotations_and_genuine_source_marks_are_preserved(self):
        for text in ('Original wording, punctuation!', 'She said "keep this".', '“Keep these marks.”'):
            with self.subTest(text=text):
                self.assertEqual(find_source_quote(text, text), text)
        self.assertEqual(find_source_quote('"keep this"', 'She said "keep this".'), '"keep this"')

    def test_extra_balanced_surrounding_marks(self):
        original = "People's opinions aren't established facts."
        for left, right in [('"', '"'), ("'", "'"), ('“', '”'), ('‘', '’'), ('«', '»')]:
            with self.subTest(left=left):
                self.assertEqual(find_source_quote(left + original + right, original), original)
        self.assertEqual(find_source_quote('"“' + original + '”"', original), original)
        self.assertEqual(find_source_quote('"“Keep these marks.”"', '“Keep these marks.”'), '“Keep these marks.”')

    def test_whitespace_differences_return_original_whitespace(self):
        original = "Before: exact\twords.\nSecond  line! After."
        for quote in ('exact words. Second line!', '  "exact words.\tSecond line!"  ', '“exact words. Second line!”'):
            with self.subTest(quote=quote):
                self.assertEqual(find_source_quote(quote, original), 'exact\twords.\nSecond  line!')

    def test_fabricated_paraphrased_or_noncontiguous_quotations_are_rejected(self):
        original = "The author does not support this proposal."
        for quote in ('"The author supports this proposal."', 'The writer opposes this proposal.',
                      'The author supports a different policy.', 'The author this proposal.'):
            with self.subTest(quote=quote):
                self.assertIsNone(find_source_quote(quote, original))

    def test_case_internal_punctuation_apostrophes_and_unbalanced_marks_stay_strict(self):
        original = "People's opinions aren't established facts."
        for quote in ('"peoples opinions arent established facts."', 'People’s opinions aren’t established facts.',
                      '"people\'s opinions aren\'t established facts."', '"People\'s opinions aren\'t established facts!"',
                      '"' + original, '“' + original + '"'):
            with self.subTest(quote=quote):
                self.assertIsNone(find_source_quote(quote, original))
        for quote in (None, 123, '', ' \n\t ', '""', "''", '“ ”'):
            with self.subTest(quote=quote):
                self.assertIsNone(find_source_quote(quote, original))

    def test_obama_011_extraction_returns_exact_original_quote(self):
        tweet = next(t for t in load_tweets("Obama") if t.id == "Obama_011")
        client = Mock()
        client.chat.return_value = {"output": [{"content": [{"text": json.dumps({
            "status": "extracted", "claims": [{
                "quote": '"' + tweet.text + '"', "claim": "The author expresses a derogatory opinion.",
                "target": "Obama and Tiger", "stance": "negative",
            }],
        })}]}]}
        result = extract_one(tweet, client)
        self.assertEqual(result["claims"][0]["source_quote"], tweet.text)
        self.assertEqual(result["claims"][0]["tweet_id"], "Obama_011")
        self.assertEqual(result["claims"][0]["dialect_label"], tweet.dialect_label)
        client.chat.assert_called_once()

    def test_all_dataset_tweets_accept_added_wrappers_without_changing_text(self):
        # Read-only corpus check: no extraction requests or checkpoint writes.
        for topic in TOPICS:
            for tweet in load_tweets(topic):
                with self.subTest(tweet=tweet.id):
                    expected = find_source_quote(tweet.text, tweet.text)
                    for left, right in [('"', '"'), ('“', '”'), ("'", "'")]:
                        self.assertEqual(find_source_quote(left + tweet.text + right, tweet.text), expected)

    def test_summary_extraction_reuses_matcher_and_retains_original_span(self):
        original = "The author\tlikes the speech."
        client = Mock()
        client.chat.return_value = {"output": [{"content": [{"text": json.dumps({
            "status": "extracted", "claims": [{"quote": '"The author likes the speech."',
                "claim": "The author likes the speech.", "target": "speech", "stance": "positive"}],
        })}]}]}
        result = extract_sentence({"text": original, "source_ids": ["T1"]}, 1, client)
        self.assertEqual(result["claims"][0]["summary_quote"], original)

    def test_retries_copy_exact_source_and_stop_after_validated_success(self):
        tweet = load_tweets('Obama')[29]
        def response(quote):
            return {'output': [{'content': [{'text': json.dumps({'status': 'extracted', 'claims': [{
                'quote': quote, 'claim': 'Synthetic unverified interpretation.', 'target': 'Obama', 'stance': 'neutral',
            }]})}]}]}
        for failures in (0, 1, 2):
            with self.subTest(failures=failures):
                client, notify = Mock(), Mock()
                client.chat.side_effect = [response('A fabricated quotation.')] * failures + [response(unescape(tweet.text))]
                result = extract_with_retries(tweet, client, on_retry=notify)
                self.assertEqual(result['claims'][0]['source_quote'], tweet.text)
                self.assertEqual(client.chat.call_count, failures + 1)
                self.assertEqual(notify.call_count, failures)
                for index in range(1, failures + 1):
                    payload = client.chat.call_args_list[index].args[0]
                    self.assertIn('exact contiguous substring', payload['messages'][0]['content'])
                    self.assertEqual(json.loads(payload['messages'][-1]['content'])['retry_attempt'], index + 1)

    def test_three_persistent_failures_raise_explicit_error_not_no_claim(self):
        tweet = load_tweets('Obama')[29]
        client = Mock()
        client.chat.return_value = {'output': [{'content': [{'text': json.dumps({'status': 'extracted', 'claims': [{
            'quote': 'Fabricated evidence.', 'claim': 'Unsupported.', 'target': 'Obama', 'stance': 'neutral',
        }]})}]}]}
        with self.assertRaises(SourceExtractionError) as caught:
            extract_with_retries(tweet, client)
        self.assertEqual(caught.exception.tweet_id, tweet.id)
        self.assertEqual(caught.exception.failed_attempt_count, 3)
        self.assertEqual(len(caught.exception.attempt_errors), 3)
        self.assertEqual(client.chat.call_count, 3)

    def test_malformed_model_output_is_bounded_and_explicit(self):
        tweet = load_tweets('Obama')[0]
        client = Mock()
        client.chat.return_value = {'output': []}
        with self.assertRaises(SourceExtractionError) as caught:
            extract_with_retries(tweet, client)
        self.assertEqual(client.chat.call_count, 3)
        self.assertIn('malformed extraction response', str(caught.exception))


if __name__ == "__main__":
    unittest.main()
