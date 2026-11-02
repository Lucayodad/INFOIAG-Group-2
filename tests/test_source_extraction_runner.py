"""Checkpoint/retry integration in temporary directories; no live model calls."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from data_loader import load_tweets
from run_source_extraction import main


def record(tweet):
    return {'tweet_id': tweet.id, 'dialect_label': tweet.dialect_label, 'status': 'extracted', 'claims': [{
        'claim_id': tweet.id + '_C01', 'tweet_id': tweet.id, 'dialect_label': tweet.dialect_label,
        'source_quote': tweet.text, 'claim': 'Synthetic test interpretation.', 'target': 'Obama', 'stance': 'neutral',
    }]}


def reply(tweet, invalid=False):
    claims = [{'quote': tweet.text, 'claim': 'Synthetic test interpretation.', 'target': 'Obama', 'stance': 'neutral'}]
    if invalid:
        # A valid first claim must not be saved if another claim is fabricated.
        claims.append({**claims[0], 'quote': 'This fabricated evidence never appeared in this tweet.'})
    return {'output': [{'content': [{'text': json.dumps({'status': 'extracted', 'claims': claims})}]}]}


class ExtractionRunnerTests(unittest.TestCase):
    def invoke(self, root, client, limit):
        with patch('run_source_extraction.ROOT', root), patch('run_source_extraction.OllamaClient', return_value=client), \
             patch('sys.argv', ['run_source_extraction.py', '--topic', 'Obama', '--limit', str(limit)]), \
             redirect_stdout(io.StringIO()) as console:
            status = main()
        client.close.assert_called_once()
        return status, console.getvalue()

    def checkpoint(self, root, records):
        output = root / 'results/source_claims/Obama.json'
        output.parent.mkdir(parents=True)
        output.write_text(json.dumps(records, indent=2), encoding='utf-8')
        return output

    def test_persistent_failure_continues_and_later_resume_skips_every_success(self):
        tweets = load_tweets('Obama')[:4]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = record(tweets[0])
            original = deepcopy(saved)
            checkpoint = self.checkpoint(root, [saved])
            client = Mock()
            client.chat.side_effect = [reply(tweets[1], invalid=True)] * 3 + [reply(tweets[2]), reply(tweets[3])]
            status, printed = self.invoke(root, client, 4)
            self.assertEqual(status, 1)
            self.assertEqual(client.chat.call_count, 5)
            rows = json.loads(checkpoint.read_text(encoding='utf-8'))
            self.assertEqual([row['tweet_id'] for row in rows], [tweets[i].id for i in (0, 2, 3)])
            self.assertEqual(rows[0], original)
            self.assertTrue(all(row['status'] == 'extracted' for row in rows))
            self.assertIn('Unresolved tweets (1): Obama_002', printed)
            failure_path = checkpoint.with_name('Obama.failures.json')
            failure = json.loads(failure_path.read_text(encoding='utf-8'))[0]
            self.assertEqual(failure['tweet_id'], 'Obama_002')
            self.assertEqual(failure['failed_attempt_count'], 3)
            self.assertFalse(failure['resolved'])
            self.assertEqual(len(failure['attempt_errors']), 3)
            recovery = Mock()
            recovery.chat.return_value = reply(tweets[1])
            status, printed = self.invoke(root, recovery, 4)
            self.assertEqual(status, 0)
            self.assertEqual(recovery.chat.call_count, 1)
            self.assertEqual(json.loads(recovery.chat.call_args.args[0]['messages'][1]['content'])['id'], 'Obama_002')
            recovered = json.loads(checkpoint.read_text(encoding='utf-8'))
            self.assertEqual(recovered[:3], rows)
            self.assertEqual(recovered[-1]['tweet_id'], 'Obama_002')
            self.assertTrue(json.loads(failure_path.read_text(encoding='utf-8'))[0]['resolved'])
            self.assertIn('Unresolved tweets: none.', printed)
            before = checkpoint.read_bytes()
            skip = Mock()
            self.assertEqual(self.invoke(root, skip, 4)[0], 0)
            skip.chat.assert_not_called()
            self.assertEqual(checkpoint.read_bytes(), before)

    def test_all_failed_attempts_leave_success_checkpoint_byte_for_byte_unchanged(self):
        tweets = load_tweets('Obama')[:2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self.checkpoint(root, [record(tweets[0])])
            before = checkpoint.read_bytes()
            client = Mock()
            client.chat.return_value = reply(tweets[1], invalid=True)
            self.assertEqual(self.invoke(root, client, 2)[0], 1)
            self.assertEqual(checkpoint.read_bytes(), before)
            again = Mock()
            again.chat.return_value = reply(tweets[1], invalid=True)
            self.assertEqual(self.invoke(root, again, 2)[0], 1)
            failure = json.loads(checkpoint.with_name('Obama.failures.json').read_text(encoding='utf-8'))[0]
            self.assertEqual(failure['failed_attempt_count'], 3)
            self.assertEqual(failure['total_failed_attempt_count'], 6)

    def test_crash_recovery_resolves_stale_failure_without_repeating_checkpointed_tweet(self):
        tweet = load_tweets('Obama')[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self.checkpoint(root, [record(tweet)])
            failure_path = checkpoint.with_name('Obama.failures.json')
            failure_path.write_text(json.dumps([{'tweet_id': tweet.id, 'resolved': False,
                                                 'error': 'Earlier failed attempt', 'failed_attempt_count': 3}]), encoding='utf-8')
            before = checkpoint.read_bytes()
            client = Mock()
            self.assertEqual(self.invoke(root, client, 1)[0], 0)
            client.chat.assert_not_called()
            self.assertEqual(checkpoint.read_bytes(), before)
            self.assertTrue(json.loads(failure_path.read_text(encoding='utf-8'))[0]['resolved'])


if __name__ == '__main__':
    unittest.main()
