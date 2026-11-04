"""Controlled live validation: synthetic evidence, real local models.

This is a mechanics/model probe, never a real-topic experimental result.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from demo_revision_loop import fixture
from ollama_backend import OllamaClient
from model_usage import RecordingClient
from summary_agent import run_summary_agent

class VisibleClient(OllamaClient):
    number = 0
    def chat(self, payload):
        self.number += 1
        print(f"REAL Qwen call {self.number}: {list(payload.get('format', {}).get('properties', {}))}", flush=True)
        return super().chat(payload)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/evidence/final_validation') /
                        ('controlled_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json'))
    output = parser.parse_args().output
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    client = VisibleClient(timeout=300)
    try:
        models = client.check_models()
        graph, draft, _ = fixture('factual')
        result = run_summary_agent(graph, draft, topic='Obama', prompt=draft['summary_prompt'],
            client=client, max_revisions=1, max_model_calls=12)  # no group check: labels are evaluation-only
        result['validation'] = {'mode': 'synthetic evidence, REAL local Qwen and embeddings',
            'models': models, 'not_a_90_tweet_experiment': True,
            'group_check': 'none: dialect labels are used only by the final evaluation'}
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'status': result['status'], 'stop_reason': result['stop_reason'],
            'revisions': result['revision_count'], 'usage_complete': result['usage_complete'],
            'nodes': [r['node'] for r in result['execution_history']], 'output': str(output)}, indent=2), flush=True)
    finally:
        client.close()


if __name__ == '__main__':
    main()
