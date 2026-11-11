# Copyright 2026 Maxrio9. SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
import os
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


path = Path(__file__).resolve().parents[2] / 'server/training/collectors/artifacts.py'
spec = importlib.util.spec_from_file_location('training_artifacts', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ArtifactDiscoveryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.proc = self.root / 'proc'
        self.proc.mkdir()
        (self.proc / 'stat').write_text('btime 1000\n')
        self.jobs = self.root / 'jobs'
        self.jobs.mkdir()

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def process(self, pid=123, args=None, cwd=None, start=100, state='S'):
        directory = self.proc / str(pid)
        directory.mkdir(exist_ok=True)
        uid = os.getuid()
        (directory / 'status').write_text(f'Uid:\t{uid}\t{uid}\t{uid}\t{uid}\n')
        fields = [state] + ['0'] * 18 + [str(start)] + ['0'] * 4
        (directory / 'stat').write_text(f'{pid} (python training) ' + ' '.join(fields))
        (directory / 'cmdline').write_bytes(('\0'.join(args or ['python', 'train.py']) + '\0').encode())
        link = directory / 'cwd'
        if link.is_symlink():
            link.unlink()
        link.symlink_to(cwd or self.jobs, target_is_directory=True)
        return directory

    def collect(self):
        return module.collect([self.jobs], 'worker-a', self.proc, now=3000000)

    def test_status_document_preserves_custom_stages_and_tracks_process_exit(self):
        self.process()
        file = self.jobs / 'experiment/training-progress.json'
        self.write_json(file, {
            'schema_version': 1, 'id': 'shared-distributed-job', 'campaign': 'Pretraining',
            'model': 'model-x', 'pid': 123, 'process_start_ticks': 100,
            'stages': {
                'data': [{'id': 'shards', 'label': 'Dataset shards', 'state': 'complete', 'completed': 8, 'total': 8, 'unit': 'shards'}],
                'training': [{'id': 'epochs', 'label': 'Epochs', 'state': 'running', 'completed': 1.5, 'total': 3, 'unit': 'epochs',
                              'metrics': [{'label': 'Tokens per second', 'value': 45.8, 'unit': 'tokens/s'}, {'label': 'Strategy', 'value': 'LoRA'}]}],
                'evaluation': [],
            },
        })
        run, = self.collect()
        row, = run['sources'][0]['stages']['training']
        self.assertEqual((row['completed'], row['total'], row['state']), (1.5, 3, 'running'))
        self.assertEqual(row['metrics'][0]['unit'], 'tokens/s')
        self.assertEqual(row['metrics'][1]['value'], 'LoRA')
        self.assertEqual(run['campaign'], 'Pretraining')
        shutil.rmtree(self.proc / '123')
        stopped, = self.collect()
        self.assertEqual(stopped['sources'][0]['stages']['training'][0]['state'], 'stopped')

    def test_missing_process_evidence_preserves_unknown_running_state_and_saved_results(self):
        self.write_json(self.jobs / 'custom/training-progress.json', {
            'schema_version': 1, 'pid': 123, 'process_start_ticks': 100,
            'stages': {
                'training': [{'id': 'fit', 'label': 'Training', 'state': 'running'}],
                'evaluation': [{'id': 'validated', 'label': 'Saved validation', 'state': 'complete'}],
            },
        })
        self.write_json(self.jobs / 'native/trainer_state.json', {'global_step': 1, 'max_steps': 10})
        csv = self.jobs / 'csv/history.csv'
        csv.parent.mkdir()
        csv.write_text('epoch,loss,val_loss\n0,0.6,0.7\n')
        for proc in (self.root / 'missing-proc', self.proc):
            if proc == self.proc:
                (proc / 'stat').unlink()
            runs = module.collect([self.jobs], 'worker-a', proc, now=3000000)
            self.assertEqual(len(runs), 3)
            states = [stage['state'] for run in runs for source in run['sources']
                      for stage in source['stages']['training']]
            self.assertEqual(states, ['unknown'] * 3)
            custom = next(run for run in runs if run['sources'][0]['label'] == 'custom')
            self.assertEqual(custom['sources'][0]['stages']['evaluation'][0]['state'], 'complete')
            self.assertEqual(custom['sources'][0]['stages']['training'][0]['detail'], 'Live process identity unavailable.')

    def test_unreadable_or_other_user_process_is_unknown_until_absent_or_dead(self):
        process = self.process()
        self.write_json(self.jobs / 'training-progress.json', {
            'schema_version': 1, 'pid': 123, 'process_start_ticks': 100,
            'stages': {'training': [{'id': 'fit', 'label': 'Training', 'state': 'running'}]},
        })
        different_user = os.getuid() + 1
        (process / 'status').write_text(f'Uid:\t{different_user}\t{different_user}\t{different_user}\t{different_user}\n')
        self.assertEqual(self.collect()[0]['sources'][0]['stages']['training'][0]['state'], 'unknown')
        self.process()
        original = module.read_process
        def unreadable(pid, *args, **kwargs):
            return None if pid == 123 else original(pid, *args, **kwargs)
        with patch.object(module, 'read_process', side_effect=unreadable):
            self.assertEqual(self.collect()[0]['sources'][0]['stages']['training'][0]['state'], 'unknown')
        self.process(state='Z')
        self.assertEqual(self.collect()[0]['sources'][0]['stages']['training'][0]['state'], 'stopped')
        shutil.rmtree(process)
        self.assertEqual(self.collect()[0]['sources'][0]['stages']['training'][0]['state'], 'stopped')

    def test_reused_pid_or_old_status_cannot_establish_running(self):
        self.process(start=100)
        file = self.jobs / 'training-progress.json'
        document = {'schema_version': 1, 'pid': 123, 'process_start_ticks': 99,
                    'stages': {'training': [{'id': 'train', 'label': 'Training', 'state': 'running'}]}}
        self.write_json(file, document)
        self.assertEqual(self.collect()[0]['sources'][0]['stages']['training'][0]['state'], 'stopped')
        document.pop('process_start_ticks')
        self.write_json(file, document)
        os.utime(file, (900, 900))
        self.assertEqual(self.collect()[0]['sources'][0]['stages']['training'][0]['state'], 'stopped')
        document.pop('pid')
        self.write_json(file, document)
        self.assertEqual(self.collect()[0]['sources'][0]['stages']['training'][0]['state'], 'unknown')

    def test_manifest_preserves_valid_completed_evaluation_comparisons(self):
        comparison = dict(metric='accuracy', metricLabel='Accuracy', value=0, seriesId='lora', seriesLabel='LoRA',
                          checkpoint=0, cohortId='dev-set-v1', unit='%', lower=0, upper=1, private='ignored')
        file = self.jobs / 'training-progress.json'
        self.write_json(file, {'schema_version': 1, 'stages': {'evaluation': [
            {'id': 'eval', 'label': 'Validation', 'state': 'complete', 'comparison': comparison},
            {'id': 'bad', 'label': 'Malformed comparison', 'state': 'complete', 'comparison': {**comparison, 'lower': 2}},
            {'id': 'running', 'label': 'Pending result', 'state': 'running', 'comparison': comparison},
        ]}})
        rows = self.collect()[0]['sources'][0]['stages']['evaluation']
        self.assertEqual(rows[0]['comparison'], {key: value for key, value in comparison.items() if key != 'private'})
        self.assertNotIn('comparison', rows[1])
        self.assertNotIn('comparison', rows[2])

    def test_manifest_eta_counts_down_from_the_document_measurement(self):
        self.process()
        file = self.jobs / 'training-progress.json'
        self.write_json(file, {'schema_version': 1, 'pid': 123, 'process_start_ticks': 100,
            'stages': {'training': [{'id': 'train', 'label': 'Training', 'state': 'running', 'etaSeconds': 100}]}})
        os.utime(file, (2990, 2990))
        first = module.collect([self.jobs], 'worker-a', self.proc, now=3000000)[0]['sources'][0]['stages']['training'][0]
        second = module.collect([self.jobs], 'worker-a', self.proc, now=3010000)[0]['sources'][0]['stages']['training'][0]
        self.assertEqual((first['etaSeconds'], second['etaSeconds']), (90, 80))
        expired = module.collect([self.jobs], 'worker-a', self.proc, now=3100000)[0]['sources'][0]['stages']['training'][0]
        self.assertNotIn('etaSeconds', expired)

    def test_manifest_preserves_declared_worker_context_health_inputs_and_multiple_objectives(self):
        self.process()
        comparison = dict(metric='accuracy', metricLabel='Accuracy', value=.85, seriesId='sft', seriesLabel='SFT',
                          checkpoint=2, checkpointUnit='epochs', goal='maximize', contextId='seed-42', contextLabel='Seed 42')
        row = {'id': 'save', 'label': 'Save checkpoint', 'state': 'running',
               'resources': [{'kind': 'cpu', 'id': '0-3', 'scope': 'assigned'}],
               'activity': {'kind': 'saving', 'label': 'Saving checkpoint', 'since': 2990000},
               'dependencies': [{'id': 'optimizer', 'label': 'Optimizer step', 'state': 'ready'}],
               'heartbeatAt': 2999000, 'heartbeatTimeoutSeconds': 60,
               'lastProgressAt': 2990000, 'progressTimeoutSeconds': 300,
               'health': {'state': 'healthy'}}
        self.write_json(self.jobs / 'training-progress.json', {'schema_version': 1, 'pid': 123, 'process_start_ticks': 100,
            'resources': [{'kind': 'gpu', 'id': 'GPU-explicit', 'label': 'Training GPU', 'scope': 'assigned'}],
            'stages': {'training': [row], 'evaluation': [{'id': 'validation', 'label': 'Validation', 'state': 'complete',
                'comparisons': [comparison, {**comparison, 'metric': 'kl', 'metricLabel': 'KL', 'value': .05,
                                           'goal': 'target', 'target': .1}]}]}})
        source = self.collect()[0]['sources'][0]
        self.assertEqual(source['resources'][0]['id'], 'GPU-explicit')
        stage = source['stages']['training'][0]
        for field in ('resources', 'activity', 'dependencies', 'heartbeatAt', 'heartbeatTimeoutSeconds', 'lastProgressAt', 'progressTimeoutSeconds'):
            self.assertEqual(stage[field], row[field])
        self.assertNotIn('health', stage)
        results = source['stages']['evaluation'][0]['comparisons']
        self.assertEqual((results[0]['goal'], results[1]['goal'], results[1]['target']), ('maximize', 'target', .1))
        self.assertEqual(results[0]['checkpointUnit'], 'epochs')

    def test_native_validation_retains_accuracy_loss_perplexity_and_observational_metrics_per_checkpoint(self):
        output = self.jobs / 'sft'
        self.write_json(output / 'run_config.json', {'run_name': 'Classification SFT', 'evaluation_dataset': 'heldout-v2',
            'evaluation_context': {'temperature': 0, 'max_new_tokens': 128}, 'evaluation_context_label': 'Greedy, 128 token budget',
            'evaluation_metrics': {'accuracy': {'unit': 'fraction'}}})
        self.write_json(output / 'trainer_state.json', {'global_step': 20, 'max_steps': 100, 'log_history': [
            {'step': 10, 'eval_loss': .8, 'eval_accuracy': .7, 'eval_perplexity': 2.2, 'eval_kl': .12, 'eval_custom_metric': -2},
            {'step': 20, 'eval_loss': .4, 'eval_accuracy': .85, 'eval_perplexity': 1.5, 'eval_kl': .09, 'eval_custom_metric': -1},
        ]})
        source = self.collect()[0]['sources'][0]
        results = source['stages']['evaluation']
        self.assertEqual(len(results), 2)
        latest = {value['metricLabel']: value for value in results[0]['comparisons']}
        self.assertEqual({name: value['goal'] for name, value in latest.items()},
                         {'Loss': 'minimize', 'Accuracy': 'maximize', 'Perplexity': 'minimize', 'Kl': 'observe', 'Custom metric': 'observe'})
        self.assertEqual(latest['Accuracy']['unit'], 'fraction')
        self.assertEqual(latest['Loss']['checkpoint'], 20)
        self.assertEqual(results[1]['comparisons'][0]['checkpoint'], 10)
        self.assertTrue(all('baseline' not in item for result in results for item in result['comparisons']))
        original = latest['Loss']['contextId']
        settings = module.run_config(output)
        settings['evaluation_context']['temperature'] = .8
        changed = module.validation_stages([{'step': 20, 'eval_loss': .4}], settings, output)
        self.assertNotEqual(original, changed[0]['comparisons'][0]['contextId'])

    def test_vision_speech_reward_and_custom_target_metrics_keep_native_units_and_epoch_coordinates(self):
        output = self.jobs / 'vision'
        output.mkdir()
        (output / 'history.csv').write_text('epoch,loss,val_f1,val_iou,val_wer,val_reward,val_kl,val_custom_score\n0,1,.7,.6,.2,2,.15,4\n1,.5,.8,.7,.1,3,.08,5\n')
        self.write_json(output / 'run_config.json', {'run_name': 'Vision and custom metrics',
            'evaluation_metrics': {'kl': {'goal': 'target', 'target': .1}, 'custom_score': {'goal': 'minimize', 'unit': 'ms'}}})
        results = self.collect()[0]['sources'][0]['stages']['evaluation']
        metrics = {entry['metricLabel']: entry for entry in results[0]['comparisons']}
        self.assertEqual((metrics['F1']['goal'], metrics['Iou']['goal'], metrics['Wer']['goal'], metrics['Reward']['goal']),
                         ('maximize', 'maximize', 'minimize', 'maximize'))
        self.assertEqual((metrics['Kl']['goal'], metrics['Kl']['target']), ('target', .1))
        self.assertEqual((metrics['Custom score']['goal'], metrics['Custom score']['unit']), ('minimize', 'ms'))
        self.assertTrue(all(item['checkpointUnit'] == 'epochs' and item['checkpoint'] == 2 for item in metrics.values()))
        self.assertNotIn('unit', metrics['F1'])

    def test_process_resources_report_affinity_and_visible_devices_without_exposing_environment(self):
        output = self.jobs / 'native'
        self.write_json(output / 'trainer_state.json', {'global_step': 1, 'max_steps': 10})
        process = self.process(args=['python', 'train.py', '--output_dir', str(output)])
        with (process / 'status').open('a') as status:
            status.write('Cpus_allowed_list:\t0-3,8\n')
        (process / 'environ').write_bytes(b'CUDA_VISIBLE_DEVICES=2,GPU-visible\0SECRET_TOKEN=never-expose\0')
        source = self.collect()[0]['sources'][0]
        self.assertEqual({value['id'] for value in source['resources']}, {'cpus-0-3,8', '2', 'GPU-visible'})
        self.assertEqual({value['scope'] for value in source['resources']}, {'affinity', 'visible'})
        self.assertNotIn('never-expose', json.dumps(source))

    def test_validation_history_drops_pre_resume_measurements_and_never_invents_a_checkpoint(self):
        results = module.validation_stages([{'step': 10, 'eval_accuracy': .5}, {'step': 40, 'eval_accuracy': .8},
            {'step': 5, 'eval_accuracy': .3}], {}, self.jobs, current=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['comparisons'][0]['checkpoint'], 5)
        unknown = module.validation_stages([{'eval_accuracy': .4}], {}, self.jobs)
        self.assertNotIn('comparisons', unknown[0])
        self.assertEqual(unknown[0]['metrics'][0]['value'], .4)

    def test_native_unknown_settings_are_not_given_a_fabricated_context_identity(self):
        validation = module.validation_stages([{'epoch': 0, 'val_loss': .5, 'test_loss': .6}], {}, self.jobs, 'epochs')
        comparisons = [value for row in validation for value in row['comparisons']]
        self.assertTrue(all('contextId' not in value for value in comparisons))
        self.assertEqual(len({value['metric'] for value in comparisons}), 2, 'Unknown validation/test contexts must not merge')

    def test_explicit_native_comparison_ids_can_join_common_baseline_documents(self):
        config = {'evaluation_cohort_id': 'dataset-dev-v2', 'evaluation_context_id': 'greedy-128',
                  'evaluation_metrics': {'accuracy': {'id': 'accuracy', 'goal': 'maximize', 'unit': 'fraction'}}}
        native = module.validation_stages([{'step': 20, 'eval_accuracy': .8}], config, self.jobs)[0]['comparisons'][0]
        baseline = module.normalized_comparison({'metric': 'accuracy', 'metricLabel': 'Accuracy', 'value': .7,
            'seriesId': 'released', 'seriesLabel': 'Released', 'baseline': True, 'goal': 'maximize', 'unit': 'fraction',
            'cohortId': 'dataset-dev-v2', 'contextId': 'greedy-128', 'checkpointUnit': 'steps'})
        self.assertEqual([(native[key], baseline[key]) for key in ('metric', 'cohortId', 'contextId', 'unit', 'goal', 'checkpointUnit')],
                         [(native[key], native[key]) for key in ('metric', 'cohortId', 'contextId', 'unit', 'goal', 'checkpointUnit')])

    def test_colocated_training_and_evaluation_reporters_share_a_run_without_overwriting(self):
        self.process(pid=123)
        self.process(pid=456)
        for directory, group, pid in (('trainer', 'training', 123), ('evaluator', 'evaluation', 456)):
            self.write_json(self.jobs / directory / 'training-progress.json', {
                'schema_version': 1, 'run_id': 'vision-transfer', 'source_id': directory,
                'source_label': directory.title(), 'pid': pid, 'process_start_ticks': 100,
                'stages': {group: [{'id': 'work', 'label': directory.title(), 'state': 'running',
                                   'completed': .5, 'total': None, 'unit': 'epochs'}]},
            })
        run, = self.collect()
        self.assertEqual(len(run['sources']), 2)
        self.assertEqual(len({source['id'] for source in run['sources']}), 2)
        self.assertEqual({source['sparkId'] for source in run['sources']}, {'worker-a'})
        self.assertEqual({source['label'] for source in run['sources']}, {'Trainer', 'Evaluator'})
        training = next(source for source in run['sources'] if source['stages']['training'])
        evaluation = next(source for source in run['sources'] if source['stages']['evaluation'])
        self.assertEqual(training['stages']['training'][0]['completed'], .5)
        self.assertEqual(evaluation['stages']['evaluation'][0]['attemptId'], '456:100')

    def test_multiple_independent_runs_on_one_device_remain_separate(self):
        self.process()
        for index in range(4):
            self.write_json(self.jobs / f'experiment-{index}/training-progress.json', {
                'schema_version': 1, 'pid': 123, 'process_start_ticks': 100,
                'stages': {'training': [{'label': 'Train', 'state': 'running'}]},
            })
        runs = self.collect()
        self.assertEqual(len(runs), 4)
        self.assertEqual(len({run['id'] for run in runs}), 4)
        self.assertEqual(len({run['sources'][0]['id'] for run in runs}), 4)
        self.assertEqual({run['sources'][0]['sparkId'] for run in runs}, {'worker-a'})

    def test_stage_process_overrides_follow_independent_process_lifetimes(self):
        self.process(pid=123, start=100)
        self.process(pid=456, start=200)
        file = self.jobs / 'training-progress.json'
        document = {'schema_version': 1, 'pid': 123, 'process_start_ticks': 100, 'stages': {
            'training': [{'id': 'train', 'label': 'Train', 'state': 'running'}],
            'evaluation': [{'id': 'eval', 'label': 'Eval', 'state': 'running', 'pid': 456}],
        }}
        self.write_json(file, document)
        groups = self.collect()[0]['sources'][0]['stages']
        # PID456 must not inherit PID123's start ticks from the document.
        self.assertEqual((groups['training'][0]['state'], groups['evaluation'][0]['state']), ('running', 'running'))
        self.assertEqual(groups['evaluation'][0]['attemptId'], '456:200')
        document['stages']['evaluation'][0]['process_start_ticks'] = 200
        self.write_json(file, document)
        shutil.rmtree(self.proc / '123')
        groups = self.collect()[0]['sources'][0]['stages']
        self.assertEqual((groups['training'][0]['state'], groups['evaluation'][0]['state']), ('stopped', 'running'))
        self.process(pid=123, start=100)
        self.process(pid=456, start=300)
        groups = self.collect()[0]['sources'][0]['stages']
        self.assertEqual((groups['training'][0]['state'], groups['evaluation'][0]['state']), ('running', 'stopped'))
        self.assertNotIn('attemptId', groups['evaluation'][0])
        self.process(pid=456, start=200, state='T')
        self.assertEqual(self.collect()[0]['sources'][0]['stages']['evaluation'][0]['state'], 'stopped')

    def test_explicit_native_run_identity_joins_ranks_and_an_evaluator_across_hosts(self):
        for index in range(2):
            output = self.jobs / f'rank-{index}'
            self.write_json(output / 'run_config.json', {'run_id': 'distributed-fit', 'run_name': 'Distributed fit',
                'source_label': f'Rank {index}'})
            self.write_json(output / 'trainer_state.json', {'global_step': 10, 'max_steps': 100})
            self.process(pid=123 + index, args=['python', 'train.py', '--output_dir', str(output)])
        self.write_json(self.jobs / 'evaluator/training-progress.json', {
            'schema_version': 1, 'run_id': 'distributed-fit', 'source_id': 'held-out-evaluator',
            'source_label': 'Held-out evaluator', 'stages': {'evaluation': [
                {'id': 'validation', 'label': 'Validation', 'state': 'complete', 'metrics': [{'label': 'Accuracy', 'value': .8}]}]},
        })
        local, = self.collect()
        remote, = module.collect([self.jobs], 'worker-b', self.proc, now=3000000)
        self.assertEqual(local['id'], remote['id'])
        self.assertEqual(len(local['sources']), 3)
        self.assertTrue({source['id'] for source in local['sources']}.isdisjoint(source['id'] for source in remote['sources']))
        self.assertEqual(len([source for source in local['sources'] if source['stages']['training']]), 2)

    def test_explicit_reporter_ids_deduplicate_copies_but_path_defaults_do_not(self):
        for directory in ('original', 'copy'):
            self.write_json(self.jobs / directory / 'training-progress.json', {
                'schema_version': 1, 'run_id': 'shared', 'source_id': 'one-logical-reporter',
                'stages': {'training': [{'id': 'train', 'label': 'Train', 'state': 'complete'}]},
            })
        run, = self.collect()
        self.assertEqual(len(run['sources']), 1)
        self.assertNotEqual(module.reporter_identity('worker-a', 'path-a'), module.reporter_identity('worker-a', 'path-b'))
        # Identity fields are hashed in full, rather than truncated like labels.
        self.assertNotEqual(module.reporter_identity('worker-a', 'same', {'source_id': 'x' * 300 + 'a'}),
                            module.reporter_identity('worker-a', 'same', {'source_id': 'x' * 300 + 'b'}))

    def test_explicit_manifest_overrides_native_logs_for_the_same_output_directory(self):
        output = self.jobs / 'experiment'
        self.write_json(output / 'trainer_state.json', {'global_step': 10, 'max_steps': 20})
        self.write_json(output / 'training-progress.json', {'schema_version': 1, 'campaign': 'Explicit job',
            'stages': {'training': [{'id': 'train', 'label': 'Training', 'state': 'complete'}]}})
        run, = self.collect()
        self.assertEqual(run['campaign'], 'Explicit job')

    def test_payload_bound_keeps_live_reporters_instead_of_dropping_an_entire_distributed_run(self):
        self.process()
        for index in range(4):
            self.write_json(self.jobs / f'worker-{index}/training-progress.json', {
                'schema_version': 1, 'run_id': 'many-workers', 'source_id': f'worker-{index}',
                'pid': 123, 'process_start_ticks': 100,
                'stages': {'training': [{'id': 'train', 'label': 'Training',
                    'state': 'running' if index == 3 else 'complete',
                    'metrics': [{'label': 'Settings', 'value': 'x' * 100}]}]},
            })
        full, = self.collect()
        active = next(source for source in full['sources'] if source['stages']['training'][0]['state'] == 'running')
        budget = len(json.dumps([{**full, 'sources': [active]}]).encode()) + 8
        with patch.object(module, 'MAX_PAYLOAD_BYTES', budget):
            bounded, = self.collect()
        self.assertEqual(len(bounded['sources']), 1)
        self.assertEqual(bounded['sources'][0]['stages']['training'][0]['state'], 'running')

    def test_transformers_output_argument_proves_liveness_and_newest_state_after_resume(self):
        output = self.jobs / 'transformers'
        self.write_json(output / 'checkpoint-30/trainer_state.json', {'global_step': 30, 'max_steps': 30})
        self.write_json(output / 'trainer_state.json', {
            'global_step': 3, 'max_steps': 40, 'epoch': .3,
            'log_history': [{'step': 3, 'loss': 2.5, 'learning_rate': .0001}, {'step': 3, 'eval_loss': 2.1, 'eval_accuracy': .85}],
        })
        os.utime(output / 'checkpoint-30/trainer_state.json', (2000, 2000))
        self.process(args=['python', 'train.py', '--output_dir=' + str(output)])
        run, = self.collect()
        row, = run['sources'][0]['stages']['training']
        self.assertEqual((row['completed'], row['total'], row['state'], row['loss']), (3, 40, 'running', 2.5))
        metrics = {m['label']: m['value'] for m in row['metrics']}
        self.assertEqual(metrics['Epoch'], .3)
        self.assertEqual(run['sources'][0]['stages']['evaluation'][0]['metrics'][1]['label'], 'Accuracy')
        self.process(args=['python', 'train.py', '--output_dir', str(output)], state='T')
        self.assertEqual(self.collect()[0]['sources'][0]['stages']['training'][0]['state'], 'unknown')

    def test_process_output_arguments_do_not_expand_configured_discovery_roots(self):
        output = self.root / 'not-admitted'
        self.write_json(output / 'trainer_state.json', {'global_step': 3, 'max_steps': 40})
        self.process(args=['python', 'train.py', '--output_dir=' + str(output)])
        self.assertEqual(self.collect(), [])

    def test_unrelated_process_cwd_does_not_activate_a_trainer(self):
        output = self.jobs / 'experiment'
        self.write_json(output / 'trainer_state.json', {'global_step': 4, 'max_steps': 10})
        self.process(args=['sleep', '100'], cwd=output)
        self.assertEqual(self.collect()[0]['sources'][0]['stages']['training'][0]['state'], 'unknown')

    def test_lightning_sparse_csv_uses_hparams_and_output_process(self):
        output = self.jobs / 'lightning'
        version = output / 'lightning_logs/version_2'
        version.mkdir(parents=True)
        (version / 'hparams.yaml').write_text('run_name: "Vision encoder transfer"\nmodel_name: resnet50\ntrainer:\n  max_steps: 200\n')
        (version / 'metrics.csv').write_text('epoch,step,train_loss,val_loss,val_accuracy\n0,49,0.9,,\n0,49,,0.8,0.76\n1,99,0.6,,\n')
        self.process(args=['python', 'vision_train.py', '--default_root_dir', str(output)])
        run, = self.collect()
        row, = run['sources'][0]['stages']['training']
        self.assertEqual(run['campaign'], 'Vision encoder transfer')
        self.assertEqual(run['model'], 'resnet50')
        self.assertEqual((row['completed'], row['total'], row['unit'], row['state'], row['loss']), (99, 200, 'steps', 'running', .6))
        validation, = run['sources'][0]['stages']['evaluation']
        self.assertEqual({m['label']: m['value'] for m in validation['metrics']}, {'Loss': .8, 'Accuracy': .76})
        self.process(args=['python', 'vision_train.py', '--default_root_dir', str(output)], state='T')
        row = self.collect()[0]['sources'][0]['stages']['training'][0]
        self.assertEqual(row['state'], 'unknown')

    def test_keras_epoch_csv_names_unknown_total_and_process_exit(self):
        output = self.jobs / 'image-classifier'
        output.mkdir()
        file = output / 'training.log'
        file.write_text('epoch,accuracy,loss,val_accuracy,val_loss\n0,0.7,0.6,0.68,0.7\n1,0.8,0.4,0.77,0.5\n2,0.99,')
        self.write_json(output / 'run_config.json', {'experiment_name': 'Image classifier · augmentation', 'model_name': 'CNN'})
        self.process(args=['python', 'train.py', '--output-dir', str(output)])
        run, = self.collect()
        row, = run['sources'][0]['stages']['training']
        self.assertEqual(run['campaign'], 'Image classifier · augmentation')
        self.assertEqual((row['completed'], row['total'], row['unit'], row['state']), (2, None, 'epochs', 'running'))
        self.assertEqual(row['loss'], .4)
        validation = run['sources'][0]['stages']['evaluation'][0]
        self.assertEqual(len(run['sources'][0]['stages']['evaluation']), 2)
        self.assertEqual({m['label']: m['value'] for m in validation['metrics']}['Accuracy'], .77)
        shutil.rmtree(self.proc / '123')
        self.assertEqual(self.collect()[0]['sources'][0]['stages']['training'][0]['state'], 'unknown')

    def test_csv_uses_configured_total_but_does_not_fabricate_final_completion(self):
        output = self.jobs / 'keras'
        output.mkdir()
        file = output / 'history.csv'
        file.write_text('epoch,loss,val_loss\n4,0.3,0.4\n')
        self.write_json(output / 'run_config.json', {'run_name': 'Last epoch', 'epochs': 5})
        self.process(args=['python', 'train.py', '--output-dir', str(output)])
        row = self.collect()[0]['sources'][0]['stages']['training'][0]
        self.assertEqual((row['completed'], row['total'], row['state']), (5, 5, 'running'))
        # Reusing the output path cannot activate a CSV older than the new process.
        os.utime(file, (900, 900))
        self.assertEqual(self.collect()[0]['sources'][0]['stages']['training'][0]['state'], 'unknown')

    def test_trl_preserves_dpo_and_grpo_metrics_and_latest_validation(self):
        for name, method, measurements in (
            ('dpo', 'DPO', {'rewards/accuracies': .87, 'rewards/margins': 1.8, 'entropy': .4}),
            ('grpo', 'GRPO', {'reward': .75, 'kl': .03, 'completions/mean_length': 128}),
        ):
            output = self.jobs / name
            self.write_json(output / 'training_args.json', {'run_name': f'{method} alignment', 'method': method})
            self.write_json(output / 'adapter_config.json', {'peft_type': 'LORA', 'base_model_name_or_path': 'example/model'})
            self.write_json(output / 'trainer_state.json', {'global_step': 42, 'max_steps': 100,
                'log_history': [{'step': 20, 'eval_loss': .7, 'eval_accuracy': .6},
                                {'step': 40, 'eval_loss': .5}, {'step': 42, 'loss': .3, **measurements}]})
        self.process(args=['python', 'dpo_train.py', '--output_dir', str(self.jobs / 'dpo'), '--run_name', 'Chosen preference run'])
        runs = {run['campaign']: run for run in self.collect()}
        dpo = runs['Chosen preference run']
        row = dpo['sources'][0]['stages']['training'][0]
        self.assertEqual(row['label'], 'DPO training')
        self.assertEqual(row['state'], 'running')
        metrics = {m['label']: m['value'] for m in row['metrics']}
        self.assertEqual((metrics['Rewards / accuracies'], metrics['Rewards / margins'], metrics['Adapter']), (.87, 1.8, 'LORA'))
        self.assertEqual(dpo['model'], 'example/model')
        validation = dpo['sources'][0]['stages']['evaluation'][0]
        self.assertEqual(validation['metrics'], [{'label': 'Loss', 'value': .5}])
        grpo = runs['GRPO alignment']['sources'][0]['stages']['training'][0]
        self.assertEqual({m['label']: m['value'] for m in grpo['metrics']}['Kl'], .03)

    def test_csv_tail_is_bounded_and_safe_yaml_and_manifest_names(self):
        output = self.jobs / 'large'
        output.mkdir()
        file = output / 'metrics.csv'
        file.write_text('epoch,step,train_loss\n' + ''.join(f'0,{step},0.5\n' for step in range(30000)))
        rows = module.csv_values(file)
        self.assertEqual(len(rows), 128)
        self.assertEqual(rows[-1]['step'], 29999)
        (output / 'hparams.yaml').write_text('name: !!python/object:bad\ntrainer:\n  max_steps: 40000\n')
        config = module.load_config(output / 'hparams.yaml')
        self.assertNotIn('name', config)
        self.assertEqual(config['trainer.max_steps'], 40000)
        self.write_json(output / 'training-progress.json', {'schema_version': 1, 'run_name': 'Custom distributed pretraining',
            'stages': {'training': [{'label': 'Train', 'state': 'complete'}]}})
        run, = self.collect()
        self.assertEqual(run['campaign'], 'Custom distributed pretraining')

    def test_csv_separators_resume_and_metric_label_bounds(self):
        output = self.jobs / 'separated'
        output.mkdir()
        for separator in (';', '\t'):
            file = output / 'training.csv'
            header = separator.join(('epoch', 'loss', 'val_loss', 'metric_' + 'x' * 100)) + '\n'
            file.write_text(header + separator.join(('4', '.1', '.2', '12')) + '\n'
                            + separator.join(('0', '.8', '', '13')) + '\n')
            run, = self.collect()
            row = run['sources'][0]['stages']['training'][0]
            self.assertEqual(row['completed'], 1)
            self.assertFalse(run['sources'][0]['stages']['evaluation'])
            self.assertTrue(all(len(item['label']) <= 80 for item in row['metrics']))
        self.assertFalse(module.number(10**400))

    def test_discovery_is_bounded_and_skips_dependency_trees_and_symlinks(self):
        for name in ('one', 'two', 'three'):
            self.write_json(self.jobs / name / 'trainer_state.json', {'global_step': 1})
        self.write_json(self.jobs / 'node_modules/hidden/trainer_state.json', {'global_step': 1})
        (self.jobs / 'linked').symlink_to(self.jobs / 'one', target_is_directory=True)
        found = module.discover([self.jobs], max_files=2)
        self.assertEqual(len(found), 2)
        self.assertFalse(any('node_modules' in path.parts or 'linked' in path.parts for path in found))
        self.assertEqual(module.discover([self.jobs], max_entries=1), [])
        self.assertEqual(len(module.discover([self.jobs / 'linked'])), 1)


if __name__ == '__main__':
    unittest.main()
