#!/usr/bin/env python3
# Copyright 2026 Maxrio9. SPDX-License-Identifier: Apache-2.0
"""Discover progress documents, Transformers state and framework CSV logs."""

import csv
import hashlib
import io
from itertools import islice
import json
import math
import os
from pathlib import Path
import re
import sys
import time


MAX_DEPTH = 5
MAX_ENTRIES = 3000
MAX_FILES = 48
MAX_RUNS = 24
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_PAYLOAD_BYTES = 1024 * 1024
MAX_CSV_BYTES = 256 * 1024
SKIP_DIRS = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', '.cache',
             'models', 'model', 'datasets', 'recordings', 'wandb', 'vendor', 'weights', '_archive'}
STATES = {'complete', 'running', 'waiting', 'failed', 'stopped', 'unknown'}
GROUPS = ('data', 'training', 'evaluation')
OUTPUT_FLAGS = ('--output_dir', '--output-dir', '--run-dir', '--run_dir', '--default_root_dir', '--log_dir', '--log-dir', '--save-dir')
CONFIG_FLAGS = ('--config', '--config-file', '--config_file')
CSV_NAMES = {'metrics.csv', 'history.csv', 'training.csv', 'training.log'}
CONFIG_NAMES = ('config.json', 'training_args.json', 'run_config.json', 'hparams.json', 'adapter_config.json', 'hparams.yaml', 'config.yaml')


def load_json(path):
    try:
        if not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
            return None
        return json.loads(path.read_text())
    except (OSError, UnicodeError, ValueError):
        return None


def load_config(path):
    """Read JSON or a bounded YAML scalar subset; never construct YAML objects."""
    try:
        if not path.is_file() or path.stat().st_size > 64 * 1024:
            return {}
    except OSError:
        return {}
    if path.suffix not in ('.yaml', '.yml'):
        value = load_json(path)
        return value if isinstance(value, dict) else {}
    try:
        result, stack = {}, []
        for line in path.read_text().splitlines()[:1024]:
            match = re.fullmatch(r'( *)([A-Za-z_][A-Za-z0-9_-]*):(?: +(.*))?', line)
            if not match:
                continue
            indent, key, value = len(match[1]), match[2], match[3]
            while stack and stack[-1][0] >= indent:
                stack.pop()
            if value is None or value == '':
                stack.append((indent, key))
                continue
            # Tags, aliases, containers and multiline values are deliberately ignored.
            if value[0] in '!&*[{|>':
                continue
            try:
                scalar = json.loads(value)
            except ValueError:
                scalar = value[1:-1].replace("''", "'") if value.startswith("'") and value.endswith("'") else value.split(' #', 1)[0].strip()
            if isinstance(scalar, (str, int, float, bool)) or scalar is None:
                result['.'.join([item[1] for item in stack] + [key])] = scalar
        return result
    except (OSError, UnicodeError):
        return {}


def config_value(config, *keys):
    for key in keys:
        value = config.get(key)
        if value is None and '.' in key:
            value = config
            for part in key.split('.'):
                value = value.get(part) if isinstance(value, dict) else None
        if value is not None:
            return value
    return None


def run_config(root, live=None):
    config = {}
    # Later files provide run-specific settings over a model's config.json.
    for name in CONFIG_NAMES:
        config.update(load_config(root / name))
    if live:
        config.update(live.get('metadata', {}))
        for name in ('run_name', 'experiment_name', 'display_name'):
            value = next(flag_values(live['args'], ('--' + name, '--' + name.replace('_', '-'))), None)
            if value is not None:
                config[name] = value
    return config


def run_name(config, fallback):
    return text(config_value(config, 'display_name', 'run_name', 'experiment_name', 'logger.name', 'name')) or fallback


def identity_value(value):
    # Identity fields must not be truncated like labels: equal prefixes are not
    # evidence that two reports describe the same job.
    if isinstance(value, str) and 0 < len(value.strip()) <= 1024 and not re.search(r'[\x00-\x1f\x7f-\x9f]', value):
        return value.strip()
    return None


def run_identity(config, host, path, prefix):
    """Explicit IDs join a logical run across hosts; paths keep other jobs apart."""
    identity = identity_value(config_value(config, 'run_id', 'sparkdash.run_id'))
    value = identity if identity else host + '\0' + str(path)
    return ('manifest-' if identity else prefix) + hashlib.sha256(value.encode()).hexdigest()[:24]


def reporter_identity(host, path, config=None):
    config = config or {}
    identity = identity_value(config_value(config, 'source_id', 'job_id', 'sparkdash.source_id')) or str(path)
    return 'reporter-' + hashlib.sha256((host + '\0' + identity).encode()).hexdigest()[:24]


def reporter_label(config, fallback):
    return text(config_value(config, 'source_label', 'job_label', 'sparkdash.source_label')) or text(fallback) or 'Training process'


def model_name(config):
    return text(config_value(config, 'model_name_or_path', '_name_or_path', 'base_model_name_or_path', 'model_name', 'model.name'))


def metric_label(key):
    return key.replace('_', ' ').replace('/', ' / ').capitalize()


def logged_metrics(values, excluded=()):
    return [item for key, value in values.items() if key not in excluded and (item := metric(metric_label(key), value))][:16]


def count(value):
    return value if type(value) is int and 0 <= value <= 2**53 - 1 else None


def number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def text(value, maximum=120):
    if not isinstance(value, str):
        return None
    clean = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value).strip()
    return clean[:maximum] or None


def metric(label, value, unit=None):
    if not number(value):
        return None
    label = text(label, 80)
    if not label:
        return None
    result = {'label': label, 'value': value}
    if unit:
        result['unit'] = unit
    return result


def normalized_comparison(raw):
    if not isinstance(raw, dict):
        return None
    result = {}
    for key in ('metric', 'seriesId'):
        value = raw.get(key)
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,159}', value):
            return None
        result[key] = value
    for key, maximum in (('metricLabel', 80), ('seriesLabel', 120)):
        value = text(raw.get(key), maximum)
        if not value:
            return None
        result[key] = value
    if not number(raw.get('value')):
        return None
    result['value'] = raw['value']
    baseline = raw.get('baseline')
    if 'baseline' in raw:
        if type(baseline) is not bool:
            return None
        result['baseline'] = baseline
    checkpoint = raw.get('checkpoint')
    if baseline is True:
        if 'checkpoint' in raw:
            return None
    elif not number(checkpoint) or not 0 <= checkpoint <= 2**53 - 1:
        return None
    else:
        result['checkpoint'] = checkpoint
    if 'cohortId' in raw:
        cohort = raw['cohortId']
        if not isinstance(cohort, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,159}', cohort):
            return None
        result['cohortId'] = cohort
    if 'unit' in raw:
        unit = text(raw['unit'], 32)
        if not unit:
            return None
        result['unit'] = unit
    if 'lower' in raw or 'upper' in raw:
        lower, upper = raw.get('lower'), raw.get('upper')
        if not number(lower) or not number(upper) or lower > upper:
            return None
        result.update(lower=lower, upper=upper)
    if 'goal' in raw:
        if raw['goal'] not in ('maximize', 'minimize', 'target', 'observe'):
            return None
        result['goal'] = raw['goal']
    if 'target' in raw:
        if raw.get('goal') != 'target' or not number(raw['target']):
            return None
        result['target'] = raw['target']
    if raw.get('goal') == 'target' and 'target' not in result:
        return None
    for name in ('contextId',):
        value = raw.get(name)
        if name in raw:
            if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,159}', value):
                return None
            result[name] = value
    for name, maximum in (('contextLabel', 120), ('checkpointUnit', 32)):
        if name in raw:
            value = text(raw[name], maximum)
            if not value:
                return None
            result[name] = value
    return result


def normalized_resources(raw):
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw[:32]:
        if not isinstance(item, dict) or item.get('kind') not in ('cpu', 'gpu', 'accelerator'):
            continue
        identity = text(item.get('id'), 160)
        if not identity:
            continue
        value = {'kind': item['kind'], 'id': identity}
        if label := text(item.get('label')):
            value['label'] = label
        if item.get('scope') in ('assigned', 'visible', 'affinity'):
            value['scope'] = item['scope']
        result.append(value)
    return result


def stage_context(raw, row):
    resources = normalized_resources(raw.get('resources'))
    if isinstance(raw.get('resources'), list):
        row['resources'] = resources
    activity = raw.get('activity')
    if isinstance(activity, dict):
        kind, label = activity.get('kind'), text(activity.get('label'))
        if isinstance(kind, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,159}', kind) and label:
            row['activity'] = {'kind': kind, 'label': label}
            if detail := text(activity.get('detail'), 240):
                row['activity']['detail'] = detail
            since = count(activity.get('since'))
            if since is not None and since <= 8.64e15:
                row['activity']['since'] = since
    dependencies = []
    for dependency in raw.get('dependencies', [])[:32] if isinstance(raw.get('dependencies'), list) else []:
        if not isinstance(dependency, dict):
            continue
        identity, label = dependency.get('id'), text(dependency.get('label'))
        if (isinstance(identity, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,159}', identity)
                and label and dependency.get('state') in ('waiting', 'ready', 'failed', 'unknown')):
            dependencies.append({'id': identity, 'label': label, 'state': dependency['state']})
    if dependencies:
        row['dependencies'] = dependencies
    for name in ('heartbeatAt', 'lastProgressAt'):
        value = count(raw.get(name))
        if value is not None and value <= 8.64e15:
            row[name] = value
    for name in ('heartbeatTimeoutSeconds', 'progressTimeoutSeconds'):
        value = raw.get(name)
        if number(value) and 0 < value <= 604800:
            row[name] = value
    return row


def normalized_stage(raw, updated):
    if not isinstance(raw, dict) or raw.get('state') not in STATES:
        return None
    label = text(raw.get('label'))
    if not label:
        return None
    identity = text(raw.get('id'), 80) or hashlib.sha256(label.encode()).hexdigest()[:20]
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,79}', identity):
        identity = hashlib.sha256(identity.encode()).hexdigest()[:20]
    completed = raw.get('completed') if number(raw.get('completed')) and 0 <= raw['completed'] <= 2**53 - 1 else None
    total = raw.get('total') if number(raw.get('total')) and 0 <= raw['total'] <= 2**53 - 1 else None
    if total is not None and completed is not None and completed > total:
        total = None
    row = {'id': identity, 'label': label, 'state': raw['state'], 'completed': completed,
           'total': total, 'unit': text(raw.get('unit'), 32), 'updatedAt': updated}
    detail = text(raw.get('detail'), 240)
    if detail:
        row['detail'] = detail
    for name in ('loss', 'etaSeconds'):
        value = raw.get(name)
        if number(value) and (name == 'loss' or value >= 0):
            row[name] = value
    for name in ('latestCheckpoint', 'nextCheckpoint', 'promptConcurrency'):
        value = count(raw.get(name))
        if value is not None and (name != 'promptConcurrency' or value > 0):
            row[name] = value
    metrics = []
    for item in raw.get('metrics', [])[:16] if isinstance(raw.get('metrics'), list) else []:
        if not isinstance(item, dict):
            continue
        label, value = text(item.get('label'), 80), item.get('value')
        if not label:
            continue
        if not number(value) and value is not None:
            value = text(value)
            if value is None:
                continue
        unit = text(item.get('unit'), 32)
        item = {'label': label, 'value': value}
        if unit:
            item['unit'] = unit
        metrics.append(item)
    if metrics:
        row['metrics'] = metrics
    return stage_context(raw, row)


def flag_values(args, flags):
    for i, arg in enumerate(args):
        if arg in flags and i + 1 < len(args):
            yield args[i + 1]
        else:
            for flag in flags:
                if arg.startswith(flag + '='):
                    yield arg[len(flag) + 1:]


def metric_goal(key):
    """Recognized optimization directions; unknown objectives stay observational."""
    key = re.sub(r'[^a-z0-9]+', '_', key.lower()).strip('_')
    if re.search(r'(^|_)(loss|perplexity|mae|mse|rmse|wer|cer|error|errors)($|_)', key):
        return 'minimize'
    if re.search(r'(^|_)(accuracy|accuracies|precision|recall|f1|f1score|auroc|auc|auprc|bleu|rouge|iou|miou|map|ndcg|mrr|reward|rewards)($|_|[0-9])', key):
        return 'maximize'
    if re.search(r'(^|_)rouge(?:[0-9]+|l(?:sum)?)($|_)', key):
        return 'maximize'
    return 'observe'


def comparison_metadata(config, split):
    result = {}
    cohort = config_value(config, 'evaluation_cohort_id', 'cohort_id', 'sparkdash.cohort_id')
    if identity_value(cohort):
        result['cohortId'] = cohort if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,159}', cohort) else 'cohort-' + hashlib.sha256(cohort.encode()).hexdigest()[:24]
    else:
        dataset = config_value(config, 'evaluation_dataset', 'eval_dataset', 'dataset_name')
        if identity_value(dataset):
            result['cohortId'] = 'cohort-' + hashlib.sha256((str(dataset) + '\0' + split).encode()).hexdigest()[:24]
    context = config_value(config, 'evaluation_context_id', 'sparkdash.context_id')
    configured = config_value(config, 'evaluation_context', 'sparkdash.evaluation_context')
    if identity_value(context):
        identity = str(context)
    elif isinstance(configured, dict) and configured:
        identity = json.dumps(configured, sort_keys=True, separators=(',', ':'))
    else:
        identity = None
    if identity is not None:
        result['contextId'] = context if identity_value(context) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,159}', context) else 'context-' + hashlib.sha256(identity.encode()).hexdigest()[:24]
    label = text(config_value(config, 'evaluation_context_label', 'sparkdash.context_label'))
    if label:
        result['contextLabel'] = label
    elif identity is not None:
        result['contextLabel'] = 'Configured evaluation settings'
    return result


def validation_stages(history, config, root, coordinate='steps', current=None):
    """Keep real validation checkpoints and separate every metric/goal/context."""
    overrides = config_value(config, 'evaluation_metrics', 'sparkdash.evaluation_metrics')
    overrides = overrides if isinstance(overrides, dict) else {}
    series = identity_value(config_value(config, 'evaluation_series_id', 'sparkdash.series_id'))
    series = series or identity_value(config_value(config, 'run_id', 'sparkdash.run_id')) or str(root)
    series_id = 'series-' + hashlib.sha256(series.encode()).hexdigest()[:24]
    series_label = text(config_value(config, 'evaluation_series_label', 'sparkdash.series_label')) or run_name(config, root.name)
    history = history[-128:]
    previous, resumed_at = None, 0
    counter = 'step' if coordinate == 'steps' else 'epoch'
    for index, entry in enumerate(history):
        point = entry.get(counter) if isinstance(entry, dict) else None
        if number(point):
            if previous is not None and point < previous:
                resumed_at = index
            previous = point
    history = history[resumed_at:]
    records = {}
    for entry_index, entry in enumerate(history[-128:]):
        if not isinstance(entry, dict):
            continue
        point = entry.get('step') if coordinate == 'steps' else entry.get('epoch')
        point = point + 1 if coordinate == 'epochs' and number(point) else point
        if number(point) and current is not None and point > current:
            continue
        if not number(point) or not 0 <= point <= 2**53 - 1:
            point = None
        for key, value in entry.items():
            match = re.match(r'^(eval_|val_|val/|test_|test/)(.+)$', key)
            if not match or not number(value):
                continue
            prefix, name = match.groups()
            split = 'evaluation' if prefix == 'eval_' else 'validation' if prefix.startswith('val') else 'test'
            record = records.setdefault((split, point), {'metrics': {}, 'comparisons': {}})
            if point is None and record.get('entry') != entry_index:
                record.update(metrics={}, comparisons={}, entry=entry_index)
            setting = overrides.get(key, overrides.get(name, {}))
            setting = setting if isinstance(setting, dict) else {}
            label = text(setting.get('label'), 80) or text(metric_label(name), 80)
            unit = text(setting.get('unit'), 32)
            record['metrics'][name] = dict(label=label, value=value, **({'unit': unit} if unit else {}))
            if point is None:
                continue
            metric_id = setting.get('id')
            if not isinstance(metric_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,159}', metric_id):
                # A split name is not evidence of equal evaluation settings.
                metric_id = split + '-metric-' + hashlib.sha256(name.encode()).hexdigest()[:24]
            comparison = {'metric': metric_id,
                          'metricLabel': label, 'value': value, 'seriesId': series_id, 'seriesLabel': series_label,
                          'checkpoint': point, 'checkpointUnit': coordinate,
                          'goal': setting.get('goal', metric_goal(name)), **comparison_metadata(config, split)}
            if unit:
                comparison['unit'] = unit
            if 'target' in setting:
                comparison['target'] = setting['target']
            if normalized := normalized_comparison(comparison):
                record['comparisons'][name] = normalized
    result = []
    for index, ((split, point), values) in enumerate(reversed(list(records.items()))):
        suffix = f' · {"Step" if coordinate == "steps" else "Epoch"} {point:g}' if point is not None else ''
        row = {'id': 'validation' if index == 0 else 'validation-' + hashlib.sha256(f'{split}:{point}'.encode()).hexdigest()[:20],
               'label': 'Latest validation' if index == 0 else split.capitalize() + suffix,
               'state': 'complete', 'completed': None, 'total': None, 'unit': None,
               'metrics': list(values['metrics'].values())[:16]}
        if point is not None:
            row['detail'] = f'Recorded at {"step" if coordinate == "steps" else "epoch"} {point:g}.'
        comparisons = list(values['comparisons'].values())[:20]
        if comparisons:
            row['comparisons'] = comparisons
            row['comparison'] = comparisons[0]
        result.append(row)
    return result[:48]


def process_resources(process):
    if not process:
        return []
    resources = []
    affinity = process.get('affinity')
    if isinstance(affinity, str) and re.fullmatch(r'[0-9,-]{1,128}', affinity):
        resources.append({'kind': 'cpu', 'id': 'cpus-' + affinity, 'label': 'CPU cores ' + affinity, 'scope': 'affinity'})
    path = process.get('proc_path')
    try:
        with (path / 'environ').open('rb') as handle:
            environment = handle.read(65537)
        if len(environment) <= 65536:
            for part in environment.split(b'\0'):
                key, separator, value = part.partition(b'=')
                if key not in (b'CUDA_VISIBLE_DEVICES', b'HIP_VISIBLE_DEVICES', b'ROCR_VISIBLE_DEVICES') or not separator:
                    continue
                for identifier in value.decode(errors='replace').split(',')[:32]:
                    identifier = identifier.strip()
                    if identifier == '-1' or not re.fullmatch(r'[A-Za-z0-9_.:/-]{1,160}', identifier):
                        continue
                    item = {'kind': 'gpu', 'id': identifier, 'label': 'GPU ' + identifier, 'scope': 'visible'}
                    if not any(old['kind'] == item['kind'] and old['id'] == identifier for old in resources):
                        resources.append(item)
    except (OSError, TypeError):
        pass
    return normalized_resources(resources)


def source_resources(config, process):
    declared = config_value(config, 'resources', 'sparkdash.resources')
    return normalized_resources(declared) if isinstance(declared, list) else process_resources(process)


def resolved(value, cwd):
    try:
        path = Path(value).expanduser()
        return (cwd / path).resolve() if not path.is_absolute() else path.resolve()
    except (OSError, ValueError, RuntimeError):
        return None


def boot_time(proc):
    try:
        for line in (proc / 'stat').read_text().splitlines():
            if line.startswith('btime '):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


def read_process(pid, proc=Path('/proc'), uid=None, boot=None, ticks=None):
    if count(pid) is None or pid < 1:
        return None
    uid = os.getuid() if uid is None else uid
    path = proc / str(pid)
    try:
        status = (path / 'status').read_text()
        owner = next(line for line in status.splitlines() if line.startswith('Uid:')).split()[1:3]
        if owner != [str(uid), str(uid)]:
            return None
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] in ('Z', 'X', 'x'):
            return None
        start_ticks = int(fields[19])
        args = (path / 'cmdline').read_bytes().decode(errors='replace').rstrip('\0').split('\0')
        cwd = (path / 'cwd').resolve(strict=True)
    except (OSError, ValueError, IndexError, StopIteration, RuntimeError):
        return None
    start = None
    proc_path = path
    if boot is not None:
        start = (boot + start_ticks / (ticks or os.sysconf('SC_CLK_TCK'))) * 1000
    output_dirs = {path for value in flag_values(args, OUTPUT_FLAGS) if (path := resolved(value, cwd))}
    metadata = {}
    for value in flag_values(args, CONFIG_FLAGS):
        config_path = resolved(value, cwd)
        config = load_config(config_path) if config_path else {}
        metadata.update(config)
        if config:
            for key in ('output_dir', 'output-dir', 'run_dir', 'default_root_dir', 'trainer.default_root_dir', 'log_dir'):
                value = config_value(config, key)
                if isinstance(value, str) and (path := resolved(value, cwd)):
                    output_dirs.add(path)
    training_name = re.compile(r'(^|[/_.-])(train|trainer|training|finetune|fine_tune|accelerate|torchrun|deepspeed|lightning|sft|pretrain|run_clm|run_mlm)([/_.-]|$)', re.I)
    trainer = any(training_name.search(Path(arg).name) for arg in args[:3] if not arg.startswith('-'))
    return {'pid': pid, 'start_ticks': start_ticks, 'started': start, 'cwd': cwd,
            'outputs': output_dirs, 'args': args, 'trainer': trainer, 'metadata': metadata, 'paused': fields[0] in ('T', 't'),
            'proc_path': proc_path, 'affinity': next((line.split(':', 1)[1].strip() for line in status.splitlines()
                                               if line.startswith('Cpus_allowed_list:')), None)}


def process_index(proc=Path('/proc'), limit=2048):
    boot, processes = boot_time(proc), {}
    try:
        paths = sorted((path for path in proc.iterdir() if path.name.isdecimal()), key=lambda p: int(p.name))
    except OSError:
        return processes
    for path in paths[:limit]:
        process = read_process(int(path.name), proc, boot=boot)
        if process:
            processes[process['pid']] = process
    return processes


def discover(roots, max_entries=MAX_ENTRIES, max_files=MAX_FILES):
    found, seen, visited = [], set(), 0
    stack = [(path, 0) for root in reversed(roots) if (path := resolved(root, Path.home()))]
    while stack and visited < max_entries and len(found) < max_files:
        directory, depth = stack.pop()
        try:
            key = str(directory.resolve())
            if key in seen or directory.is_symlink():
                continue
            seen.add(key)
            with os.scandir(directory) as iterator:
                entries = sorted((Path(entry.path) for entry in islice(iterator, max_entries - visited)), key=lambda path: path.name)
        except OSError:
            continue
        children = []
        for path in entries:
            visited += 1
            if visited > max_entries or len(found) >= max_files:
                break
            try:
                if path.is_symlink():
                    continue
                if path.is_file() and path.name in {'training-progress.json', 'trainer_state.json', *CSV_NAMES}:
                    found.append(path)
                elif depth < MAX_DEPTH and path.is_dir() and path.name not in SKIP_DIRS and not path.name.startswith('.'):
                    children.append((path, depth + 1))
            except OSError:
                continue
        stack.extend(reversed(children))
    return found


def confirmed_inactive_process(pid, proc):
    """Missing PID or an explicit dead/paused state, never a permission failure."""
    directory = proc / str(pid)
    try:
        fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] in ('Z', 'X', 'x', 'T', 't')
    except FileNotFoundError:
        try:
            directory.stat()
        except FileNotFoundError:
            return True
        except OSError:
            pass
    except (OSError, ValueError, IndexError):
        pass
    return False


def alive_for_document(document, path, processes, inactive_processes=()):
    pid = count(document.get('pid'))
    process = processes.get(pid)
    if process is None:
        return False if pid in inactive_processes else None
    if process['paused']:
        return False
    identity = document.get('process_start_ticks')
    if identity is not None:
        return type(identity) is int and identity == process['start_ticks']
    if process['started'] is None:
        return None
    return path.stat().st_mtime_ns // 1_000_000 >= process['started'] - 1000


def manifest_run(path, document, processes, host, now, inactive_processes=()):
    if not isinstance(document, dict) or type(document.get('schema_version')) is not int or document['schema_version'] != 1:
        return None
    groups = document.get('stages')
    if not isinstance(groups, dict):
        return None
    updated = path.stat().st_mtime_ns // 1_000_000
    stages = {}
    for group in GROUPS:
        rows = groups.get(group, [])
        stages[group] = []
        for raw in rows[:48] if isinstance(rows, list) else []:
            row = normalized_stage(raw, updated)
            if row is None:
                continue
            # Different stages may run in different processes on the same host.
            # A stage PID override must never inherit another process's start ticks.
            identity = dict(document)
            if 'pid' in raw:
                identity['pid'] = raw['pid']
                identity.pop('process_start_ticks', None)
            if 'process_start_ticks' in raw:
                identity['process_start_ticks'] = raw['process_start_ticks']
            live = alive_for_document(identity, path, processes, inactive_processes)
            if group == 'evaluation' and row['state'] == 'complete':
                comparison = normalized_comparison(raw.get('comparison'))
                if comparison:
                    row['comparison'] = comparison
                comparisons = [normalized for item in raw.get('comparisons', [])[:20]
                               if (normalized := normalized_comparison(item))] if isinstance(raw.get('comparisons'), list) else []
                if comparisons:
                    row['comparisons'] = comparisons
                    row.setdefault('comparison', comparisons[0])
            needs_live_process = row['state'] == 'running' or (group == 'evaluation' and row['state'] == 'waiting')
            if needs_live_process and live is not True:
                row['state'] = 'stopped' if live is False else 'unknown'
                row['detail'] = 'Stage process is inactive.' if live is False else 'Live process identity unavailable.'
            if row['state'] == 'running' and 'etaSeconds' in row:
                remaining = row['etaSeconds'] - max(0, now - updated) / 1000
                if remaining >= 0:
                    row['etaSeconds'] = remaining
                else:
                    del row['etaSeconds']
            if row['state'] == 'running' and group == 'evaluation':
                process = processes[identity['pid']]
                row['attemptId'] = f"{process['pid']}:{process['start_ticks']}"
            if row['state'] == 'running' and group == 'training':
                process = processes[identity['pid']]
                row['attemptId'] = f"{process['pid']}:{process['start_ticks']}"
            if 'pid' in raw and 'resources' not in row:
                resources = process_resources(processes.get(count(identity.get('pid'))))
                if resources:
                    row['resources'] = resources
            stages[group].append(row)
    if not any(stages.values()):
        return None
    identity = identity_value(document.get('id')) or identity_value(document.get('run_id'))
    digest = hashlib.sha256((identity if identity else host + '\0' + str(path)).encode()).hexdigest()[:24]
    return {'id': 'manifest-' + digest,
            'campaign': text(document.get('campaign')) or text(document.get('label')) or run_name(document, path.parent.name),
            'model': text(document.get('model')), 'sources': [
                {'id': reporter_identity(host, path, document), 'sparkId': host,
                 'label': reporter_label(document, path.parent.name), 'kind': 'training', 'status': 'live',
                 'updatedAt': now, 'stages': stages,
                 'resources': source_resources(document, processes.get(count(document.get('pid'))))}]}


def trainer_run(path, document, processes, host, now):
    if not isinstance(document, dict) or count(document.get('global_step')) is None:
        return None
    root = path.parent.parent if path.parent.name.startswith('checkpoint-') else path.parent
    updated = path.stat().st_mtime_ns // 1_000_000
    completed, total = count(document['global_step']), count(document.get('max_steps'))
    live = next((process for process in processes.values() if not process['paused']
                 and (root in process['outputs'] or (process['trainer'] and process['cwd'] == root))), None)
    if total is not None and total > 0 and completed >= total:
        state = 'complete'
    else:
        state = 'running' if live else 'unknown'
    row = {'id': 'trainer', 'label': 'Training', 'state': state,
           'completed': completed, 'total': total if total and completed <= total else None,
           'unit': 'steps', 'updatedAt': updated}
    if live and state == 'running':
        row['attemptId'] = f"{live['pid']}:{live['start_ticks']}"
    history = document.get('log_history')
    history = history[-64:] if isinstance(history, list) else []
    latest = {}
    for entry in history:
        if isinstance(entry, dict):
            latest.update(entry)
    if number(latest.get('loss')):
        row['loss'] = latest['loss']
    known = [('Epoch', document.get('epoch'), None), ('Learning rate', latest.get('learning_rate'), None),
             ('Gradient norm', latest.get('grad_norm'), None), ('Training throughput', latest.get('train_samples_per_second'), 'samples/s')]
    metrics = [value for label, value_, unit in known if (value := metric(label, value_, unit))]
    # TRL and custom Trainer subclasses retain their native objective metrics.
    extra = {key: value for key, value in latest.items()
             if not key.startswith('eval_') and key not in {'loss', 'epoch', 'step', 'learning_rate', 'grad_norm', 'train_samples_per_second', 'total_flos'}}
    metrics.extend(logged_metrics(extra))
    if metrics:
        row['metrics'] = metrics[:16]
    if not live and state != 'complete':
        row['detail'] = 'Trainer state found. Live process identity unavailable.'
    config = run_config(root, live)
    for key in ('run_name', 'experiment_name', 'display_name', 'trial_name'):
        if document.get(key):
            config[key] = document[key]
    if 'trial_name' in config and not any(config.get(key) for key in ('run_name', 'experiment_name', 'display_name')):
        config['run_name'] = config['trial_name']
    method = text(config_value(config, 'training_method', 'method', 'algorithm'), 100)
    if not method and all(key in latest for key in ('rewards/accuracies', 'rewards/margins')):
        method = 'Preference optimization'
    if method:
        row['label'] = method + ' training'
    peft = text(config.get('peft_type'))
    if peft:
        row.setdefault('metrics', []).append({'label': 'Adapter', 'value': peft})
        row['metrics'] = row['metrics'][:16]
    evaluation = validation_stages(history, config, root, current=completed)
    return {'id': run_identity(config, host, root, 'transformers-'), 'campaign': run_name(config, root.name), 'model': model_name(config),
            'sources': [{'id': reporter_identity(host, root, config), 'sparkId': host,
                         'label': reporter_label(config, root.name), 'kind': 'training',
                         'status': 'live', 'updatedAt': now,
                         'stages': {'data': [], 'training': [row], 'evaluation': evaluation},
                         'resources': source_resources(config, live)}]}


def csv_values(path):
    """Read a fixed-size tail while retaining the original CSV header."""
    try:
        with path.open('rb') as handle:
            header = handle.readline(8192)
            if not header.endswith(b'\n'):
                return []
            start = handle.tell()
            size = path.stat().st_size
            if size - start > MAX_CSV_BYTES:
                handle.seek(size - MAX_CSV_BYTES)
                handle.readline(MAX_CSV_BYTES)  # Discard the partial first row.
            body = handle.read(MAX_CSV_BYTES)
        # The writer may still be appending the last row.
        if not body.endswith(b'\n'):
            body = body.rsplit(b'\n', 1)[0] + b'\n' if b'\n' in body else b''
        # Keras CSVLogger supports a configurable one-character separator.
        decoded_header = header.decode()
        delimiter = max((',', ';', '\t'), key=decoded_header.count)
        reader = csv.DictReader(io.StringIO((header + body).decode()), delimiter=delimiter)
        if reader.fieldnames is None or len(reader.fieldnames) > 128 or any(key is None or len(key) > 120 for key in reader.fieldnames):
            return []
        rows = []
        for row in reader:
            values = {}
            for key, value in row.items():
                if key is None or value is None:
                    continue
                try:
                    parsed = float(value)
                except ValueError:
                    continue
                if math.isfinite(parsed):
                    values[key] = parsed
            if values:
                rows.append(values)
        return rows[-128:]
    except (OSError, UnicodeError, csv.Error):
        return []


def csv_run(path, processes, host, now):
    rows = csv_values(path)
    if not rows or not any('epoch' in row or 'step' in row for row in rows):
        return None
    # Lightning logs contain step/epoch; Keras CSVLogger writes zero-based epochs.
    lightning = any('step' in row for row in rows)
    root = path.parent
    updated = path.stat().st_mtime_ns // 1_000_000
    live = next((process for process in processes.values() if not process['paused']
                 and (root in process['outputs'] or any(parent in process['outputs'] for parent in list(root.parents)[:MAX_DEPTH])
                      or (process['trainer'] and process['cwd'] == root))
                 and process['started'] is not None and updated >= process['started'] - 1000), None)
    config = run_config(root, live)
    counter = 'step' if lightning else 'epoch'
    previous_counter, resumed_at = None, 0
    for index, entry in enumerate(rows):
        if counter in entry:
            if previous_counter is not None and entry[counter] < previous_counter:
                resumed_at = index
            previous_counter = entry[counter]
    rows = rows[resumed_at:]
    latest = {}
    for entry in rows:
        latest.update(entry)
    # Avoid treating arbitrary CSV datasets with an epoch column as training logs.
    if not any(key in latest for key in ('loss', 'train_loss', 'train_loss_step', 'train_loss_epoch', 'val_loss', 'accuracy', 'train/loss', 'val/loss')):
        return None
    step = latest.get('step')
    epoch = latest.get('epoch')
    completed = step if lightning and step is not None else epoch + 1 if epoch is not None else None
    if completed is not None and not 0 <= completed <= 2**53 - 1:
        return None
    total_value = config_value(config, 'max_steps', 'trainer.max_steps') if lightning and step is not None else config_value(config, 'epochs', 'max_epochs', 'trainer.max_epochs')
    total = total_value if number(total_value) and 0 < total_value <= 2**53 - 1 else None
    if total is not None and completed is not None and completed > total:
        total = None
    # CSVLogger does not record fit termination. Never infer completion from age.
    state = 'running' if live else 'unknown'
    row = {'id': 'fit', 'label': 'Training', 'state': state, 'completed': completed, 'total': total,
           'unit': 'steps' if lightning and step is not None else 'epochs', 'updatedAt': updated}
    if live:
        row['attemptId'] = f"{live['pid']}:{live['start_ticks']}"
    train = {key: value for key, value in latest.items() if not key.startswith(('val_', 'val/', 'test_', 'test/')) and key not in ('step', 'epoch')}
    for key in ('train_loss_epoch', 'train_loss_step', 'train_loss', 'train/loss', 'loss'):
        if key in train:
            row['loss'] = train.pop(key)
            break
    row['metrics'] = logged_metrics(train)
    if lightning and epoch is not None:
        row['metrics'] = [{'label': 'Epoch', 'value': epoch}, *row['metrics']][:16]
    if not live:
        row['detail'] = 'CSV metrics found. Live process identity unavailable; final status is unknown.'
    evaluation = validation_stages(rows, config, root, 'steps' if lightning else 'epochs', completed)
    framework = 'Lightning CSV' if lightning and path.name == 'metrics.csv' else 'Step CSV' if lightning else 'Keras CSVLogger'
    row['metrics'] = [{'label': 'Logger', 'value': framework}, *row['metrics']][:16]
    fallback = f'{root.parent.name} / {root.name}' if lightning and re.fullmatch(r'version_\d+', root.name) else root.name
    return {'id': run_identity(config, host, path, 'csv-'), 'campaign': run_name(config, text(fallback) or 'Training run'), 'model': model_name(config),
            'sources': [{'id': reporter_identity(host, path, config), 'sparkId': host,
                         'label': reporter_label(config, text(fallback) or root.name), 'kind': 'training', 'status': 'live', 'updatedAt': now,
                         'stages': {'data': [], 'training': [row], 'evaluation': evaluation},
                         'resources': source_resources(config, live)}]}


def collect(roots, host, proc=Path('/proc'), now=None):
    now = time.time_ns() // 1_000_000 if now is None else now
    # Linux process evidence may be absent (macOS) or inaccessible. A missing
    # process in that situation is unknown, rather than proof of termination.
    process_evidence_available = boot_time(proc) is not None
    inactive_processes = set()
    processes = process_index(proc)
    # Only operator-configured roots are scanned. Process arguments prove liveness;
    # they never grant access to unrelated directories.
    files = discover(roots)
    manifests, trainers, csv_runs, covered, manifested = [], {}, [], set(), set()
    for path in files:
        try:
            if path.name in CSV_NAMES:
                run = csv_run(path, processes, host, now)
                if run:
                    csv_runs.append((path, run))
                continue
            document = load_json(path)
            if path.name == 'training-progress.json':
                identities = [document] if isinstance(document, dict) else []
                for group in GROUPS:
                    rows = document.get('stages', {}).get(group, []) if isinstance(document, dict) and isinstance(document.get('stages'), dict) else []
                    if isinstance(rows, list):
                        identities.extend(row for row in rows[:48] if isinstance(row, dict))
                for identity in identities:
                    pid = count(identity.get('pid'))
                    if pid is not None and pid not in processes:
                        process = read_process(pid, proc, boot=boot_time(proc))
                        if process:
                            processes[pid] = process
                        elif process_evidence_available and confirmed_inactive_process(pid, proc):
                            inactive_processes.add(pid)
                run = manifest_run(path, document, processes, host, now, inactive_processes)
                if run:
                    manifests.append(run)
                    covered.add(path.parent)
                    manifested.add(path.parent)
            else:
                run = trainer_run(path, document, processes, host, now)
                if run:
                    root = path.parent.parent if path.parent.name.startswith('checkpoint-') else path.parent
                    covered.add(root)
                    key = (run['id'], run['sources'][0]['id'])
                    previous = trainers.get(key)
                    current_stage = run['sources'][0]['stages']['training'][0]
                    old_stage = previous[1]['sources'][0]['stages']['training'][0] if previous else None
                    if old_stage is None or (current_stage['updatedAt'], current_stage['completed']) > (old_stage['updatedAt'], old_stage['completed']):
                        trainers[key] = (root, run)
        except OSError:
            continue
    # A richer manifest or Trainer state takes precedence over a duplicate CSV.
    csv_runs = [run for path, run in csv_runs if path.parent not in covered]
    candidates = [*manifests, *(run for root, run in trainers.values() if root not in manifested), *csv_runs]
    grouped = {}
    for run in candidates:
        existing = grouped.get(run['id'])
        if existing is None:
            grouped[run['id']] = run
            continue
        for source in run['sources']:
            previous = next((item for item in existing['sources'] if item['id'] == source['id']), None)
            if previous is None:
                existing['sources'].append(source)
            else:
                measured = lambda item: max((stage.get('updatedAt', 0) for group in GROUPS for stage in item['stages'][group]), default=0)
                if measured(source) >= measured(previous):
                    existing['sources'][existing['sources'].index(previous)] = source
    def priority(source):
        training = any(row['state'] == 'running' for row in source['stages']['training'])
        evaluation = any(row['state'] in ('running', 'waiting') for row in source['stages']['evaluation'])
        running = any(row['state'] == 'running' for group in GROUPS for row in source['stages'][group])
        measured = max((row.get('updatedAt', 0) for group in GROUPS for row in source['stages'][group]), default=0)
        return training or evaluation, running, measured

    candidates = list(grouped.values())
    candidates.sort(key=lambda run: max((priority(source) for source in run['sources']), default=(False, False, 0)), reverse=True)
    selected, size = [], 2
    for run in candidates:
        bounded = {**run, 'sources': []}
        for source in sorted(run['sources'], key=priority, reverse=True)[:64]:
            candidate = {**bounded, 'sources': [*bounded['sources'], source]}
            if size + len(json.dumps(candidate, allow_nan=False).encode()) + 2 <= MAX_PAYLOAD_BYTES:
                bounded['sources'].append(source)
        if bounded['sources']:
            selected.append(bounded)
            size += len(json.dumps(bounded, allow_nan=False).encode()) + 2
        if len(selected) >= MAX_RUNS:
            break
    return selected


if __name__ == '__main__':
    settings = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    roots = settings.get('roots', [])
    host = settings.get('host', 'local')
    print(json.dumps(collect(roots, host), allow_nan=False))
