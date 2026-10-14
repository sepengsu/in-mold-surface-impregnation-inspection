"""Blind Train-Audit planning, Dev policy freeze and one-shot Check validation."""

from collections import Counter, defaultdict
from pathlib import Path
import shutil

import numpy as np

from .disk import DiskArray
from .labels import OriginalLabels, read_mask
from .protocol import fingerprint, read_json, sha256, write_json


def select_common(raw, original, probability, rules):
    if probability.shape != (2, *original.shape) or raw.shape != original.shape:
        raise ValueError('Aligned native class probabilities and candidate masks are required.')
    if (not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any()
            or not np.allclose(probability.sum(0), 1, atol=1e-5)):
        raise ValueError('Invalid OOF probabilities.')
    proposal = probability.argmax(0).astype(np.uint8)  # ties select Background
    selected = np.zeros_like(raw, dtype=bool)
    for label, name in ((0, '0_to_1'), (1, '1_to_0')):
        rule = rules[name]
        if rule['enabled']:
            selected |= (raw & (original == label) & (proposal != original)
                         & (probability[1-label] >= rule['min_proposal_probability'])
                         & (probability[label] <= rule['max_self_confidence']))
    return selected, proposal


def validate_rules(policy):
    if policy.get('approved') is not True or not str(policy.get('reviewer', '')).strip():
        raise ValueError('The Dev-only policy requires explicit human approval and reviewer identity.')
    if policy.get('selection_basis') != 'Train-Audit-Dev only; never Gold or Audit-Check':
        raise ValueError('Policy selection must be based exclusively on Train-Audit-Dev.')
    if set(policy['rules']) != {'0_to_1', '1_to_0'}:
        raise ValueError('Separate rules for both correction directions are required.')
    for rule in policy['rules'].values():
        if set(rule) != {'enabled', 'min_proposal_probability', 'max_self_confidence'}:
            raise ValueError('This version supports direction/probability rules only; no implicit boundary exclusion.')
        if type(rule['enabled']) is not bool or any(not 0 <= rule[k] <= 1 for k in
                ('min_proposal_probability', 'max_self_confidence')):
            raise ValueError('Invalid direction rules.')
    goal = policy['check_operating_target']
    if (not 0 < goal['minimum_case_bootstrap_precision_lower'] <= 1
            or goal['minimum_cases_per_enabled_direction'] < 2
            or goal['minimum_confirmed_pixels_per_enabled_direction'] < 1):
        raise ValueError('Invalid prospective Audit-Check operating targets.')


def audit_sources(experiment):
    config = read_json(experiment / 'experiment.json')
    rows = read_json(experiment / 'splits.json')['splits']['train']
    surface = read_json(experiment / 'oof/surface.json')
    from .protocol import verified
    from .oof import validate_oof
    validate_oof(experiment, rows, config['settings']['data'])
    if not verified(experiment/'issues',('ignore.npy','scoring.json','label_quality.npy')):
        raise ValueError('Raw CL artifacts are incomplete or changed.')
    return (config, rows, OriginalLabels(config['settings']['data']),
            DiskArray(experiment / 'issues/ignore.npy'),
            DiskArray(experiment / 'oof' / surface['probability_file']))


def plan_audit(experiment, authorization=None):
    config = read_json(experiment / 'experiment.json')
    rows = read_json(experiment / 'splits.json')['splits']['train']
    gold = read_json(experiment / 'gold_scope.json')
    excluded_cases = set(gold['cases'])
    exception = None
    if authorization:
        exception = read_json(authorization)
        if (exception.get('approved') is not True or exception.get('allow_gold_case_overlap') is not True
                or not str(exception.get('human_user_instruction', '')).strip()):
            raise ValueError('A Gold Case overlap exception needs an explicit recorded human instruction.')
        exception = dict(exception, path=str(Path(authorization).resolve()), sha256=sha256(authorization))
        excluded_cases = set()
    eligible = [i for i, r in enumerate(rows) if r['group'] not in excluded_cases
                and r['image_sha256'] not in gold['image_sha256']]
    path = experiment / 'audit/plan.json'
    if path.exists():
        raise ValueError('Audit selection is immutable; use a new experiment for a different audit plan.')
    wanted = config['design']['audit_representative_tiles'] + config['design']['audit_diagnostic_tiles']
    groups = sorted({rows[i]['group'] for i in eligible})
    if len(groups) < 4 or len(eligible) < wanted:
        result = {'status': 'blocked_no_independent_train_audit', 'eligible_tiles': len(eligible),
                  'eligible_cases': groups, 'gold_cases': gold['cases'], 'scope_exception': exception,
                  'reason': 'Gold covers every current Case. Strict exclusion leaves no Train-Audit. '
                            'New independent Case data or an explicit human scope exception is required.',
                  'desired_representative_tiles': config['design']['audit_representative_tiles'],
                  'desired_diagnostic_tiles': config['design']['audit_diagnostic_tiles'], 'tiles': []}
        write_json(experiment / 'audit/blocker.json', result)
        return result
    _, _, original, raw, probability = audit_sources(experiment)
    rng = np.random.default_rng(config['settings']['seed'])
    rng.shuffle(groups)
    dev_cases = set(groups[:len(groups)//2])
    selected = []
    for partition in ('dev', 'check'):
        pool = [i for i in eligible if (rows[i]['group'] in dev_cases) == (partition == 'dev')]
        representative_n = config['design']['audit_representative_tiles']//2
        diagnostic_n = config['design']['audit_diagnostic_tiles']//2
        if len(pool) < representative_n + diagnostic_n:
            raise ValueError('Insufficient tiles in a Case-disjoint Audit partition; split is not altered.')
        representative = list(map(int, rng.choice(pool, representative_n, replace=False)))
        remainder = [i for i in pool if i not in representative]
        # Directions, boundary, small components and self-confidence strata.
        # Diagnostic selection is intentionally not a population estimator.
        from scipy.ndimage import binary_erosion, label
        buckets = defaultdict(list)
        for i in remainder:
            y = original.load(rows[i])[1].astype(bool)
            issue = raw[i]
            labels, _ = label(y)
            sizes = np.bincount(labels.ravel())
            small = (labels > 0) & (sizes[labels] <= 64)
            boundary = y ^ binary_erosion(y)
            direction = 'remove' if int((issue & y).sum()) >= int((issue & ~y).sum()) else 'add'
            structure = 'boundary' if (issue & boundary).any() else 'small' if (issue & small).any() else 'other'
            high = bool(issue.any()) and float(np.max(probability[i][:, issue], axis=0).mean()) >= .9
            confidence = 'high' if high else 'mixed'
            buckets[(direction, structure, confidence)].append(i)
        for bucket in buckets.values():
            rng.shuffle(bucket)
        diagnostic = []
        keys = sorted(buckets)
        while len(diagnostic) < diagnostic_n:
            for key in keys:
                if buckets[key] and len(diagnostic) < diagnostic_n:
                    diagnostic.append(buckets[key].pop())
        for kind, indexes in (('representative', representative), ('diagnostic', diagnostic)):
            for i in indexes:
                row = rows[i]
                folder = experiment / 'audit' / partition / f'tile_{i:04d}'
                folder.mkdir(parents=True)
                shutil.copyfile(Path(config['settings']['data']) / row['image'], folder / 'image.png')
                selected.append({'index': i, 'image': row['image'], 'image_sha256': row['image_sha256'],
                                 'original_mask_sha256': row['mask_sha256'], 'case': row['group'],
                                 'partition': partition, 'sample_kind': kind,
                                 'inclusion_probability': representative_n / len(pool) if kind == 'representative' else None,
                                 'blind_image': str(folder / 'image.png'),
                                 'reviewed': False, 'reviewer': '', 'judged_before_cl_display': False,
                                 'human_alpha_mask': '', 'human_alpha_sha256': '',
                                 'uncertain_mask': '', 'uncertain_sha256': ''})
    plan = {'status': 'awaiting_blind_review', 'tiles': selected, 'scope_exception': exception,
            'gold_images_always_excluded': True, 'dev_cases': sorted(dev_cases),
            'check_cases': sorted(set(groups)-dev_cases), 'population_claim': False,
            'instruction': 'Review whole 256x256 masks before CL display; record uncertainty separately. '
                           'Complete review.json, preserving every selection field. These masks never become training targets.',
            'sample_size_note': '30 representative + 30 diagnostic are a starting design, not evidence of adequacy.'}
    write_json(path, plan)
    write_json(experiment / 'audit/review.template.json', plan)
    return plan


def load_review(experiment, manifest):
    plan = read_json(experiment / 'audit/plan.json')
    if not plan['tiles']:
        raise ValueError(plan['reason'])
    review = read_json(manifest)
    if len(review['tiles']) != len(plan['tiles']):
        raise ValueError('Review must cover the full frozen sample.')
    mutable = {'reviewed', 'reviewer', 'judged_before_cl_display', 'human_alpha_mask', 'human_alpha_sha256',
               'uncertain_mask', 'uncertain_sha256'}
    project = Path(__file__).resolve().parents[2]
    gold_root = (project / read_json(experiment/'experiment.json')['design']['gold']).resolve()
    for before, after in zip(plan['tiles'], review['tiles']):
        if {k:v for k,v in after.items() if k not in mutable} != {k:v for k,v in before.items() if k not in mutable}:
            raise ValueError('Review changed frozen sample membership, ordering or provenance.')
        if not after['reviewed'] or not after['reviewer'] or not after['judged_before_cl_display']:
            raise ValueError('Whole-mask independent human review is required.')
        if sha256(after['blind_image']) != after['image_sha256']:
            raise ValueError('Blind review image changed from the frozen selected image.')
        for name in ('human_alpha', 'uncertain'):
            mask_key = 'human_alpha_mask' if name == 'human_alpha' else 'uncertain_mask'
            path = (Path(manifest).parent / after[mask_key]).resolve()
            if path.is_relative_to(gold_root):
                raise ValueError('Gold masks cannot serve as Train-Audit policy-development GT.')
            if not after[mask_key] or sha256(path) != after[name + '_sha256']:
                raise ValueError('Reviewed mask/uncertainty provenance changed.')
            after[mask_key] = str(path)
    dev = {r['case'] for r in review['tiles'] if r['partition'] == 'dev'}
    check = {r['case'] for r in review['tiles'] if r['partition'] == 'check'}
    if dev & check:
        raise ValueError('Audit Dev and Check must be Case-disjoint.')
    return review


def quality_counts(original, human, uncertain, candidate, proposal):
    valid = ~uncertain
    wrong = original != human
    fix = candidate & valid & wrong & (proposal == human)
    harm = candidate & valid & ~wrong & (proposal != human)
    return {'candidate_pixels': int(candidate.sum()), 'confirmed_candidate_pixels': int((candidate & valid).sum()),
            'ambiguous_candidate_pixels': int((candidate & uncertain).sum()),
            'corrected': int(fix.sum()), 'harmed': int(harm.sum()),
            'original_errors': int((wrong & valid).sum()), 'caught_errors': int((candidate & wrong & valid).sum()),
            'noncandidate_exposure_fn': int((~candidate & valid & (original == 0) & human).sum()),
            'valid_pixels': int(valid.sum()), 'reviewed_pixels': int(human.size)}


def quality_summary(rows, repetitions=2000, seed=42):
    keys = ('candidate_pixels', 'confirmed_candidate_pixels', 'ambiguous_candidate_pixels', 'corrected', 'harmed',
            'original_errors', 'caught_errors', 'noncandidate_exposure_fn', 'valid_pixels', 'reviewed_pixels')
    total = {k: sum(r[k] for r in rows) for k in keys}
    cases = sorted({r['case'] for r in rows if r['candidate_pixels']})
    total.update(cases=cases, net_label_error_reduction=total['corrected']-total['harmed'],
                 precision_confirmed=total['caught_errors']/total['confirmed_candidate_pixels'] if total['confirmed_candidate_pixels'] else None,
                 proposal_accuracy_confirmed=total['corrected']/total['confirmed_candidate_pixels'] if total['confirmed_candidate_pixels'] else None,
                 error_recall=total['caught_errors']/total['original_errors'] if total['original_errors'] else None,
                 candidate_density=total['candidate_pixels']/total['reviewed_pixels'] if total['reviewed_pixels'] else None,
                 errors_per_million_reviewed_pixels=1e6*total['original_errors']/total['reviewed_pixels'] if total['reviewed_pixels'] else None,
                 case_bootstrap_pessimistic_precision_interval=None)
    if len(cases) >= 2:
        counts = np.array([[sum(r['corrected'] for r in rows if r['case']==c),
                            sum(r['candidate_pixels'] for r in rows if r['case']==c)] for c in cases])
        rng = np.random.default_rng(seed)
        values = []
        for _ in range(repetitions):
            sampled = counts[rng.integers(len(cases), size=len(cases))].sum(0)
            values.append(float(sampled[0]/sampled[1]))
        total['case_bootstrap_pessimistic_precision_interval'] = np.quantile(values, [.025,.975]).tolist()
    return total


def evaluate_audit(experiment, manifest, partition, policy=None):
    config, rows, original, raw, probability = audit_sources(experiment)
    review = load_review(experiment, manifest)
    output = experiment / 'audit' / f'{partition}_quality.json'
    if partition == 'check' and output.exists():
        raise ValueError('Audit-Check is one-shot; its result must not be used to tune and rerun the policy.')
    frozen_policy = read_json(experiment / 'policy/frozen.json') if partition == 'check' else read_json(policy) if policy else None
    if frozen_policy:
        validate_rules(frozen_policy)
    if partition == 'check':
        if frozen_policy['dev_report_sha256'] != sha256(experiment/'audit/dev_quality.json'):
            raise ValueError('Frozen Dev evaluation changed before Audit-Check.')
    tables = defaultdict(list)
    from scipy.ndimage import binary_erosion, distance_transform_edt, label
    for entry in review['tiles']:
        if entry['partition'] != partition:
            continue
        i = entry['index']
        y = original.load(rows[i])[1]
        human = read_mask(entry['human_alpha_mask']) > 0
        uncertain = read_mask(entry['uncertain_mask']) > 0
        if human.shape != uncertain.shape or human.shape != (256,256):
            raise ValueError('Review masks must be complete native 256x256 masks.')
        common, proposal = select_common(raw[i], y, probability[i], frozen_policy['rules']) if frozen_policy else (raw[i], probability[i].argmax(0).astype(np.uint8))
        boundary = human ^ binary_erosion(human)
        distance = distance_transform_edt(~boundary) if boundary.any() else np.full(human.shape,np.inf)
        components, _ = label(human)
        sizes = np.bincount(components.ravel())
        regions = {'all': np.ones_like(human), 'boundary_0_2': distance<=2, 'boundary_2_5': (distance>2)&(distance<=5),
                   'boundary_gt5': distance>5, 'small_exposure_le64': (components>0)&(sizes[components]<=64),
                   'other_structure': ~((components>0)&(sizes[components]<=64))}
        for scope, selection in (('raw', raw[i]), ('common', common)):
            for direction, region in (('all', np.ones_like(human)), ('0_to_1', y==0), ('1_to_0', y==1)):
                for name, area in regions.items():
                    eligible = region & area
                    counts = quality_counts(y, human, uncertain | ~eligible, selection & eligible, proposal)
                    counts['reviewed_pixels'] = int(eligible.sum())
                    tables[f'{scope}/{direction}/{name}'].append(dict(counts, case=entry['case'], index=i,
                                                                     sample_kind=entry['sample_kind']))
    report = {'partition': partition, 'review_manifest': str(Path(manifest).resolve()),
              'review_sha256': sha256(manifest), 'plan_sha256': sha256(experiment/'audit/plan.json'),
              'policy_sha256': sha256(experiment/'policy/frozen.json') if partition=='check' else sha256(policy) if policy else None,
              'population_claim': False, 'scope': 'Mixed representative/diagnostic sample only; no unweighted population precision claim.',
              'uncertainty': 'Ambiguous pixels excluded from confirmed counts; counted as failures in conservative gate denominator.',
              'bootstrap_note': 'Cases resampled, never pixels; few Cases and spatial dependence limit confirmation.',
              'summaries': {k:quality_summary(v, config['design']['case_bootstrap_repetitions']) for k,v in tables.items()},
              'per_tile': dict(tables)}
    write_json(output, report)
    return report


def freeze_policy(experiment, draft):
    policy = read_json(draft)
    validate_rules(policy)
    dev = read_json(experiment / 'audit/dev_quality.json')
    if dev['partition'] != 'dev' or dev['policy_sha256'] != sha256(draft):
        raise ValueError('Evaluate this exact approved draft on Audit-Dev before freezing it.')
    if sha256(dev['review_manifest']) != dev['review_sha256']:
        raise ValueError('Human review changed since Dev evaluation.')
    load_review(experiment,dev['review_manifest'])
    if (experiment/'audit/check_quality.json').exists() or (experiment/'policy/frozen.json').exists():
        raise ValueError('Policy freeze is one-shot and must precede Audit-Check.')
    policy.update(dev_report_sha256=sha256(experiment/'audit/dev_quality.json'),
                  audit_plan_sha256=sha256(experiment/'audit/plan.json'),
                  draft_sha256=sha256(draft), experiment_fingerprint=read_json(experiment/'experiment.json')['fingerprint'])
    policy['fingerprint'] = fingerprint(policy)
    write_json(experiment/'policy/frozen.json', policy)
    return policy


def require_check_evidence(experiment):
    policy = read_json(experiment/'policy/frozen.json')
    check = read_json(experiment/'audit/check_quality.json')
    validate_rules(policy)
    if (check['policy_sha256'] != sha256(experiment/'policy/frozen.json') or check['plan_sha256'] != policy['audit_plan_sha256']
            or policy['dev_report_sha256'] != sha256(experiment/'audit/dev_quality.json')
            or check['review_sha256'] != sha256(check['review_manifest'])):
        raise ValueError('Audit-Check did not validate this exact frozen policy and sample.')
    load_review(experiment,check['review_manifest'])
    goal = policy['check_operating_target']
    for direction, rule in policy['rules'].items():
        if not rule['enabled']:
            continue
        result = check['summaries'][f'common/{direction}/all']
        interval = result['case_bootstrap_pessimistic_precision_interval']
        if (len(result['cases']) < goal['minimum_cases_per_enabled_direction']
                or result['confirmed_candidate_pixels'] < goal['minimum_confirmed_pixels_per_enabled_direction']
                or interval is None or interval[0] < goal['minimum_case_bootstrap_precision_lower']):
            raise ValueError(f'{direction}: insufficient evidence for automatic correction; no candidate padding or fallback.')
    return policy
