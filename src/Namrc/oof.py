"""Validate case-held-out provenance before reuse; source outputs stay immutable."""

from pathlib import Path
import shutil

from .disk import DiskArray
from .labels import OriginalLabels
from .protocol import read_json, seal, sha256, verified


def validate_oof(directory, rows, data, *, expected_settings=None):
    directory = Path(directory)
    if not verified(directory/'oof', ('labels.npy','probabilities.npy','provenance.json')):
        raise ValueError('OOF predictions are incomplete.')
    config = read_json(directory/'experiment.json')
    if config['manifest_sha256'] != sha256(Path(data)/'manifest.json'):
        raise ValueError('OOF data manifest mismatch.')
    if expected_settings:
        if bool(config['settings'].get('central_only',False)) != bool(expected_settings.get('central_only',False)):
            raise ValueError('OOF central_only selection differs. Use completed central-only OOF or omit --oof-source to train central-only OOF.')
        keys = ('folds','oof_epochs','patience','batch_size','base_channels','augmentation','seed','device','no_amp','smoke_test')
        if any(config['settings'][key] != expected_settings[key] for key in keys):
            raise ValueError('Imported OOF training settings differ from this paired experiment.')
        for key,default in (('gradient_accumulation_steps',1),('activation_checkpointing',False)):
            if config['settings'].get(key,default) != expected_settings.get(key,default):
                raise ValueError('Imported OOF accumulation/checkpointing settings differ.')
    provenance = read_json(directory/'oof/provenance.json')
    if provenance['train_images'] != [r['image'] for r in rows] or provenance.get('heldout_used_for_checkpoint_selection') is not False:
        raise ValueError('OOF order or held-out checkpoint selection is invalid.')
    folds = read_json(directory/'folds.json')
    predicted, all_cases = [], {r['group'] for r in rows}
    if len(provenance['folds']) != len(folds):
        raise ValueError('OOF fold/provenance count mismatch.')
    for index, (entry, fold) in enumerate(zip(provenance['folds'], folds), 1):
        sets = {key:{r['group'] for r in fold[key]} for key in ('train','val','test')}
        if (sets['test'] & (sets['train']|sets['val']) or sets['train'] & sets['val']
                or sets['train']|sets['val']|sets['test'] != all_cases):
            raise ValueError('OOF case leakage or incomplete fold.')
        for key, recorded in (('train','fit_cases'),('val','selection_cases'),('test','predicted_cases')):
            if sets[key] != set(entry[recorded]):
                raise ValueError('OOF provenance does not match frozen fold membership.')
        fold_directory = directory/'oof'/f'fold_{index}'
        if not verified(fold_directory, ('best.pt','result.json','splits.json','config.json')):
            raise ValueError('OOF checkpoint fit is not completed.')
        if sha256(fold_directory/'best.pt') != entry['checkpoint_sha256']:
            raise ValueError('OOF checkpoint provenance hash mismatch.')
        if read_json(fold_directory/'splits.json')['splits'] != fold:
            raise ValueError('Checkpoint training split differs from its claimed OOF fold.')
        predicted.extend(r['image'] for r in fold['test'])
    if sorted(predicted) != sorted(r['image'] for r in rows) or len(set(predicted)) != len(predicted):
        raise ValueError('Each TRAIN tile must occur in exactly one held-out prediction fold.')
    labels, probabilities = DiskArray(directory/'oof/labels.npy'), DiskArray(directory/'oof/probabilities.npy')
    if labels.shape != (len(rows),256,256) or probabilities.shape != (len(rows),2,256,256):
        raise ValueError('OOF array shape mismatch.')
    original = OriginalLabels(data)
    for index,row in enumerate(rows):
        _, target, valid = original.load(row)
        if not (labels[index][valid] == target[valid]).all():
            raise ValueError('OOF original labels differ from current valid TRAIN labels.')
    return config, provenance


def import_oof(source, output, rows, data, settings):
    source = Path(source).resolve()
    validate_oof(source, rows, data, expected_settings=settings)
    destination = output/'oof'
    destination.mkdir(parents=True, exist_ok=True)
    filenames = ['labels.npy','probabilities.npy','provenance.json']
    for index in range(1,len(read_json(source/'folds.json'))+1):
        for name in ('best.pt','result.json','splits.json','config.json','complete.json'):
            filenames.append(f'fold_{index}/{name}')
    for name in filenames:
        target = destination/name
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary = target.with_name(target.name+'.tmp')
        shutil.copyfile(source/'oof'/name, temporary)
        temporary.replace(target)
    seal(destination, ('labels.npy','probabilities.npy','provenance.json'),
         {'imported_from':str(source),'source_experiment_sha256':sha256(source/'experiment.json')})
