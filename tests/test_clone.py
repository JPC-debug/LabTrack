import sqlite3
import pytest
from labtrack.database import ExperimentStore
from labtrack.parser import parse_log
from tests.test_metrics import create_experiment


def test_clone_copies_configuration_and_resets_results(tmp_path):
    store = ExperimentStore(tmp_path / 'clone.db')
    uid = create_experiment(store)
    store.update(uid, {'status': 'completed', 'tags': ['VLA', 'baseline']})
    store.add_metrics(uid, parse_log('Epoch 1 loss=1 accuracy=85%').metrics)
    source = store.get(uid)
    source_metrics = store.get_metrics(uid)
    clone_id = store.clone(uid)
    clone = ExperimentStore(store.path).get(clone_id)
    assert clone['name'] == source['name'] + '_copy'
    for field in ['model', 'dataset', 'batch_size', 'learning_rate', 'epochs', 'gpu',
                  'checkpoint_path', 'train_command', 'eval_command', 'notes', 'tags']:
        assert clone[field] == source[field]
    assert clone['status'] == 'planned'
    assert clone['success_rate'] is None
    assert clone['id'] != source['id']
    assert clone['created_time'] != source['created_time']
    assert clone['updated_time'] != source['updated_time']
    assert clone['created_time'] == clone['updated_time']
    assert store.get_metrics(clone_id) == []
    assert store.get(uid) == source and store.get_metrics(uid) == source_metrics
    assert len(store.list_tags()) == 2
    second = store.clone(uid)
    assert store.get(second)['name'] == source['name'] + '_copy_2'
    store.update(clone_id, {'name': 'Edited clone', 'tags': ['debug'], 'batch_size': 32})
    assert store.get(uid) == source
    store.delete(clone_id)
    assert store.get(uid) == source and store.get_metrics(uid) == source_metrics


def test_missing_clone_and_atomic_failure(tmp_path):
    store = ExperimentStore(tmp_path / 'clone.db')
    with pytest.raises(ValueError):
        store.clone(999)
    assert store.list() == []
    uid = create_experiment(store)
    store.update(uid, {'tags': ['VLA']})
    before = store.get(uid)
    with store.connect() as conn:
        conn.execute(f'''CREATE TRIGGER fail_clone BEFORE INSERT ON experiment_tags
                         WHEN NEW.experiment_id != {uid}
                         BEGIN SELECT RAISE(ABORT,'simulated failure'); END''')
    with pytest.raises(sqlite3.IntegrityError):
        store.clone(uid)
    assert store.list() == [before]
    assert store.list_tags()[0]['experiment_count'] == 1
