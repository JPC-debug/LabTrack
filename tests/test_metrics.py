import sqlite3
import pytest
from labtrack.database import ExperimentStore
from labtrack.parser import parse_log


def create_experiment(store):
    return store.create(dict(name='metrics experiment', model='VLA', dataset='LIBERO', batch_size=16,
                            learning_rate=.001, epochs=8, gpu='A100', checkpoint_path='ckpt.pt',
                            train_command='python train.py', eval_command='python eval.py',
                            success_rate=80, notes='notes'))


def test_metrics_persistence_upsert_and_isolation(tmp_path):
    path = tmp_path / 'db.sqlite'
    store = ExperimentStore(path)
    uid = create_experiment(store)
    other = create_experiment(store)
    before = store.get(uid)
    store.add_metrics(uid, parse_log('Epoch 2 loss=1\nEpoch 1 loss=2 lr=.001').metrics)
    assert [m['step'] for m in ExperimentStore(path).get_metrics(uid)] == [1, 1, 2]
    store.add_metrics(uid, parse_log('Epoch 2 loss=.5').metrics)
    assert len(store.get_metrics(uid)) == 3
    assert store.get_metrics(uid)[-1]['metric_value'] == .5
    assert store.get_metrics(other) == []
    assert store.get(uid)['created_time'] == before['created_time']
    assert store.get(uid)['updated_time'] != before['updated_time']
    assert store.get(uid)['success_rate'] == 80
    store.add_metrics(uid, parse_log('Epoch 3 accuracy=.9').metrics, replace=True)
    assert len(store.get_metrics(uid)) == 1
    store.delete_metrics(uid)
    assert store.get_metrics(uid) == []
    assert store.get(uid)


def test_delete_cascades_and_foreign_key(tmp_path):
    store = ExperimentStore(tmp_path / 'db.sqlite')
    uid, other = create_experiment(store), create_experiment(store)
    metrics = parse_log('Epoch 1 loss=1').metrics
    store.add_metrics(uid, metrics)
    store.add_metrics(other, metrics)
    with store.connect() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute('INSERT INTO experiment_metrics (experiment_id,step,metric_name,metric_value) VALUES (999,1,?,?)', ('loss', 1))
    store.delete(uid)
    assert store.get_metrics(uid) == []
    assert len(store.get_metrics(other)) == 1
    with store.connect() as conn:
        assert conn.execute('PRAGMA foreign_key_check').fetchall() == []
    with pytest.raises(ValueError):
        store.add_metrics(uid, metrics)
    with pytest.raises(ValueError):
        store.delete_metrics(uid)


@pytest.mark.parametrize('bad', [[], [{'step': -1, 'metric_name': 'loss', 'metric_value': 1}],
    [{'step': 1, 'metric_name': 'unknown', 'metric_value': 1}],
    [{'step': 1, 'metric_name': 'loss', 'metric_value': float('nan')}],
    [{'step': 1, 'metric_name': 'accuracy', 'metric_value': 101}],
    [{'step': True, 'metric_name': 'loss', 'metric_value': 1}]])
def test_invalid_replace_keeps_old_metrics(tmp_path, bad):
    store = ExperimentStore(tmp_path / 'db.sqlite')
    uid = create_experiment(store)
    valid = parse_log('Epoch 1 loss=1').metrics
    store.add_metrics(uid, valid)
    before = store.get_metrics(uid)
    with pytest.raises(ValueError):
        store.add_metrics(uid, valid + bad if bad else [], replace=True)
    assert store.get_metrics(uid) == before


def test_upgrade_existing_database(tmp_path):
    path = tmp_path / 'old.sqlite'
    store = ExperimentStore(path)
    uid = create_experiment(store)
    before = store.get(uid)
    # Mimic phase-one database: experiments exists, metrics table does not.
    with store.connect() as conn:
        conn.execute('DROP TABLE experiment_metrics')
    upgraded = ExperimentStore(path)
    assert upgraded.get(uid) == before
    upgraded.add_metrics(uid, parse_log('Epoch 1 loss=1').metrics)
    assert len(upgraded.get_metrics(uid)) == 1


def test_replace_rolls_back_on_database_failure(tmp_path):
    store = ExperimentStore(tmp_path / 'db.sqlite')
    uid = create_experiment(store)
    store.add_metrics(uid, parse_log('Epoch 1 loss=2').metrics)
    before = store.get_metrics(uid)
    before_row = store.get(uid)
    with store.connect() as conn:
        conn.execute('''CREATE TRIGGER fail_insert BEFORE INSERT ON experiment_metrics
                        WHEN NEW.step=3 BEGIN SELECT RAISE(ABORT, 'simulated failure'); END''')
    with pytest.raises(sqlite3.IntegrityError):
        store.add_metrics(uid, parse_log('Epoch 2 loss=1\nEpoch 3 loss=.5').metrics, replace=True)
    assert store.get_metrics(uid) == before
    assert store.get(uid) == before_row
