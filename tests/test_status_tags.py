import sqlite3
import pytest
from labtrack.database import ExperimentStore
from labtrack.parser import parse_log
from tests.test_metrics import create_experiment


def test_status_and_tag_lifecycle(tmp_path):
    store = ExperimentStore(tmp_path / 'db.sqlite')
    uid = create_experiment(store)
    assert store.get(uid)['status'] == 'planned'
    assert store.get(uid)['tags'] == []
    row = store.get(uid)
    store.update(uid, dict(row, status='running', tags=[' VLA ', 'VLA', '', 'baseline', ' baseline ']))
    assert store.get(uid)['status'] == 'running'
    assert store.get(uid)['tags'] == ['VLA', 'baseline']
    other = store.create(dict(row, status='completed', tags=['VLA', 'LIBERO']))
    assert [r['id'] for r in store.list(status='running', tag='baseline')] == [uid]
    assert len(store.list(tag='VLA')) == 2
    assert store.list(tag="' OR 1=1 --") == []
    assert next(t for t in store.list_tags() if t['name'] == 'VLA')['experiment_count'] == 2
    store.add_metrics(uid, parse_log('Epoch 1 loss=1').metrics)
    store.delete(uid)
    assert store.get_metrics(uid) == []
    with store.connect() as conn:
        assert conn.execute('SELECT * FROM experiment_tags WHERE experiment_id=?', (uid,)).fetchall() == []
        assert conn.execute('SELECT count(*) FROM tags WHERE name=?', ('VLA',)).fetchone()[0] == 1
    assert store.get(other)['tags'] == ['LIBERO', 'VLA']
    assert store.cleanup_unused_tags() == 1
    assert {t['name'] for t in store.list_tags()} == {'LIBERO', 'VLA'}
    store.update(other, {'notes': 'legacy partial update'})
    assert store.get(other)['status'] == 'completed'
    assert store.get(other)['tags'] == ['LIBERO', 'VLA']


@pytest.mark.parametrize('changes', [{'status': 'unknown'}, {'status': ''}, {'tags': 'VLA,LIBERO'}, {'tags': ['bad|tag']}])
def test_invalid_status_tags(tmp_path, changes):
    store = ExperimentStore(tmp_path / 'db.sqlite')
    uid = create_experiment(store)
    before = store.get(uid)
    with pytest.raises(ValueError):
        store.update(uid, changes)
    assert store.get(uid) == before


def test_real_legacy_migration_preserves_metrics(tmp_path):
    path = tmp_path / 'legacy.sqlite'
    # Exact phase-two schema without status or tag tables, not a phase-three mock.
    with sqlite3.connect(path) as conn:
        conn.execute('''CREATE TABLE experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, model TEXT NOT NULL,
            dataset TEXT NOT NULL, batch_size INTEGER NOT NULL, learning_rate REAL NOT NULL,
            epochs INTEGER NOT NULL, gpu TEXT NOT NULL, checkpoint_path TEXT NOT NULL,
            train_command TEXT NOT NULL, eval_command TEXT NOT NULL, success_rate REAL,
            notes TEXT NOT NULL, created_time TEXT NOT NULL, updated_time TEXT NOT NULL)''')
        conn.execute('''INSERT INTO experiments VALUES
            (7,'old experiment','VLA','LIBERO',16,.001,8,'A100','','','',80,'old notes','2025-01-01','2025-01-02')''')
        conn.execute('''CREATE TABLE experiment_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
            step INTEGER NOT NULL, metric_name TEXT NOT NULL, metric_value REAL NOT NULL,
            UNIQUE(experiment_id,step,metric_name))''')
        conn.execute("INSERT INTO experiment_metrics VALUES (9,7,1,'loss',1.5)")
        original = conn.execute('SELECT * FROM experiments').fetchone()
    store = ExperimentStore(path)
    row = store.get(7)
    assert row['status'] == 'planned' and row['tags'] == []
    assert row['created_time'] == '2025-01-01' and row['updated_time'] == '2025-01-02'
    assert store.get_metrics(7)[0]['id'] == 9
    with store.connect() as conn:
        after = conn.execute('SELECT * FROM experiments').fetchone()
        assert tuple(after)[:-1] == original
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE experiments SET status='unknown' WHERE id=7")
    assert ExperimentStore(path).get(7) == row
    store.delete(7)
    assert store.get_metrics(7) == []
