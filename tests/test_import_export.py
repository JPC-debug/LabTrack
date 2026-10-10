import csv
import io
import sqlite3
import pytest
from labtrack.database import ExperimentStore
from labtrack.parser import parse_log
from labtrack.import_export import (EXPERIMENT_COLUMNS, METRIC_COLUMNS, export_experiments,
                                   export_metrics, parse_experiments_csv, import_experiments_csv)
from tests.test_metrics import create_experiment


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=EXPERIMENT_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def test_export_and_roundtrip(tmp_path):
    store = ExperimentStore(tmp_path / 'db.sqlite')
    uid = create_experiment(store)
    store.update(uid, {'status': 'completed', 'tags': ['VLA', 'LIBERO'], 'notes': '中文, notes\nsecond line', 'success_rate': None})
    store.add_metrics(uid, parse_log('Epoch 1 loss=.5').metrics)
    exported = export_experiments(store.list())
    rows = list(csv.DictReader(io.StringIO(exported.decode('utf-8-sig'))))
    assert list(rows[0]) == EXPERIMENT_COLUMNS
    assert rows[0]['tags'] == 'LIBERO|VLA'
    assert rows[0]['notes'] == '中文, notes\nsecond line'
    metric_rows = list(csv.DictReader(io.StringIO(export_metrics(store.get_all_metrics()).decode('utf-8-sig'))))
    assert list(metric_rows[0]) == METRIC_COLUMNS
    assert metric_rows[0]['experiment_id'] == str(uid)
    assert parse_experiments_csv(exported).errors == []
    ids = import_experiments_csv(store, exported)
    assert len(ids) == 1 and ids[0] != uid
    imported = store.get(ids[0])
    assert imported['tags'] == ['LIBERO', 'VLA']
    assert imported['status'] == 'completed' and imported['notes'] == rows[0]['notes']
    assert imported['success_rate'] is None
    assert store.get_metrics(ids[0]) == []
    assert imported['created_time'] != store.get(uid)['created_time']
    assert export_experiments([]).decode('utf-8-sig').strip() == ','.join(EXPERIMENT_COLUMNS)
    assert export_metrics([]).decode('utf-8-sig').strip() == ','.join(METRIC_COLUMNS)


def test_invalid_csv_never_writes_partial_data(tmp_path):
    store = ExperimentStore(tmp_path / 'db.sqlite')
    uid = create_experiment(store)
    row = dict(store.get(uid), tags='new_tag| new_tag ||baseline')
    content = csv_bytes([row, dict(row, learning_rate='nan', status='unknown', epochs='1.5')])
    result = parse_experiments_csv(content)
    assert len(result.preview) == 2
    assert any('第 3 行：learning_rate' in e for e in result.errors)
    assert any('第 3 行：status' in e for e in result.errors)
    assert any('第 3 行：epochs' in e for e in result.errors)
    with pytest.raises(ValueError):
        import_experiments_csv(store, content)
    assert len(store.list()) == 1
    assert store.list_tags() == []


def test_atomic_database_failure_rolls_back_tags_and_experiments(tmp_path):
    store = ExperimentStore(tmp_path / 'db.sqlite')
    uid = create_experiment(store)
    row = dict(store.get(uid), tags='new_tag')
    with store.connect() as conn:
        conn.execute('''CREATE TRIGGER fail_csv BEFORE INSERT ON experiments WHEN NEW.name='fail'
                        BEGIN SELECT RAISE(ABORT,'simulated failure'); END''')
    with pytest.raises(sqlite3.IntegrityError):
        import_experiments_csv(store, csv_bytes([row, dict(row, name='fail')]))
    assert len(store.list()) == 1 and store.list_tags() == []


@pytest.mark.parametrize('content', [b'', b'\xff', b'name,model\na,b\n',
    b'name,model,dataset,batch_size,learning_rate,epochs\na,b,c,16,.1\n',
    b'name,model,dataset,batch_size,learning_rate,epochs\n"unclosed',
    b'name,name,model,dataset,batch_size,learning_rate,epochs\n',
    b'x' * (5*1024*1024+1)])
def test_invalid_csv_structure(content):
    assert parse_experiments_csv(content).errors


def test_minimum_columns_and_defaults(tmp_path):
    content = b'name,model,dataset,batch_size,learning_rate,epochs\nminimal,VLA,LIBERO,16,1e-4,8\n'
    result = parse_experiments_csv(content)
    assert not result.errors
    assert result.rows[0]['status'] == 'planned'
    assert result.rows[0]['tags'] == []
    assert result.rows[0]['success_rate'] is None
    store = ExperimentStore(tmp_path / 'db.sqlite')
    assert len(import_experiments_csv(store, content)) == 1
