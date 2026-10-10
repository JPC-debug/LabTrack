import pytest
from labtrack.database import ExperimentStore


@pytest.fixture
def values():
    return dict(name='VLA baseline', model='OpenVLA', dataset='LIBERO', batch_size=16,
                learning_rate=0.0001, epochs=10, gpu='A100', checkpoint_path='/models/best.pt',
                train_command='python train.py', eval_command='python eval.py', success_rate=80.5, notes='baseline')


def test_crud_persistence(tmp_path, values):
    path = tmp_path / 'experiments.db'
    store = ExperimentStore(path)
    uid = store.create(values)
    original = store.get(uid)
    assert ExperimentStore(path).get(uid)['success_rate'] == 80.5
    store.update(uid, dict(values, name='updated', success_rate=None))
    assert store.get(uid)['name'] == 'updated'
    assert store.get(uid)['success_rate'] is None
    assert store.get(uid)['created_time'] == original['created_time']
    store.delete(uid)
    assert store.get(uid) is None
    assert store.list() == []
    with pytest.raises(ValueError):
        store.update(uid, values)
    with pytest.raises(ValueError):
        store.delete(uid)


def test_search(tmp_path, values):
    store = ExperimentStore(tmp_path / 'db.sqlite')
    uid = store.create(values)
    store.create(dict(values, name="literal 100%_test", model='Other'))
    assert len(store.list(name='baseline', model='openvla')) == 1
    assert len(store.list(name='%_')) == 1
    assert store.list(name="' OR 1=1 --") == []
    day = store.get(uid)['created_time'][:10]
    assert len(store.list(start=day, end=day)) == 2
    assert store.list(end='2000-01-01') == []
    assert store.list(start='2999-01-01') == []


@pytest.mark.parametrize('changes', [{'name': ' '}, {'model': ''}, {'dataset': ''},
    {'batch_size': 0}, {'epochs': 1.5}, {'learning_rate': 0}, {'learning_rate': float('nan')},
    {'success_rate': -1}, {'success_rate': 101}, {'success_rate': float('inf')}])
def test_invalid(tmp_path, values, changes):
    store = ExperimentStore(tmp_path / 'db.sqlite')
    with pytest.raises(ValueError):
        store.create(dict(values, **changes))
    assert store.list() == []


@pytest.mark.parametrize('rate', [None, 0, 100])
def test_rate_boundaries(tmp_path, values, rate):
    store = ExperimentStore(tmp_path / 'db.sqlite')
    uid = store.create(dict(values, success_rate=rate))
    assert store.get(uid)['success_rate'] == rate
