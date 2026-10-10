from labtrack.database import ExperimentStore
from labtrack.parser import parse_log
from labtrack.report import generate_report, summarize_metrics
from tests.test_metrics import create_experiment


def test_summary_uses_highest_epoch():
    metrics = parse_log('Epoch 8 loss=.4 accuracy=.91 success rate=95%\nEpoch 1 loss=.2 accuracy=.5').metrics
    assert summarize_metrics(metrics) == dict(final_loss=.4, best_loss=.2, final_accuracy=91, final_success_rate=95)
    assert summarize_metrics([]) == {}
    assert summarize_metrics(parse_log('Epoch 1 lr=.001').metrics) == {}


def test_report_contains_fields_and_missing_results(tmp_path):
    store = ExperimentStore(tmp_path / 'db.sqlite')
    uid = create_experiment(store)
    row = store.get(uid)
    metrics = parse_log('Epoch 1 loss=.4 accuracy=90% success rate=85%').metrics
    report = generate_report(row, metrics)
    for text in ['metrics experiment', 'VLA', 'LIBERO', 'Batch size：16', 'Learning rate：0.001',
                 'Epochs：8', 'A100', 'ckpt.pt', 'python train.py', 'python eval.py', '80%',
                 'Final Loss：0.4', 'Best Loss：0.4', 'Final Accuracy (%)：90',
                 'Final Success Rate (%)：85', 'notes', row['created_time'], row['updated_time']]:
        assert text in report
    empty = generate_report(dict(row, success_rate=None), [])
    assert '尚未评测' in empty and '暂无训练指标' in empty
    assert 'Final Loss' not in empty


def test_markdown_preserves_multiline_code(tmp_path):
    store = ExperimentStore(tmp_path / 'db.sqlite')
    uid = create_experiment(store)
    row = dict(store.get(uid), train_command='echo ```\npython train.py', notes='first\nsecond', name='a [link]')
    report = generate_report(row, [])
    assert '````text\necho ```\npython train.py\n````' in report
    assert 'first\nsecond' in report
    assert r'a \[link\]' in report


def test_report_phase_three(tmp_path):
    store = ExperimentStore(tmp_path / 'db.sqlite')
    uid = create_experiment(store)
    store.update(uid, {'status': 'running', 'tags': ['VLA', 'baseline'],
                       'train_command': 'python train.py --batch_size 32 --lr=.001 --epochs 8 --gpu A100 --fp16'})
    report = generate_report(store.get(uid), [])
    for text in ['## Experiment Status\n\nrunning', '## Tags', '- VLA', '- baseline',
                 '## Parsed Training Arguments', 'fp16: True', '## Configuration Consistency',
                 'Batch size mismatch: LabTrack = 16; Command']:
        assert text in report
    row = dict(store.get(uid), train_command='python t.py --batch_size 16 --lr 0.001')
    assert 'All provided tracked parameters match' in generate_report(row, [])
    assert '无法确认一致性' in generate_report(dict(row, train_command='python t.py'), [])
    assert '命令解析失败，无法核对' in generate_report(dict(row, train_command='python t.py --model "bad'), [])
