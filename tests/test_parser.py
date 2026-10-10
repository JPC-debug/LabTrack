import pytest
from labtrack.parser import parse_log, parse_uploaded_log


def test_common_formats():
    result = parse_log('Epoch 1/8, Loss: 1.3526, LR: 0.001\nEpoch 2/8, Loss: 1.0128, LR: 0.001')
    assert result.parsed_lines == 2
    assert result.skipped_lines == 0
    assert result.metrics == [
        {'step': 1, 'metric_name': 'learning_rate', 'metric_value': .001},
        {'step': 1, 'metric_name': 'loss', 'metric_value': 1.3526},
        {'step': 2, 'metric_name': 'learning_rate', 'metric_value': .001},
        {'step': 2, 'metric_name': 'loss', 'metric_value': 1.0128}]
    assert [m['metric_value'] for m in parse_log('Epoch 1, loss=1.3526\nEpoch 2, loss=1.0128').metrics] == [1.3526, 1.0128]


def test_aliases_scientific_rates_and_duplicates():
    result = parse_log('epoch: 0, learning rate=1e-4 accuracy=.85 success_rate=90%\n'
                       'Epoch 2, Loss=0.5, acc=1, success rate=0\nEpoch 2 loss=0.4')
    values = {(m['step'], m['metric_name']): m['metric_value'] for m in result.metrics}
    assert values[(0, 'learning_rate')] == .0001
    assert values[(0, 'accuracy')] == 85
    assert values[(0, 'success_rate')] == 90
    assert values[(2, 'accuracy')] == 100
    assert values[(2, 'success_rate')] == 0
    assert values[(2, 'loss')] == .4
    assert len(result.warnings) == 1


def test_partial_and_failed_logs():
    result = parse_log('python train.py\nEpoch 1 loss=nan\nEpoch 2 loss=-1 LR=1e999 accuracy=105\nEpoch 3 Loss=0.4 LR=0\n')
    assert result.parsed_lines == 1
    assert result.skipped_lines == 3
    assert len(result.metrics) == 2
    assert result.warnings
    assert not parse_log('').metrics
    assert not parse_log('no metrics here').metrics
    assert not parse_log('Epoch 9223372036854775808 loss=1').metrics
    assert not parse_log('Epoch 2 loss=1.2.3').metrics
    assert not parse_log('Epoch 1.5 loss=1').metrics
    assert not parse_log('Epoch ' + '9' * 5000 + ' loss=1').metrics


def test_upload_validation():
    assert parse_uploaded_log(b'\xef\xbb\xbfEpoch 1 loss=1', 'TRAIN.LOG').metrics
    for content, filename in [(b'abc', 'file.py'), (b'\xff', 'file.log'),
                              (b'\x00', 'file.txt'), (b'x' * (5 * 1024 * 1024 + 1), 'file.log')]:
        with pytest.raises(ValueError):
            parse_uploaded_log(content, filename)
