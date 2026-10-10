import pytest
from labtrack.command_parser import parse_train_command, check_consistency


def test_formats_boolean_and_quoted_values():
    command = 'python train.py \\\n--batch_size 16 \\\n--lr 0.0001 --gpu 2 --epochs=50 --model "open vla" --fp16'
    assert parse_train_command(command) == {'batch_size': '16', 'lr': '0.0001', 'gpu': '2',
                                             'epochs': '50', 'model': 'open vla', 'fp16': True}
    assert parse_train_command('python train.py --offset -2 --epochs=2 --epochs 3')['epochs'] == '3'
    assert parse_train_command('python train.py -- --ignored value') == {}
    assert parse_train_command('') == {}


@pytest.mark.parametrize('command', ['python train.py --model "bad', 'python a.py && python b.py', 'python a.py --=3', 'python a.py;python b.py'])
def test_invalid_commands(command):
    with pytest.raises(ValueError):
        parse_train_command(command)


def test_consistency_numeric_aliases_gpu_and_missing():
    row = dict(batch_size=16, learning_rate=.0001, epochs=50, gpu='2')
    checks = check_consistency(row, parse_train_command('python t.py --batch-size 16 --learning_rate=1e-4 --epochs 50 --gpu 2'))
    assert len(checks) == 4 and all(c['matches'] for c in checks)
    checks = check_consistency(row, parse_train_command('python t.py --batch_size 32 --lr .001 --epochs=20 --gpu=3'))
    assert len(checks) == 4 and not any(c['matches'] for c in checks)
    assert check_consistency(row, parse_train_command('python t.py --model vla')) == []
    assert not check_consistency(row, {'epochs': True})[0]['valid']
    assert not check_consistency(row, {'lr': 'NaN'})[0]['matches']
    assert not check_consistency(row, {'epochs': '50.1'})[0]['matches']
    checks = check_consistency(row, {'lr': '.0001', 'learning_rate': '.1'})
    assert checks[0]['matches'] is False and checks[1]['matches'] is True
    assert row == dict(batch_size=16, learning_rate=.0001, epochs=50, gpu='2')
