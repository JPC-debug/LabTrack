"""Parse a single command as text only; never launch a shell or subprocess."""
import re
import shlex
from decimal import Decimal, InvalidOperation

TRACKED = {'batch_size': ('batch_size', 'batch-size'),
           'learning_rate': ('learning_rate', 'learning-rate', 'lr'),
           'epochs': ('epochs',), 'gpu': ('gpu',)}
PARAMETER_LABELS = {'batch_size': 'Batch size', 'learning_rate': 'Learning rate',
                    'epochs': 'Epochs', 'gpu': 'GPU'}


def parse_train_command(command):
    """Return long-option values from POSIX text, or raise an actionable ValueError."""
    command = command.replace('\\\r\n', '').replace('\\\n', '')
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=';&|<>')
        lexer.whitespace_split = True
        lexer.commenters = ''
        tokens = list(lexer)
    except ValueError as exc:
        raise ValueError(f'训练命令无法解析：{exc}。请检查引号或转义符。') from exc
    if any(token in (';', '&&', '||', '|', '>', '>>', '<', '&') for token in tokens):
        raise ValueError('仅支持单条命令，不支持管道、重定向或 shell 控制符。')
    arguments = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == '--':
            break
        if token.startswith('--'):
            key, separator, value = token[2:].partition('=')
            if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_-]*', key):
                raise ValueError(f'训练命令参数名称无效：{token}')
            if not separator:
                value = True
                if index + 1 < len(tokens):
                    next_token = tokens[index + 1]
                    negative_number = re.fullmatch(r'-\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', next_token)
                    if not next_token.startswith('-') or negative_number:
                        index += 1
                        value = tokens[index]
            arguments[key] = value
        index += 1
    return arguments


def check_consistency(experiment, arguments):
    """Check only explicitly provided tracked flags, including every LR alias."""
    checks = []
    for field, aliases in TRACKED.items():
        for alias in aliases:
            if alias not in arguments:
                continue
            command_value = arguments[alias]
            recorded = experiment[field]
            valid = True
            if field == 'gpu':
                matches = not isinstance(command_value, bool) and str(recorded).strip() == str(command_value).strip()
                valid = not isinstance(command_value, bool)
            else:
                try:
                    if isinstance(command_value, bool):
                        raise InvalidOperation()
                    left, right = Decimal(str(recorded)), Decimal(str(command_value))
                    valid = right.is_finite() and right > 0
                    if field in ('epochs', 'batch_size'):
                        valid = valid and right == right.to_integral_value()
                    matches = valid and left == right
                except (InvalidOperation, ValueError):
                    matches = valid = False
            checks.append(dict(field=field, parameter=alias, label=PARAMETER_LABELS[field],
                               recorded=recorded, command=command_value, matches=matches, valid=valid))
    return checks
