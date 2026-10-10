"""Shared validation for forms, database writes and CSV imports."""
import math

BASE_FIELDS = ['name', 'model', 'dataset', 'batch_size', 'learning_rate', 'epochs', 'gpu',
               'checkpoint_path', 'train_command', 'eval_command', 'success_rate', 'notes']
FIELDS = BASE_FIELDS + ['status']
STATUSES = ('planned', 'running', 'completed', 'failed')


def normalize_tags(tags):
    if tags is None:
        return []
    if not isinstance(tags, (list, tuple)) or any(not isinstance(t, str) for t in tags):
        raise ValueError('tags 必须是文本标签列表')
    result = []
    for tag in tags:
        tag = tag.strip()
        if not tag or tag in result:
            continue
        if any(char in tag for char in '|\n\r'):
            raise ValueError('tags 不能包含竖线或换行符')
        result.append(tag)
    return result


def validate_errors(values):
    data = {key: values.get(key) for key in FIELDS}
    errors = {}
    for key in ['name', 'model', 'dataset']:
        data[key] = str(data[key] or '').strip()
        if not data[key]:
            errors[key] = '不能为空'
    for key in ['gpu', 'checkpoint_path', 'train_command', 'eval_command', 'notes']:
        data[key] = str(data[key] or '').strip()
    for key in ['batch_size', 'epochs']:
        value = data[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not 1 <= value <= 2**63 - 1 or not math.isfinite(value) or int(value) != value):
            errors[key] = '必须是 SQLite 范围内的正整数'
        else:
            data[key] = int(value)
    for key in ['learning_rate', 'success_rate']:
        if key == 'success_rate' and data[key] is None:
            continue
        try:
            if isinstance(data[key], bool):
                raise ValueError()
            value = float(data[key])
            valid = math.isfinite(value) and (value > 0 if key == 'learning_rate' else 0 <= value <= 100)
            if not valid:
                raise ValueError()
            data[key] = value
        except (TypeError, ValueError, OverflowError):
            errors[key] = '必须是有限正数' if key == 'learning_rate' else '必须在 0–100 之间'
    data['status'] = values.get('status', 'planned')
    if data['status'] not in STATUSES:
        errors['status'] = f"不支持 {data['status']}，可选值：{', '.join(STATUSES)}"
    try:
        data['tags'] = normalize_tags(values.get('tags', []))
    except ValueError as exc:
        errors['tags'] = str(exc)
    return data, errors


def validate(values):
    data, errors = validate_errors(values)
    if errors:
        raise ValueError('; '.join(f'{key} {message}' for key, message in errors.items()))
    return data


def validate_metrics(metrics):
    """Validate the entire batch before any database mutation."""
    import math
    from labtrack.parser import METRIC_NAMES
    cleaned = []
    for metric in metrics:
        step = metric.get('step')
        name = metric.get('metric_name')
        value = metric.get('metric_value')
        if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= 2**63 - 1:
            raise ValueError('指标 step 必须是非负整数，且不超过 SQLite 整数范围')
        if name not in METRIC_NAMES:
            raise ValueError('不支持的指标名称')
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError('指标值必须为数值')
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError('指标值必须为有限非负数')
        if name in ('accuracy', 'success_rate') and value > 100:
            raise ValueError('Accuracy / Success rate 必须在 0–100 之间')
        cleaned.append((step, name, value))
    return cleaned
