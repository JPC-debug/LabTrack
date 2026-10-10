"""CSV format and validation; database transaction ownership stays in ExperimentStore."""
import csv
import io
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import List
import pandas as pd
from labtrack.validation import BASE_FIELDS, validate_errors

EXPERIMENT_COLUMNS = ['id', 'name', 'model', 'dataset', 'status', 'tags', 'batch_size',
                      'learning_rate', 'epochs', 'gpu', 'checkpoint_path', 'train_command',
                      'eval_command', 'success_rate', 'notes', 'created_time', 'updated_time']
METRIC_COLUMNS = ['experiment_id', 'step', 'metric_name', 'metric_value']
REQUIRED_COLUMNS = ['name', 'model', 'dataset', 'batch_size', 'learning_rate', 'epochs']


@dataclass
class CSVImportResult:
    rows: List[dict] = field(default_factory=list)
    preview: List[dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def export_experiments(rows):
    data = [dict(row, tags='|'.join(row.get('tags', []))) for row in rows]
    return pd.DataFrame(data, columns=EXPERIMENT_COLUMNS).to_csv(index=False).encode('utf-8-sig')


def export_metrics(rows):
    return pd.DataFrame(rows, columns=METRIC_COLUMNS).to_csv(index=False).encode('utf-8-sig')


def parse_experiments_csv(content):
    result = CSVImportResult()
    if len(content) > 5 * 1024 * 1024:
        result.errors.append('文件：CSV 不能超过 5 MB')
        return result
    try:
        text = content.decode('utf-8-sig')
    except UnicodeDecodeError:
        result.errors.append('文件：必须使用 UTF-8 编码')
        return result
    if '\x00' in text:
        result.errors.append('文件：包含二进制内容，必须是纯文本 CSV')
        return result
    csv.field_size_limit(5 * 1024 * 1024)
    reader = csv.DictReader(io.StringIO(text, newline=''), strict=True)
    try:
        headers = reader.fieldnames
        if not headers:
            result.errors.append('第 1 行：缺少 CSV 表头')
            return result
        if len(set(headers)) != len(headers) or any(not h for h in headers):
            result.errors.append('第 1 行：表头包含重复或空字段')
        for column in REQUIRED_COLUMNS:
            if column not in headers:
                result.errors.append(f'第 1 行：{column} 缺少必需字段')
        if result.errors:
            return result
        for row_number, raw in enumerate(reader, 2):
            # Physical start line for multiline quoted fields is less useful than record number.
            result.preview.append({k: v for k, v in raw.items() if k is not None})
            if None in raw or any(value is None for value in raw.values()):
                result.errors.append(f'第 {row_number} 行：CSV 列数与表头不一致')
                continue
            values = {key: raw.get(key, '') for key in BASE_FIELDS}
            values['status'] = raw.get('status', 'planned').strip()
            values['tags'] = raw.get('tags', '').split('|')
            conversion_errors = {}
            for key in ('batch_size', 'epochs'):
                try:
                    number = Decimal(values[key].strip())
                    if not number.is_finite() or not 1 <= number <= 2**63 - 1 or number != number.to_integral_value():
                        raise InvalidOperation()
                    values[key] = int(number)
                except (InvalidOperation, ValueError):
                    conversion_errors[key] = '必须是 SQLite 范围内的正整数'
                    values[key] = None
            if not values['success_rate'].strip():
                values['success_rate'] = None
            data, errors = validate_errors(values)
            errors.update(conversion_errors)
            for key, reason in errors.items():
                result.errors.append(f'第 {row_number} 行：{key} {reason}')
            if not errors:
                result.rows.append(data)
    except csv.Error as exc:
        result.errors.append(f'第 {reader.line_num} 行：CSV 格式错误：{exc}')
    if not result.preview and not result.errors:
        result.errors.append('文件：CSV 没有实验数据')
    return result


def import_experiments_csv(store, content):
    result = parse_experiments_csv(content)
    if result.errors:
        raise ValueError('\n'.join(result.errors))
    return store.create_many(result.rows)
