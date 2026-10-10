"""Small, line-based parser. Add aliases in PATTERNS to extend supported logs."""
import math
import re
from dataclasses import dataclass, field
from typing import List

METRIC_NAMES = ('loss', 'learning_rate', 'accuracy', 'success_rate')
NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
EPOCH = re.compile(r'\bepoch\s*(?:[:=]\s*)?(\d+)\b(?![\w.])', re.I)
PATTERNS = {
    'loss': r'loss',
    'learning_rate': r'learning[ _-]+rate|lr',
    'accuracy': r'accuracy|acc',
    'success_rate': r'success[ _-]+rate',
}
COMPILED = {name: re.compile(
    rf'(?<![\w])(?:{aliases})\s*[:=]\s*({NUMBER})(\s*%)?(?![\w.])', re.I)
    for name, aliases in PATTERNS.items()}


@dataclass
class ParseResult:
    metrics: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    parsed_lines: int = 0
    skipped_lines: int = 0


def parse_log(text):
    """Extract metrics on lines containing an explicit epoch; never execute text.

    Rate values with '%' are percentages. Bare values in [0, 1] are fractions;
    other bare values are percentages. Last duplicate (epoch, metric) wins.
    """
    result = ParseResult()
    records = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        epoch = EPOCH.search(line)
        if not epoch:
            result.skipped_lines += 1
            continue
        digits = epoch.group(1).lstrip('0') or '0'
        if len(digits) > 19 or int(digits) > 2**63 - 1:
            result.warnings.append(f'第 {line_number} 行：Epoch 超出支持范围，已跳过。')
            result.skipped_lines += 1
            continue
        step = int(digits)
        found = False
        for name, pattern in COMPILED.items():
            match = pattern.search(line)
            if not match:
                continue
            value = float(match.group(1))
            percent = bool(match.group(2))
            if name in ('accuracy', 'success_rate') and not percent and 0 <= value <= 1:
                value *= 100
            invalid = not math.isfinite(value) or value < 0
            invalid |= name == 'learning_rate' and percent
            invalid |= name == 'loss' and percent
            invalid |= name in ('accuracy', 'success_rate') and value > 100
            if invalid:
                result.warnings.append(f'第 {line_number} 行：{name} 数值无效，已跳过。')
                continue
            key = (step, name)
            if key in records:
                result.warnings.append(f'第 {line_number} 行：Epoch {step} 的 {name} 重复，采用最后一个值。')
            records[key] = dict(step=step, metric_name=name, metric_value=value)
            found = True
        if found:
            result.parsed_lines += 1
        else:
            result.skipped_lines += 1
            result.warnings.append(f'第 {line_number} 行：存在 Epoch，但没有可识别的有效指标。')
    result.metrics = [records[key] for key in sorted(records)]
    return result


def parse_uploaded_log(content, filename):
    """Validate uploaded bytes separately from the text parser."""
    if not filename.lower().endswith(('.txt', '.log')):
        raise ValueError('仅支持 .txt 或 .log 文件。')
    if len(content) > 5 * 1024 * 1024:
        raise ValueError('日志文件不能超过 5 MB。')
    try:
        text = content.decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        raise ValueError('日志编码无法识别，请转换为 UTF-8 后重新上传。') from exc
    if '\x00' in text:
        raise ValueError('文件包含二进制内容，请上传纯文本日志。')
    return parse_log(text)
