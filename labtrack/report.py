"""Metric summaries and downloadable Markdown reports, independent of Streamlit."""
import re
from labtrack.command_parser import parse_train_command, check_consistency

SUMMARY_LABELS = {'final_loss': 'Final Loss', 'best_loss': 'Best Loss',
                  'final_accuracy': 'Final Accuracy (%)', 'final_success_rate': 'Final Success Rate (%)'}


def summarize_metrics(metrics):
    summary = {}
    for name in ('loss', 'accuracy', 'success_rate'):
        values = sorted((m for m in metrics if m['metric_name'] == name), key=lambda m: m['step'])
        if values:
            summary['final_' + name] = values[-1]['metric_value']
            if name == 'loss':
                summary['best_loss'] = min(m['metric_value'] for m in values)
    return summary


def _text(value):
    text = str(value if value is not None else '未填写')
    return re.sub(r'([\\`*_{}\[\]<>#|])', r'\\\1', text).replace('\n', '  \n')


def _code(value):
    text = str(value or '未填写')
    longest = max([len(m.group()) for m in re.finditer(r'`+', text)] or [0])
    fence = '`' * max(3, longest + 1)
    return f'{fence}text\n{text}\n{fence}'


def generate_report(experiment, metrics):
    summary = summarize_metrics(metrics)
    lines = [f"# LabTrack 实验报告：{_text(experiment['name'])}", '', '## 实验信息', '',
             f"- 实验 ID：{experiment['id']}", f"- 模型：{_text(experiment['model'])}",
             f"- 数据集：{_text(experiment['dataset'])}", f"- GPU：{_text(experiment['gpu'])}",
             f"- Checkpoint：{_text(experiment['checkpoint_path'])}",
             f"- 创建时间 (UTC)：{_text(experiment['created_time'])}",
             f"- 更新时间 (UTC)：{_text(experiment['updated_time'])}", '', '## 超参数', '',
             f"- Batch size：{experiment['batch_size']}",
             f"- Learning rate：{experiment['learning_rate']}", f"- Epochs：{experiment['epochs']}",
             '', '## 训练命令', '', _code(experiment['train_command']),
             '', '## 评测命令', '', _code(experiment['eval_command']), '', '## 最终结果', '']
    rate = experiment['success_rate']
    lines.append('- 手动记录的 Success rate：' + ('尚未评测' if rate is None else f'{rate:g}%'))
    lines += ['', '## 训练指标摘要', '']
    if not metrics:
        lines.append('暂无训练指标。')
    else:
        lines.append(f'- 指标记录数：{len(metrics)}')
        lines.append(f"- Epoch 范围：{min(m['step'] for m in metrics)}–{max(m['step'] for m in metrics)}")
        for key, label in SUMMARY_LABELS.items():
            if key in summary:
                lines.append(f'- {label}：{summary[key]:g}')
        lines.append('- Final 指对应指标最大 Epoch 的值；Best Loss 指最小 Loss。')
    lines += ['', '## Experiment Status', '', _text(experiment.get('status', 'planned')),
              '', '## Tags', '']
    lines += [f'- {_text(tag)}' for tag in experiment.get('tags', [])] or ['无标签。']
    lines += ['', '## Parsed Training Arguments', '']
    try:
        arguments = parse_train_command(experiment['train_command'])
    except ValueError as exc:
        lines += [_text(str(exc)), '', '## Configuration Consistency', '', '命令解析失败，无法核对。']
    else:
        lines += [f'- {_text(key)}: {_text(value)}' for key, value in arguments.items()] or ['没有识别到长参数。']
        checks = check_consistency(experiment, arguments)
        lines += ['', '## Configuration Consistency', '']
        mismatches = [check for check in checks if not check['matches']]
        if mismatches:
            for check in mismatches:
                lines.append(f"- {check['label']} mismatch: LabTrack = {_text(check['recorded'])}; Command (--{_text(check['parameter'])}) = {_text(check['command'])}")
        elif checks:
            lines.append('All provided tracked parameters match the training command.')
        else:
            lines.append('训练命令未提供待核对参数，无法确认一致性。')
    lines += ['', '## 备注', '', _code(experiment['notes']), '',
              '日志中的 Accuracy 和 Success rate 已统一为百分比。训练指标不会自动覆盖手动实验结果。', '']
    return '\n'.join(lines)
