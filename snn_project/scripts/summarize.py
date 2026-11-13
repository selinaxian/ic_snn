"""Summarize completed reservoir sweeps with paired test-day bootstrap."""
import argparse
import json
from pathlib import Path

import pandas as pd

from snn_core.metrics import paired_interval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    lines = [
        '# IC 逐分钟 STDP 储备池探索',
        '',
        '同一 IC 样本，过去 30 分钟的发放记录预测未来 5 分钟收益。固定逐分钟预测、原训练验证测试日期、Gao 式速率编码及初始增益 2.5。',
        '',
        '表中为验证与测试 Rank IC。测试差值等于 STDP 减同配置随机冻结储备池。区间按测试交易日成对重采样 400 次，保留日内重叠标签。区间为探索性估计，未做多重比较校正。',
        '',
        '| 种子 | 神经元 | STDP 抑制比 | 输入 | 读出 | 验证 STDP | 测试随机 | 测试 STDP | 测试差值区间 | 测试原始岭回归 |',
        '|---:|---:|---:|---|---|---:|---:|---:|---:|---|---:|',
    ]
    activity = []
    for path in sorted([*args.directory.glob('sweep_*'), *args.directory.glob('check_*')]):
        if not (path / 'predictions.csv').exists():
            continue
        result = json.loads((path / 'metrics.json').read_text(encoding='utf-8'))
        predictions = pd.read_csv(path / 'predictions.csv')
        config = result['config']
        neurons, depression = config['side'] ** 3, config['depression']
        for name, variant in result['variants'].items():
            test = predictions[(predictions.variant == name) & (predictions.split == 'test')]
            days = test.timestamp.str[:10].to_numpy()
            actual = test.actual_bp.to_numpy()
            val_metrics, test_metrics = (variant['metrics'][key] for key in ('validation', 'test'))
            pairs = [(feature, f'stdp_{feature}_ridge', f'random_{feature}_ridge')
                     for feature in ('signature', 'count')]
            if 'stdp_cube' in test_metrics:
                pairs.append(('signature 3NN', 'stdp_cube', 'random_cube'))
            for feature, trained, control in pairs:
                interval = paired_interval(actual, test[f'{trained}_bp'].to_numpy(),
                                           test[f'{control}_bp'].to_numpy(), days)
                lines.append(f'| {config["seed"]} | {neurons} | {depression} | {name} | {feature} | '
                             f'{val_metrics[trained]["rank_ic"]:+.3f} | {test_metrics[control]["rank_ic"]:+.3f} | '
                             f'{test_metrics[trained]["rank_ic"]:+.3f} | [{interval[0]:+.3f}, {interval[1]:+.3f}] | '
                             f'{test_metrics["raw_ridge"]["rank_ic"]:+.3f} |')
            trained_rate, random_rate = variant['internal_spike_rate_stdp_random']
            activity.append(f'| {config["seed"]} | {neurons} | {depression} | {name} | {variant["mean_in_degree"]:.1f} | '
                            f'{random_rate:.2%} | {trained_rate:.2%} |')
    if not activity:
        raise ValueError('No completed sweeps')
    lines.extend(['', '## 储备池活动', '',
                  '| 种子 | 神经元 | STDP 抑制比 | 输入 | 平均入度 | 随机池内部发放率 | STDP 池内部发放率 |',
                  '|---:|---:|---:|---|---:|---:|---:|', *activity, ''])
    (args.directory / 'exploration.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()

