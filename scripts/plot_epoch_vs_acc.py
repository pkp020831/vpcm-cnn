
import os
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

def plot_results(log_dir="logs/train/runs", output_file="ill_conditioning_plot.png"):
    results = {}
    log_path = Path(log_dir)

    for run_dir in log_path.iterdir():
        if not run_dir.is_dir():
            continue

        metrics_file = run_dir / "csv/version_0/metrics.csv"
        config_file = run_dir / ".hydra/config.yaml"

        if not metrics_file.exists() or not config_file.exists():
            continue

        # 설정 파일에서 깊이 정보 읽기
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            # model.net.cfg의 길이를 깊이로 간주 (입력/출력층 제외)
            try:
                depth = len(config['model']['net']['cfg']) - 2
            except (KeyError, TypeError):
                print(f"Skipping {run_dir}: could not determine depth.")
                continue

        # 메트릭 파일 읽기
        df = pd.read_csv(metrics_file)
        
        # test/acc 데이터 추출 (NaN 값은 제외)
        if 'val/acc' in df.columns:
            test_acc = df[['epoch', 'val/acc']].dropna()
            if not test_acc.empty:
                if depth not in results:
                    results[depth] = []
                results[depth].append(test_acc)

    if not results:
        print("No valid results found to plot.")
        return

    plt.figure(figsize=(12, 8))

    # 깊이 순으로 정렬하여 플로팅
    for depth in sorted(results.keys()):
        # 동일 깊이에 대해 여러 실험이 있을 경우, 평균을 내거나 대표 실험 하나를 선택할 수 있습니다.
        # 여기서는 간단하게 첫 번째 실험 결과를 사용합니다.
        # 여러 실험의 평균을 내려면 추가적인 처리가 필요합니다.
        if results[depth]:
            df_plot = results[depth][0] # 첫 번째 실험 결과 사용
            plt.plot(df_plot['epoch'], df_plot['val/acc'] * 100, label=f'Depth {depth}', marker='o', linestyle='-')

    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy vs. Epoch for Different Model Depths')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    plot_results()
