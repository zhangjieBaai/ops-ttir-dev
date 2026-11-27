# norm_quant: Triton源码回写

[实测报告](report.md) · [源码](source_v1.py) · [汇总](summary.json) · [共同实验协议](../../complex_suite/source_port_plan.md)

从仓库根运行 `complex_suite/source_port_runner.py --op norm_quant --run <新目录>`；固定官方venv与独占GPU，脚本自动先核验源码及文件路径pipeline，再正确性和三路配对性能。`--check-only`只验证。
