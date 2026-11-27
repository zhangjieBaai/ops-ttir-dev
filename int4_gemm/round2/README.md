# INT4 TTIR 第二轮

[结果报告](report.md) · [30点扩展合同](expanded_plan.md) · [候选计划](optimization_plan.md) · [最终汇总](summary.json)

原7点只作为pilot生成记录。正式性能/精度/sanitizer均采用expanded/manifest.json的30点，所有候选都是直接TTIR文本变换；新点使用明确声明的固定配置，不声称已经逐shape充分调优。

复现：先隔离空闲GPU，Python固定 /root/venvs/ir-ops-dev/bin/python。运行 verify_pipeline.py --output <新文件>，然后 run.py --run <新目录> --names baseline,parent,r2_group_pair --rounds 5 --rep 20 --seed 234。新输出不得覆盖旧证据。
