# 执行说明

- 输入TTIR从头冻结于新目录，旧候选和基线未覆盖。初版registry与格式化过程的registry全部保留，当前输入为inputs_v4.json；前两轮check为准备记录，当前主测绑定check_v4.json。原始生成器/runner初版快照保存在results/，格式化修复未改变候选计算。
- 主测与独立进程复测均完整测量两个候选，不能称新holdout。primary_finalists.json只读取main.json生成（生成时独立复测已启动，但不读取其结果），按预定主测全组几何比选择；原始无效点仍待一次固定重测，最终接受必须主/复所有要求点、A/A和数值/内存门槛通过。没有按复测结果在候选之间切换。
- 主测无效点与复测无效点（包括original_pointer对照）取并集16点，写repair_plan.json后固定，两进程仅各重测一次、7轮；不按速度选择更快数据。原始数据保留。
- memcheck为0 errors；racecheck为72 warnings、0 errors。初始shell的run_step在set+e下会返回尾部ensure_idle的0，因此results/racecheck_exit_status.txt记录的是wrapper的0，不能当成compute-sanitizer实际返回码或PASS。工具日志的72 warnings为判断来源，未清除告警。run_measurements.sh已修复未来重放的返回码采集，本次不为清零告警重复测工具。同步资格仍为false。
