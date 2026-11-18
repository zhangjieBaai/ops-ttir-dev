# MM baseline TTIR

当前活跃的6份baseline分别位于 `bm*_bn*_bk*_g8_w*_s*/baseline.ttir`。每份与官方源码按所属配置生成的TTIR逐字节一致，路径、SHA256及31个shape的映射冻结在 `../baseline.json`。

本目录根部的 `baseline.ttir` 是迁移前的历史单配置原件，保留字节用于追溯；当前代码不读取它进行自动dispatch。历史manifest/脚本见 `../setup_attempts/v3/`。

首轮候选 `r1_*.ttir` 保留全部尝试（含资源失败、回退和无效计时），不可静默覆盖。每组 `optimized.ttir` 为 `../results/round1_selection.json` 所选源文件的字节副本；其中5组为baseline fallback，只有1组获得已验证局部收益，详见 `../report_round1.md`。
