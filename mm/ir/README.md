# MM baseline TTIR

当前活跃的6份baseline分别位于 `bm*_bn*_bk*_g8_w*_s*/baseline.ttir`。每份与官方源码按所属配置生成的TTIR逐字节一致，路径、SHA256及31个shape的映射冻结在 `../baseline.json`。

本目录根部的 `baseline.ttir` 是迁移前的历史单配置原件，保留字节用于追溯；当前代码不读取它进行自动dispatch。历史manifest/脚本见 `../setup_attempts/v3/`。
