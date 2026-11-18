# ops-ttir-dev

直接修改 Triton TTIR 的算子优化实验，首个算子为 LayerNorm forward。

当前入口：[layer_norm/README.md](layer_norm/README.md)。代码、冻结 TTIR、候选生成器和结果均由 Git 管理；可再生的 TTGIR/LLVM/PTX/cubin 放在忽略的 `layer_norm/artifacts/`。

MM 基线实验入口：[mm/README.md](mm/README.md)。本阶段只跑通官方教程到 TTIR 的原生编译、验证和计时流程，尚未开始优化。

跨算子的统一实验要求见 [TTIR_EXPERIMENT_REQUIREMENTS.md](TTIR_EXPERIMENT_REQUIREMENTS.md)，包含环境、baseline 冻结、Triton 原生编译/加载/启动链及验证规范。
