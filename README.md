# ops-ttir-dev

直接修改 Triton TTIR 的算子优化实验，首个算子为 LayerNorm forward。

当前入口：[layer_norm/README.md](layer_norm/README.md)。代码、冻结 TTIR、候选生成器和结果均由 Git 管理；可再生的 TTGIR/LLVM/PTX/cubin 放在忽略的 `layer_norm/artifacts/`。
