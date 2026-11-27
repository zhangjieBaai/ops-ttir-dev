# LayerNorm：原版 Triton 与六步优化逐项对照

这些是前面六种TTIR优化的可执行Triton源码表达。为了看清改动，每个版本都是完整、可独立导入的文件；从第2步开始保留前面已经加入的改动。原教程、原始TTIR基线和最终选择均未修改。

| 对照 | 完整Triton实现 | 相对上一步的diff | 相对原版的diff |
|---|---|---|---|
| 原版：8192宽动态循环 | [baseline.py](baseline.py) | — | — |
| 只复用1/N | [reciprocal_only.py](reciprocal_only.py) | [diff](diffs/baseline_to_reciprocal_only.diff) | 同左 |
| 只使用rsqrt | [rsqrt_only.py](rsqrt_only.py) | [diff](diffs/baseline_to_rsqrt_only.diff) | 同左 |
| 1. 倒数复用＋rsqrt | [arithmetic.py](arithmetic.py) | [diff](diffs/baseline_to_arithmetic.diff) | 同左 |
| 2. 2048动态分组 | [grouped.py](grouped.py) | [diff](diffs/arithmetic_to_grouped.diff) | [diff](diffs/baseline_to_grouped.diff) |
| 3. 范围静态展开 | [unroll.py](unroll.py) | [diff](diffs/grouped_to_unroll.diff) | [diff](diffs/baseline_to_unroll.diff) |
| 4. 均值加载.ca | [mean_ca.py](mean_ca.py) | [diff](diffs/unroll_to_mean_ca.diff) | [diff](diffs/baseline_to_mean_ca.diff) |
| 5. 输出.cs | [output_cs.py](output_cs.py) | [diff](diffs/mean_ca_to_output_cs.diff) | [diff](diffs/baseline_to_output_cs.diff) |
| 6. 小N使用1024分组 | [mixed.py](mixed.py) | [diff](diffs/output_cs_to_mixed.diff) | [diff](diffs/baseline_to_mixed.diff) |

`baseline.py`来自冻结教程函数，只有函数重命名、注释移除和格式化；逻辑不改。每个文件导出同名`kernel`及对应`BLOCK_SIZE`常量。所有版本都保留一行一个CTA、同一kernel中三遍处理，不增加kernel、workspace或host形状分派；`N`不是`tl.constexpr`。

1. **倒数复用与rsqrt。**

   原版：

   ```python
   mean = tl.sum(_mean, axis=0) / N
   var = tl.sum(_var, axis=0) / N
   rstd = 1 / tl.sqrt(var + eps)
   ```

   改成：

   ```python
   inv_n = 1.0 / N
   mean = tl.sum(_mean, axis=0) * inv_n
   var = tl.sum(_var, axis=0) * inv_n
   rstd = tl.rsqrt(var + eps)
   ```

   两个单项版本只改其中对应的表达式，便于区分贡献。

2. **2048动态分组。**

   ```diff
   - BLOCK_SIZE = 8192
   + BLOCK_SIZE = 2048
   ```

   原版已经有求均值、求方差、写输出的三遍动态循环，因此源码在这一步只需缩小加载/累加向量的宽度。三遍都仍使用：

   ```python
   acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
   for off in range(0, N, BLOCK_SIZE):
       cols = off + tl.arange(0, BLOCK_SIZE)
       # 本遍的加载和计算
   ```

   `BLOCK_SIZE`表示CTA处理的逻辑向量宽度，不是线程数；启动依然8warps。

3. **按N范围静态展开。**

   三遍循环分别作同样的改写：

   ```diff
   - for off in range(0, N, BLOCK_SIZE):
   -     cols = off + tl.arange(0, BLOCK_SIZE)
   + for block in tl.static_range(K):
   +     cols = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
   ```

   将三遍计算封装在`_static_body`中，参数`BLOCK_SIZE`、`K`均为`tl.constexpr`。设备kernel按运行时N进入分支，例如：

   ```python
   if N <= 4096:
       _static_body(X, Y, W, B, Mean, Rstd, stride, N, eps, 2048, 2)
   elif N <= 6144:
       _static_body(X, Y, W, B, Mean, Rstd, stride, N, eps, 2048, 3)
   ```

   这里`K=2/3`是编译期固定块数，循环可展开；N依然是运行时值，每次load/store仍检查`cols<N`。完整实现包含所有范围及N>16384的2048动态fallback，不是用Python host按shape选择kernel。

4. **只给均值阶段的输入加载加.ca。**

   ```diff
   - a = tl.load(X + cols, mask=cols < N, other=0.0)
   + a = tl.load(X + cols, mask=cols < N, other=0.0, cache_modifier=".ca")
   ```

   求方差和写输出阶段的X加载保留默认属性。动态fallback和静态展开主体均作相同修改。`.ca`是缓存提示，不是通用禁止CSE指令；本程序中其影响还包括加载合并和数据存活范围。

5. **只给Y输出加.cs。**

   ```diff
   - tl.store(Y + cols, y, mask=mask)
   + tl.store(Y + cols, y, mask=mask, cache_modifier=".cs")
   ```

   Mean和Rstd的两个标量store不改。`.cs`是streaming缓存策略提示，不表示完全绕过缓存，也不保证对其他程序有收益。

6. **N≤3072改用1024分组。**

   保留第5步主体，只改变设备分派的前三个范围：

   ```python
   if N <= 1024:
       _static_body(X, Y, W, B, Mean, Rstd, stride, N, eps, 1024, 1)
   elif N <= 2048:
       _static_body(X, Y, W, B, Mean, Rstd, stride, N, eps, 1024, 2)
   elif N <= 3072:
       _static_body(X, Y, W, B, Mean, Rstd, stride, N, eps, 1024, 3)
   ```

   N>3072仍采用2048宽度及对应块数，较大N的动态fallback也不变。

运行任一版本时使用对应模块的`BLOCK_SIZE`，其余启动参数相同：

```python
import importlib.util
import torch

spec = importlib.util.spec_from_file_location("example", "layer_norm/triton_steps/mixed.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)

M, N = 128, 4096
x = torch.randn((M, N), device="cuda", dtype=torch.float16)
w = torch.ones(N, device="cuda", dtype=x.dtype)
b = torch.zeros(N, device="cuda", dtype=x.dtype)
y = torch.empty_like(x)
mean = torch.empty(M, device="cuda", dtype=torch.float32)
rstd = torch.empty_like(mean)
example.kernel[(M,)](
    x, y, w, b, mean, rstd, x.stride(0), N, 1e-5,
    BLOCK_SIZE=example.BLOCK_SIZE,
    num_warps=8, num_stages=3, num_ctas=1, enable_fp_fusion=True,
)
```

验证程序使用正常源码JIT导出TTIR，再经同一原生文件编译链检查LLVM/PTX/cubin逐字节一致，随后运行原142例数值检查及120个范围边界用例。数值通过不表示与历史手写TTIR具有完全相同机器指令或性能；原报告中的加速比不能直接标到这些源码版本上。

复现时先隔离空闲H100，固定官方Triton3.7.0环境。以下结果目录必须不存在：

```bash
export CUDA_VISIBLE_DEVICES=0
/root/venvs/ir-ops-dev/bin/python layer_norm/verify_pipeline.py \
  --output /root/tmp/layernorm_source_pipeline_new.json
/root/venvs/ir-ops-dev/bin/python layer_norm/triton_steps/verify.py \
  --output /root/tmp/layernorm_source_steps_new --benchmark
```

全部示例和diff可以用`python layer_norm/triton_steps/generate.py`重建；该步骤使用系统Python的black统一格式，GPU运行仍只使用指定官方venv。候选链的SHA256在[manifest.json](manifest.json)，数值与计时协议见[plan.md](plan.md)。

本次验证记录：[原教程编译链](results/original_pipeline_20261124.json)、[九个源码版本与2620项数值检查/30点计时](results/source_steps_20261124/results.json)、[独立的源码示例性能记录](results/source_steps_20261124/report.md)。原始数值与范围边界检查全部通过，计时A/A通过；九种源码的寄存器/spill依次为60/0、60/0、60/0、60/0、32/0、80/0、48/0、48/0、48/0。资源相同不代表机器码或性能相同；本次源码示例为单会话观测，没有独立复测，不替换原TTIR报告的正式加速比。
