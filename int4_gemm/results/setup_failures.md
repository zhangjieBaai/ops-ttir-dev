# 搭建失败记录

1. 第一次 baseline 导出被 Python 模块路径遮蔽阻止：公共 environment 路径插到
   sys.path 最前，导致 import tutorial 读取 LayerNorm tutorial。原日志 `setup.log`；
   改成 append 后成功，未冻结任何错误 baseline，未修改 kernel 算术。
2. 第一次候选生成的 scalar 变换断言失败：arith.cmpi 在文本 IR 的类型后缀是输入
   `xi32` 而不是输出 `xi1`。在写 scalar 候选前失败；修正文本匹配后重新运行，
   已生成的同名 shared_scale 文件通过字节相等校验保留，未重写候选。
   这是生成器错误，不是设备数值失败；不产生性能样本。
