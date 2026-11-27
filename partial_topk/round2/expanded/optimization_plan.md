继承本轮共同协议与父目录数值合同；候选全是直接TTIR改写。

30点为rows={1,4,16,64,128}乘(N,K)={(65536,8),(65537,8),(131072,8),(262144,8),(262144,16),(1048576,8)}，保留全部原6点。固定4warps/3stages与chunk1024；新大N的merge block由原公式计算，无autotune。旧unroll与packed_unroll因数字开头SSA重命名不合法编译失败，保留原件；修正为unroll_v2/packed_unroll_v2，另测试已获益static_ids与展开的组合static_unroll_v2。没有改变数值合同。

正式证据在本expanded目录，父目录小矩阵结果为pilot。先所有候选全矩阵正确性，再筛选，冻结后7轮20ms主测、独立5轮20ms复测。基线/parent/候选同轮对照，完整A/A和sanitizer门槛沿用共同协议。
