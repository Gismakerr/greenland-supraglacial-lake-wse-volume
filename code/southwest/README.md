# Code

公开入口为 `01`–`07`、`09`、`12`、`13`。编号缺口是有意保留的：西南没有第08步
AICc面积–水位拟合，也没有后续水量和相位处理。

每个入口默认核验对应冻结结果目录的文件集合、大小和SHA-256；添加 `--show-engine`
可显示该步骤实际使用的原算法文件。原算法集中在 `_engine`，未保留 `__pycache__`、临时
修复脚本、下载调试脚本和历史副本。

全包检查入口为：

```powershell
C:\ProgramData\anaconda3\envs\geo_env\python.exe code\00_validate_package.py
```

本整理包不自动全量重跑，因为原始S2和完整PIXC数据没有复制进投稿包。需要重算时应以
`ENGINE_INDEX.csv`为路由，显式提供外部原始数据目录，并先在Lake 17或Lake 29等小样本
上测试。
