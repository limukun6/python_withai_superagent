# A-15：今日最终源码无实训回归（2026-11-21）

完成A12动态Flash与A14对称CSV契约后，冻结生产源码运行接口检查：既有AB/固定调度/Flash46项、对称输入4项、general_schema/native_schema/image_files。这些schema检查只有1次ImageFlattener预处理fit，不训练监督分类器。预留0分类器fits、0API/模拟HTTP。先journal保存计划/源摘要/测试清单，再执行；失败按实际处理，不能伪报通过。scripts/run_a09_interface_checks.py --include-fixed --include-flash --include-schema NEW_DIR。历史A12/A13证据保留原源码指纹，不因最后schema变更冒称全部同源重跑。

首轮回归通过后，补充审阅发现只按当前task/shared核验仍可用新task ID绕过别的历史未知scope。新增全项目unknown/pending预检与一项直接反例测试，再冻结新版新目录回归；仍0fit/0API。不重跑47fit模拟端到端，仅增收紧入口的无训练反例，旧端到端记录保持旧摘要。
