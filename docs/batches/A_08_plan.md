# A-08 执行协议：A侧动作边界与执行分发

远端已fetch，B分支仍d750c7b，未发现新的Scheduler交接模块。本批不自行实现B调度，不声称真实B已接通。
新增A侧可复用scheduler_bridge：接受ab_schema_version=1状态与约定动作，按执行器持有的可信能力目录/划分白名单/预算检查，再调用注入的执行器。记录输入、动作、证据、预算和返回观察；执行报错保留，不自动重试。
用固定动作fixture测试合法explore/confirm/stop，未知候选、重复探索、确认失败候选、伪造trial、非法划分、重复确认、预算不一致、成本低报、内部未知fit、refit预留、超出11/3上限。实际训练0，API0。通过只证明A侧接入边界，真实Scheduler→CLI循环仍待B适配。
运行前写runner scripts/run_a08_ab_acceptance_batch.py，validate-only/full，输出协议、全部case和三动作真实分发日志；fixture执行器只返回模拟观察，明确classifier_fits=0。
任何B新动作目录只能通过同一边界进入，保留AB v1字段，不因本批另创调度策略。
