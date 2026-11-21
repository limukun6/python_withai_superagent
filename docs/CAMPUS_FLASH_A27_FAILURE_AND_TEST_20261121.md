# 本次Flash失败原因与本机测试方法（2026-11-21）

后续更新A28：用户调整网络后已用同一客户端成功连通，并完成真实SMS raw/checked两臂（7成功请求含连通、14fit、¥.043046）。当前不再DNS阻塞；本文保留当时失败证据及通用测试方法，172行账本是旧快照，最新179行/11历史未知。见batches/A28_flash_network_resume_results.md；不要因此重复已完成的连通测试或用旧账本分叉。

## 实际结果

补齐对称文本方法后，使用正式Agent的ExperimentLLM、正确模型DeepSeek-V4.1-Flash、同一真实累计账本，重新发起**一次**SSE连通测试。直连Session.trust_env=False，连接超时10秒、读取超时60秒、无自动重试，0模型训练。原始结果：

```text
NameResolutionError: Failed to resolve 'aigc-api.hkust-gz.edu.cn'
[Errno 11001] getaddrinfo failed
```

失败在DNS解析阶段，约1.515秒。没有建立接口HTTP连接、没有HTTP状态、没有收到模型响应。报错里的“Max retries exceeded”是底层异常描述；本脚本只有一次请求尝试，没有重试循环，requests默认适配器也未配置自动重试。

随后只读诊断（约6.102秒）发现：

| 检查 | 结果 | 能说明什么 |
|---|---|---|
| 学校域名socket.getaddrinfo | gaierror 11001 | 当前执行环境不能解析该域名 |
| 用户原说明的10.121.10.250:443 | 连接超时 | 当时连已知校园IP也不通，不能只归因于模型参数或DNS记录 |

现有证据不能确定具体是校园接入、VPN隧道/路由、DNS服务还是学校端网络状态。没有HTTP响应，API Key、模型权限和模型推理速度尚未被这次请求检验；不能据此宣称它们有问题。浏览器可能使用不同DNS/代理，浏览器打开页面也不等于Python直连可达。

没有修改代理、VPN、hosts、DNS或防火墙。历史11未知费用仍保留。本次原请求ticket和原传输日志证明未发出后，追加not_dispatched审计，保留reserved→unknown→reviewed_not_dispatched链；没有把一般超时当零。账本现在172行（170历史+两次已证明未发出的连接尝试）、known5.1277825元、11历史未知、pending0，完整累计费用和平台扣费仍null。

## 第一步：只读网络检查，不花模型调用费用

在你打算运行实验的同一台机器、同一个网络环境，打开PowerShell。先运行这两条：

```powershell
nslookup aigc-api.hkust-gz.edu.cn
Test-NetConnection 10.121.10.250 -Port 443 -InformationLevel Detailed
```

将DNS结果与学校当前提供的地址核对。10.121.10.250来自本项目已有学校接入说明；若学校已变更地址，以学校说明为准，不要把IP写进hosts硬绕过。TcpTestSucceeded=False表示该机器当前不能建立到这个IP的443端口连接；True只说明TCP通，不代表TLS、认证和聊天一定成功。

也可以运行仓库里已写好的只读诊断，额外检查TLS并保存JSON（不会读取key、不发HTTP）：

```powershell
Set-Location "D:\codex\python_withai\runs\integration"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
& "D:\codex\python_withai\.venv\Scripts\python.exe" -m scripts.diagnose_a24_campus_readonly "D:\codex\python_withai\runs\network_check_$stamp"
```

输出目录中看results.json和completion.json。DNS失败且IP443也超时，应先核实你所在网络是否能访问学校内网、学校认可的远程接入是否生效；不要为了测试去改OpenAI代理配置。具体该开/关哪个VPN取决于其功能，不能把普通外网代理当作校园接入。

## 第二步：手动只发一次Flash请求

脚本复用正式客户端、SSE解析和真实累计费用账本，0训练，最多1请求。默认仅预检，不读密钥、不发请求；加--execute才会调用。以下均为完整单行命令，避免PowerShell续行符复制错误：

```powershell
Set-Location "D:\codex\python_withai\runs\integration"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
& "D:\codex\python_withai\.venv\Scripts\python.exe" -m scripts.test_campus_flash_once --ledger "D:\codex\python_withai\runs\a23_real_local_history\ledger.sqlite" --api-key-file "D:\codex\python_withai\api_key.txt" --output "D:\codex\python_withai\runs\flash_check_preflight_$stamp"
```

预检通过后，如果你选择在当前网络做一次真实测试，单独执行：

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
& "D:\codex\python_withai\.venv\Scripts\python.exe" -m scripts.test_campus_flash_once --ledger "D:\codex\python_withai\runs\a23_real_local_history\ledger.sqlite" --api-key-file "D:\codex\python_withai\api_key.txt" --output "D:\codex\python_withai\runs\flash_check_once_$stamp" --execute
```

成功时completion.json为passed，answer.json包含ok:true，llm_usage.jsonl有实际token；失败看completion.json、events.jsonl、llm_usage.jsonl，不要反复点运行。0.10元是估计预留/结算停止阈值，不是平台实际扣费硬上界。本次计入即将进行的sms_spam开发小试，沿用任务费用及已有共享费；历史课程总费用仍未核验。

若新请求超时或缺usage，新增unknown会按设计挡住后续请求；这是失败保护，不是让你换任务ID或清账本。只有DNS原证据严格证明未发出时才可审计追加not_dispatched，不能对ReadTimeout这样做。

你把**网络检查输出与新测试目录的completion.json**发回来即可，必要时再看events/usage；不要发送api_key.txt、Authorization头或含密钥的截图。另机测试需私下同步最新账本，不能并发分叉，也不能用旧history_audit重新导入清掉新增记录。单次连通成功后再安排正式SMS raw/checked闭环，它本身不证明Agent效果。
