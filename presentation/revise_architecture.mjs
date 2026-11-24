// Architecture revision of the saved deck. Keeps its slide IDs and existing Q&A pages.
const C={ink:'#152D3E',muted:'#536776',teal:'#087F72',blue:'#355D88',orange:'#AA572D',light:'#EDF4F4',navy:'#0D2233'};
function tx(s,t,x,y,w,h,size=24,color=C.ink,bold=false){const v=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});v.text=t;v.text.style={typeface:'Microsoft YaHei',fontSize:size,color,bold,autoFit:'none',insets:0};return v;}
function box(s,t,x,y,w,h,owner='A',size=22){const color=owner==='B'?C.blue:owner==='A'?C.teal:C.muted;const v=s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill:owner==='B'?'#F0F4FA':owner==='A'?C.light:'#F5F6F7',line:{fill:color,width:1.4}});v.text=t;v.text.style={typeface:'Microsoft YaHei',fontSize:size,color,alignment:'center',verticalAlignment:'middle',autoFit:'none',insets:7};return v;}
function wire(s,points,color=C.muted,dashed=false){for(let i=1;i<points.length;i++){const [x,y]=points[i-1],[a,b]=points[i];if(x===a&&y===b)continue;const backwards=a<x||b<y;const v={geometry:'connector',kind:'straight',position:{left:Math.min(x,a),top:Math.min(y,b),width:Math.abs(x-a),height:Math.abs(y-b)},line:{fill:color,width:2,style:dashed?'dashed':'solid'}};if(i===points.length-1)v[backwards?'head':'tail']={type:'triangle',width:'med',length:'med'};s.shapes.add(v);}}
async function clear(p,i,title){const s=p.slides.getItem(i);const q=await p.inspect({select:`$.slides[${i}].elements[*]`,include:['id']});if(q.records.length)p.delete(q.records.map(v=>v.id));s.background.fill='#FFFFFF';s.hidden=i>=5;tx(s,title,56,43,1168,70,42,C.ink,true);tx(s,i<5?String(i+1).padStart(2,'0')+' / 05':'附页 '+String(i-4).padStart(2,'0'),1102,677,125,28,18,C.muted);return s;}
function tab(s,values,x,y,w,h,widths,size=23){const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values});for(let r=0;r<values.length;r++)for(let c=0;c<values[r].length;c++){const z=t.getCell(r,c);z.fill=r===0?C.navy:r%2===0?'#F0F5F5':'#FFFFFF';z.text.style={typeface:'Microsoft YaHei',fontSize:r===0?23:size,color:r===0?'#FFFFFF':C.ink,bold:r===0};}t.borders.assign({fill:'#DFE8EB',width:1,style:'solid'});return t;}

export const revisedSpeech=[
'先看整个项目的数据流。任务包在本地解析，生成任务契约和训练内划分；原始特征与标签进入训练Worker。规划器收到的是任务摘要、合法候选、历史结果和预算。B线调度决定尝试、复核还是结束；探索时由本地策略或Flash选候选。A线负责核验和执行，把真实分数、模型身份与预测证据写回下一轮。结束后，按复核证据选模、全量重训并导出。正式测试始终与搜索隔离。',
'这些结构对应六个瓶颈。目标理解不稳，用统一契约固定指标和标签；一次生成的候选有限，用共同方法库和预算调度；错误事实继续影响决策，用绑定来源的事实核验；旧错误只剩警告，用带状态的问题记录；重复反馈太长，用可逆压缩；中断丢失成果，用检查点和独立预测包。其中方法与调度由B负责，事实核验和紧凑反馈是可选增强。',
'再看处理循环。调度先决定动作，A线检查候选、划分与剩余拟合预算，通过后才训练。开启grounded模式时，明确矛盾会形成问题，要求模型纠正；成功返回且用量已知时，最多再用一次语义修复，仍占三次调用额度。训练后记录真实证据，再进入下一轮。网络、解析或用量失败不会自动重试。纠正提案也不直接等于执行后问题已关闭。',
'效果分别验证：目标契约十八例符合预期；反馈传递的受控验收由四项通过变成九项。事实核验保留组二十四例全部符合预期，真实Flash明确纠正三类预设错误。紧凑反馈少百分之六点一五输入token；恢复交付六种成功、三种正确拒绝。这些分别支持结构能力，不等于普遍提分。',
'跟进训练也发现局限：短信新诊断动作平均低三点一八个百分点，花卉结果一正一负。新核验协议本轮费用还增加约百分之七十三，所以保留为可选项。下一步把知识纠正与必须训练的修复分开，避免可靠性检查挤占有效探索。'
];

export async function revise({presentation:p,input}){
 const phase=input.phase;
 if(phase==='project'){
  const s=await clear(p,0,'项目全景：数据、决策与证据回路');
  tx(s,'训练数据留在本地；Flash 接收摘要与证据上下文，不接收整份训练数据',57,112,1163,37,23,C.muted);
  box(s,'任务包\n数据＋说明',56,170,190,74,'X');
  box(s,'A 接入与解析\nCSV／数组／图像',335,170,220,74);
  box(s,'A 训练内划分\n主划分＋两组复核',635,170,240,74);
  box(s,'A 执行器 / Worker\n特征→拟合→评估',965,170,260,74);
  wire(s,[[246,207],[335,207]],C.blue);wire(s,[[555,207],[635,207]],C.blue);wire(s,[[875,207],[965,207]],C.blue);
  tx(s,'原始数据',252,173,82,27,18,C.blue);tx(s,'分区数据',878,173,88,27,18,C.blue);
  box(s,'A 任务契约\n类型·标签·指标\n目标来源与未知状态',335,335,220,100);
  wire(s,[[445,244],[445,335]]);tx(s,'元数据',455,280,90,27,18,C.muted);
  box(s,'A 实验记录与核对\n分数·对象·预测·来源',965,335,260,90);
  wire(s,[[1095,244],[1095,335]],C.teal);tx(s,'实际观测',1107,283,104,27,18,C.teal);
  box(s,'复核选模→全量重训\nA 导出：模型＋预测',56,335,245,100,'X',21);
  box(s,'B 合法方法库\n表格／文本／图像',56,490,245,90,'B');
  box(s,'B 调度＋本地 / Flash\n尝试新法／复核／结束',410,490,285,90,'B');
  box(s,'A 执行准入\n候选·分区·拟合预算\n[可选] 绑定事实核验',780,480,310,100,'A',22);
  wire(s,[[445,435],[445,490]]);wire(s,[[301,550],[410,550]]);wire(s,[[695,535],[780,535]]);
  tx(s,'候选',320,549,70,25,18,C.blue);tx(s,'动作',710,507,65,25,18,C.muted);
  wire(s,[[965,380],[880,380],[880,455],[600,455],[600,490]],C.teal);tx(s,'历史与证据反馈',635,427,228,27,19,C.teal);
  wire(s,[[1090,535],[1244,535],[1244,207],[1225,207]],C.muted);tx(s,'准入后执行',1110,538,130,28,18,C.muted);
  wire(s,[[410,509],[320,509],[320,385],[301,385]]);tx(s,'结束',322,447,76,26,18,C.muted);
  tx(s,'全程守卫：实际拟合 ≤11（含重训） · Flash ≤3 · 共享账本 · 无自动重试',56,611,1165,36,23,C.ink,true);
  tx(s,'A / B 为实现分工；[可选] 非默认开启；当前97源完整真实 grounded 路径与交付仍待验',56,666,1030,28,19,C.muted);
 }
 if(phase==='matrix'){
  const s=await clear(p,1,'瓶颈 → 对应结构 → 改变的行为');
  tab(s,[['瓶颈','提出 / 接入的结构','改变的行为'],
   ['目标、标签与评分漂移','A 任务契约\nTaskSpec / MetricSpec','统一指标、正类和语义\n未声明意图保留未知'],
   ['一次生成的候选有限','B 共同方法库＋预算调度','本地 / Flash 共用合法目录\n支持探索、复核和结束'],
   ['错数值 / 对象 / 参数含义','A 绑定事实＋提案核验 [可选]','引用具体来源与参照\n矛盾不当真，未知不冒充确认'],
   ['反馈丢失 / 旧错只剩警告','A 结构化状态与历史\n＋问题状态 [可选]','结果写回下一轮\n区分待纠正、待执行、已确认'],
   ['重复核对信息增加输入','A 可逆紧凑反馈 [可选]','表格与引用去重\n保持核对字段可反解'],
   ['中断后成果无法交付','A 检查点＋恢复 / 独立预测包','核对源码、模型和资产\n恢复交付，不冒充自动续训']],56,147,1168,490,[270,410,488],23);
  tx(s,'预算/来源/划分守卫贯穿全程；“知识纠正与训练修复分层”是下一步，尚未实现',56,663,1030,30,20,C.muted);
 }
 if(phase==='loop'){
  const s=await clear(p,2,'处理循环：先准入，再执行，再更新证据');
  box(s,'统一状态\n任务＋历史＋候选＋预算',65,158,285,74);
  box(s,'B 调度：探索 / 复核 / 结束',455,158,350,74,'B',23);
  box(s,'匹配复核证据选模\n全量重训→冻结交付',950,158,270,74,'X',22);
  wire(s,[[350,195],[455,195]]);wire(s,[[805,195],[950,195]]);tx(s,'结束',847,161,77,28,19,C.muted);
  box(s,'探索：本地 / Flash 选未试候选\n复核：固定候选，换登记分区',455,310,350,78,'B',22);
  wire(s,[[630,232],[630,310]]);
  box(s,'A 核验与执行准入\n合法候选·分区·预算',455,455,350,78);
  wire(s,[[630,388],[630,455]]);tx(s,'grounded 可选核对事实断言',459,415,355,27,19,C.teal);
  box(s,'Worker：拟合与预测',950,455,270,78,'A',23);
  box(s,'A 观测与记忆\n实际模型·分数·预测·检查',950,310,270,78,'A',22);
  wire(s,[[805,494],[950,494]]);tx(s,'通过',853,463,72,26,19,C.teal);wire(s,[[1085,455],[1085,388]],C.teal);
  wire(s,[[1220,349],[1242,349],[1242,125],[208,125],[208,158]],C.teal);tx(s,'真实证据写回下一轮',887,129,300,24,18,C.teal);
  box(s,'[可选] 矛盾→结构化问题\n明确纠正，仍待执行确认',65,455,285,78,'A',21);
  wire(s,[[455,494],[350,494]],C.orange);tx(s,'矛盾',371,463,65,27,19,C.orange);
  wire(s,[[207,455],[207,349],[455,349]],C.orange);tx(s,'已知用量时，最多一次语义修复',65,307,375,27,18,C.orange);
  tx(s,'网络 / 解析 / 用量异常 → 停止付费；若启用 partial 且有合法模型，仅交付检查点',56,586,1160,63,23,C.orange);
  tx(s,'修复仍占原3次Flash额度；执行后问题确认代码已接入，完整真实闭环尚待验证',56,663,1030,29,19,C.muted);
 }
 if(phase==='evidence'){
  const s=await clear(p,3,'用不同实验，分别验证结构的作用');
  tab(s,[['结构','已有数据','证据支持范围'],
   ['目标契约','18 / 18 案例符合预期','目标 / 标签 / 正类的受控工程检查'],
   ['反馈传递','验收 4 / 9 → 9 / 9','模拟客户端＋真实训练 Worker'],
   ['事实核验与问题状态','保留24/24；23断言确认\n3类受控旧错误明确纠正','程序 oracle＋真实 Flash 受控历史纠错'],
   ['可逆紧凑反馈','输入 token −6.15%','3个已见状态，本次下一候选相同'],
   ['恢复与独立交付','6种恢复 / 3种拒绝','受控快照；恢复时0拟合、0 API']],56,163,1168,420,[260,365,543],23);
  tx(s,'结构能力有证据；自然纠错率、稳定提分与新任务迁移分别验证',56,599,1160,43,27,C.teal,true);
  tx(s,'A138 / A1-03 / A145–146 / A140 / A141各自版本；不能拼成当前97源的一轮完整对照',56,665,1030,28,18,C.muted);
 }
 if(phase==='data_detail'){
  if(p.slides.items.length===8)p.slides.add({width:1280,height:720});
  const s=await clear(p,8,'完整数据边界：原始数据、上下文与预测');
  tx(s,'规划上下文由任务契约、合法目录与已记录观测生成；整份训练数据不发送给 Flash',56,113,1167,29,21,C.muted);
  box(s,'任务包 / 说明\n本地读取与适配',56,160,225,76);
  box(s,'任务契约＋划分登记\n指标·类序·行号·来源',375,160,265,76);
  box(s,'B 共同方法配置\n表征＋分类器',735,160,255,76,'B');
  wire(s,[[281,198],[375,198]]);
  box(s,'训练特征 X ＋标签 y',56,310,225,80,'A',23);
  box(s,'登记的训练内 Payload\n内训 X/y ＋评估 X/y',375,310,265,80);
  box(s,'真实 Worker\n特征→拟合→预测',735,310,255,80,'A',21);
  box(s,'观测与逐行证据\n指标＋预测＋指纹',1050,310,175,80,'A',20);
  wire(s,[[168,236],[168,310]],C.blue);wire(s,[[281,350],[375,350]],C.blue);wire(s,[[640,350],[735,350]],C.blue);wire(s,[[990,350],[1050,350]],C.teal);
  wire(s,[[508,236],[508,310]]);wire(s,[[862,236],[862,310]]);
  box(s,'外层验证 / 正式测试\n隔离于搜索\n冻结后才预测 / 评价',56,495,225,100,'X',21);
  box(s,'规划上下文（非整份数据）\n任务·候选·历史·预算\n可选核对 / 压缩 / 事实表',375,495,265,100,'A',20);
  box(s,'[可选] Flash\n只返回候选与提案',735,495,255,100,'B',22);
  wire(s,[[640,545],[735,545]]);
  wire(s,[[1137,390],[1137,631],[508,631],[508,595]],C.teal);tx(s,'实际观测汇总后进入上下文',710,637,445,29,19,C.teal);
  wire(s,[[862,495],[862,390]],C.muted);tx(s,'提案经准入后执行',875,439,300,27,19,C.muted);
  tx(s,'蓝线：本地训练数据；绿线：证据回传；灰线：契约 / 配置 / 提案；测试标签不进入规划器',56,673,1030,29,18,C.muted);
 }
 if(phase==='implementation'){
  if(p.slides.items.length===9)p.slides.add({width:1280,height:720});
  const s=await clear(p,9,'处理顺序与对应的真实代码对象');
  tab(s,[['阶段','对应模块','交付给下一阶段的对象'],
   ['1. 接入、登记','data / contracts / task_goal\nab_catalog / freeze_partitions','TaskSpec / MetricSpec\ncapabilities / split payloads'],
   ['2. 状态与调度','ab_run / ab_scheduler\n以及已选定的调度变体','state：task/history/catalog/budget\naction：explore/confirm/stop'],
   ['3. 提案与核验','ab_flash；可选 fact_registry\ngrounded_proposal / grounding_issues','候选 / 引用 / 断言 / 问题状态\n通过后才进入训练准入'],
   ['4. 真实执行','scheduler_bridge / Executor\nexperiment_worker / fit_guard','模型＋metrics＋prediction_record\n实际拟合次数与来源凭据'],
   ['5. 证据反馈','estimator_facts / hypothesis_memory\n可选 issues 与 compact_feedback','history / checks / memory / issues\n下一轮仍保留未知和失败'],
   ['6. 选模、重训、交付','ab_comparison / finalization\npredict / export_model_bundle','最终模型 / 源码与资产清单 / CSV\n恢复工具仅重交付，不续训']],56,148,1168,490,[220,455,493],22);
  tx(s,'可选增强已接入，但当前97源完整真实 grounded 路径与独立交付未验；各批证据保持原版本',56,664,1030,28,18,C.muted);
 }
 if(phase==='notes'){
  const times=['00:00—00:35','00:35—01:10','01:10—01:40','01:40—02:05','02:05—02:30'];
  for(let i=0;i<5;i++)p.slides.getItem(i).speakerNotes.text=times[i]+'\n\n'+revisedSpeech[i]+'\n\n'+input.sources;
  if(input.qa)p.slides.getItem(5).speakerNotes.text=input.qa;
  p.slides.getItem(7).speakerNotes.text='附页：原A140压缩图表与A141恢复交付。不同批次，不是A146费用比较。\n'+input.sources;
  p.slides.getItem(8).speakerNotes.text=input.architecture+'\n\n'+input.sources;
  p.slides.getItem(9).speakerNotes.text=input.architecture+'\n\n'+input.sources;
 }
 return {phase,mainSlides:5,totalSlides:p.slides.items.length,existing_QA_pages_preserved:true};
}
