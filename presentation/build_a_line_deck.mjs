// Reusable Artifact Session callback. Edit the existing bound deck; no local PPTX builder.
const C={navy:'#0D2233',teal:'#087F72',ink:'#152D3E',muted:'#536776',light:'#E6EEEE',white:'#FFFFFF',orange:'#AB572D'};
function text(s,txt,x,y,w,h,size=28,color=C.ink,bold=false){
 const t=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 t.text=txt;t.text.style={typeface:'Microsoft YaHei',fontSize:size,color,bold,autoFit:'none',insets:0};return t;
}
function page(p,title,index){
 const s=p.slides.add({width:1280,height:720});s.background.fill=C.white;
 text(s,title,66,56,1150,80,44,C.ink,true);
 text(s,index<=5?String(index).padStart(2,'0')+' / 05':'附页 '+String(index-5).padStart(2,'0'),1104,658,120,30,18,C.muted);
 return s;
}
function table(s,values,x,y,width,height,widths){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width,height,columnWidths:widths,values});
 for(let r=0;r<values.length;r++)for(let c=0;c<values[r].length;c++){
  const cell=t.getCell(r,c);cell.fill=r===0?C.navy:(r%2===0?'#F0F5F5':C.white);
  cell.text.style={typeface:'Microsoft YaHei',fontSize:r===0?24:27,color:r===0?C.white:C.ink,bold:r===0};
 }
 t.borders.assign({fill:C.light,style:'solid',width:1});return t;
}
const script=[
'接下来介绍我们负责的A线：让算法Agent能正确理解任务、依据实验做决定，并交付可用模型。我们的重点是让每一步决定都有证据，而且能被检查。',
'架构上，任务首先转成统一契约，固定标签含义、评分指标和数据划分。语言模型提出下一方法，A线在执行前核对引用的事实，训练后再把真实结果写回记忆。比如，模型可能说，C从零点三增大到三，正则化更强。对于同一路线，这个方向是错的。我们绑定参数、参照实验和来源，要求它明确纠正；证据不足就保留未知。候选库和预算调度仍由同一套模块负责。',
'我们先冻结案例再验收。未参与开发的二十四例中，十二条合法事实全部确认，六条错误没有被误确认，六条证据不足都保留为未知。随后，在优惠券、短信和花卉三个真实任务的固定历史上，调用Flash比较新旧协议。新协议的二十三条结构化断言都通过核验，三类预设旧错误都得到明确纠正。这证明了受控核验和纠错能力，还不能代表所有自然错误都能解决。',
'可靠性也有成本。我们将重复的核对信息压缩成可恢复记录，三个任务的输入token减少百分之六点一五，本次下一候选都没有改变。针对中断交付，六种可恢复状态成功导出一致预测，三种损坏或不一致状态正确拒绝，恢复时没有重新训练或调用API。',
'最后，我们没有把解释正确等同于成绩提高。八次跟进训练中，短信的新诊断动作平均低三点一八个百分点；花卉一正一负，没有稳定优势。新核验协议本轮费用还增加约百分之七十三。因此它保留为可选项。下一步是把知识纠正和必须训练的修复分开，减少不必要的实验。'
];
export async function build({presentation:p,input}){
 const phase=input.phase;
 if(phase==='cover'){
  if(p.slides.items.length===1 && p.slides.getItem(0).shapes.items.length===0)p.slides.delete([0]);
  if(p.slides.items.length!==0)throw new Error('Cover creation requires an empty deck; preserve existing saved slides');
  const s=p.slides.add({width:1280,height:720});s.background.fill=C.navy;
  text(s,'SUPERAGENT · A 线',80,88,1110,44,25,'#67D9CA',true);
  text(s,'让算法 Agent 的决策\n有据可查',76,203,1120,208,66,C.white,true);
  text(s,'任务契约 → 事实核验 → 可靠交付',80,465,1100,60,32,'#BED2E0');
  text(s,'2026.11.24  ·  2 分 30 秒',80,624,1100,35,20,'#9EB6C8');
 }
 if(phase==='architecture'){
  if(p.slides.items.length===2 && p.slides.getItem(0).shapes.items.length===0)p.slides.delete([0]);
  const s=page(p,'在规划与执行之间核对事实',2);
  text(s,'任务契约',72,180,210,54,30,C.teal,true);
  text(s,'→',276,180,55,54,32,C.muted);
  text(s,'LLM 提案',338,180,225,54,30,C.teal,true);
  text(s,'→',553,180,55,54,32,C.muted);
  text(s,'证据核验',618,180,210,54,30,C.teal,true);
  text(s,'→',831,180,55,54,32,C.muted);
  text(s,'执行与交付',905,180,270,54,30,C.teal,true);
  text(s,'固定标签、指标、划分；真实执行结果再写回历史',72,254,1110,52,28,C.muted);
  text(s,'C：0.3 → 3.0',72,365,1080,70,42,C.ink,true);
  text(s,'旧陈述：正则化更强',72,454,510,60,31,C.orange);
  text(s,'核验：正则化更弱',645,454,535,60,31,C.teal,true);
  text(s,'前提：同一受支持的正则化参数化；不据此推断准确率必然提高',72,540,1120,58,24,C.muted);
  text(s,'A：契约、事实与交付     B：共同候选库与预算调度',72,644,1005,32,20,C.muted);
 }
 if(phase==='evidence'){
  const s=page(p,'先检验事实，再观察真实模型',3);
  text(s,'冻结后的 24 个保留案例',72,161,640,50,30,C.teal,true);
  table(s,[['事实类别','确认结果'],['合法事实','12 / 12 确认'],['错误事实','0 / 6 被误确认'],['证据不足','0 / 6 被误确认']],72,231,625,297,[285,340]);
  text(s,'真实 Flash',767,162,430,51,30,C.teal,true);
  text(s,'23 条断言',765,239,435,70,44,C.ink,true);
  text(s,'均有记录支持',768,312,420,46,29,C.muted);
  text(s,'3 类预设错误',765,396,445,65,37,C.ink,true);
  text(s,'明确纠正，待执行确认',768,468,425,65,27,C.muted);
  text(s,'A145：24例程序 oracle，非人工金标准；A146：3个已见任务，新旧各6回复',72,588,1130,37,21,C.muted);
  text(s,'受控纠错证据；尚非自然错误率或完整闭环收益',72,635,1010,37,23,C.teal,true);
 }
 if(phase==='efficiency'){
  const s=page(p,'减少重复输入，保留可交付成果',4);
  text(s,'反馈输入 token',72,157,590,48,30,C.teal,true);
  s.charts.add('bar',{position:{left:65,top:220,width:610,height:306},categories:['完整反馈','紧凑反馈'],
   series:[{name:'输入 token',values:[13157,12348],fill:C.teal}],barOptions:{direction:'column',grouping:'clustered',gapWidth:110},
   hasLegend:false,titlePlacement:'none',chartFill:C.white,plotAreaFill:C.white,
   xAxis:{textStyle:{typeface:'Microsoft YaHei',fontSize:24,fill:C.ink},majorGridlines:null},
   yAxis:{min:0,max:16000,majorUnit:4000,numberFormatCode:'#,##0',textStyle:{typeface:'Microsoft YaHei',fontSize:19,fill:C.muted},majorGridlines:{fill:C.light,width:1}},
   dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'#,##0',textStyle:{typeface:'Microsoft YaHei',fontSize:24,bold:true,fill:C.ink}}});
  text(s,'减少 6.15%',73,552,590,55,34,C.teal,true);
  text(s,'本次三个状态：下一候选均相同',73,612,615,39,23,C.muted);
  text(s,'中断后的模型交付',744,157,465,50,30,C.teal,true);
  table(s,[['受控状态','结果'],['可恢复的 6 种','预测一致'],['损坏等 3 种','正确拒绝']],744,235,465,246,[266,199]);
  text(s,'0 次重训 · 0 次 API',747,527,455,56,30,C.ink,true);
  text(s,'A140：3任务、9请求；A141：9个受控快照，非真实断电测试',72,671,1020,32,18,C.muted);
 }
 if(phase==='conclusion'){
  const s=page(p,'受控纠错已验证，选模收益尚不稳定',5);
  text(s,'相同两分区跟进不同动作：8 次实际拟合',72,161,1120,51,29,C.teal,true);
  table(s,[['状态 / 指标','新动作减旧动作','证据边界'],['短信受控 / macro-F1','−3.18 pp','诊断 vs 改进'],['花卉自然 / accuracy','+0.63 pp','两划分一负一正']],72,239,1136,223,[440,290,406]);
  text(s,'核验协议本轮费用 +73.1%',72,492,1125,53,31,C.orange,true);
  text(s,'新旧各 6 回复：¥0.102702 / ¥0.059341（课程折算）',72,550,1120,39,23,C.muted);
  text(s,'保留为可选项；下一步分开知识纠正与训练修复',72,612,1030,49,28,C.teal,true);
 }
 if(phase==='appendices'){
  let s=page(p,'技术问答：为什么这样设计',6);s.hidden=true;
  const qa=[
   ['为什么不再加一个 Reviewer？','用绑定记录核验，不用另一个模型的赞同替代证据；因果猜想仍未知。'],
   ['这证明了通用性吗？','三个已见任务支持机制复用；受控案例和不同划分不等于新任务迁移。'],
   ['比较公平吗？','共享原始历史、候选、预算；新组额外有事实表和输出约束，未逐项隔离。'],
   ['为什么不默认开启？','未见稳定选模收益且成本增加；知识纠正可能挤占低收益诊断训练。'],
   ['失败会重试吗？','网络 / 解析 / 用量失败不自动重试；完整路径仍 ≤11 fit、≤3 Flash。']
  ];
  qa.forEach(([q,a],i)=>{text(s,q,72,153+i*94,1136,38,27,C.teal,true);text(s,a,72,193+i*94,1136,45,25,C.ink);});
  text(s,'问答附页 · 不计入 2 分 30 秒主讲',72,668,1010,29,19,C.muted);
  s=page(p,'数据来源与实现边界',7);s.hidden=true;
  table(s,[['批次','证据','不能推出什么'],['A145 / A146','保留24例；12次真实Flash','自然错误率为零 / 真实闭环已完成'],['A140','输入token少6.15%，候选相同','准确率普遍无损 / 稳定省15.7%费用'],['A141','6恢复、3拒绝，预测一致','任意断电都安全 / 已支持续训'],['A147','2任务不同动作，8fit','完整Agent提分 / 新任务泛化']],72,165,1136,340,[225,405,506]);
  text(s,'原始结果：Git 500983f；各批保留其原源码与输入身份',72,541,1110,48,27,C.teal,true);
  text(s,'当前 grounded-v1 显式启用；新97源完整路径与独立交付尚待验证',72,602,1110,43,24,C.muted);
  text(s,'问答附页 · 所有完整来源路径见备注及演讲稿',72,667,1010,30,19,C.muted);
 }
 if(phase==='notes'){
  const times=['00:00—00:15','00:15—00:55','00:55—01:30','01:30—02:00','02:00—02:30'];
  for(let i=0;i<5;i++){
   const s=p.slides.getItem(i);s.speakerNotes.text=times[i]+'\n\n'+script[i]+'\n\n'+input.sources;
  }
  p.slides.getItem(5).speakerNotes.text=input.qa;
  p.slides.getItem(6).speakerNotes.text=input.sources+'\n\n'+input.boundaries;
 }
 return {phase,slideCount:p.slides.items.length,mainSlides:5,appendixSlides:2};
}
export {script};
