"""Verify row-level results, paired decisions and source freeze; never fit a model."""
import statistics
from study import ROOT,SOURCE,OUT,OLD,GUARD,PACKAGES,SEEDS,load,subset,take,sha,read,save,lines,manifest,grouped_partition
from superagent.evidence_schedule import catalog_for,schedule,duel
from superagent.models import predict_with_model
import joblib
import numpy as np
import zipfile

def same_features(first,second):
    left,right=np.asarray(first),np.asarray(second)
    if left.shape!=right.shape or left.dtype!=right.dtype:return False
    if left.dtype!=object:return np.array_equal(left,right)
    return all(type(a) is type(b) and (a==b or isinstance(a,(float,np.floating)) and np.isnan(a) and np.isnan(b))
               for a,b in zip(left.flat,right.flat))

def verify():
    assert manifest()==read(OUT/'freeze.json')
    assert 'Ran 280 tests' in (ROOT/'tests.log').read_text() and '\nOK\n' in (ROOT/'tests.log').read_text()
    past=read(OUT/'replay.json')
    assert len(past)==6 and all(r['reproduced'] for r in past)
    notes=['v0.42｜固定复核 vs. 按证据探索、复核或停止','',
        '纯本地、真实分类器拟合；不读取课程验证集或正式测试集。双方都从目录首项开始，使用同一 expanded 候选库、相同的训练划分、探索规则和时间预算，每臂最多10次测量加1次完整重训。外层评分只在两臂选择已封存后进行。',
        f"历史回放：旧策略 {len(past)}/6 重现历史选型；新策略只有 {sum(r['missing'] is None for r in past)}/6 条路径有完整历史观测，其余停在首个缺失请求，没有补分。",
        '探索和复核共用预算，比较的是整套调度和晋级规则；不能把效果单独归因于提前复核。']
    verified={};total_fits=0
    for name in PACKAGES:
        whole,access=load(name);root=OUT/name
        assert read(root/'done.json')['pairs']==3 and set(access)==set(read(root/'package_access.json'))
        keep,reserve,_=grouped_partition(whole,41101,'shared_text')
        dev=subset(whole,keep);rows=read(root/'paired.json')
        assert [r['seed'] for r in rows]==list(SEEDS)
        details=[];notes.extend(['',name+'（以下均为训练内外层保留集）'])
        for row in rows:
            folder=root/str(row['seed']);bounds=read(folder/'outer.json')
            fit,held,_=grouped_partition(dev,row['seed'],'shared_text')
            assert bounds['train']==keep[fit].tolist() and bounds['held']==keep[held].tolist()
            assert not set(bounds['train']+bounds['held']).intersection(reserve)
            training=subset(dev,fit);correct={}
            for tag,arm in row['arms'].items():
                assert arm['complete'] and 1<arm['fits']<=11
                path=folder/tag;bundle_path=path/'selected_model.joblib'
                assert sha(bundle_path)==arm['model_hash']==read(path/'final_selection.json')['model_sha256']
                bundle=joblib.load(bundle_path);assert bundle['candidate']==arm['selected']
                preds=predict_with_model(training.kind,bundle['model'],take(dev.X_train,held))
                assert np.array_equal(preds,read(folder/f'{tag}_outer_predictions.json'))
                correct[tag]=preds==dev.y_train[held]
                assert int(correct[tag].sum())==arm['correct']
                total_fits+=arm['fits']
            fixed=folder/'fixed';adaptive=folder/'adaptive'
            for split in range(3):
                left=read(adaptive/f'split_{split}.json')
                right=read(fixed/('split.json' if not split else f'confirmation_split_{split}.json'))
                inner_train,inner_held,_=grouped_partition(training,707+split*10000,'shared_text')
                assert left['train_indices']==right['train_indices']==inner_train.tolist()
                assert left['search_indices']==right['search_indices']==inner_held.tolist()
                for payload_path in [adaptive/f'payload_{split}.joblib',fixed/(f'confirmation_data_{split}.joblib' if split else 'inner_data.joblib')]:
                    payload=joblib.load(payload_path)
                    for side,indices in [('train',inner_train),('search',inner_held)]:
                        assert np.array_equal(payload['y_'+side],training.y_train[indices])
                        assert same_features(payload['X_'+side],take(training.X_train,indices))
            trace=lines(adaptive/'events.jsonl'); logs=[]
            def demand(candidate,split):
                target=joblib.load(adaptive/f'payload_{split}.joblib')['y_search']
                return joblib.load(adaptive/f'measured_{split}_{candidate}.joblib')['search_predictions']==target
            chosen=schedule(catalog_for(training.kind),demand,logs.append)
            assert logs+[dict(event='selection_frozen',candidate=chosen)]==trace and chosen==row['arms']['adaptive']['selected']
            assert [r['event'] for r in lines(fixed/'llm_usage.jsonl')]==['ledger_initialized']
            d=duel(correct['fixed'],correct['adaptive'])
            assert all(row[k]==v for k,v in d.items())
            assert len([e for e in logs if e['event']=='request'])+1==row['arms']['adaptive']['fits']
            assert row['arms']['fixed']['fits']==11
            primary=[e for e in logs if e['event']=='observed' and e['split']==0]
            best=min(primary,key=lambda e:(-e['accuracy'],e['candidate']))['candidate']
            duels=[e for e in logs if e['event']=='duel']
            old_trials=read(fixed/'trials.json')
            old_top=min(old_trials,key=lambda e:(-e['search_accuracy'],e['candidate']))['candidate']
            old_meas=read(fixed/'confirmation.json')['candidates']
            opp=flips=0
            for other in old_meas:
                lead=next(c for c in old_meas if c['candidate']==old_top)['measurements']
                primary_delta=lead[0]['search_accuracy']-other['measurements'][0]['search_accuracy']
                if primary_delta>0:
                    opp+=1;flips+=lead[1]['search_accuracy']<=other['measurements'][1]['search_accuracy']
            value=dict(seed=row['seed'],coverage=[len(old_trials),len(primary)],duels=len(duels),
                adaptive_positive=sum(e['deltas'][0]>0 for e in duels),adaptive_first_gain_failed=sum(e['first_gain_failed_recheck'] for e in duels),
                fixed_positive=opp,fixed_first_gain_failed=flips,adaptive_differs_from_own_primary=chosen!=best,
                fixed_differs_from_own_primary=row['arms']['fixed']['selected']!=old_top,
                stop=logs[-1]['reason'],promotions=sum(e['promoted'] for e in duels))
            details.append(value)
            old,new=[row['arms'][k] for k in ('fixed','adaptive')]
            notes.extend([f"  {row['seed']}：{old['correct']}/{row['rows']}={old['accuracy']:.2%} → {new['correct']}/{row['rows']}={new['accuracy']:.2%}；修正{row['fixed']}/新增错误{row['lost']}；拟合{old['fits']}→{new['fits']}，总秒数{old['elapsed']:.1f}→{new['elapsed']:.1f}。",
                f"  选型 {old['selected']} → {chosen}；探索候选 {value['coverage']}；提前复核{len(duels)}次，首轮正差后次轮不胜 {value['adaptive_first_gain_failed']}/{value['adaptive_positive']}；停止理由 {value['stop']}。"])
            if new['selected']!=old['selected']:
                scores={m['candidate']:[s['search_accuracy'] for s in m['measurements']] for m in old_meas}
                if all(item['selected'] in scores for item in (old,new)):
                    diff=[a-b for a,b in zip(scores[old['selected']],scores[new['selected']])]
                    notes.append('  仅事后诊断，未回传两臂：固定臂所选相对于自适应臂所选，在固定臂的三个共同内部划分上差值依次为 '+
                        ' / '.join(f'{v*100:+.3f}' for v in diff)+' 个百分点。')
        means={tag:statistics.mean(row['arms'][tag]['accuracy'] for row in rows) for tag in ('fixed','adaptive')}
        fits={tag:sum(row['arms'][tag]['fits'] for row in rows) for tag in ('fixed','adaptive')}
        times={tag:sum(row['arms'][tag]['elapsed'] for row in rows) for tag in ('fixed','adaptive')}
        delta=statistics.mean(r['delta'] for r in rows);wins=sum(r['delta']>0 for r in rows)
        saving=1-fits['adaptive']/fits['fixed']
        passed=(delta>=.002 and wins>=2 and saving>=0) or (delta>=-.002 and saving>=.2)
        verified[name]=dict(means=means,fits=fits,times=times,mean_delta=delta,wins=wins,fit_saving=saving,screen_pass=passed,details=details)
        notes.append(f"  三组等权平均 {means['fixed']:.3%} → {means['adaptive']:.3%}，差 {100*delta:+.3f} 个百分点；总拟合 {fits['fixed']} → {fits['adaptive']}（节省{saving:.1%}）；预设初筛门槛：{'通过' if passed else '未过'}。")
    both=all(r['screen_pass'] for r in verified.values())
    notes.extend(['',f'本轮共 {total_fits} 次实测分类器拟合，12 次完整流程；历史回放与复核没有新增拟合。280项测试通过，源代码与冻结清单一致。新增模型API调用为0。',
        '初筛门槛为每任务均值至少提升0.2个百分点且2/3胜、不增加拟合；或均值损失不超过0.2个百分点且至少省20%拟合。不作统计等效声明。'+('两项均过，可另做确认。' if both else '未满足两任务共同晋级要求，维持原默认。'),
        '两个策略复核的候选对不同，反转数是描述，不是错误采纳率的因果比较。三组训练数据有重叠；不能据此声称无人见过的任务也会如此。新策略目前仅支持accuracy和本地规划，没有动历史费用或新评课程验证集。'])
    save(ROOT/'verification.json',dict(verified=True,datasets=verified,model_fits=total_fits,both_pass=both))
    output=ROOT.parents[1]/'outputs'
    report=output/'superagent-v0.42-REPORT.txt';report.write_text('\n'.join(notes)+'\n')
    package=output/'superagent-v0.42-study.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or 'feature_cache' in path.parts or path.suffix in ('.joblib','.pyc'):continue
            archive.write(path,'work/superagent_v42/'+str(path.relative_to(ROOT)))
        archive.write(GUARD/'sitecustomize.py','work/superagent_v38/offline_guard/sitecustomize.py')
        for name in PACKAGES:
            archive.write(OLD/name/'reserve.json',f'work/superagent_v41/results/{name}/reserve.json')
        archive.write(report,'REPORT.txt')
    with zipfile.ZipFile(package) as archive:assert archive.testzip() is None
    print('\n'.join(notes))
if __name__=='__main__':verify()
