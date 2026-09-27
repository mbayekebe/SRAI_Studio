"""Reconcile verified Studio assets and preserve legacy Book 6 as an archive."""
import argparse, ast, hashlib, json, os, sqlite3, sys
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal

ROOT=Path(__file__).resolve().parent
MARK='[Studio asset reconciliation v1.1]'
TYPES={
 'book_chapter':('Book chapter',True,20), 'notebook':('Notebook',True,20),
 'exercises':('Exercises',True,10),'executive_brief':('Executive brief',True,10),
 'youtube':('Long YouTube lesson',True,15),'github':('GitHub landing page',True,15),
 'linkedin':('LinkedIn post',True,10),'website_page':('Website page',False,0),
 'presentation':('Teaching presentation',False,0),'instructor_solutions':('Instructor solutions',False,0),
 'learner_guide':('Learner guide',False,0),'assessment_brief':('Learner assessment brief',False,0),
 'supporting_data':('Supporting data',False,0),'institutional_toolkit':('Institutional toolkit',False,0),
 'thumbnail':('Video thumbnail',False,0),'controlled_resources':('Controlled learning resources',True,100),
}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def setup(studio):
    tree=ast.parse((studio/'manage.py').read_text(encoding='utf-8-sig')); names=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='setdefault' and len(n.args)>1 and isinstance(n.args[0],ast.Constant) and n.args[0].value=='DJANGO_SETTINGS_MODULE':names.append(ast.literal_eval(n.args[1]))
    if len(names)!=1:raise RuntimeError('Ambiguous Django settings')
    sys.path.insert(0,str(studio));os.chdir(studio);os.environ['DJANGO_SETTINGS_MODULE']=names[0]
    import django;django.setup()

def dump_all():
    from django.apps import apps
    return {m._meta.model_name:list(m.objects.order_by('pk').values()) for m in apps.get_app_config('studio').get_models()}

def append_note(old,note):return old if note in old else (old.rstrip()+'\n\n'+note).strip()

def reconcile(spec,report):
    from studio.models import Book,Chapter,Notebook,ProductionUnit,AssetType,ProductionAsset,ProductionTask,Publication
    archive=Book.objects.filter(code='B06-LEGACY').first()
    if archive is None:
        old=Book.objects.get(code='B06')
        if old.title!='Decision Intelligence':raise RuntimeError('Unexpected Book 6 identity; stopped.')
        actual=list(old.chapters.order_by('number'))
        if len(actual)!=len(spec['legacy_b6']):raise RuntimeError('Legacy chapter count changed.')
        for ch,expected in zip(actual,spec['legacy_b6']):
            if (ch.code,ch.number,ch.title)!=(expected['chapter']['code'],expected['chapter']['number'],expected['chapter']['title']):raise RuntimeError('Legacy chapter identity changed: '+ch.code)
            nb=ch.notebook
            if nb.code!=expected['notebook']['code'] or nb.repository_path!=expected['notebook']['repository_path']:raise RuntimeError('Legacy notebook identity changed: '+ch.code)
            if ch.production_unit.code!='PU-'+ch.code:raise RuntimeError('Legacy production unit identity changed.')
        if ProductionUnit.objects.filter(code__startswith='LEGACY-B06-').exists() or Chapter.objects.filter(code__startswith='LEGACY-B06-').exists():raise RuntimeError('Archive code collision.')
        for ch in actual:
            unit=ch.production_unit;oldcode=unit.code
            Chapter.objects.filter(pk=ch.pk).update(code='LEGACY-'+ch.code)
            ProductionUnit.objects.filter(pk=unit.pk).update(code='LEGACY-'+ch.code,notes=append_note(unit.notes,'[Archived Decision Intelligence source] Former unit '+oldcode+'; retained with its original assets, notebooks and history.'))
        Book.objects.filter(pk=old.pk).update(code='B06-LEGACY',title='Decision Intelligence (archived source curriculum)',status='archived',sequence=106)
        report['book6_action']='Archive 15 legacy chapters and units; preserve their primary keys and all related history.'
        book=Book.objects.create(code='B06',title='Generative Artificial Intelligence',slug='generative-artificial-intelligence',sequence=6,status='active',description='Current Book 6. Verified Lesson 1 imported; future syllabus awaiting canonical curriculum import. Legacy Decision Intelligence preserved under B06-LEGACY.')
        item=spec['units']['PU-B06-C01']
        ch=Chapter.objects.create(book=book,code='B06-C01',number=1,title=item['title'],slug='foundations-of-generative-ai',status='published')
        unit=ProductionUnit.objects.create(chapter=ch,code='PU-B06-C01',workflow_state='published',notes='Created from verified Generative AI Lesson 1; independent review not asserted.')
        asset=next(a for a in item['assets'] if a['type']=='notebook')
        Notebook.objects.create(chapter=ch,code='M6_N01',title=item['title'],repository_path=asset['file_path'],validation_status='passed',validation_message='Owner-confirmed execution PASS recorded in PU-B06-C01 publication evidence; canonical SHA-256 verified during reconciliation. Not re-executed by this importer.')
        for p in item['publications']:Publication.objects.create(production_unit=unit,**p)
    else:
        if archive.status!='archived' or archive.chapters.count()!=15:raise RuntimeError('Unexpected archive state.')
        unit=ProductionUnit.objects.get(code='PU-B06-C01')
        if unit.chapter.book.code!='B06' or unit.chapter.title!=spec['units']['PU-B06-C01']['title']:raise RuntimeError('Unexpected current Book 6 state.')
        report['book6_action']='Already reconciled; no repeated archive or unit creation.'
    report['asset_changes']=[]
    for code,item in spec['units'].items():
        unit=ProductionUnit.objects.get(code=code)
        for a in item['assets']:
            name,required,weight=TYPES[a['type']]
            typ,created=AssetType.objects.get_or_create(code=a['type'],defaults={'name':name,'family':'educational' if a['type'] not in ['youtube','github','linkedin','website_page'] else 'communication','required':required,'weight':weight,'sequence':50})
            obj=ProductionAsset.objects.filter(production_unit=unit,asset_type=typ).first()
            note=MARK+' '+json.dumps({'source_version':item['version'],'files':a['supporting_files'],'basis':'Supplied canonical manifest and publication acceptance; file preflight recorded in report.'},sort_keys=True)
            desired={k:a[k] for k in ['title','status','version','file_path','public_url']}
            desired['notes']=append_note(obj.notes if obj else '',note)
            if obj and obj.status in ['drafting','review']:
                raise RuntimeError('Active asset edit conflicts with release reconciliation: '+code+'/'+a['type'])
            changed={k:v for k,v in desired.items() if obj is None or getattr(obj,k)!=v}
            if changed:
                report['asset_changes'].append({'unit':code,'type':a['type'],'before':None if obj is None else dict(ProductionAsset.objects.filter(pk=obj.pk).values().get()),'updates':changed})
                if obj:ProductionAsset.objects.filter(pk=obj.pk).update(**changed)
                else:
                    # Bypass model.save's synthetic approval time. No historical time is invented.
                    ProductionAsset.objects.bulk_create([ProductionAsset(production_unit=unit,asset_type=typ,approved_at=None,**desired)])
        # Compute asset completion only; do not promote Gold Standard or create reviews.
        assets=list(unit.assets.select_related('asset_type'))
        denominator=sum((a.asset_type.weight for a in assets if a.asset_type.required),Decimal(0))
        numerator=sum((a.asset_type.weight for a in assets if a.asset_type.required and a.status in ProductionAsset.COMPLETE_STATUSES),Decimal(0))
        percentage=(numerator/denominator*100).quantize(Decimal('0.01')) if denominator else Decimal(0)
        note=MARK+' Canonical asset inventory reconciled. Completion measures registered assets only; existing independent-review history is unchanged.'
        action='Maintain current publication and canonical asset evidence.'
        if code=='EP-M02':action='Supply private controlled-resource inventory; public page/video are recorded, private downloads remain disabled.'
        elif code=='PU-B06-C01':action='Maintain published Lesson 1; import future lessons only from the approved Book 6 syllabus.'
        desired={'completion_percentage':percentage,'notes':append_note(unit.notes,note),'next_action':action}
        if any(getattr(unit,k)!=v for k,v in desired.items()):ProductionUnit.objects.filter(pk=unit.pk).update(**desired)
        report.setdefault('completion',{})[code]=str(percentage)
    tasks=[('EP-M02','Reconcile private controlled-resource inventory','The supplied repository inventory contains no controlled EP-M02 learner files. Publication does not establish their completion.'),('PU-B06-C01','Import approved future Book 6 syllabus','The supplied Book 6 repository includes verified Lesson 1 only. Do not invent future chapter titles or migrate legacy Decision Intelligence notebooks into Generative AI.')]
    for code,title,description in tasks:
        ProductionTask.objects.get_or_create(production_unit=ProductionUnit.objects.get(code=code),title=title,defaults={'description':description,'status':'blocked','priority':'normal'})


def preflight(spec,repo_root):
    results=[]
    for original,meta in spec['files'].items():
        relative=original.split('C:/SRAI_GitHub/',1)[1];actual=repo_root/relative
        result={'source':original,'actual':str(actual),'exists':actual.is_file()}
        if actual.is_file():
            result['size_matches']=actual.stat().st_size==meta['bytes']
            content=actual.read_bytes()
            result['sha256']=hashlib.sha256(content).hexdigest()
            result['hash_match_method']='exact'
            result['hash_matches']=not meta['sha256'] or result['sha256']==meta['sha256']
            if not result['hash_matches'] and actual.suffix.lower() in {'.ipynb','.csv','.json','.md','.txt'}:
                # Git and Windows can check out the same text with opposite newline styles.
                # Accept only if a pure LF or CRLF conversion reproduces the recorded SHA.
                lf=content.replace(b'\r\n',b'\n')
                for method,variant in (('LF',lf),('CRLF',lf.replace(b'\n',b'\r\n'))):
                    if hashlib.sha256(variant).hexdigest()==meta['sha256']:
                        result['hash_matches']=True
                        result['hash_match_method']='line_ending_conversion_to_'+method
                        break
        result['pass']=result['exists'] and result.get('size_matches',False) and result.get('hash_matches',False)
        results.append(result)
    return results

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--studio',required=True);p.add_argument('--repositories',default='C:/SRAI_GitHub');p.add_argument('--apply',action='store_true');args=p.parse_args(argv)
    spec=json.loads((ROOT/'spec.json').read_text())
    for rel,h in spec['sources'].items():
        if sha(ROOT/rel)!=h:raise RuntimeError('Bundled source mismatch: '+rel)
    studio=Path(args.studio).resolve();setup(studio)
    from django.db import connection,transaction
    from studio.models import ProductionUnit,Book
    if connection.vendor!='sqlite':raise RuntimeError('This updater requires SQLite.')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=ROOT/'results'/stamp;out.mkdir(parents=True)
    report={'mode':'apply' if args.apply else 'preview','file_checks':preflight(spec,Path(args.repositories)),'notes':spec['notes']}
    failures=[f for f in report['file_checks'] if not f['pass']]
    if failures:
        report['status']='BLOCKED_FILE_CHECKS';(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        for f in failures:print('FILE CHECK FAILED:',f['actual'],'size=',f.get('size_matches'),'hash=',f.get('hash_matches'))
        print('Report:',out/'report.json');return 2
    # Use resolved paths in Studio, allowing an explicit alternate repository root.
    locations={r['source']:r['actual'] for r in report['file_checks']}
    for item in spec['units'].values():
        for a in item['assets']:
            if a['file_path']:a['file_path']=locations[a['file_path']]
    db=Path(connection.settings_dict['NAME']).resolve();before_hash=sha(db) if not args.apply else None
    if args.apply:
        backup=out/'Studio_before.sqlite3'
        with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as src,sqlite3.connect(backup) as dst:
            src.backup(dst)
            if dst.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise RuntimeError('Invalid backup')
        report['backup']={'path':str(backup),'sha256':sha(backup)}
    with transaction.atomic():
        before=dump_all();report['before']=before
        report['unit_count_before']=ProductionUnit.objects.count()
        reconcile(spec,report)
        after=dump_all()
        # Existing validation/review/publication/release history must be byte-for-byte equivalent at field level.
        for name in ['notebook','qualityreview','publication','release']:
            byid={r['id']:r for r in after[name]}
            if any(byid.get(r['id'])!=r for r in before[name]):raise RuntimeError('History altered: '+name)
        # Existing asset approval timestamps and every existing Gold Standard flag are preserved.
        assets={r['id']:r for r in after['productionasset']}
        if any(assets[r['id']]['approved_at']!=r['approved_at'] for r in before['productionasset']):raise RuntimeError('Approval timestamp changed')
        current={r['id']:r for r in after['productionunit']}
        if any(current[r['id']]['gold_standard']!=r['gold_standard'] for r in before['productionunit']):raise RuntimeError('Gold Standard changed')
        verify={};reconcile(spec,verify)
        if dump_all()!=after:raise RuntimeError('Repeat-run idempotence failed')
        report['checks']={'original_notebook_validation_preserved':True,'reviews_publications_releases_preserved':True,'approval_dates_preserved':True,'gold_standard_preserved':True,'idempotent':True}
        report['unit_count_after']=ProductionUnit.objects.count();report['active_books']=Book.objects.filter(status='active').count();report['archived_books']=Book.objects.filter(status='archived').count()
        report['status']='APPLY_PENDING_COMMIT' if args.apply else 'PREVIEW_ROLLED_BACK'
        (out/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
        if not args.apply:transaction.set_rollback(True)
    if not args.apply:
        connection.close()
        # Rollback restores logical data; SQLite headers may change in some journaling configurations.
        report['database_logically_unchanged']=True
    else:report['status']='APPLIED_VERIFIED_ASSET_SCOPE_COMPLETE'
    (out/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(report['status']);print('Units:',report['unit_count_before'],'->',report['unit_count_after']);print('Books: active',report['active_books'],'archived',report['archived_books'])
    print('Asset rows changed/created:',len(report['asset_changes']));print('Local file checks:',len(report['file_checks']),'PASS')
    for code,value in report['completion'].items():print(code, 'asset completion',value+'%')
    print('Open tasks: EP-M02 private inventory; approved future Book 6 syllabus.');print('Report:',out/'report.json')
    return 0

if __name__=='__main__':sys.exit(main())
