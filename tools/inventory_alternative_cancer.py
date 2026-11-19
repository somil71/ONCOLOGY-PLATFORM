"""Metadata-only feasibility audit for an explicitly separate GBM/LGG study."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


def get_json(url, payload=None):
    body=None if payload is None else json.dumps(payload).encode()
    request=Request(url,data=body,headers={'User-Agent':'CancerFusion research feasibility audit','Content-Type':'application/json'})
    with urlopen(request,timeout=90) as response:
        raw=response.read()
        if not raw:
            raise RuntimeError(f'Official endpoint returned HTTP {response.status} with an empty body; no availability can be inferred.')
        return json.loads(raw)


def file_case_ids(project, datatype, strategy=None):
    terms=[{'op':'in','content':{'field':'cases.project.project_id','value':[project]}},
        {'op':'in','content':{'field':'access','value':['open']}},
        {'op':'in','content':{'field':'data_type','value':[datatype]}}]
    if datatype=='Gene Expression Quantification':
        terms.append({'op':'in','content':{'field':'analysis.workflow_type','value':['STAR - Counts']}})
    if strategy:terms.append({'op':'in','content':{'field':'experimental_strategy','value':[strategy]}})
    cases=set();offset=0;files=0
    while True:
        data=get_json('https://api.gdc.cancer.gov/files',{'filters':{'op':'and','content':terms},'fields':'file_id,cases.submitter_id','size':1000,'from':offset,'sort':'file_id:asc'})
        hits=data['data']['hits'];files+=len(hits)
        cases.update(c['submitter_id'] for h in hits for c in h.get('cases',[]))
        offset+=len(hits)
        if offset>=data['data']['pagination']['total']:break
        if not hits:raise RuntimeError('GDC pagination stopped prematurely')
    return cases,files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cancers',nargs='+',default=['GBM','LGG'])
    parser.add_argument('--modality',choices=['MR','CT'],default='MR')
    parser.add_argument('--output',type=Path,default=Path('data/alternative-cohort-feasibility'))
    args=parser.parse_args()
    output=args.output;output.mkdir(exist_ok=True,parents=True)
    source=Path('data/tcga-brca/TCGA-CDR-SupplementalTableS1.xlsx');cdr=pd.read_excel(source,sheet_name='TCGA-CDR')
    result={'queried_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Metadata-only alternative cancer feasibility; not a change of the BRCA study',
        'cdr_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cohorts':{}}
    for cancer in args.cancers:
        project='TCGA-'+cancer
        url='https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeries?'+urlencode({'Collection':project,'Modality':args.modality,'format':'json'})
        try:
            series=get_json(url)
            if not isinstance(series,list):raise RuntimeError(f'Unexpected TCIA response: {type(series)}')
            (output/f'{project}-{args.modality.lower()}-metadata.json').write_text(json.dumps(series,indent=2),encoding='utf-8')
            radiology={s['PatientID'] for s in series}
            tcia_status='metadata_retrieved'
        except (RuntimeError,ValueError) as exc:
            series=[];radiology=set();tcia_status=str(exc)
        rna,rna_files=file_case_ids(project,'Gene Expression Quantification')
        slides,slide_files=file_case_ids(project,'Slide Image','Diagnostic Slide')
        outcomes=cdr[cdr.type.eq(cancer)].copy();outcomes['OS.time']=pd.to_numeric(outcomes['OS.time'],errors='coerce')
        valid=outcomes['OS'].isin([0,1])&np.isfinite(outcomes['OS.time'])&outcomes['OS.time'].gt(0)
        usable=outcomes[valid]
        radiology_usable=usable[usable.bcr_patient_barcode.isin(radiology)]
        paired=radiology_usable[radiology_usable.bcr_patient_barcode.isin(rna&slides)]
        result['cohorts'][project]={'tcia_url':url,'tcia_api_status':tcia_status,'radiology_series':len(series) if radiology else None,'radiology_patients':len(radiology) if radiology else None,'cdr_cases':len(outcomes),
            'cdr_positive_os_cases':len(usable),'cdr_positive_os_deaths':int(usable.OS.sum()),
            'radiology_cdr_positive_os_cases':len(radiology_usable) if radiology else None,'radiology_cdr_positive_os_deaths':int(radiology_usable.OS.sum()) if radiology else None,
            'open_rna_cases':len(rna),'open_rna_files':rna_files,'diagnostic_slide_cases':len(slides),'diagnostic_slide_files':slide_files,
            'radiology_rna_slide_cdr_cases':len(paired) if radiology else None,'radiology_rna_slide_cdr_deaths':int(paired.OS.sum()) if radiology else None,
            'rna_slide_cdr_cases_without_mri_requirement':int(usable.bcr_patient_barcode.isin(rna&slides).sum()),
            'rna_slide_cdr_deaths_without_mri_requirement':int(usable.loc[usable.bcr_patient_barcode.isin(rna&slides),'OS'].sum()),
            'collection_page':f'https://www.cancerimagingarchive.net/collection/{project.lower()}/',
            'modality':args.modality,
            'published_collection_subjects_not_matched':{'GBM':262,'LGG':199}.get(cancer),
            'access_warning':'Brain MRI face-reconstruction access restrictions apply; current derivative access requires verification.' if args.modality=='MR' else 'Public metadata is not proof of raw download, protocol suitability, or baseline acquisition timing.',
            'interpretation':'Metadata overlap only; prediction-time series/sample selection, histology definition, raw decoding and event origin need independent review.'}
        if radiology:
            paired[['bcr_patient_barcode','OS.time','OS']].to_csv(output/f'{project}-candidate-overlap.csv',index=False)
        print(json.dumps({project:result['cohorts'][project]}),flush=True)
        (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
