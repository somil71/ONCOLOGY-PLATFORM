"""Download one public KIRC renal CT series for technical feasibility only."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import stat
from urllib.parse import urlencode
from urllib.request import urlopen
import zipfile

import pandas as pd


def main():
    inventory=Path('data/separate-cancer-feasibility')
    source=inventory/'TCGA-KIRC-ct-metadata.json'
    ids=set(pd.read_csv(inventory/'TCGA-KIRC-candidate-overlap.csv').bcr_patient_barcode)
    series=json.loads(source.read_text())
    candidates=[s for s in series if s['PatientID'] in ids
        and 'renal' in s.get('StudyDesc','').casefold()
        and 'nephrographic' in s.get('SeriesDescription','').casefold()
        and 30<=int(s.get('ImageCount',0))<=150]
    if not candidates:raise ValueError('No explicit nephrographic renal CT candidate in metadata')
    selected=min(candidates,key=lambda s:(int(s.get('FileSize',10**15)),s['SeriesInstanceUID']))
    out=Path('data/tcia-kirc');out.mkdir(exist_ok=True)
    selected['selection']='Smallest metadata-listed nephrographic renal-mass CT series with30-150images among candidate patients; no outcome used; engineering feasibility only.'
    selected['prediction_time_eligible']=False
    selected['prediction_time_note']='Study/series timing versus diagnosis not yet validated; not admitted to survival training by this pilot.'
    (out/'pilot-ct-selection.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    url='https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?'+urlencode({'SeriesInstanceUID':selected['SeriesInstanceUID']})
    archive=out/'pilot-series.zip'
    if not archive.exists():
        partial=out/'pilot-series.zip.part';size=0
        with urlopen(url,timeout=120) as response,partial.open('wb') as target:
            while True:
                block=response.read(1024*1024)
                if not block:break
                size+=len(block)
                if size>50*1024**2:raise ValueError('Pilot exceeded50MiB budget')
                target.write(block)
        if not zipfile.is_zipfile(partial):raise ValueError('Public endpoint did not return a valid archive')
        partial.replace(archive)
    destination=(out/'pilot-dicom'/selected['PatientID']).resolve();destination.mkdir(exist_ok=True,parents=True)
    with zipfile.ZipFile(archive) as bundle:
        seen=set()
        for info in bundle.infolist():
            target=(destination/info.filename).resolve()
            if not target.is_relative_to(destination) or Path(info.filename).is_absolute() or stat.S_ISLNK(info.external_attr>>16):
                raise ValueError('Unsafe archive entry')
            if target in seen:raise ValueError('Duplicate archive entry')
            seen.add(target)
        bundle.extractall(destination)
    provenance={**selected,'download_url':url,'downloaded_at_utc':datetime.now(timezone.utc).isoformat(),
        'zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'downloaded_bytes':archive.stat().st_size,
        'metadata_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'extracted_files':len([p for p in destination.rglob('*') if p.is_file()]),
        'output_directory':str(destination),'scope':'Public CT pilot only; clinical protocol, prediction-time and tumor visibility remain unreviewed.'}
    (out/'pilot-download-provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps(provenance))


if __name__=='__main__':main()
