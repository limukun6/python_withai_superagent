"""Match actual local credential bytes against intended Git changes; never print the credential."""
import argparse
from pathlib import Path
import subprocess


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--api-key-file',type=Path,required=True);args=p.parse_args()
    key=args.api_key_file.read_bytes().strip()
    if len(key)<8:raise ValueError('Cannot validate unexpectedly short credential')
    names=set()
    for command in (['git','diff','--name-only'],['git','diff','--cached','--name-only'],['git','ls-files','--others','--exclude-standard']):
        names.update(subprocess.check_output(command).decode('utf-8').splitlines())
    checked=0;bad=[]
    for name in names:
        path=Path(name)
        if not path.is_file():continue
        checked+=1
        if key in path.read_bytes():bad.append(name)
    if bad:raise ValueError('Credential detected in intended changes; refuse staging/publication')
    print(dict(files_checked=checked,actual_credential_found=False))
