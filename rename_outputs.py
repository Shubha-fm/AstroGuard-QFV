#!/usr/bin/env python3
"""Rename AstroGuard-QFV report ZIPs and change future report ZIP filenames.

Run from anywhere: python3 rename_outputs.py --project /path/to/AstroGuard_18K_macOS_Fresh_Start
Only report naming and references are touched. Training data and checkpoints are untouched.
"""
import argparse
from pathlib import Path

NEW_ZIP = {
    'RESULTS_18K/SEND_ME_18K_COLLECTION.zip': 'RESULTS_18K/AstroGuard_QFV_18K_Collection_Summary.zip',
    'RESULTS_18K/classical/SEND_ME_LARGE_RESULTS.zip': 'RESULTS_18K/classical/AstroGuard_QFV_18K_Classical_Results.zip',
    'RESULTS_18K/matched/SEND_ME_LARGE_RESULTS.zip': 'RESULTS_18K/matched/AstroGuard_QFV_18K_Matched_Results.zip',
    'RESULTS_18K/smoke/SEND_ME_LARGE_RESULTS.zip': 'RESULTS_18K/smoke/AstroGuard_QFV_18K_Smoke_Results.zip',
    'RESULTS_18K/quantum_sanity/SEND_ME_LARGE_RESULTS.zip': 'RESULTS_18K/quantum_sanity/AstroGuard_QFV_18K_Quantum_Sanity_Results.zip',
    'RESULTS_18K/independent/SEND_ME_INDEPENDENT_LABELS.zip': 'RESULTS_18K/independent/AstroGuard_QFV_18K_Independent_Labels.zip',
    'RESULTS/SEND_ME_BULK_REPORT.zip': 'RESULTS/AstroGuard_QFV_18K_Bulk_Collection_Summary.zip',
}

PATCHES = {
    'run_18k_collect.py': [
        ('SEND_ME_18K_COLLECTION.zip', 'AstroGuard_QFV_18K_Collection_Summary.zip'),
    ],
    'run_18k_external_labels.py': [
        ('SEND_ME_INDEPENDENT_LABELS.zip', 'AstroGuard_QFV_18K_Independent_Labels.zip'),
    ],
    'run_imbalanced_bulk.py': [
        ('SEND_ME_BULK_REPORT.zip', 'AstroGuard_QFV_18K_Bulk_Collection_Summary.zip'),
    ],
    'run_large.py': [
        ("z=output/'SEND_ME_LARGE_RESULTS.zip'",
         "tag={'classical':'Classical', 'matched':'Matched', 'smoke':'Smoke',\n"
         "         'quantum_sanity':'Quantum_Sanity'}.get(output.name,output.name)\n"
         "    z=output/f'AstroGuard_QFV_18K_{tag}_Results.zip'"),
        ("if file.suffix.lower() in ('.pt','.joblib'",
         "if file.suffix.lower() in ('.zip','.pt','.joblib'"),
    ],
    'README_MAC.md': [
        ('SEND_ME_18K_COLLECTION.zip','AstroGuard_QFV_18K_Collection_Summary.zip'),
        ('RESULTS_18K/classical/SEND_ME_LARGE_RESULTS.zip',
         'RESULTS_18K/classical/AstroGuard_QFV_18K_Classical_Results.zip'),
        ('RESULTS_18K/matched/SEND_ME_LARGE_RESULTS.zip',
         'RESULTS_18K/matched/AstroGuard_QFV_18K_Matched_Results.zip'),
    ],
    'README_NEXT_STAGE.md': [
        ('SEND_ME_BULK_REPORT.zip', 'AstroGuard_QFV_18K_Bulk_Collection_Summary.zip'),
    ],
    'README_MAC_FRESH.md': [
        ('SEND_ME_18K_COLLECTION.zip','AstroGuard_QFV_18K_Collection_Summary.zip'),
        ('RESULTS_18K/classical/SEND_ME_LARGE_RESULTS.zip',
         'RESULTS_18K/classical/AstroGuard_QFV_18K_Classical_Results.zip'),
        ('RESULTS_18K/matched/SEND_ME_LARGE_RESULTS.zip',
         'RESULTS_18K/matched/AstroGuard_QFV_18K_Matched_Results.zip'),
    ],
}


def apply(project):
    project = Path(project).expanduser().resolve()
    if not (project/'mac_run.sh').exists():
        raise SystemExit(f'Not the AstroGuard project folder: {project}')
    print('Project:', project)
    for rel, transforms in PATCHES.items():
        p=project/rel
        if not p.exists():
            print('SKIP (not present):', rel)
            continue
        old=p.read_text(encoding='utf-8')
        new=old
        for source,target in transforms:
            if source in new: new=new.replace(source,target)
            elif target not in new: raise SystemExit(f'Expected naming text not found in {rel}: {source}')
        if new!=old:
            backup=p.with_name(p.name+'.before_filename_update')
            if not backup.exists(): backup.write_text(old,encoding='utf-8')
            p.write_text(new,encoding='utf-8')
            print('UPDATED:',rel,' (backup:',backup.name+')')
        else: print('ALREADY UPDATED:',rel)
    for src,dst in NEW_ZIP.items():
        a,b=project/src,project/dst
        if not a.exists(): continue
        if b.exists():
            print('SKIP rename (target exists):',dst)
        else:
            a.rename(b)
            print('RENAMED:',src,'->',dst)
    print('Report naming finished. No input, model checkpoints, or predictions were modified.')


if __name__ == '__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--project',default='.',help='Path to Mac experiment folder')
    args=cli.parse_args()
    apply(args.project)
