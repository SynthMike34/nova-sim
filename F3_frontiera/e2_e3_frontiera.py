#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e2_e3_frontiera.py - figura della frontiera E2 vs E3.
Confronta i passi del ciclo a evento (E2) con quelli della marcia avanti
comandata (E3). I due conteggi sono letti dalla chiave canone_rettificato di
config/joint_limits_v2.json, non scritti a mano.

    python e2_e3_frontiera.py           genera la figura
    python e2_e3_frontiera.py --test    genera e verifica (conteggi > 0, rapporto coerente)
"""
import os as _os


def _MP(_n):
    _qui = _os.path.dirname(_os.path.abspath(__file__))
    _base = _os.path.dirname(_qui)
    for _c in (_n, _os.path.join(_base, 'config', _os.path.basename(_n))):
        if _os.path.exists(_c):
            return _c
    return _os.path.join(_base, 'config', _os.path.basename(_n))


def _OUT(_n):
    _qui = _os.path.dirname(_os.path.abspath(__file__))
    _d = _os.path.join(_os.path.dirname(_qui), 'outputs')
    if not _os.path.isdir(_d):
        try:
            _os.makedirs(_d)
        except Exception:
            _d = _qui
    return _os.path.join(_d, _os.path.basename(_n))


import sys
import re
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VERSIONE = '1.0'


def dal_canone():
    """Legge E1/E2/E3 dalla chiave di canone. Nessun valore a memoria."""
    d = json.load(open(_MP('joint_limits_v2.json'), encoding='utf-8'))
    testo = json.dumps(d.get('_nova_sim_campaign_v1', {}), ensure_ascii=False)
    e2 = re.search(r'E2\s+(\d+)\s+cicli', testo)
    e3 = re.search(r'E3\s+(\d+)\s+passi', testo)
    if not (e2 and e3):
        raise SystemExit('conteggi E2/E3 non trovati nella chiave di canone')
    return int(e2.group(1)), int(e3.group(1))


def main():
    test = '--test' in sys.argv
    print('e2_e3_frontiera.py - versione ' + VERSIONE)

    n2, n3 = dal_canone()
    rap = n2 / float(n3)
    print(' E2 %d cycles | E3 %d steps | ratio %.1f' % (n2, n3, rap))

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    barre = ax.bar(['E2\nevent-triggered timing,\nretrograde equilibrium',
                    'E3\ncommanded forward\nwalking'],
                   [n2, n3], color=['#1f77b4', '#2ca02c'], width=0.5)
    for b, v in zip(barre, (n2, n3)):
        ax.text(b.get_x() + b.get_width() / 2, v + max(n2, n3) * 0.03,
                '%d steps' % v, ha='center', fontsize=13, fontweight='bold')
    ax.annotate('', xy=(1, n3 + max(n2, n3) * 0.10),
                xytext=(0, n2 - max(n2, n3) * 0.06),
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=2))
    ax.text(0.52, (n2 + n3) / 2.0 + max(n2, n3) * 0.06, 'ratio %.1f' % rap,
            ha='center', fontsize=15, fontweight='bold', color='#d62728')
    ax.set_ylabel('steps [C]')
    ax.set_ylim(0, max(n2, n3) * 1.16)
    ax.set_title('E2 vs E3 frontier @66.23 kg - computed in simulation',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(_OUT('e2_e3_frontiera.png'), dpi=150)
    print('Salvato e2_e3_frontiera.png')

    json.dump(dict(versione=VERSIONE, E2_cicli=n2, E3_passi=n3,
                   rapporto=round(rap, 1)),
              open(_OUT('e2_e3_frontiera.json'), 'w'), indent=1)

    if test:
        ok = n2 > 0 and n3 > 0 and 1.0 < rap < 10.0
        print('TEST ' + ('PASS' if ok else 'FAIL'))
        sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
