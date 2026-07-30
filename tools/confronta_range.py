#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""confronta_range.py — XML vs joint_limits_v2.json (range SW), tabella discrepanze.
   python confronta_range.py [--test]"""
import os as _os
def _MP(_n):
    if _os.path.isabs(_n) or _os.sep in _n or '/' in _n:
        if _os.path.exists(_n): return _n
    _qui = _os.path.dirname(_os.path.abspath(__file__))
    _base = _os.path.dirname(_qui)
    for _c in (_n,
               _os.path.join(_base, 'models', _os.path.basename(_n)),
               _os.path.join(_base, 'config', _os.path.basename(_n)),
               _os.path.join(_qui, _os.path.basename(_n))):
        if _os.path.exists(_c): return _c
    return _os.path.join(_base, 'models', _os.path.basename(_n))
def _DER(_n):
    _qui = _os.path.dirname(_os.path.abspath(__file__))
    _d = _os.path.join(_os.path.dirname(_qui), 'models')
    if not _os.path.isdir(_d): _d = _qui
    return _os.path.join(_d, _os.path.basename(_n))
def _OUT(_n):
    _qui = _os.path.dirname(_os.path.abspath(__file__))
    _d = _os.path.join(_os.path.dirname(_qui), 'outputs')
    if not _os.path.isdir(_d):
        try: _os.makedirs(_d)
        except Exception: _d = _qui
    return _os.path.join(_d, _os.path.basename(_n))
import json
import re
import sys
import math

VERSIONE = '1.2'
XML = _MP('tx34_v1.xml')
JSN = _MP('joint_limits_v2.json')

ALIAS = {'waist_yaw': 'waist_rotation', 'toe_pitch': 'toes'}
def nome_canonico(n):
    for a, b in ALIAS.items():
        n = n.replace(a, b)
    n = n.lower().replace('_l_', '_L_').replace('_r_', '_R_')
    parti = set(re.split(r'[_]', n.lower()))
    lato = 'l' if ('l' in parti or n.lower().startswith('l_') or '_l' in n.lower()) else ''
    lato = 'r' if (n.lower().startswith('r_') or '_r_' in n.lower() or n.lower().endswith('_r')) else lato
    resto = sorted(p for p in parti if p not in ('l', 'r'))
    return lato + ':' + '-'.join(resto)

def main():
    test = '--test' in sys.argv
    print('confronta_range VERSIONE ' + VERSIONE)
    jl = json.load(open(JSN))
    lim = jl.get('limits', jl)
    js = {}
    for nome, v in lim.items():
        if not isinstance(v, dict):
            continue
        lo = hi = None
        for k in ('range_sw_deg', 'sw_deg', 'range_sw', 'sw'):
            if k in v and isinstance(v[k], (list, tuple)) and len(v[k]) == 2:
                lo, hi = v[k]
        if lo is None:
            for kmin, kmax in (('sw_min_deg', 'sw_max_deg'), ('min_sw', 'max_sw'), ('min_deg', 'max_deg')):
                if kmin in v and kmax in v:
                    lo, hi = v[kmin], v[kmax]
        if lo is not None:
            c = nome_canonico(nome)
            js[c] = (float(lo), float(hi), nome)
            if c.startswith(':'):  # voce simmetrica: anche sui due lati
                js['l' + c] = (float(lo), float(hi), nome)
                js['r' + c] = (float(lo), float(hi), nome)
    xml = open(XML).read()
    xr = {}
    for m in re.finditer(r'<joint name="([^"]+)"[^/]*?range="([\-0-9\. ]+)"', xml):
        lo, hi = (float(x) for x in m.group(2).split())
        xr[nome_canonico(m.group(1))] = (math.degrees(lo), math.degrees(hi), m.group(1))
    print('%-26s | %-16s | %-16s | delta' % ('giunto', 'JSON sw [deg]', 'XML [deg]'))
    print('-' * 78)
    disc = 0
    comuni = sorted(set(js) & set(xr))
    for c in comuni:
        jlo, jhi, jn = js[c]
        xlo, xhi, xn = xr[c]
        d = max(abs(jlo - xlo), abs(jhi - xhi))
        flag = '  <-- DISCREPANZA' if d > 2.0 else ''
        if d > 2.0:
            disc += 1
        print('%-26s | %7.1f %7.1f | %7.1f %7.1f | %5.1f%s' % (jn + '/' + xn, jlo, jhi, xlo, xhi, d, flag))
    solo_j = sorted(set(js) - set(xr))
    solo_x = sorted(set(xr) - set(js))
    if solo_j:
        print('solo nel JSON (%d): %s' % (len(solo_j), ', '.join(js[c][2] for c in solo_j)))
    if solo_x:
        print('solo nell XML (%d): %s' % (len(solo_x), ', '.join(xr[c][2] for c in solo_x)))
    print('CONFRONTATI: %d | DISCREPANZE >2 gradi: %d' % (len(comuni), disc))
    if test:
        ok = len(comuni) >= 8
        print('TEST ' + ('PASS' if ok else 'FAIL'))
        sys.exit(0 if ok else 1)

if __name__ == '__main__':
    main()
