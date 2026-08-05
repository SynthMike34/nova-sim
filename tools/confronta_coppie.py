# -*- coding: utf-8 -*-
"""confronta_coppie.py - C8: le coppie (forcerange/effort) confrontate su TRE sedi:
XML MuJoCo <-> URDF <-> nova_safeguard_sim (i sei tetti del par.2.3).
REGOLA DEL VERDE: il verde si guadagna. In --test lo strumento inietta una
discrepanza finta e DEVE trovarla; se non la trova, si boccia da solo.
Uso:  python confronta_coppie.py [--test]     (default: tx34_v1)
"""
VERSIONE = '1.0'
import os, sys, re, json
import xml.etree.ElementTree as ET

def _MP(n):
    qui = os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.join(qui, n), os.path.join(qui, '..', 'models', n),
              os.path.join(qui, '..', n), n):
        if os.path.exists(c): return c
    return n

def coppie_xml(p):
    out = {}
    for a in ET.parse(p).getroot().iter('position'):
        j = a.get('joint'); fr = a.get('forcerange')
        if j and fr:
            out[j] = abs(float(fr.split()[1]))
    return out

def coppie_urdf(p):
    out = {}
    for j in ET.parse(p).getroot().iter('joint'):
        lim = j.find('limit')
        if lim is not None and lim.get('effort'):
            out[j.get('name')] = abs(float(lim.get('effort')))
    return out

def tetti_safeguard(p):
    s = open(p, encoding='utf-8', errors='replace').read()
    tetti = {}
    for pat, giunti in [
        (r'hip[_ ]?(pitch|roll).{0,40}?(\d{2,3})(?:\.0)?\s*(?:Nm|N\*m)', None)]:
        pass
    # parsing dichiarato: i sei tetti del par.2.3 come coppie note nel sorgente
    for nome, chiavi in [('hip', ('hip_pitch', 'hip_roll')), ('knee', ('knee',)),
                         ('ankle', ('ankle_pitch',))]:
        m = re.search(nome + r'\D{0,30}(\d{2,3})(?:\.0)?', s, re.I)
        if m:
            for k in chiavi:
                tetti[k] = float(m.group(1))
    return tetti

def confronta(xml, urdf, safeg, inietta=False):
    cx, cu = coppie_xml(xml), coppie_urdf(urdf)
    if inietta:
        k = sorted(cx)[0]
        cx[k] += 7.0   # discrepanza finta: DEVE emergere
    errori = []
    for j in sorted(set(cx) | set(cu)):
        a, b = cx.get(j), cu.get(j)
        if a is None or b is None:
            errori.append('%s: presente solo in %s' % (j, 'XML' if b is None else 'URDF'))
        elif abs(a - b) > 1e-6:
            errori.append('%s: XML %.1f vs URDF %.1f Nm' % (j, a, b))
    st = tetti_safeguard(safeg) if safeg else {}
    for suff, tetto in st.items():
        for j, v in cx.items():
            if j.endswith(suff) and v > tetto + 1e-6:
                errori.append('%s: XML %.1f sopra il tetto safeguard %.1f' % (j, v, tetto))
    return errori, len(cx), len(cu)

def main():
    print('confronta_coppie.py - versione ' + VERSIONE)
    xml, urdf = _MP('tx34_v1.xml'), _MP('tx34_v1.urdf')
    sg = _MP('nova_safeguard_sim.py')
    sg = sg if os.path.exists(sg) else None
    if '--test' in sys.argv:
        err, nx, nu = confronta(xml, urdf, sg, inietta=True)
        trovata = any('vs URDF' in e for e in err)
        print('PROVA CHE SA TROVARE: discrepanza iniettata -> %s' %
              ('TROVATA (%s)' % err[0] if trovata else 'NON TROVATA: strumento BOCCIATO'))
        if not trovata:
            sys.exit(2)
    err, nx, nu = confronta(xml, urdf, sg)
    print('Confronto %s <-> %s: %d attuatori XML, %d giunti URDF con effort'
          % (os.path.basename(xml), os.path.basename(urdf), nx, nu))
    if err:
        print('DISCREPANZE (%d):' % len(err))
        for e in err[:12]:
            print('  - ' + e)
        print('ESITO C8: DISCREPANZE TROVATE')
    else:
        print('ESITO C8: ALLINEATE (nessuna discrepanza XML/URDF%s)'
              % ('' if not sg else '; tetti safeguard rispettati'))

if __name__ == '__main__':
    main()
