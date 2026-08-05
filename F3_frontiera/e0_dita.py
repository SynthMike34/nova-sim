#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e0_dita.py v0.1 - CANTIERE E0-TOES (proposta di MIKE): le dita si SVEGLIANO.
Le dita collettive (toe_pitch, 0-45 gradi, 6 Nm) esistono nel modello da
sempre ma sono state PASSIVE per tutta la campagna. Questo modulo le attiva
con un INIETTORE universale (intercetta mj_step) SENZA toccare i moduli
calcolati in simulazione, e confronta SENZA vs CON dita su tre banchi (priorita' MIKE):
  A) camminata E3   B) salto (quota + residuo decollo)   C) atterraggio E6.

    python e0_dita.py            demo 3D: il salto CON le dita attive (loop)
    python e0_dita.py --test     i tre confronti con tabella + json + png

TOE_PUSH in spinta/late-stance · TOE_BRAKE in atterraggio. Stop-loss: sessione.
"""
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
import sys
import json
import importlib
import numpy as np
import mujoco

VERSIONE = '0.7'
TOE_PUSH = 0.45        # rad di flessione nella spinta (solo tratto FINALE)
H_PUSH, T_PUSH2 = 0.70, 0.42   # Step 2: alluce profondo (40 gr), dita 2-5 leggere (24)
H_BRAKE, T_BRAKE, T_RITARDO = 0.50, 0.50, 0.025  # brake asimmetrico (toes +25 ms)
XML_TOES = _MP('tx34_v1_toes.xml')
TOE_PUSH_W = 0.28      # push dolce in camminata (finestra stretta)
TOE_BRAKE = 0.55       # rad al contatto in atterraggio
TOE_BASE = 0.03

_orig_step = mujoco.mj_step
_toe_cache = {}

def _toe_ids(m):
    k = id(m)
    if k not in _toe_cache:
        ids = []
        for i in range(m.nu):
            nome = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT,
                                     m.actuator_trnid[i][0]) or ''
            if nome.endswith('toe_pitch'):
                ids.append((nome[0], 'toe', i))
            elif nome.endswith('hallux_pitch'):
                ids.append((nome[0], 'hallux', i))
            elif nome.endswith('toes_pitch'):
                ids.append((nome[0], 'toes', i))
        _toe_cache[k] = ids
    return _toe_cache[k]

class Iniettore:
    """Con-testo: dentro il blocco, ogni mj_step riceve prima la legge dita."""
    def __init__(self, legge):
        self.legge = legge
    def __enter__(self):
        def passo(m, d):
            for lato, gruppo, i in _toe_ids(m):
                d.ctrl[i] = float(np.clip(self.legge(lato, gruppo), 0.0, 0.873))
            _orig_step(m, d)
        mujoco.mj_step = passo
        return self
    def __exit__(self, *a):
        mujoco.mj_step = _orig_step
        return False

def smooth(u):
    return 0.5 - 0.5*np.cos(np.pi*np.clip(u, 0.0, 1.0))

# ---------------- A) CAMMINATA E3 ----------------
def prova_camminata(dita):
    import e3_accoppiato as E3
    importlib.reload(E3)
    c = E3.Camminatrice()
    dt = E3.m.opt.timestep
    def legge(lato, gruppo='toe'):
        if lato != getattr(c, 'stance', 'r'):
            return 0.0                       # piede di volo: dita neutre
        u = (c.t_f - 0.13)/0.05              # SOLO l'ultimo soffio di stance
        return TOE_BASE + (TOE_PUSH_W - TOE_BASE)*smooth(u)
    ctx = Iniettore(legge) if dita else _Nulla()
    x0 = float(c.d.qpos[0])
    with ctx:
        for i in range(int(30.0/dt)):
            if not c.frame():
                return dict(passi=int(c.passi), x=round(float(c.d.qpos[0])-x0, 2),
                            caduta_s=round(i*dt, 1))
    return dict(passi=int(c.passi), x=round(float(c.d.qpos[0])-x0, 2), caduta_s=None)

# ---------------- B) SALTO ----------------
def prova_salto(dita):
    import salto as S
    importlib.reload(S)
    s = S.Saltatrice()
    dt = s.m.opt.timestep
    def legge(lato, gruppo='toe'):
        if s.fase == 'spinta':
            u = (s.t_f - 0.6*S.T_PUSH)/(0.4*S.T_PUSH)   # RITARDATO: ultimo 40%
            return TOE_BASE + (TOE_PUSH - TOE_BASE)*smooth(u)
        if s.fase in ('atterra', 'risale'):
            return TOE_BRAKE
        return TOE_BASE
    ctx = Iniettore(legge) if dita else _Nulla()
    z0 = float(s.d.xpos[s.fid[0]][2])
    clear, t_volo, vx_dec = 0.0, 0.0, None
    with ctx:
        for i in range(int(4.0/dt)):
            fase_pre = s.fase
            vivo = s.frame()
            if s.fase == 'volo' and fase_pre == 'spinta':
                vx_dec = round(float(s.d.qvel[0]), 2)
            if s.fase == 'volo':
                t_volo += dt
                zp = min(float(s.d.xpos[s.fid[0]][2]),
                         float(s.d.xpos[s.fid[1]][2])) - z0
                clear = max(clear, zp)
            if not vivo:
                break
    return dict(volo_ms=round(t_volo*1000, 0), quota_cm=round(clear*100, 1),
                vx_decollo=vx_dec)

# ---------------- C) ATTERRAGGIO E6 (il test definitivo) ----------------
def prova_atterraggio(dita):
    import e6_supervisore as E6
    importlib.reload(E6)
    g = E6.Atterratrice()
    dt = E6.m.opt.timestep
    def legge(lato, gruppo='toe'):
        if g.fase == 'spinta':
            u = (g.t_f - 0.6*E6.T_PUSH)/(0.4*E6.T_PUSH)
            return TOE_BASE + (TOE_PUSH - TOE_BASE)*smooth(u)
        if g.fase == 'atterrata':
            if g.modo == 'FERMA':
                d = g.d
                com = d.subtree_com[0]
                feet = 0.5*(d.xpos[E6.BID['r_foot']][0] + d.xpos[E6.BID['l_foot']][0])
                resto = float(com[0] + d.qvel[0]/g.om) - feet
                return float(np.clip(3.0*resto, 0.0, 0.60))   # frena solo se scappa AVANTI
            return TOE_BRAKE
        return TOE_BASE
    ctx = Iniettore(legge) if dita else _Nulla()
    with ctx:
        for i in range(int(10.0/dt)):
            if not g.frame():
                break
    return dict(passi=int(g.passi), assestamenti=int(g.assestamenti),
                fermo_s=round(float(getattr(g, 'fermo_max', g.t_fermo)), 2),
                in_piedi=bool(g.d.qpos[2] > 0.85 and not g.caduta))

BODY_R = """<body name="r_toe" pos="0.16 0 -0.05">
              <joint name="r_toe_pitch" axis="0 1 0" range="0.0349 0.6981"/>
              <geom name="r_toe_geom" type="box" pos="0.03 0 0" size="0.035 0.045 0.018"
                    mass="0.3" contype="1" conaffinity="1"/>
            </body>"""

def _blocco_split(lato):
    segno = 1.0 if lato == 'r' else -1.0     # mediale: dx -> +y, sx -> -y
    yh, yt = segno*0.02925, -segno*0.01575
    return ('<body name="%s_hallux" pos="0.16 0 -0.05">\n'
            '              <joint name="%s_hallux_pitch" axis="0 1 0" range="0.0 0.873"/>\n'
            '              <geom name="%s_hallux_geom" type="box" pos="0.03 %.5f 0" '
            'size="0.035 0.01575 0.018" mass="0.105" contype="1" conaffinity="1"/>\n'
            '            </body>\n'
            '            <body name="%s_toes" pos="0.16 0 -0.05">\n'
            '              <joint name="%s_toes_pitch" axis="0 1 0" range="0.0349 0.6981"/>\n'
            '              <geom name="%s_toes_geom" type="box" pos="0.03 %.5f 0" '
            'size="0.035 0.02925 0.018" mass="0.195" contype="1" conaffinity="1"/>\n'
            '            </body>' % (lato, lato, lato, yh, lato, lato, lato, yt))

def genera_toes_xml():
    """2A: deriva il modello con alluce indipendente e ne verifica la stabilita."""
    s = open(_MP('tx34_v1.xml')).read()
    for lato in ('r', 'l'):
        blocco = BODY_R.replace('r_', lato + '_')
        assert blocco in s, 'blocco toe %s non trovato' % lato
        s = s.replace(blocco, _blocco_split(lato), 1)
        import re as _re
        patt = _re.compile(r'<position (?:name="%s_toe_pitch" )?joint="%s_toe_pitch"'
                           r' kp="30" forcerange="-6 6"/>' % (lato, lato))
        assert len(patt.findall(s)) == 1, 'attuatore toe %s: attese 1 riga' % lato
        s = patt.sub(
            '<position name="%s_hallux_pitch" joint="%s_hallux_pitch" kp="30" forcerange="-4 4"/>\n'
            '    <position name="%s_toes_pitch" joint="%s_toes_pitch" kp="30" forcerange="-3 3"/>'
            % (lato, lato, lato, lato), s, count=1)
    open(_DER(XML_TOES), 'w').write(s)
    m = mujoco.MjModel.from_xml_path(XML_TOES)
    total_mass = m.body_mass.sum()
    print(f"[MASSA MODELLO] {total_mass:.3f} kg")
    assert 65.5 < total_mass < 67.0, f"MASSA FUORI RANGE: {total_mass:.3f} kg - verifica XML"
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for _ in range(int(2.0/m.opt.timestep)):
        _orig_step(m, d)
    z = float(d.qpos[2])
    cerniere = sum(1 for j in range(m.njnt)
                   if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE)
    print('2A %s: massa %.2f kg | cerniere %d (attese 27) | z dopo 2 s: %.3f -> %s'
          % (XML_TOES, float(sum(m.body_mass)), cerniere, z,
             'STABILE' if z > 0.88 and np.isfinite(z) else 'INSTABILE'))
    return z > 0.88

def prova_salto_alluce(h=H_PUSH, t2=T_PUSH2, frac=0.6):
    import salto as S
    importlib.reload(S)
    S.XML = XML_TOES
    s = S.Saltatrice()
    dt = s.m.opt.timestep
    def legge(lato, gruppo):
        if s.fase == 'spinta':
            u = smooth((s.t_f - frac*S.T_PUSH)/((1-frac)*S.T_PUSH))
            alto = h if gruppo == 'hallux' else t2
            return TOE_BASE + (alto - TOE_BASE)*u
        if s.fase in ('atterra', 'risale'):
            return 0.5
        return TOE_BASE
    z0 = float(s.d.xpos[s.fid[0]][2])
    clear, t_volo, vx_dec = 0.0, 0.0, None
    with Iniettore(legge):
        for i in range(int(4.0/dt)):
            fase_pre = s.fase
            vivo = s.frame()
            if s.fase == 'volo' and fase_pre == 'spinta':
                vx_dec = round(float(s.d.qvel[0]), 2)
            if s.fase == 'volo':
                t_volo += dt
                zp = min(float(s.d.xpos[s.fid[0]][2]),
                         float(s.d.xpos[s.fid[1]][2])) - z0
                clear = max(clear, zp)
            if not vivo:
                break
    return dict(volo_ms=round(t_volo*1000, 0), quota_cm=round(clear*100, 1),
                vx_decollo=vx_dec)

def _e6_su_toes():
    src = open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'e6_supervisore.py')).read()
    src = src.replace("XML = _MP('tx34_v1.xml')", "XML = _MP('%s')" % _os.path.basename(XML_TOES))
    _dst = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '_e6_toes.py')
    open(_dst, 'w').write(src)
    sys.path.insert(0, _os.path.dirname(_dst))
    if '_e6_toes' in sys.modules:
        return importlib.reload(sys.modules['_e6_toes'])
    return importlib.import_module('_e6_toes')

def prova_atterraggio_alluce(asimmetrico=True):
    E6 = _e6_su_toes()
    g = E6.Atterratrice()
    dt = E6.m.opt.timestep
    t_att = [None]
    def legge(lato, gruppo):
        if g.fase == 'spinta':
            u = smooth((g.t_f - 0.6*E6.T_PUSH)/(0.4*E6.T_PUSH))
            alto = H_PUSH if gruppo == 'hallux' else T_PUSH2
            return TOE_BASE + (alto - TOE_BASE)*u
        if g.fase == 'atterrata':
            if t_att[0] is None:
                t_att[0] = 0.0
            if g.modo == 'FERMA':
                d = g.d
                com = d.subtree_com[0]
                feet = 0.5*(d.xpos[E6.BID['r_foot']][0] + d.xpos[E6.BID['l_foot']][0])
                resto = float(com[0] + d.qvel[0]/g.om) - feet
                return float(np.clip(3.0*resto, 0.0, 0.60))
            if not asimmetrico:
                return H_BRAKE
            if gruppo == 'hallux':
                return H_BRAKE
            return T_BRAKE if t_att[0] >= T_RITARDO else TOE_BASE
        t_att[0] = None
        return TOE_BASE
    def passo_orologio(m, d):
        if t_att[0] is not None:
            t_att[0] += m.opt.timestep
    with Iniettore(legge):
        vecchio = mujoco.mj_step
        def con_orologio(m, d):
            vecchio(m, d)
            passo_orologio(m, d)
        mujoco.mj_step = con_orologio
        try:
            for i in range(int(10.0/dt)):
                if not g.frame():
                    break
        finally:
            mujoco.mj_step = vecchio
    return dict(passi=int(g.passi), assestamenti=int(g.assestamenti),
                fermo_s=round(float(getattr(g, 'fermo_max', g.t_fermo)), 2),
                in_piedi=bool(g.d.qpos[2] > 0.85 and not g.caduta))

class _Nulla:
    def __enter__(self): return self
    def __exit__(self, *a): return False

def campagna():
    print('e0_dita.py - versione ' + VERSIONE)
    print('E0-TOES: le dita si svegliano (TOE_PUSH %.2f rad ~%d gradi, 6 Nm).'
          % (TOE_PUSH, round(np.degrees(TOE_PUSH))))
    ris = {}
    ok2a = genera_toes_xml()
    for nome, fn in (('camminata_E3', prova_camminata),
                     ('salto', prova_salto),
                     ('atterraggio_E6', prova_atterraggio)):
        ris[nome] = dict(senza=fn(False), con=fn(True))
        print('--- %s ---' % nome)
        print('  SENZA dita:', ris[nome]['senza'])
        print('  CON   dita:', ris[nome]['con'])
    if ok2a:
        ris['salto']['alluce'] = prova_salto_alluce()
        print('  ALLUCE ind.:', ris['salto']['alluce'])
        ris['atterraggio_E6']['alluce'] = prova_atterraggio_alluce()
        print('  ALLUCE ind. (att.):', ris['atterraggio_E6']['alluce'])
    a, b = ris['camminata_E3']['senza'], ris['camminata_E3']['con']
    print('CAMMINATA: passi %d -> %d | distanza %+.2f -> %+.2f m'
          % (a['passi'], b['passi'], a['x'], b['x']))
    a, b = ris['salto']['senza'], ris['salto']['con']
    print('SALTO: volo %.0f -> %.0f ms | quota %.1f -> %.1f cm | vx decollo %s -> %s'
          % (a['volo_ms'], b['volo_ms'], a['quota_cm'], b['quota_cm'],
             a['vx_decollo'], b['vx_decollo']))
    a, b = ris['atterraggio_E6']['senza'], ris['atterraggio_E6']['con']
    print('ATTERRAGGIO: fermo %.2f -> %.2f s | in piedi a fine: %s -> %s'
          % (a['fermo_s'], b['fermo_s'], a['in_piedi'], b['in_piedi']))
    json.dump(dict(versione=VERSIONE, toe_push=TOE_PUSH, risultati=ris),
              open(_OUT('e0_dita.json'), 'w'), indent=1)
    print('Salvato e0_dita.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.5))
    coppie = [('camminata_E3', 'passi', 'E3 steps - passive vs active toes'),
              ('salto', 'quota_cm', 'jump height [cm]'),
              ('atterraggio_E6', 'fermo_s', 'time to upright rest [s] - claim withdrawn')]
    for ax, (k, campo, tit) in zip(axs, coppie):
        eti = ['passive\ntoes', 'ACTIVE\ntoes']
        vals = [ris[k]['senza'][campo], ris[k]['con'][campo]]
        col = ['tab:gray', 'tab:green']
        if 'alluce' in ris[k]:
            eti.append('HALLUX\nindep.')
            vals.append(ris[k]['alluce'][campo])
            col.append('tab:purple')
        ax.bar(eti, vals, color=col, width=0.55)
        for xx, vv in enumerate(vals):
            ax.text(xx, vv, str(vv), ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
        ax.set_title(tit, fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
    fig.suptitle('E0 toes - canon @66.23 kg, computed in simulation', fontsize=13)
    fig.tight_layout()
    fig.savefig(_OUT('e0_dita.png'), dpi=150)
    print('Salvato e0_dita.png')

def demo():
    import salto as S
    importlib.reload(S)
    s_rif = [None]
    def legge(lato, gruppo='toe'):
        s = s_rif[0]
        if s is None:
            return TOE_BASE
        if s.fase == 'spinta':
            u = (s.t_f - 0.6*S.T_PUSH)/(0.4*S.T_PUSH)   # RITARDATO: ultimo 40%
            return TOE_BASE + (TOE_PUSH - TOE_BASE)*smooth(u)
        if s.fase in ('atterra', 'risale'):
            return TOE_BRAKE
        return TOE_BASE
    print(__doc__)
    print('DEMO: il salto CON le dita attive (guarda le punte in spinta!).')
    with Iniettore(legge):
        import time
        import mujoco.viewer
        s = S.Saltatrice()
        s_rif[0] = s
        dt = s.m.opt.timestep
        t_tot = 0.0
        with mujoco.viewer.launch_passive(s.m, s.d) as v:
            while v.is_running():
                t0 = time.time()
                vivo = s.frame()
                t_tot += dt
                if not vivo or t_tot > 6.0:
                    s.reset()
                    t_tot = 0.0
                v.sync()
                resto = dt - (time.time() - t0)
                if resto > 0:
                    time.sleep(resto)

if __name__ == '__main__':
    if '--test' in sys.argv:
        campagna()
    else:
        demo()
