#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tacchi_param.py v1.0 - Tacchi parametrici (TDD par.3.1 "tacchi 10-12 cm" /
par.12-T16 parziale): inviluppo di equilibrio in funzione dell'altezza del tacco.

    python tacchi_param.py            demo 3D a h=10 cm (frecce=spinte, R=reset)
    python tacchi_param.py 8          demo 3D all'altezza scelta (0/5/8/10/12)
    python tacchi_param.py --test     campagna completa: curva + tabella + PNG + JSON

Per h in {0, 5, 8, 10, 12} cm: massima spinta assorbibile avanti/indietro/
laterale [m/s], quota CoM, coppie caviglia al limite. Modelli derivati
tx34_v1_tacchi_hXX.xml rigenerati da tx34_v1.xml (regola piattaforma).
Leggi di equilibrio: h=0 la certificata piedi nudi (XB 0,05, ar 0,20, KH 0,4);
h>0 la certificata tacchi (XB 0,06, ar 0,16, KH 0) — uniforme, cosi' la curva
isola l'effetto dell'altezza [discontinuita' di legge a h=0 dichiarata].
Soglia minima operativa [A] (modalita' accompagnata): 0,30 av / 0,25 ind / 0,15 lat.
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
import sys
import re
import json
import numpy as np
import mujoco

VERSIONE = '1.1'
XML = _MP('tx34_v1.xml')
ALTEZZE = (0, 5, 8, 10, 12)
SOGLIA = dict(avanti=0.30, indietro=0.25, laterale=0.15)
K1, K2 = 4.0, 0.8

def genera_h(h_cm):
    """Genera tx34_v1_tacchi_hXX.xml; ritorna il percorso (h=0 -> modello base)."""
    if h_cm == 0:
        return XML
    r = h_cm/100.0 - 0.02            # innalzamento suolo->caviglia (h=12 -> +0,10)
    hh = (0.025 + r)/2.0             # semialtezza dello spillo
    s = open(XML).read()
    for L in ('r', 'l'):
        scarpa = (
            '<geom name="' + L + '_scarpa" type="box" pos="0.05 0 -0.03" '
            'size="0.10 0.04 0.015" mass="3.6"/>\n'
            '            <geom name="' + L + ('_tacco" type="box" pos="-0.01 0 %.4f" '
            'size="0.014 0.014 %.4f" mass="0.4" contype="1" conaffinity="1" '
            'rgba="0.85 0.08 0.15 1"/>\n' % (-0.045 - hh, hh)) +
            '            <geom name="' + L + ('_punta" type="box" pos="0.135 0 %.4f" '
            'size="0.045 0.04 0.012" mass="0.3" contype="1" conaffinity="1" '
            'rgba="0.85 0.08 0.15 1"/>' % (-(0.058 + r))))
        s = re.sub(r'<geom name="' + L + r'_sole".*?/>', scarpa, s, flags=re.S)
    s = re.sub(r'(name="[rl]_toe_geom".*?mass="0\.3")\s+contype="1" conaffinity="1"(/>)',
               r'\1\2', s, flags=re.S)
    s = s.replace('<body name="pelvis" pos="0 0 0.912">',
                  '<body name="pelvis" pos="0 0 %.3f">' % (0.912 + r))
    s = s.replace('qpos="0 0 0.912 1', 'qpos="0 0 %.3f 1' % (0.912 + r))
    s = s.replace('TX34_NOVA_v1', 'TX34_NOVA_v1_tacchi_h%d' % h_cm)
    percorso = _DER('tx34_v1_tacchi_h%d.xml' % h_cm)
    open(percorso, 'w').write(s)
    return percorso

def legge(h_cm):
    if h_cm == 0:
        return dict(XB=0.05, AR=0.20, KH=0.4)
    return dict(XB=0.06, AR=0.16, KH=0.0)

class Banco:
    def __init__(self, h_cm):
        self.h = h_cm
        self.m = mujoco.MjModel.from_xml_path(genera_h(h_cm))
        total_mass = self.m.body_mass.sum()
        print(f"[MASSA MODELLO] {total_mass:.3f} kg")
        assert 65.5 < total_mass < 67.0, f"MASSA FUORI RANGE: {total_mass:.3f} kg — verifica XML"
        self.aid = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT,
                                      self.m.actuator_trnid[i][0]): i
                    for i in range(self.m.nu)}
        self.fid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, b)
                    for b in ('r_foot', 'l_foot')]
        self.L = legge(h_cm)
        self.z_cad = 0.45 + (h_cm/100.0 - 0.02 if h_cm else 0.0)
        d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, d, 0)
        mujoco.mj_forward(self.m, d)
        self.com_z = float(d.subtree_com[0][2])

    def controlla(self, d, cp):
        dt = self.m.opt.timestep
        com = d.subtree_com[0]
        v = (com - cp)/dt
        feet = 0.5*(d.xpos[self.fid[0]] + d.xpos[self.fid[1]])
        ap = float(np.clip(K1*(com[0]-feet[0]-self.L['XB']) + K2*v[0], -0.45, 0.45))
        ar = float(np.clip(-(K1*(com[1]-feet[1]) + K2*v[1]),
                           -self.L['AR'], self.L['AR']))
        hp = float(np.clip(self.L['KH']*v[0], -0.4, 0.4))
        hr = float(np.clip(self.L['KH']*v[1], -0.3, 0.3))
        for j in ('r_ankle_pitch', 'l_ankle_pitch'):
            d.ctrl[self.aid[j]] = ap
        for j in ('r_ankle_roll', 'l_ankle_roll'):
            d.ctrl[self.aid[j]] = ar
        for j in ('r_hip_pitch', 'l_hip_pitch'):
            d.ctrl[self.aid[j]] = hp
        for j in ('r_hip_roll', 'l_hip_roll'):
            d.ctrl[self.aid[j]] = hr
        return com.copy()

    def prova(self, direzione, vel, registra=False):
        d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, d, 0)
        mujoco.mj_forward(self.m, d)
        dt = self.m.opt.timestep
        cp = d.subtree_com[0].copy()
        picchi = dict(pitch=0.0, roll=0.0)
        for i in range(int(8.0/dt)):
            t = i*dt
            if abs(t - 3.0) < dt/2:
                if direzione == 'avanti':
                    d.qvel[0] += vel
                elif direzione == 'indietro':
                    d.qvel[0] -= vel
                else:
                    d.qvel[1] += vel
            cp = self.controlla(d, cp)
            mujoco.mj_step(self.m, d)
            if t > 3.0:
                picchi['pitch'] = max(picchi['pitch'],
                    abs(float(d.actuator_force[self.aid['r_ankle_pitch']])),
                    abs(float(d.actuator_force[self.aid['l_ankle_pitch']])))
                picchi['roll'] = max(picchi['roll'],
                    abs(float(d.actuator_force[self.aid['r_ankle_roll']])),
                    abs(float(d.actuator_force[self.aid['l_ankle_roll']])))
            if d.qpos[2] < self.z_cad:
                return False, picchi
        return True, picchi

    def inviluppo(self):
        out = dict(h_cm=self.h, com_z_m=round(self.com_z, 3))
        for direzione in ('avanti', 'indietro', 'laterale'):
            vmax = 0.0
            pk = dict(pitch=0.0, roll=0.0)
            v = 0.05
            while v <= 0.85:
                ok, picchi = self.prova(direzione, v)
                if not ok:
                    break
                vmax = v
                pk = picchi
                v = round(v + 0.05, 2)
            out['v_' + direzione] = round(vmax, 2)
            out['Nm_cav_' + direzione] = dict(pitch=round(pk['pitch'], 1),
                                              roll=round(pk['roll'], 1))
        out['pass_soglia'] = bool(all(out['v_' + k] >= SOGLIA[k] for k in SOGLIA))
        return out

def campagna():
    print('tacchi_param.py - versione ' + VERSIONE)
    print('Inviluppo di equilibrio vs altezza tacco. Soglia minima [A]: '
          '0,30 av / 0,25 ind / 0,15 lat (accompagnata).')
    print('%4s | %7s | %7s %8s %9s | %10s %9s | %s' %
          ('h', 'CoM [m]', 'avanti', 'indietro', 'laterale',
           'cav.pitch', 'cav.roll', 'soglia'))
    ris = []
    for h in ALTEZZE:
        b = Banco(h)
        r = b.inviluppo()
        ris.append(r)
        print('%4d | %7.3f | %7.2f %8.2f %9.2f | %10.1f %9.1f | %s' %
              (h, r['com_z_m'], r['v_avanti'], r['v_indietro'], r['v_laterale'],
               max(r['Nm_cav_avanti']['pitch'], r['Nm_cav_indietro']['pitch']),
               max(r['Nm_cav_laterale']['roll'], r['Nm_cav_avanti']['roll']),
               'PASS' if r['pass_soglia'] else 'SOTTO'))
    esito = 'PASS' if all(r['pass_soglia'] for r in ris) else 'PARZIALE'
    print('ESITO tacchi parametrici: %s (curva completa nel PNG/JSON)' % esito)
    json.dump(dict(versione=VERSIONE, esito=esito, soglia_A=SOGLIA, curva=ris),
              open('tacchi_param.json', 'w'), indent=1)
    print('Salvato tacchi_param.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    H = [r['h_cm'] for r in ris]
    fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for k, col in (('avanti', 'tab:blue'), ('indietro', 'tab:green'),
                   ('laterale', 'tab:red')):
        axs[0].plot(H, [r['v_' + k] for r in ris], 'o-', color=col, label=k)
        axs[0].axhline(SOGLIA[k], color=col, ls=':', lw=0.8)
    axs[0].set_ylabel('spinta massima assorbita [m/s]')
    axs[0].set_title('TX-34: inviluppo di equilibrio vs altezza tacco '
                     '(punteggiate = soglie minime [A])')
    axs[0].legend()
    axs[1].plot(H, [max(r['Nm_cav_avanti']['pitch'], r['Nm_cav_indietro']['pitch'])
                    for r in ris], 's-', color='tab:purple', label='caviglia pitch')
    axs[1].plot(H, [max(r['Nm_cav_laterale']['roll'], r['Nm_cav_avanti']['roll'])
                    for r in ris], 'd-', color='tab:orange', label='caviglia roll')
    axs[1].axhline(80, color='r', ls='--', lw=1, label='limite RMD-X8')
    axs[1].set_xlabel('altezza tacco [cm]')
    axs[1].set_ylabel('coppia al limite inviluppo [Nm]')
    axs[1].legend()
    fig.tight_layout()
    fig.savefig('tacchi_param.png', dpi=150)
    print('Salvato tacchi_param.png')

def demo(h):
    import time
    import mujoco.viewer
    print(__doc__)
    b = Banco(h)
    L = legge(h)
    sp = dict(av=0.50, ind=0.45, lat=0.35) if h == 0 else dict(av=0.30, ind=0.25, lat=0.12)
    print('Demo a h=%d cm (CoM %.3f m). Frecce=spinte (%.2f/%.2f/%.2f), R=reset.'
          % (h, b.com_z, sp['av'], sp['ind'], sp['lat']))
    d = mujoco.MjData(b.m)
    mujoco.mj_resetDataKeyframe(b.m, d, 0)
    mujoco.mj_forward(b.m, d)
    stato = dict(cp=d.subtree_com[0].copy(), caduta=None)

    def reset():
        mujoco.mj_resetDataKeyframe(b.m, d, 0)
        mujoco.mj_forward(b.m, d)
        stato['cp'] = d.subtree_com[0].copy()
        stato['caduta'] = None

    def tasto(k):
        if k == 265:
            d.qvel[0] += sp['av']
        elif k == 264:
            d.qvel[0] -= sp['ind']
        elif k in (262, 263):
            d.qvel[1] += sp['lat'] if k == 263 else -sp['lat']
        elif k in (82, 114):
            reset()
    dt = b.m.opt.timestep
    with mujoco.viewer.launch_passive(b.m, d, key_callback=tasto) as v:
        while v.is_running():
            t0 = time.time()
            stato['cp'] = b.controlla(d, stato['cp'])
            mujoco.mj_step(b.m, d)
            if d.qpos[2] < b.z_cad and stato['caduta'] is None:
                stato['caduta'] = time.time()
            if stato['caduta'] and time.time() - stato['caduta'] > 2.0:
                reset()
            v.sync()
            resto = dt - (time.time() - t0)
            if resto > 0:
                time.sleep(resto)

if __name__ == '__main__':
    if '--test' in sys.argv:
        campagna()
    else:
        h = 10
        for a in sys.argv[1:]:
            if a.isdigit() and int(a) in ALTEZZE:
                h = int(a)
        demo(h)
