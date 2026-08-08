# -*- coding: utf-8 -*-
"""
C1.11 — INVILUPPO DI EQUILIBRIO sul piede nuovo
Campagna C — NOVA-SIM — modulo di BUSSOLA

Protocollo IDENTICO a `avvia_tx34.py` v2.4 (quello con cui e' stato misurato il
canone): 4 s di quiete, impulso di velocita' sul giunto libero, 6 s di verifica.
Controllore di stazione e costanti presi tali e quali, e DIVERSI fra nudo e tacchi:
  nudo   : XB 0,05 · AR_CLIP 0,20 · KH 0,4 · Z_CADUTA 0,45
  tacchi : XB 0,06 · AR_CLIP 0,16 · KH 0,0 · Z_CADUTA 0,60
K1, K2 = 4,0 / 0,8 in entrambi.

Il controllo intermedio (originale-qui) e' incluso: senza, non si sa se il canone
riproduce prima di attribuire la differenza al piede.
"""
VERSIONE = '1.0'
import os, sys, json
os.environ.setdefault('MUJOCO_GL', 'disable')
import re
import numpy as np
import mujoco

K1, K2 = 4.0, 0.8
COST = dict(nudo=dict(XB=0.05, AR_CLIP=0.20, KH=0.4, ZC=0.45),
            tacchi=dict(XB=0.06, AR_CLIP=0.16, KH=0.0, ZC=0.60))
DIR = {'avanti': (0, +1), 'indietro': (0, -1), 'lat.destro': (1, -1), 'lat.sinistro': (1, +1)}


def genera_tacchi(sorgente, uscita):
    """Trasformazione di `avvia_tx34._genera_tacchi`, applicata a un sorgente scelto."""
    s = open(sorgente).read()
    scarpa = (
        '<geom name="{L}_scarpa" type="box" pos="0.05 0 -0.03" size="0.10 0.04 0.015" mass="3.6"/>\n'
        '            <geom name="{L}_tacco" type="box" pos="-0.01 0 -0.115" size="0.014 0.014 0.055"\n'
        '                  mass="0.4" contype="1" conaffinity="1" rgba="0.85 0.08 0.15 1"/>\n'
        '            <geom name="{L}_punta" type="box" pos="0.135 0 -0.158" size="0.045 0.04 0.012"\n'
        '                  mass="0.3" contype="1" conaffinity="1" rgba="0.85 0.08 0.15 1"/>')
    for L in ('r', 'l'):
        s = re.sub(r'<geom name="' + L + r'_sole".*?/>', scarpa.format(L=L), s, flags=re.S)
    s = re.sub(r'(name="[rl]_toe_geom".*?mass="0\.3")\s+contype="1" conaffinity="1"(/>)',
               r'\1\2', s, flags=re.S)
    s = s.replace('<body name="pelvis" pos="0 0 0.912">', '<body name="pelvis" pos="0 0 1.012">')
    s = s.replace('qpos="0 0 0.912 1', 'qpos="0 0 1.012 1')
    open(uscita, 'w').write(s)
    return uscita


class Banco:
    def __init__(self, xml, tipo):
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        self.c = COST[tipo]
        self.aid = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT,
                                      self.m.actuator_trnid[i][0]): i for i in range(self.m.nu)}
        self.fid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, b)
                    for b in ('r_foot', 'l_foot')]
        self.gid = {}
        for lato, gg in (('r', ('r_sole', 'r_toe_geom', 'r_scarpa', 'r_tacco', 'r_punta')),
                         ('l', ('l_sole', 'l_toe_geom', 'l_scarpa', 'l_tacco', 'l_punta'))):
            for g in gg:
                j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, g)
                if j >= 0:
                    self.gid[j] = lato
        self.dt = self.m.opt.timestep

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        mujoco.mj_forward(self.m, self.d)
        self.cp = self.d.subtree_com[0].copy()
        self.caduta = None

    def frame(self):
        m, d, c = self.m, self.d, self.c
        com = d.subtree_com[0]
        v = (com - self.cp) / self.dt
        self.cp = com.copy()
        feet = 0.5 * (d.xpos[self.fid[0]] + d.xpos[self.fid[1]])
        ap = float(np.clip(K1 * (com[0] - feet[0] - c['XB']) + K2 * v[0], -0.45, 0.45))
        ar = float(np.clip(-(K1 * (com[1] - feet[1]) + K2 * v[1]), -c['AR_CLIP'], c['AR_CLIP']))
        hp = float(np.clip(c['KH'] * v[0], -0.4, 0.4))
        hr = float(np.clip(c['KH'] * v[1], -0.3, 0.3))
        for j in ('r_ankle_pitch', 'l_ankle_pitch'):
            d.ctrl[self.aid[j]] = ap
        for j in ('r_ankle_roll', 'l_ankle_roll'):
            d.ctrl[self.aid[j]] = ar
        for j in ('r_hip_pitch', 'l_hip_pitch'):
            d.ctrl[self.aid[j]] = hp
        for j in ('r_hip_roll', 'l_hip_roll'):
            d.ctrl[self.aid[j]] = hr
        mujoco.mj_step(m, d)
        if d.qpos[2] < c['ZC'] and self.caduta is None:
            self.caduta = True
        return self.caduta is None

    def cop(self):
        r6 = np.zeros(6)
        tot = mom = 0.0
        for i in range(self.d.ncon):
            g1, g2 = int(self.d.contact.geom1[i]), int(self.d.contact.geom2[i])
            g = g1 if g1 in self.gid else (g2 if g2 in self.gid else None)
            if g is None:
                continue
            mujoco.mj_contactForce(self.m, self.d, i, r6)
            fn = abs(float(r6[0]))
            if fn <= 1e-6:
                continue
            b = self.fid[0] if self.gid[g] == 'r' else self.fid[1]
            tot += fn
            mom += fn * float(self.d.contact.pos[i][0] - self.d.xpos[b][0])
        return mom / tot if tot > 50 else float('nan')


def spinta(b, asse, segno, vel, T_quiete=4.0, T_ver=6.0):
    b.reset()
    n_q = int(T_quiete / b.dt)
    n_t = int((T_quiete + T_ver) / b.dt)
    arp = 0.0
    cop_sp = float('nan')
    esc = 0.0
    for i in range(n_t):
        if i == n_q:
            b.d.qvel[asse] += segno * vel
            cop_sp = b.cop()
        if not b.frame():
            return False, arp, cop_sp, esc, i * b.dt
        if i >= n_q:
            arp = max(arp, max(abs(float(b.d.actuator_force[b.aid['r_hip_roll']])),
                               abs(float(b.d.actuator_force[b.aid['l_hip_roll']]))))
            feet = 0.5 * (b.d.xpos[b.fid[0]] + b.d.xpos[b.fid[1]])
            esc = max(esc, abs(float(b.d.subtree_com[0][asse] - feet[asse])))
    return True, arp, cop_sp, esc, T_quiete + T_ver


def inviluppo(xml, tipo, passo=0.05, vmax=1.20):
    b = Banco(xml, tipo)
    out = {}
    for nome, (asse, segno) in DIR.items():
        v = passo
        best = 0.0
        det = {}
        cad = None
        while v <= vmax + 1e-9:
            ok, arp, cp, esc, t = spinta(b, asse, segno, v)
            if not ok:
                cad = dict(caduta_a=v, t_caduta=t)
                break
            best = v
            det = dict(ar=arp, cop=cp, esc=esc)   # ultimo run RIUSCITO
            v += passo
        out[nome] = dict(vmax=best, **det, **(cad or {}))
    return out
def main():
    print('c19_inviluppo.py - versione ' + VERSIONE)
    qui = os.path.dirname(os.path.abspath(__file__))
    radice = os.path.dirname(qui)
    nudo = os.path.join(radice, 'models', 'tx34_piedevero.xml')
    tac = os.path.join(radice, 'models', 'tx34_v1_tacchi_h10.xml')

    print('\nINVILUPPO DI EQUILIBRIO - piede anatomico 16,5/6,0')
    for etichetta, xml, tipo in (('piede nudo', nudo, 'nudo'),
                                 ('tacchi 12 cm', tac, 'tacchi')):
        if not os.path.exists(xml):
            print('  %-14s modello assente: %s' % (etichetta, xml))
            continue
        r = inviluppo(xml, tipo)
        print('  %-14s avanti %.2f | indietro %.2f | laterale %.2f m/s'
              % (etichetta, r['avanti']['vmax'], r['indietro']['vmax'],
                 min(r['lat.destro']['vmax'], r['lat.sinistro']['vmax'])))
    print('\nESITO C19: MISURATO')


if __name__ == '__main__':
    main()
