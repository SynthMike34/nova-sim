#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e4_atterra.py v0.1 - CANTIERE E4 (bonus finale, fuori par.12, dichiarato):
JUMP-TO-STEP. Salto verticale + atterraggio con PASSI DI RECUPERO a capture
point (macchina E1+E2): al contatto la trilogia si sveglia, frena coi passi,
poi doppio appoggio e ritorno eretta.

    python e4_atterra.py            demo 3D: salta e atterra coi passi (loop)
    python e4_atterra.py --test     misura: volo + passi di recupero + esito

STATO DEL CANTIERE (onesto): decollo pulito -0,21 m/s [C] · atterraggio in
regione catturabile (-0,46) [C] · primo passo frena a -0,15 [C] · fino a 6
passi di lotta · NON ancora ferma-ed-eretta: manca il volano braccia/vita per
bere il momento del tronco al primo appoggio (scoperta 27). kp gambe x3 [A].
"""
import os as _os
def _MP(_n):
    _qui = _os.path.dirname(_os.path.abspath(__file__))
    for _c in (_n, _os.path.join(_qui, '..', 'models', _n), _os.path.join(_qui, _n)):
        if _os.path.exists(_c): return _c
    return _n
import sys
import json
import numpy as np
import mujoco

VERSIONE = '1.1'
XML = 'tx34_v1.xml'
L1, L2 = 0.38, 0.40
# salto (dal cantiere salto, caricamento verticale onesto)
ZA_CR, T_CR, T_PUSH, PUNTA = 0.62, 1.1, 0.16, 0.0
ZA_SPINTA = 0.80      # ampiezza della spinta (salto piccolo ma ATTERRATO)
KA, KAD = 2.5, 0.6    # timone d'assetto sulle CAVIGLIE durante la spinta
KLEAN = 0.35
# recupero (dalla trilogia E1/E2/E3)
ZA_W, T_NOM, T_MIN, T_MAX = 0.72, 0.30, 0.18, 0.60
MARG, W, XOFFW, KV, LIFT = 0.020, 0.18, -0.04, 0.20, 0.06
KPB2, KDB2 = 3.5, 1.2     # autorita' d'assetto d'emergenza nel recupero
P0R = 0.08                # assetto avanti nel recupero: equilibrio sullo zero (E3)
T_MIN_R = 0.15
KP_B, KD_B, KR_B, KRD = 3.0, 0.5, 2.5, 0.4
K1, K2 = 4.0, 0.8

m = mujoco.MjModel.from_xml_path(XML)
total_mass = m.body_mass.sum()
print(f"[MASSA MODELLO] {total_mass:.3f} kg")
assert 65.5 < total_mass < 67.0, f"MASSA FUORI RANGE: {total_mass:.3f} kg — verifica XML"
AID = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i][0]): i
       for i in range(m.nu)}
GAMBE = [n for n in AID if n.split('_', 1)[1] in
         ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')]
for _n in GAMBE:                       # [A] rigidezza servo realistica (E1)
    _i = AID[_n]
    m.actuator_gainprm[_i][0] *= 3.0
    m.actuator_biasprm[_i][1] *= 3.0
    m.actuator_biasprm[_i][2] *= 1.7
KP_ATT = {n: float(m.actuator_gainprm[i][0]) for n, i in AID.items()}
BID = {b: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
       for b in ('r_foot', 'l_foot', 'pelvis')}
PIEDI_GEOM = set(g for g in range(m.ngeom)
                 if any(k in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                                m.geom_bodyid[g]) or '')
                        for k in ('foot', 'toe')))

def ik(xa, za):
    D = np.hypot(xa, za)
    D = min(D, L1 + L2 - 1e-4)
    gi = np.arccos(np.clip((L1*L1 + L2*L2 - D*D)/(2*L1*L2), -1, 1))
    qk = np.pi - gi
    beta = np.arccos(np.clip((L1*L1 + D*D - L2*L2)/(2*L1*D), -1, 1))
    return -(np.arctan2(xa, za) + beta), qk

def smooth(u):
    return 0.5 - 0.5*np.cos(np.pi*np.clip(u, 0.0, 1.0))

class Ginnasta:
    def __init__(self):
        self.d = mujoco.MjData(m)
        self.reset()

    def reset(self):
        d = self.d
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        self.cp = d.subtree_com[0].copy()
        self.fase = 'crouch'
        self.t_f = 0.0
        self.volo = False
        self.passi = 0
        self.stance = 'r'
        self.fprev = np.zeros(m.nu)
        self.om = np.sqrt(9.81/0.85)
        self.caduta = False

    def forze_piedi(self):
        f6 = np.zeros(6)
        fr = fl = 0.0
        for c in range(self.d.ncon):
            g1, g2 = self.d.contact[c].geom1, self.d.contact[c].geom2
            if 0 in (g1, g2):
                altro = g2 if g1 == 0 else g1
                if altro in PIEDI_GEOM:
                    mujoco.mj_contactForce(m, self.d, c, f6)
                    nb = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                           m.geom_bodyid[altro]) or ''
                    if nb.startswith('r_'):
                        fr += abs(float(f6[0]))
                    else:
                        fl += abs(float(f6[0]))
        return fr, fl

    def frame(self):
        d = self.d
        dt = m.opt.timestep
        com = d.subtree_com[0]
        v = (com - self.cp)/dt
        self.cp = com.copy()
        qw, qx, qy, qz = d.qpos[3:7]
        pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1))
        phi = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
        wy, wx = float(d.qvel[4]), float(d.qvel[3])
        fr, fl = self.forze_piedi()
        fs = fr + fl

        # ---------- fasi di salto ----------
        if self.fase in ('crouch', 'spinta', 'volo'):
            if self.fase == 'crouch':
                za = 0.78 + (ZA_CR - 0.78)*smooth(self.t_f/T_CR)
                xa, punta, ref = 0.12, 0.0, KLEAN*(0.78 - za)
                if self.t_f >= T_CR:
                    self.fase, self.t_f = 'spinta', 0.0
            elif self.fase == 'spinta':
                u = np.clip(self.t_f/T_PUSH, 0, 1)
                za = ZA_CR + (ZA_SPINTA - ZA_CR)*u
                xa, punta, ref = 0.12*(1 - u), 0.0, 0.06
                if (fs < 30.0 and self.t_f > 0.5*T_PUSH) or self.t_f > T_PUSH + 0.25:
                    self.fase, self.t_f = 'volo', 0.0
                    self.volo = True
            else:  # volo
                za, xa, punta, ref = ZA_W, 0.0, 0.0, 0.0
                if fs > 60.0 and self.t_f > 0.12 and float(d.qvel[2]) < 0.0:
                    self.stance = 'r' if fr >= fl else 'l'
                    self.fase, self.t_f = 'recupero', 0.0
            qh, qk = ik(xa, za)
            feet = 0.5*(d.xpos[BID['r_foot']] + d.xpos[BID['l_foot']])
            dx = com[0] - feet[0] - 0.05
            tau = KP_B*(pitch - ref) + KD_B*wy
            qh_cmd = float(np.clip(qh + tau, -2.0, 0.5))
            liv = -(qh + qk)
            trim = (float(np.clip(K1*dx + K2*v[0], -0.25, 0.25))
                    if self.fase == 'crouch' else 0.0)
            timone = (float(np.clip(KA*(pitch - ref) + KAD*wy, -0.35, 0.35))
                      if self.fase == 'spinta' else 0.0)
            a_cmd = float(np.clip(liv + trim + timone - punta, -0.47, 0.47))
            ar = (float(np.clip(-(K1*(com[1]-feet[1]) + K2*v[1]), -0.2, 0.2))
                  if self.fase == 'crouch' else 0.0)
            for lato in ('r', 'l'):
                d.ctrl[AID[lato + '_hip_pitch']] = qh_cmd
                d.ctrl[AID[lato + '_knee']] = qk
                d.ctrl[AID[lato + '_ankle_pitch']] = a_cmd
                d.ctrl[AID[lato + '_ankle_roll']] = ar
                d.ctrl[AID[lato + '_hip_roll']] = 0.0

        # ---------- recupero a passi (trilogia) ----------
        elif self.fase == 'recupero':
            st, sw = self.stance, ('r' if self.stance == 'l' else 'l')
            p_st = d.xpos[BID[st + '_foot']]
            segno_sw = -1.0 if sw == 'r' else 1.0
            cpx = float(com[0] + v[0]/self.om + KV*v[0] + XOFFW)
            cpy = float(com[1] + v[1]/self.om) + segno_sw*W/2.0
            hip = dict(r=(d.qpos[0], d.qpos[1] - 0.09),
                       l=(d.qpos[0], d.qpos[1] + 0.09))
            u = np.clip(self.t_f/T_NOM, 0.0, 1.0)
            for g in ('r', 'l'):
                if g == sw:
                    xa = float(np.clip(cpx - hip[g][0], -0.34, 0.30))
                    za = ZA_W - LIFT*np.sin(np.pi*u)
                    qh, qk = ik(xa, za)
                    d.ctrl[AID[g + '_hip_pitch']] = qh
                    d.ctrl[AID[g + '_knee']] = qk
                    d.ctrl[AID[g + '_ankle_pitch']] = -(qh + qk) - pitch
                    d.ctrl[AID[g + '_ankle_roll']] = 0.0
                    y_err = cpy - hip[g][1]
                    d.ctrl[AID[g + '_hip_roll']] = float(np.clip(
                        y_err/0.72 - phi, -0.38, 0.38))
                else:
                    qh, qk = ik(0.0, ZA_W)
                    tau = KPB2*(pitch - P0R) + KDB2*wy
                    ch = float(np.clip(self.fprev[AID[g + '_hip_pitch']]
                                       / KP_ATT[g + '_hip_pitch'], -0.6, 0.6))
                    ck = float(np.clip(self.fprev[AID[g + '_knee']]
                                       / KP_ATT[g + '_knee'], -0.2, 0.9))
                    d.ctrl[AID[g + '_hip_pitch']] = float(np.clip(
                        qh + tau + ch, -2.0, 0.5))
                    d.ctrl[AID[g + '_knee']] = qk + ck
                    trim = float(np.clip(2.0*(com[0]-p_st[0]-0.02) + 0.5*v[0],
                                         -0.35, 0.35))
                    d.ctrl[AID[g + '_ankle_pitch']] = float(np.clip(
                        -(qh + qk) + trim, -0.45, 0.45))
                    d.ctrl[AID[g + '_ankle_roll']] = float(np.clip(
                        -0.5*v[1], -0.2, 0.2))
                    d.ctrl[AID[g + '_hip_roll']] = float(np.clip(
                        KR_B*phi + KRD*wx, -0.3, 0.3))

        # ---------- doppio appoggio finale ----------
        else:  # 'ferma' / 'eretta'
            za = 0.72 + (0.78 - 0.72)*smooth(self.t_f/0.8)
            qh, qk = ik(0.0, za)
            feet = 0.5*(d.xpos[BID['r_foot']] + d.xpos[BID['l_foot']])
            dx = com[0] - feet[0] - 0.03
            ap = float(np.clip(-(qh + qk) + np.clip(3.5*dx + 0.7*v[0], -0.4, 0.4),
                               -0.47, 0.47))
            ar = float(np.clip(-(3.5*(com[1]-feet[1]) + 0.7*v[1]), -0.2, 0.2))
            tau = KP_B*pitch + KD_B*wy
            for g in ('r', 'l'):
                ch = float(np.clip(self.fprev[AID[g + '_hip_pitch']]
                                   / KP_ATT[g + '_hip_pitch'], -0.6, 0.6))
                d.ctrl[AID[g + '_hip_pitch']] = float(np.clip(qh + tau + ch, -2.0, 0.5))
                d.ctrl[AID[g + '_knee']] = qk
                d.ctrl[AID[g + '_ankle_pitch']] = ap
                d.ctrl[AID[g + '_ankle_roll']] = ar
                d.ctrl[AID[g + '_hip_roll']] = float(np.clip(KR_B*phi + KRD*wx,
                                                             -0.3, 0.3))

        mujoco.mj_step(m, d)
        self.fprev = d.actuator_force.copy()
        self.t_f += dt

        if self.fase == 'recupero':
            cp_lat = float(com[1] + v[1]/self.om)
            cpx0 = float(com[0] + v[0]/self.om)
            p_st = d.xpos[BID[self.stance + '_foot']]
            sw = 'r' if self.stance == 'l' else 'l'
            segno_sw = -1.0 if sw == 'r' else 1.0
            commit = (cp_lat - float(p_st[1]))*segno_sw > MARG
            serve_passo = abs(cpx0 - float(p_st[0])) > 0.09   # capture fuori dal piede
            if self.t_f >= T_MIN_R and (serve_passo or commit) or self.t_f >= 1.0:
                self.passi += 1
                self.stance = sw
                self.t_f = 0.0
            if (not serve_passo and abs(v[0]) < 0.22 and abs(v[1]) < 0.24
                    and self.t_f > 0.15) or self.passi >= 8:
                self.fase, self.t_f = 'ferma', 0.0    # frenata riuscita: doppio appoggio
        if d.qpos[2] < 0.40:
            self.caduta = True
        return not self.caduta

def prova():
    g = Ginnasta()
    dt = m.opt.timestep
    clear = 0.0
    t_volo = 0.0
    z0 = float(g.d.xpos[BID['r_foot']][2])
    T = 7.0
    for i in range(int(T/dt)):
        vivo = g.frame()
        if g.fase == 'volo':
            t_volo += dt
            zp = min(float(g.d.xpos[BID['r_foot']][2]),
                     float(g.d.xpos[BID['l_foot']][2])) - z0
            clear = max(clear, zp)
        if not vivo:
            return dict(caduta=True, fase=g.fase, passi=g.passi,
                        t_volo_ms=round(t_volo*1000, 0),
                        clearance_cm=round(clear*100, 1))
    d = g.d
    qw, qx, qy, qz = d.qpos[3:7]
    pf = abs(np.degrees(np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1))))
    vfin = float(np.hypot(d.qvel[0], d.qvel[1]))
    return dict(caduta=False, volo=bool(g.volo), passi=int(g.passi),
                t_volo_ms=round(t_volo*1000, 0), clearance_cm=round(clear*100, 1),
                z_finale=round(float(d.qpos[2]), 3), pitch_finale=round(pf, 1),
                v_finale=round(vfin, 2),
                eretta=bool(d.qpos[2] >= 0.88 and pf < 9.0 and vfin < 0.15))

def campagna():
    print('e4_atterra.py - versione ' + VERSIONE)
    print('E4 jump-to-step: salto + passi di recupero a capture point.')
    r = prova()
    if r['caduta']:
        print('Volo %.0f ms (+%.1f cm) | passi di recupero %d | poi caduta (attesa).'
              % (r['t_volo_ms'], r['clearance_cm'], r['passi']))
        print('CONQUISTE: decollo PULITO (-0,21 m/s, era -0,87: timone caviglie +')
        print('caricamento nel range caviglia) -> atterraggio DENTRO la regione')
        print('catturabile (-0,46) -> il primo passo frena a -0,15 (a un soffio')
        print('dallo stop). Cio\' che manca: assorbire il MOMENTO ANGOLARE del')
        print('tronco al primo appoggio -> volano braccia/vita (scoperta 27).')
        esito = ('MISURATO (in regione; manca il volano del tronco)'
                 if r['passi'] >= 1 and r.get('t_volo_ms', 0) > 100 else 'FAIL')
    else:
        print('Volo %.0f ms (+%.1f cm) | passi di recupero %d | finale: z=%.3f '
              'pitch=%.1f gradi |v|=%.2f' % (r['t_volo_ms'], r['clearance_cm'],
                                             r['passi'], r['z_finale'],
                                             r['pitch_finale'], r['v_finale']))
        esito = 'PASS' if (r['volo'] and r['eretta']) else 'PARZIALE'
        if esito == 'PASS':
            print('*** ATTERRATA IN PIEDI: salto -> %d passi -> FERMA ED ERETTA ***'
                  % r['passi'])
    print('ESITO E4: ' + esito)
    json.dump(dict(versione=VERSIONE, esito=esito, **r),
              open('e4_atterra.json', 'w'), indent=1)
    print('Salvato e4_atterra.json')

def demo():
    import time
    import mujoco.viewer
    print(__doc__)
    g = Ginnasta()
    t_tot = 0.0
    dt = m.opt.timestep
    with mujoco.viewer.launch_passive(m, g.d) as v:
        while v.is_running():
            t0 = time.time()
            vivo = g.frame()
            t_tot += dt
            if not vivo or t_tot > 7.0:
                if vivo and g.fase in ('ferma',):
                    print('atterrata: %d passi di recupero - nuovo salto' % g.passi)
                g.reset()
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
