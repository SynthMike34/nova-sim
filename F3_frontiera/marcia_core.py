#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e2_timing.py v1.0 - CANTIERE E2 (fuori campagna, cartella F3 dichiarata):
STEP-TIMING ADATTIVO: il passo non dura T_SW fisso — scatta su un
EVENTO DI STATO: si atterra quando il capture laterale supera il piede
d'appoggio verso il lato di swing (il corpo si e' "impegnato" a cadere di la').
Piazzamento: capture point (da E1, errore 2-4 cm). Obiettivo: >= 10 passi.

    python e2_timing.py            demo 3D: marcia e poi spinta avanti (auto)
    python e2_timing.py --test     criterio E1: >= 4 passi senza caduta

Obiettivo dichiarato: battere il muro dei 2 passi. Stop-loss: una sessione.
"""
import sys
import json
import numpy as np
import mujoco

VERSIONE = '2.0'
import os as _os
def _MP(_n):
    _qui = _os.path.dirname(_os.path.abspath(__file__))
    for _c in (_os.path.join(_qui, _n), _os.path.join(_qui, '..', 'models', _n), _n):
        if _os.path.exists(_c): return _c
    return _n
def _OUT(_n):
    _d = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'outputs')
    _os.makedirs(_d, exist_ok=True)
    return _os.path.join(_d, _n)
XML = _MP('tx34_piedevero.xml')
ZA0 = 0.72          # gambe leggermente flesse: margine di allungo
T_NOM = 0.30        # durata nominale (solo per la forma del lift)
T_MIN, T_MAX = 0.18, 0.60
MARG = 0.020        # [m] quanto il capture laterale deve superare il piede
LIFT = 0.06
W = 0.18            # larghezza del passo (distanza tra i piedi)
KCP = 1.0           # guadagno sul capture point
XOFF = -0.04        # centro suola avanti di 4 cm rispetto alla caviglia
KV = 0.10           # termine di Raibert su (v - v_des)
XA_MAX = 0.30
KP_B, KD_B = 3.0, 0.5   # busto pitch (solo feedback, regola 10bis)
KR_B, KRD = 2.5, 0.4    # busto roll
RULLIO = -0.05
SOGLIA_TRASF = 0.20   # [m] |com_y - piede_y| sotto cui il carico e' passato
KY_ST = 0.0           # [FORK] termine di POSIZIONE sulla caviglia d'appoggio (0 = come e2_timing)
FRAZ = 1.0            # [FORK] frazione dell'appoggio in cui il termine e' ACCESO (1,0 = sempre)
MORBIDA = False       # [FORK] False = interruttore secco, True = spegnimento a coseno
PRESS = 0.0           # [FORK] estensione della gamba in volo nell'ultima frazione: atterraggio deciso
U_PRESS = 0.7         # [FORK] da quale frazione di u comincia la pressione
T_STOP = 0.0          # [FORK] istante in cui si smette di comandare il passo (0 = mai)
T_RAMPA = 0.0         # [FORK] secondi di riduzione progressiva del passo prima dello stop (NOVA)
L1, L2 = 0.38, 0.40

m = mujoco.MjModel.from_xml_path(XML)
total_mass = m.body_mass.sum()
print(f"[MASSA MODELLO] {total_mass:.3f} kg")
assert 65.5 < total_mass < 67.0, f"MASSA FUORI RANGE: {total_mass:.3f} kg — verifica XML"
AID = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i][0]): i
       for i in range(m.nu)}
BID = {b: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
       for b in ('r_foot', 'l_foot', 'pelvis')}
GAMBE = [n for n in AID if n.split('_', 1)[1] in
         ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')]
for _n in GAMBE:                       # [A] rigidezza servo realistica: kp x3
    _i = AID[_n]
    m.actuator_gainprm[_i][0] *= 3.0
    m.actuator_biasprm[_i][1] *= 3.0
    m.actuator_biasprm[_i][2] *= 1.7
KP_ATT = {n: float(m.actuator_gainprm[i][0]) for n, i in AID.items()}

def ik(xa, za):
    D = np.hypot(xa, za)
    D = min(D, L1 + L2 - 1e-4)
    gi = np.arccos(np.clip((L1*L1 + L2*L2 - D*D)/(2*L1*L2), -1, 1))
    qk = np.pi - gi
    beta = np.arccos(np.clip((L1*L1 + D*D - L2*L2)/(2*L1*D), -1, 1))
    return -(np.arctan2(xa, za) + beta), qk

class Camminatrice:
    def __init__(self, v_des=0.0):
        self.d = mujoco.MjData(m)
        self.v_des = v_des
        self.reset()

    def reset(self):
        d = self.d
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        self.cp = d.subtree_com[0].copy()
        self.stance = 'l'
        self.fase = 'init'          # init -> seme (dondolio) -> passi
        self.t_f = 0.0
        self.passi = 0
        self.caduta = None
        self.om = np.sqrt(9.81/0.85)
        self.fprev = np.zeros(m.nu)
        self.tsw_log = []
        self.trasf = False
        self.dlat_log = []
        self.zsw_log = []
        self.t_tot = 0.0
        self.t_stop_reale = None

    def frame(self):
        d = self.d
        dt = m.opt.timestep
        self.t_tot += dt
        if T_STOP > 0.0 and self.t_tot >= T_STOP and self.fase == 'passi':
            self.fase, self.t_f = 'stop', 1.0
            self.t_stop_reale = self.t_tot
        com = d.subtree_com[0]
        v = (com - self.cp)/dt
        self.cp = com.copy()
        qw, qx, qy, qz = d.qpos[3:7]
        pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1))
        phi = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
        wy, wx = float(d.qvel[4]), float(d.qvel[3])
        # --- innesco: equilibrio, poi dondolio laterale, poi si parte ---
        if self.fase in ('init', 'seme', 'stop'):
            feet = 0.5*(d.xpos[BID['r_foot']] + d.xpos[BID['l_foot']])
            ap = float(np.clip(3.5*(com[0]-feet[0]-0.03) + 0.7*v[0], -0.4, 0.4))
            ar = float(np.clip(-(3.5*(com[1]-feet[1]) + 0.7*v[1]), -0.2, 0.2))
            if self.fase == 'init':
                za = 0.78 + (ZA0 - 0.78)*(0.5 - 0.5*np.cos(np.pi*min(self.t_f/0.9, 1.0)))
            elif self.fase == 'stop':
                za = ZA0
            else:
                za = ZA0
                ar += 0.10*np.sin(2*np.pi*0.54*self.t_f)      # risonanza laterale
            qh0, qk0 = ik(0.0, za)
            for g in ('r', 'l'):
                d.ctrl[AID[g + '_hip_pitch']] = qh0 + KP_B*pitch + KD_B*wy
                d.ctrl[AID[g + '_knee']] = qk0
                d.ctrl[AID[g + '_ankle_pitch']] = float(np.clip(-(qh0+qk0)+ap, -0.45, 0.45))
                d.ctrl[AID[g + '_ankle_roll']] = ar
                d.ctrl[AID[g + '_hip_roll']] = float(np.clip(KR_B*phi + KRD*wx, -0.3, 0.3))
            mujoco.mj_step(m, d)
            self.t_f += dt
            if self.fase == 'stop':
                pass
            elif self.fase == 'init' and self.t_f >= 1.2:
                self.fase, self.t_f = 'seme', 0.0
            elif self.fase == 'seme' and ((abs(v[1]) > 0.08 and self.t_f > 0.6)
                                          or self.t_f > 3.5):
                self.stance = 'l' if v[1] > 0 else 'r'
                self.fase, self.t_f = 'passi', 0.0
            if d.qpos[2] < 0.42 and self.caduta is None:
                self.caduta = True
            return self.caduta is None

        st, sw = self.stance, ('r' if self.stance == 'l' else 'l')
        p_st = d.xpos[BID[st + '_foot']]
        segno_sw = -1.0 if sw == 'r' else 1.0      # lato del piede di swing
        # capture point nel mondo
        cpx = float(com[0] + KCP*v[0]/self.om + KV*(v[0] - self.v_des) + XOFF)
        cpy = float(com[1] + KCP*v[1]/self.om) + segno_sw*W/2.0
        # anche nel mondo
        hip = dict(r=(d.qpos[0], d.qpos[1] - 0.09), l=(d.qpos[0], d.qpos[1] + 0.09))
        u = np.clip(self.t_f/T_NOM, 0.0, 1.0)
        for gamba in ('r', 'l'):
            if gamba == sw:
                xa = float(np.clip(cpx - hip[gamba][0], -0.25, XA_MAX))
                if T_RAMPA > 0.0 and T_STOP > 0.0:
                    xa *= float(np.clip((T_STOP - self.t_tot)/T_RAMPA, 0.0, 1.0))
                za = ZA0 - LIFT*np.sin(np.pi*u)
                if PRESS != 0.0 and u > U_PRESS:
                    za += PRESS*(u - U_PRESS)/max(1.0 - U_PRESS, 1e-9)
                qh, qk = ik(xa, za)
                d.ctrl[AID[gamba + '_hip_pitch']] = qh
                d.ctrl[AID[gamba + '_knee']] = qk
                d.ctrl[AID[gamba + '_ankle_pitch']] = -(qh + qk) - pitch
                d.ctrl[AID[gamba + '_ankle_roll']] = 0.0
                y_err = cpy - hip[gamba][1]
                d.ctrl[AID[gamba + '_hip_roll']] = float(np.clip(y_err/0.72 - phi, -0.38, 0.38))
            else:
                qh, qk = ik(0.0, ZA0)
                tau = KP_B*pitch + KD_B*wy
                comp_h = float(np.clip(self.fprev[AID[gamba + '_hip_pitch']]
                                       / KP_ATT[gamba + '_hip_pitch'], -0.6, 0.6))
                comp_k = float(np.clip(self.fprev[AID[gamba + '_knee']]
                                       / KP_ATT[gamba + '_knee'], -0.2, 0.9))
                d.ctrl[AID[gamba + '_hip_pitch']] = float(np.clip(qh + tau + comp_h,
                                                                  -2.0, 0.5))
                d.ctrl[AID[gamba + '_knee']] = qk + comp_k
                trim = float(np.clip(2.0*(com[0]-p_st[0]-0.02) + 0.5*v[0], -0.35, 0.35))
                trim += RULLIO*np.cos(np.pi*min(self.t_f/T_NOM, 1.0))
                d.ctrl[AID[gamba + '_ankle_pitch']] = float(np.clip(-(qh + qk) + trim,
                                                                    -0.45, 0.45))
                # [FORK] alternanza: il termine di posizione lavora solo nella prima
                # frazione dell'appoggio, mentre il peso passa; poi lascia il campo
                # al piazzamento del piede sul capture point. Soglia ancorata a T_MIN,
                # che coincide con la durata misurata dell'appoggio (0,180 s).
                _p = self.t_f/max(FRAZ*T_MIN, 1e-9)
                if MORBIDA:
                    _g = 0.5*(1.0 + np.cos(np.pi*min(_p, 1.0)))
                else:
                    _g = 1.0 if _p < 1.0 else 0.0
                d.ctrl[AID[gamba + '_ankle_roll']] = float(np.clip(
                    -(_g*KY_ST*(com[1]-p_st[1]) + 0.5*v[1]), -0.2, 0.2))
                d.ctrl[AID[gamba + '_hip_roll']] = float(np.clip(KR_B*phi + KRD*wx, -0.3, 0.3))
        mujoco.mj_step(m, d)
        self.fprev = d.actuator_force.copy()
        self.t_f += dt
        # EVENTO DI TOUCHDOWN: capture laterale oltre il piede d'appoggio
        cp_lat = float(com[1] + v[1]/self.om)
        commit = (cp_lat - float(p_st[1]))*segno_sw > MARG
        # [FORK] il passo successivo non parte finche' il carico non e' passato:
        # latch quando il baricentro entra entro SOGLIA_TRASF dal piede d'appoggio
        dlat = abs(float(com[1] - p_st[1]))
        if dlat <= SOGLIA_TRASF:
            self.trasf = True
        if (self.t_f >= T_MIN and commit and self.trasf) or self.t_f >= T_MAX:
            self.tsw_log.append(round(self.t_f, 3))
            self.dlat_log.append((round(dlat, 4), bool(self.trasf)))
            self.zsw_log.append(round(float(d.xpos[BID[sw + '_foot']][2]), 4))
            self.trasf = False
            self.stance = sw
            self.t_f = 0.0
            self.passi += 1
        if d.qpos[2] < 0.42 and self.caduta is None:
            self.caduta = True
        return self.caduta is None

def prova(T=30.0, v_des=0.0):
    c = Camminatrice(v_des=v_des)
    dt = m.opt.timestep
    for i in range(int(T/dt)):
        if not c.frame():
            return c.passi, i*dt, float(c.d.qpos[0]), c.tsw_log
    return c.passi, None, float(c.d.qpos[0]), c.tsw_log

def campagna():
    print('marcia_core.py - versione ' + VERSIONE)
    print('E2 step-timing a evento di stato. Criterio: >= 10 passi riproducibili.')
    p, tc, x, log = prova()
    tl = np.array(log) if log else np.array([0.0])
    print('Marcia: %d passi | %s | x finale %+.2f m' %
          (p, ('VIVA a fine prova' if tc is None else 'caduta a %.1f s' % tc), x))
    print('T_sw effettivi: medio %.3f s, min %.3f, max %.3f (%d touchdown a evento)'
          % (tl.mean(), tl.min(), tl.max(), len(log)))
    if x < -0.5:
        print('NOTA: il ciclo stabile trovato e\' RETROGRADO (deriva %.2f m/s):'
              % (x/(tc if tc else 30.0)))
        print('  il trigger laterale sincronizza la fase ma lascia libera la velocita\'')
        print('  sagittale -> equilibrio naturale all\'indietro (bias piede/CoP).')
    esito = 'PASS' if p >= 10 else 'FAIL'
    print('ESITO E2: %s (%d passi vs criterio 10)' % (esito, p))
    json.dump(dict(versione=VERSIONE, esito=esito, passi=p,
                   t_caduta_s=(round(tc, 1) if tc else None), x_m=round(x, 2),
                   tsw_medio_s=round(float(tl.mean()), 3)),
              open('e2_timing.json', 'w'), indent=1)
    print('Salvato e2_timing.json')

def demo():
    import time
    import mujoco.viewer
    print(__doc__)
    print('Record cantiere E2: 54 passi (ciclo retrogrado). R = ricomincia.')
    c = Camminatrice()
    caduta_t = None
    def tasto(k):
        if k in (82, 114):
            c.reset()
    dt = m.opt.timestep
    with mujoco.viewer.launch_passive(m, c.d, key_callback=tasto) as v:
        while v.is_running():
            t0 = time.time()
            vivo = c.frame()
            if not vivo and caduta_t is None:
                caduta_t = time.time()
                print('caduta dopo %d passi - reset tra 2 s' % c.passi)
            if caduta_t and time.time() - caduta_t > 2.0:
                c.reset()
                caduta_t = None
            v.sync()
            resto = dt - (time.time() - t0)
            if resto > 0:
                time.sleep(resto)

if __name__ == '__main__':
    if '--test' in sys.argv:
        campagna()
    else:
        demo()

# ---------------- API CANONE (Campagna C, 04/08) ----------------
def canone(kpf=2.0, t_appoggio=0.18, rullio=-0.035, xoff=-0.04, t_nom=0.30):
    """CANONE-ALTOPIANO (05/08): kp gambe = base x kpf (2.0 = kp 400), appoggio
    IMPOSTO A OROLOGIO, rullio -0,035. Regge su TRE build [C, tre run/build]:
    mujoco 3.11.0 x=+2,491220 | 3.2.7 +2,621374 | 3.1.6 +2,581697 (654, VIVA).
    L'altopiano copre rullio -0,037..-0,034. FUORI dall'altopiano la direzione
    ALTERNA ogni ~3 millesimi di rullio (pettine, non soglia) e dipende anche
    da numpy: a rullio -0,05 su 3.1.6, x = -8,53 (np 1.26) / -2,72 (np 2.4).
    MASSIMO MISURATO: +9,695622 m a rullio -0,05 su mujoco 3.11.0 + numpy
    2.4.4 - dichiarato, non canone. 'A evento' (T_MAX 0.60) NON equivale
    all'orologio fuori dalla macchina di campagna (3.1.6: caduta 35,1 s,
    +0,596): l'invocazione canonica e' A OROLOGIO."""
    global T_MIN, T_MAX, RULLIO, XOFF, T_NOM
    base = {n: (float(m.actuator_gainprm[AID[n]][0]) / _KPF_ATTUALE[0],
                float(m.actuator_biasprm[AID[n]][1]) / _KPF_ATTUALE[0])
            for n in GAMBE}
    for n in GAMBE:
        i = AID[n]
        m.actuator_gainprm[i][0] = base[n][0] * kpf
        m.actuator_biasprm[i][1] = base[n][1] * kpf
    _KPF_ATTUALE[0] = kpf
    T_MIN = T_MAX = t_appoggio
    RULLIO, XOFF, T_NOM = rullio, xoff, t_nom
    # NOTA-CANONE (quirk dichiarato): KP_ATT resta alla tabella d'import (x3).
    # La compensazione ff divide per x3 mentre i guadagni girano a x2: cosi'
    # correva la campagna (via-c18) e cosi' nasce il 654/+9,695622. NON pulire.

_KPF_ATTUALE = [3.0]   # il modello viene moltiplicato x3 all'import (storia E2)
