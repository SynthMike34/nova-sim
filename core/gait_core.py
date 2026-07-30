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
import os
import mujoco, numpy as np
L1, L2 = 0.38, 0.40

def ik(xa, za):
    D = np.hypot(xa, za); D = min(D, L1+L2-1e-4)
    gi = np.arccos(np.clip((L1*L1+L2*L2-D*D)/(2*L1*L2), -1, 1))
    qk = np.pi - gi
    beta = np.arccos(np.clip((L1*L1+D*D-L2*L2)/(2*L1*D), -1, 1))
    return -(np.arctan2(xa, za) + beta), qk

def smooth(u): return 0.5 - 0.5*np.cos(np.pi*np.clip(u, 0, 1))

def gait(step=0.10, za0=0.78, lift=0.05, t_sw=1.0,
         Ky=5.0, Kyd=2.0, Kx=2.5, y_in=0.015, lean=0.0, xb=0.02, n_passi=6,
         verbose=False, xml='tx34_v1.xml', video_ogni=0, recorder=None):
    m = mujoco.MjModel.from_xml_path(_MP(xml))
    print(f"[MASSA MODELLO] {m.body_mass.sum():.3f} kg")
    _pay = float(os.environ.get("NOVA_PAYLOAD_KG", 0))
    assert 65.5 + _pay < m.body_mass.sum() < 67.0 + _pay, f"MASSA FUORI RANGE: {m.body_mass.sum():.3f} kg (payload {_pay})"
    d = mujoco.MjData(m)
    aid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i][0]): i for i in range(m.nu)}
    bid = {b: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ('r_foot','l_foot','pelvis')}
    mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
    dt = m.opt.timestep; cp = d.subtree_com[0].copy()
    frames, ren = [], None
    if video_ogni:
        ren = mujoco.Renderer(m, 360, 480)

    stato, t_f, passi = 'warm', 0.0, 0
    stance = 'l'                      # prima stance: sinistra (parte il destro)
    xa = {'r': 0.0, 'l': 0.0}
    yr_f, xr_f = 0.0, 0.0             # riferimenti filtrati
    TAU = 0.25
    t = 0.0
    while t < 60.0:
        com = d.subtree_com[0]; v = (com-cp)/dt; cp = com.copy()
        qw,qx,qy,qz = d.qpos[3:7]
        phi = np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx*qx+qy*qy))     # rollio bacino
        th  = np.arcsin(np.clip(2*(qw*qy-qz*qx), -1, 1))         # beccheggio bacino
        rf, lf = d.xpos[bid['r_foot']], d.xpos[bid['l_foot']]
        ft = {'r': rf, 'l': lf}
        sw = 'r' if stance=='l' else 'l'
        segno = +1 if stance=='l' else -1     # verso interno

        if stato == 'warm':
            y_t, x_t = 0.5*(lf[1]+rf[1]), 0.5*(lf[0]+rf[0])
        elif stato == 'settle':
            y_t, x_t = 0.5*(lf[1]+rf[1]), max(lf[0], rf[0]) + xb - 0.04
        else:
            y_t = ft[stance][1] - segno*y_in
            x_t = ft[stance][0]
        yr_f += (y_t-yr_f)*dt/TAU; xr_f += (x_t-xr_f)*dt/TAU
        dy = com[1]-yr_f
        ar = float(np.clip(-(Ky*dy + Kyd*v[1]), -0.22, 0.22))

        tgt = {}
        if stato == 'warm':
            u = smooth(t_f/1.8)
            za_w = 0.828 + (za0-0.828)*u
            tgt = {'r': (0.0, za_w), 'l': (0.0, za_w)}
            if t_f >= 1.8: stato, t_f = 'shift', 0.0
        elif stato == 'settle':
            tgt = {'r': (xa['r'], za0), 'l': (xa['l'], za0)}
            fermo = abs(v[1]) < 0.02 and abs(v[0]) < 0.04 and abs(dy) < 0.015
            if (fermo and t_f > 1.1) or t_f > 4.0:
                stato, t_f = 'shift', 0.0
        elif stato == 'shift':
            if verbose and passi >= 1 and int(t/dt) % 25 == 0:
                print(f"    [sh t={t:.2f}] st={stance} dy={dy:+.3f} vy={v[1]:+.3f} ar={ar:+.2f} rf y={rf[1]:+.3f} z={rf[2]:.3f} lf y={lf[1]:+.3f} z={lf[2]:.3f} comx={com[0]:+.2f} xr={xr_f:+.2f}")
            tgt = {'r': (xa['r'], za0), 'l': (xa['l'], za0)}
            pronto = abs(com[1]-(ft[stance][1]-segno*y_in)) < 0.015 and abs(v[1]) < 0.04
            if (pronto and t_f > 0.4) or t_f > 2.5:
                px = d.xpos[bid['pelvis']][0]
                xa['r'], xa['l'] = rf[0]-px, lf[0]-px
                stato, t_f = 'swing', 0.0
        elif stato == 'swing':
            u = smooth(t_f/t_sw)
            tgt[stance] = (xa[stance] + ((-step/2)-xa[stance])*u, za0)
            tgt[sw] = (xa[sw] + ((step/2)-xa[sw])*u,
                       za0 - lift*np.sin(np.pi*np.clip(t_f/t_sw, 0, 1)))
            if t_f >= t_sw:
                xa[sw], xa[stance] = step/2, -step/2
                passi += 1
                if verbose: print(f"  passo {passi}: pelvi x={d.xpos[bid['pelvis']][0]:+.3f} dy={dy:+.3f} z={d.qpos[2]:.3f}")
                px = d.xpos[bid['pelvis']][0]
                xa['r'], xa['l'] = rf[0]-px, lf[0]-px
                stance, stato, t_f = sw, 'settle', 0.0
                if passi >= n_passi: break

        for leg in ('r','l'):
            qh, qk = ik(*tgt[leg])
            d.ctrl[aid[f'{leg}_hip_pitch']] = qh
            d.ctrl[aid[f'{leg}_knee']] = qk
            liv = -(qh+qk)
            if stato=='swing' and leg==sw:
                d.ctrl[aid[f'{leg}_ankle_pitch']] = liv - th
            else:
                d.ctrl[aid[f'{leg}_ankle_pitch']] = liv + np.clip(Kx*(com[0]-xr_f-xb)+0.8*v[0], -0.38, 0.38)
            if stato=='swing' and leg==sw:
                # il piede di volo deve atterrare sul suo binario: 0,18 m dal piede d'appoggio
                y_targ = ft[stance][1] - segno*0.18
                y_anca = d.qpos[1] + (0.09 if leg=='l' else -0.09)
                q_hr = (y_targ - y_anca)/0.78 - phi
                d.ctrl[aid[f'{leg}_ankle_roll']] = 0.0
                d.ctrl[aid[f'{leg}_hip_roll']] = float(np.clip(q_hr, -0.35, 0.35))
                if verbose and int(t/dt) % 30 == 0:
                    print(f"    [sw t={t:.2f}] phi={phi:+.3f} q_hr={q_hr:+.3f} piede_{leg} y={ft[leg][1]:+.3f} z={ft[leg][2]:.3f}")
            else:
                d.ctrl[aid[f'{leg}_ankle_roll']] = ar
                d.ctrl[aid[f'{leg}_hip_roll']] = float(segno*lean) if (stato=='swing' and leg==stance) else 0.0

        mujoco.mj_step(m, d)
        if recorder is not None:
            recorder(t, d)
        if video_ogni and int(t/dt) % video_ogni == 0:
            ren.update_scene(d, camera=-1)
            frames.append(ren.render().copy())
        if d.qpos[2] < 0.5:
            return passi, float(d.xpos[bid['pelvis']][0]), t, frames
        t += dt; t_f += dt
    return passi, float(d.xpos[bid['pelvis']][0]), None, frames
