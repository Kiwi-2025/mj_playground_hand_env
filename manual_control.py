#!/usr/bin/env python3
"""
Manual controller: load MJCF, run simulation+renderer in-process, accept terminal commands
to change actuator controls, and print raw metrics (palm_dist, inner/outer SDF mean,
contact_sum, lift) each render step.

Usage:
    python scripts/manual_control.py --mjcf ./para_env/xmls/grasp/hand_16dof_grasp_sites.xml

Commands (type into the terminal running this script):
    set <i> <val>    set control channel i to val
    add <i> <delta>  add delta to control channel i
    zero             zero all controls
    rand             randomize controls
    show             print current control vector
    quit             exit

Note: requires mujoco python bindings available (pip package `mujoco`).
"""
import argparse
import threading
import time
import sys
import numpy as np
import mujoco as mj
from mujoco import mjx
from para_env import para_hand_constants as consts
try:
    import mujoco.viewer as mviewer
except Exception:
    mviewer = None
try:
    from mujoco.glfw import glfw
except Exception:
    try:
        import glfw
    except Exception:
        glfw = None


def quat_to_mat(q):
    # q: [w,x,y,z] or [qx,qy,qz,qw]? MuJoCo uses xyzw or wxyz? We'll assume (qx,qy,qz,qw) as used earlier.
    # Convert quaternion [qx,qy,qz,qw] to 3x3 rotation matrix
    q = np.asarray(q, dtype=float)
    if q.size == 4:
        qx, qy, qz, qw = q
    else:
        raise ValueError('quat must have 4 elements')
    # normalize
    n = np.linalg.norm([qw, qx, qy, qz])
    if n > 0:
        qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    # build rotation matrix
    mat = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),       2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),         1 - 2*(qx**2 + qz**2),   2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),         2*(qy*qz + qx*qw),       1 - 2*(qx**2 + qy**2)]
    ])
    return mat


def cube_sdf(cube_pos, cube_quat, points, half_size):
    # cube_pos: (3,), cube_quat: (4,) in (qx,qy,qz,qw) order
    # points: (N,3), half_size: (3,) or scalar
    R = quat_to_mat(cube_quat)
    # transform points to local
    local = (points - cube_pos) @ R
    d = np.abs(local) - np.asarray(half_size)
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
    inside = np.minimum(np.max(d, axis=1), 0.0)
    return outside + inside


class ManualController:
    def __init__(self, model_path, render_width=800, render_height=600, steps_per_render=5):
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
        self.steps_per_render = steps_per_render
        self.ctrl = np.zeros(self.model.nu, dtype=float)
        self.lock = threading.Lock()
        self._frame_count = 0
        self._print_every = 10

        # ids for sites/geoms (safe lookups)
        try:
            self.palm_sid = self.model.site('palm').id
        except Exception:
            self.palm_sid = None
        # inner/outer sites: safe lookup (model.site(name) raises if missing)
        self.inner_sids = []
        for n in consts.INNER_SITE_NAMES:
            try:
                sid = self.model.site(n).id
                self.inner_sids.append(sid)
            except Exception:
                continue
        self.outer_sids = []
        for n in consts.OUTER_SITE_NAMES:
            try:
                sid = self.model.site(n).id
                self.outer_sids.append(sid)
            except Exception:
                continue
        # cube geom/body
        try:
            self.cube_geom_id = self.model.geom('cube').id
            self.cube_body_id = self.model.body('cube').id
        except Exception:
            self.cube_geom_id = None
            self.cube_body_id = None
        # fingertip geom ids: try declared names first, then fall back to any geom name containing 'tip'
        self.fingertip_geom_ids = []
        for n in getattr(consts, 'FINGERTIP_NAMES', []):
            try:
                gid = self.model.geom(n).id
                self.fingertip_geom_ids.append(gid)
            except Exception:
                continue
        if not self.fingertip_geom_ids:
            # fallback: collect geoms whose names contain 'tip'
            try:
                # model.names may contain all names; iterate to find 'tip' occurrences
                for name in getattr(self.model, 'names', []):
                    if 'tip' in name:
                        try:
                            gid = self.model.geom(name).id
                            self.fingertip_geom_ids.append(gid)
                        except Exception:
                            continue
            except Exception:
                # last resort: empty list
                self.fingertip_geom_ids = []

        # cube half size
        if self.cube_geom_id is not None:
            self.cube_half = np.array(self.model.geom_size[self.cube_geom_id])
        else:
            self.cube_half = np.array([0.03, 0.03, 0.03])

        # sensor mapping: build fast lookup for sensor adr/dim by name
        self.sensor_map = {}
        try:
            for name in [
                'cube_pos', 'cube_quat', 'palm_pos',
                'thumb_fingertip_pos', 'index_fingertip_pos', 'middle_fingertip_pos',
                'ring_fingertip_pos', 'little_fingertip_pos',
                'thumb_touch', 'index_touch', 'middle_touch', 'ring_touch', 'little_touch'
            ]:
                try:
                    sid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, name)
                except Exception:
                    sid = -1
                if sid is not None and sid >= 0:
                    adr = int(self.model.sensor_adr[sid])
                    dim = int(self.model.sensor_dim[sid])
                    self.sensor_map[name] = (sid, adr, dim)
        except Exception:
            # if sensor arrays not available for some reason, leave map empty
            self.sensor_map = {}

    def start(self):
        # startup prints
        print('ManualController starting. Model nu=', self.model.nu)
        print('Type commands (set/add/zero/rand/show/quit) in this terminal or use GUI keys when running with --glfw.')
        # print actuator index->name mapping once
        try:
            for i in range(self.model.nu):
                name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
                if isinstance(name, bytes):
                    name = name.decode()
                print(f'{i}: {name}')
        except Exception:
            pass

        # start input thread (keystrokes from terminal)
        t = threading.Thread(target=self._input_thread, daemon=True)
        t.start()

        # prefer GLFW-based interactive viewer when available; otherwise try mujoco.viewer
        # The --glfw flag will be parsed in __main__ and set as attribute on self if present.
        use_glfw = getattr(self, 'use_glfw', False) or (glfw is not None)
        if use_glfw or mviewer is None:
            # run custom GLFW-based viewer loop with keyboard callbacks
            if not glfw.init():
                raise RuntimeError('Failed to initialize GLFW')

            width, height = 900, 700
            window = glfw.create_window(width, height, 'ManualController', None, None)
            if not window:
                glfw.terminate()
                raise RuntimeError('Failed to create GLFW window')
            glfw.make_context_current(window)

            # create rendering contexts
            scene = mj.MjvScene(self.model, maxgeom=2000)
            cam = mj.MjvCamera()
            cam.type = mj.mjtCamera.mjCAMERA_FREE
            cam.distance = 1.5
            context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150)
            opt = mj.MjvOption()
            pert = mj.MjvPerturb()

            # actuator selection state
            sel = {'idx': 0, 'step': 0.05}

            def get_actuator_name(idx):
                try:
                    name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, int(idx))
                    if isinstance(name, bytes):
                        name = name.decode()
                    return name
                except Exception:
                    return str(idx)

            def key_callback(window, key, scancode, action, mods):
                if action != glfw.PRESS and action != glfw.REPEAT:
                    return
                with self.lock:
                    if key == glfw.KEY_LEFT:
                        sel['idx'] = max(0, sel['idx'] - 1)
                        print('selected', sel['idx'], get_actuator_name(sel['idx']))
                    elif key == glfw.KEY_RIGHT:
                        sel['idx'] = min(self.model.nu - 1, sel['idx'] + 1)
                        print('selected', sel['idx'], get_actuator_name(sel['idx']))
                    elif key == glfw.KEY_UP:
                        self.ctrl[sel['idx']] = float(np.clip(self.ctrl[sel['idx']] + sel['step'], -1.0, 1.0))
                    elif key == glfw.KEY_DOWN:
                        self.ctrl[sel['idx']] = float(np.clip(self.ctrl[sel['idx']] - sel['step'], -1.0, 1.0))
                    elif key == glfw.KEY_HOME:
                        self.ctrl[:] = 0.0
                        print('zeroed controls')
                    elif key == glfw.KEY_R:
                        self.ctrl[:] = np.random.uniform(-1, 1, size=self.model.nu)
                        print('randomized controls')
                    elif key == glfw.KEY_S:
                        print('ctrl[{}] {} = {:.3f}'.format(sel['idx'], get_actuator_name(sel['idx']), self.ctrl[sel['idx']]))
                    elif key == glfw.KEY_Q or key == glfw.KEY_ESCAPE:
                        glfw.set_window_should_close(window, True)

            glfw.set_key_callback(window, key_callback)

            try:
                while not glfw.window_should_close(window):
                    step_start = time.time()
                    with self.lock:
                        self.data.ctrl[:] = self.ctrl
                    for _ in range(self.steps_per_render):
                        mj.mj_step(self.model, self.data)

                    metrics = self.compute_metrics()
                    # throttle terminal prints to reduce spam
                    self._frame_count += 1
                    if self._frame_count % self._print_every == 0:
                        print_metrics(metrics)

                    # render
                    viewport = mj.MjrRect(0, 0, width, height)
                    mj.mjv_updateScene(self.model, self.data, opt, pert, cam, mj.mjtCatBit.mjCAT_ALL, scene)
                    mj.mjr_render(viewport, scene, context)

                    glfw.swap_buffers(window)
                    glfw.poll_events()

                    elapsed = time.time() - step_start
                    target = self.model.opt.timestep * self.steps_per_render
                    if target > elapsed:
                        time.sleep(target - elapsed)
            finally:
                glfw.destroy_window(window)
                glfw.terminate()
        else:
            if mviewer is None:
                raise RuntimeError('mujoco.viewer not available; install mujoco-viewer or use --glfw')
            with mviewer.launch_passive(self.model, self.data) as viewer:
                while viewer.is_running():
                    step_start = time.time()
                    with self.lock:
                        self.data.ctrl[:] = self.ctrl
                    for _ in range(self.steps_per_render):
                        mj.mj_step(self.model, self.data)

                    metrics = self.compute_metrics()
                    # throttle terminal prints to reduce spam
                    self._frame_count += 1
                    if self._frame_count % self._print_every == 0:
                        print_metrics(metrics)

                    viewer.sync()

                    elapsed = time.time() - step_start
                    target = self.model.opt.timestep * self.steps_per_render
                    if target > elapsed:
                        time.sleep(target - elapsed)

    def compute_metrics(self):
        # try to read from sensors first (framepos/framequat/touch)
        def read_sensor(name):
            meta = self.sensor_map.get(name)
            if meta is None:
                return None
            sid, adr, dim = meta
            return np.array(self.data.sensordata[adr: adr + dim])

        # palm position (prefer sensor)
        palm_pos = read_sensor('palm_pos')
        if palm_pos is None and self.palm_sid is not None:
            palm_pos = np.array(self.data.site_xpos[self.palm_sid])

        # cube position and quat (prefer sensors)
        cube_pos = read_sensor('cube_pos')
        cube_quat = read_sensor('cube_quat')
        if cube_pos is None and self.cube_body_id is not None:
            cube_pos = np.array(self.data.body_xpos[self.cube_body_id])
        if cube_quat is None and self.cube_body_id is not None:
            cube_quat = np.array(self.data.body_xquat[self.cube_body_id])
        # inner/outer positions
        inner_pts = np.array(self.data.site_xpos[self.inner_sids]) if len(self.inner_sids) else np.zeros((0,3))
        outer_pts = np.array(self.data.site_xpos[self.outer_sids]) if len(self.outer_sids) else np.zeros((0,3))

        # compute sdf
        inner_sdfs_mean = None
        outer_sdfs_mean = None
        if cube_pos is not None and cube_quat is not None and inner_pts.shape[0]>0:
            inner_sdfs = cube_sdf(cube_pos, cube_quat, inner_pts, self.cube_half)
            inner_sdfs_mean = float(np.mean(inner_sdfs))
        if cube_pos is not None and cube_quat is not None and outer_pts.shape[0]>0:
            outer_sdfs = cube_sdf(cube_pos, cube_quat, outer_pts, self.cube_half)
            outer_sdfs_mean = float(np.mean(outer_sdfs))

        # contact sum: prefer touch sensors (one per fingertip). Count fingers with nonzero touch reading.
        contact_sum = 0
        touch_names = ['thumb_touch', 'index_touch', 'middle_touch', 'ring_touch', 'little_touch']
        for tn in touch_names:
            v = read_sensor(tn)
            if v is None:
                continue
            # touch sensor may have multiple entries; consider any nonzero as contact
            try:
                if np.max(np.abs(v)) > 1e-9:
                    contact_sum += 1
            except Exception:
                continue
        # fallback: if no touch sensors available, fall back to contact parsing
        if contact_sum == 0:
            try:
                ncon = int(self.data.ncon)
                for i in range(ncon):
                    g1 = int(self.data.contact[i].geom1)
                    g2 = int(self.data.contact[i].geom2)
                    if g1 in self.fingertip_geom_ids or g2 in self.fingertip_geom_ids:
                        contact_sum += 1
            except Exception:
                contact_sum = contact_sum

        # palm distance to cube
        palm_dist = None
        if palm_pos is not None and cube_pos is not None:
            palm_dist = float(np.linalg.norm(palm_pos - cube_pos))

        # lift value: palm z
        lift = None
        if palm_pos is not None:
            lift = float(palm_pos[2])

        return {
            'palm_dist': palm_dist,
            'inner_sdfs_mean': inner_sdfs_mean,
            'outer_sdfs_mean': outer_sdfs_mean,
            'contact_sum': contact_sum,
            'lift': lift,
            'ctrl': self.ctrl.copy()
        }

    def _input_thread(self):
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                parts = line.strip().split()
                if not parts:
                    continue
                cmd = parts[0]
                if cmd == 'set' and len(parts) >= 3:
                    i = int(parts[1]); v = float(parts[2])
                    with self.lock:
                        if 0 <= i < self.model.nu:
                            self.ctrl[i] = v
                        else:
                            print('index out of range')
                elif cmd == 'add' and len(parts) >= 3:
                    i = int(parts[1]); v = float(parts[2])
                    with self.lock:
                        if 0 <= i < self.model.nu:
                            self.ctrl[i] += v
                        else:
                            print('index out of range')
                elif cmd == 'zero':
                    with self.lock:
                        self.ctrl[:] = 0
                elif cmd == 'rand':
                    with self.lock:
                        self.ctrl[:] = np.random.uniform(-1, 1, size=self.model.nu)
                elif cmd == 'show':
                    with self.lock:
                        print('ctrl=', self.ctrl)
                elif cmd == 'quit':
                    print('Exiting...')
                    # attempt to close viewer by exiting process
                    sys.exit(0)
                else:
                    print('Unknown command')
            except Exception as e:
                print('input thread error', e)
                break


def print_metrics(m):
    parts = []
    for k in ['palm_dist','inner_sdfs_mean','outer_sdfs_mean','contact_sum','lift']:
        v = m.get(k)
        parts.append(f"{k}={v}" )
    parts.append(f"ctrl_sum={np.sum(np.abs(m['ctrl'])):.3f}")
    print(' | '.join(parts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mjcf', default='./para_env/xmls/grasp/hand_16dof_grasp_sites.xml')
    parser.add_argument('--steps-per-render', type=int, default=5)
    parser.add_argument('--glfw', action='store_true', help='Force GLFW-based interactive viewer')
    args = parser.parse_args()

    mc = ManualController(args.mjcf, steps_per_render=args.steps_per_render)
    mc.use_glfw = args.glfw
    mc.start()
