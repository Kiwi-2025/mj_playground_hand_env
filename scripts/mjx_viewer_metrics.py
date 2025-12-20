#!/usr/bin/env python3
"""
Launch mjx-viewer and print mjx-testspeed metrics periodically to the same terminal.
Usage:
    python scripts/mjx_viewer_metrics.py --mjcf /path/to/model.xml --interval 5

This script requires the `mjx-viewer` and `mjx-testspeed` commands available in PATH.
"""
import argparse
import shutil
import subprocess
import threading
import time
import sys
import re

METRIC_PATTERNS = {
    'sim_time': re.compile(r'total simulation time\s*[:=]\s*([0-9\.eE+-]+)', re.I),
    'steps_per_sec': re.compile(r'steps per second\s*[:=]\s*([0-9\.eE+-]+)', re.I),
    'realtime_factor': re.compile(r'realtime factor\s*[:=]\s*([0-9\.eE+-]+)', re.I),
}


def parse_metrics(text: str) -> dict:
    out = {}
    for k, patt in METRIC_PATTERNS.items():
        m = patt.search(text)
        if m:
            try:
                out[k] = float(m.group(1))
            except Exception:
                out[k] = m.group(1)
    return out


def run_testspeed(mjcf: str, base_path: str, timeout: float) -> dict:
    cmd = ['mjx-testspeed', f'--mjcf={mjcf}', f'--base_path={base_path}']
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise
    except subprocess.TimeoutExpired:
        return {}
    out = res.stdout + "\n" + res.stderr
    return parse_metrics(out)


def metrics_loop(mjcf: str, base_path: str, interval: float, stop_event: threading.Event):
    # Run initial warm-run then loop
    while not stop_event.is_set():
        # Allow longer timeout to account for JIT/compilation overheads
        timeout = max(30.0, interval * 3)
        try:
            metrics = run_testspeed(mjcf, base_path, timeout=timeout)
        except FileNotFoundError:
            print("Error: 'mjx-testspeed' not found in PATH. Install mujoco-mjx or add to PATH.")
            return
        now = time.strftime('%Y-%m-%d %H:%M:%S')
        if metrics:
            sim_time = metrics.get('sim_time', 'N/A')
            sps = metrics.get('steps_per_sec', 'N/A')
            rt = metrics.get('realtime_factor', 'N/A')
            print(f"[{now}] sim_time={sim_time}  steps/s={sps}  realtime_factor={rt}")
        else:
            print(f"[{now}] mjx-testspeed returned no metrics (timeout or parsing failure). Tried timeout={timeout}s")
        # flush so output appears immediately in terminal running viewer
        sys.stdout.flush()
        # sleep until next interval unless stop requested
        for _ in range(int(interval*10)):
            if stop_event.is_set():
                break
            time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(description='Launch mjx-viewer and show mjx-testspeed metrics in terminal')
    parser.add_argument('--mjcf', required=True, help='Path to MJCF XML file')
    parser.add_argument('--base_path', default='.', help='Base path for mjx-testspeed')
    parser.add_argument('--interval', type=float, default=5.0, help='Seconds between metric samples')
    parser.add_argument('--viewer-args', default='', help='Additional args passed to mjx-viewer')
    parser.add_argument('--tail-log', default='', help='Path to CSV log to tail in realtime')
    args = parser.parse_args()

    viewer_cmd = shutil.which('mjx-viewer')
    testspeed_cmd = shutil.which('mjx-testspeed')
    if viewer_cmd is None:
        print("Error: 'mjx-viewer' not found in PATH. Install mujoco-mjx or add to PATH.")
        sys.exit(1)
    if testspeed_cmd is None:
        print("Warning: 'mjx-testspeed' not found in PATH. Metrics printing will fail.")

    # Start viewer subprocess
    cmd = [viewer_cmd, f'--mjcf={args.mjcf}']
    if args.viewer_args:
        cmd += args.viewer_args.split()
    print('Starting mjx-viewer: ' + ' '.join(cmd))
    viewer_proc = subprocess.Popen(cmd)

    # Start metrics thread
    stop_event = threading.Event()
    metrics_thread = threading.Thread(target=metrics_loop, args=(args.mjcf, args.base_path, args.interval, stop_event), daemon=True)
    metrics_thread.start()

    # Start tail thread if requested
    def tail_file(path: str, stop_event: threading.Event):
        import os, csv
        try:
            # wait for file to exist
            while not stop_event.is_set() and not path:
                time.sleep(0.2)
            while not stop_event.is_set() and not os.path.exists(path):
                time.sleep(0.2)
            with open(path, 'r', newline='') as f:
                reader = csv.reader(f)
                # Read header if present
                header = None
                try:
                    header = next(reader)
                except StopIteration:
                    header = None
                # move to end for tailing
                f.seek(0, 2)
                while not stop_event.is_set():
                    line = f.readline()
                    if not line:
                        time.sleep(0.2)
                        continue
                    # parse CSV line
                    try:
                        row = next(csv.reader([line]))
                    except Exception:
                        print(f"[log] {line.rstrip()}")
                        sys.stdout.flush()
                        continue
                    if header and len(header) == len(row):
                        kv = ' '.join([f"{k}={v}" for k, v in zip(header, row)])
                        print(f"[log] {kv}")
                    else:
                        print(f"[log] {line.rstrip()}")
                    sys.stdout.flush()
        except Exception as e:
            print(f"tail thread error: {e}")

    tail_thread = None
    if args.tail_log:
        import os
        tail_thread = threading.Thread(target=tail_file, args=(args.tail_log, stop_event), daemon=True)
        tail_thread.start()

    try:
        # Wait for viewer to exit
        while True:
            ret = viewer_proc.poll()
            if ret is not None:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print('Interrupted, terminating viewer...')
        try:
            viewer_proc.terminate()
        except Exception:
            pass
    finally:
        stop_event.set()
        metrics_thread.join(timeout=2.0)
        if viewer_proc.poll() is None:
            try:
                viewer_proc.kill()
            except Exception:
                pass


if __name__ == '__main__':
    main()
