#!/usr/bin/env python3
"""
make_workers.py -- build isolated NSCool worker directories.

NSCool writes its scratch files to Model_1/ under fixed names, so two
jobs sharing one NSCool tree silently overwrite each other.  That
happened once in this project and corrupted a batch of runs.  Each
worker here gets a REAL, private Model_1/ and symlinks for the large
read-only trees (EOS 104 MB, TOV 3.8 GB, I_Files, Code, Plot) plus the
binary, so N workers cost a few megabytes rather than N x 3.9 GB.

  python3 Source_Code/make_workers.py 4      # build NSCool_w1..w4
  python3 Source_Code/make_workers.py --clean

Then:
  python3 Source_Code/run_cooling_prl.py --workdir NSCool_w1 ...
"""
import os
import shutil
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER = os.path.join(BASE, "NSCool")

LINK = ["EOS", "TOV", "I_Files", "Code", "Plot", "NSCool.out"]
# Model_1 must be a real directory, private to each worker.
SEED_FROM_MASTER = ["I.dat"]          # files Model_1 needs to start


def worker_path(i):
    return os.path.join(BASE, "NSCool_w%d" % i)


def build(n):
    made = []
    for i in range(1, n + 1):
        w = worker_path(i)
        os.makedirs(w, exist_ok=True)
        for name in LINK:
            src = os.path.join(MASTER, name)
            dst = os.path.join(w, name)
            if not os.path.exists(src):
                raise SystemExit("master is missing %s" % src)
            if os.path.islink(dst) or os.path.exists(dst):
                continue
            os.symlink(src, dst)
        m1 = os.path.join(w, "Model_1")
        os.makedirs(m1, exist_ok=True)
        for f in SEED_FROM_MASTER:
            s = os.path.join(MASTER, "Model_1", f)
            d = os.path.join(m1, f)
            if os.path.exists(s) and not os.path.exists(d):
                shutil.copy2(s, d)
        # Plot_Teff subdir is written into by the solver
        os.makedirs(os.path.join(m1, "Plot_Teff"), exist_ok=True)
        made.append(w)
    return made


def clean():
    i = 1
    removed = []
    while True:
        w = worker_path(i)
        if not os.path.isdir(w):
            break
        # only ever remove a directory that looks like one of ours:
        # every entry is either a symlink we made or Model_1
        entries = set(os.listdir(w))
        if not entries <= (set(LINK) | {"Model_1", ".DS_Store"}):
            print("refusing to remove %s: unexpected contents %s"
                  % (w, sorted(entries - set(LINK) - {"Model_1"})))
            i += 1
            continue
        shutil.rmtree(w)
        removed.append(w)
        i += 1
    return removed


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--clean":
        r = clean()
        print("removed %d worker directories" % len(r))
        for w in r:
            print("   ", os.path.basename(w))
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
        made = build(n)
        print("built %d workers (symlinked to %s):" % (len(made), MASTER))
        for w in made:
            print("   ", os.path.basename(w))
        print()
        print("run with:  --workdir NSCool_w1")
