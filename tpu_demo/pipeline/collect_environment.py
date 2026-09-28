"""Collect bounded, read-only remote facts without importing TPU/AI runtimes.

Run with the existing environment activated, from the validated ChunkScan
checkout on the remote BM1690 machine. This does not compile or launch kernels.
Missing paths are recorded, never guessed or installed.
"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys


def query(command, cwd=None):
    process = subprocess.Popen(command,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                               text=True,start_new_session=True)
    try:
        stdout,stderr = process.communicate(timeout=5)
        return {"returncode":process.returncode,"stdout":stdout[-16384:],"stderr":stderr[-4096:]}
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL)
        process.communicate()
        return {"error":"timeout"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,default=Path.cwd())
    parser.add_argument("--sdk",type=Path)
    parser.add_argument("--pcie-runtime",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    paths = {}
    for label,value in (("ppl_root",args.sdk or os.environ.get("PPL_PROJECT_ROOT")),
                        ("pcie_runtime",args.pcie_runtime or os.environ.get("TILELANG_TPU_PCIE_RUNTIME_PATH"))):
        path = Path(value).expanduser().resolve() if value else None
        paths[label] = {"path":str(path) if path else None,"exists":path.exists() if path else False}
    packages = {}
    for package in ("torch","numpy","tilelang"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    commands = {name:shutil.which(name) for name in ("python","python3","cc","c++","cmake","git","tpu-smi","lspci")}
    git = {}
    if commands["git"]:
        for key,arguments in (("commit",["rev-parse","HEAD"]),("branch",["branch","--show-current"]),
                              ("status",["status","--porcelain"]),("submodules",["submodule","status"])):
            git[key] = query([commands["git"],*arguments],cwd=args.repo)
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name,value = line.split(":",1)
        if name in ("MemTotal","MemAvailable","SwapTotal","SwapFree"):
            memory[name] = value.strip()
    result = {
        "schema_version":1,"recorded_at":datetime.now(timezone.utc).isoformat(),
        "purpose":"remote BM1690 environment inventory; no kernel execution",
        "system":{"kernel":platform.release(),"machine":platform.machine(),
                  "cpu_count":os.cpu_count(),"allowed_cpus":sorted(os.sched_getaffinity(0)),"memory":memory},
        "python":{"executable":sys.executable,"version":platform.python_version(),"packages":packages},
        "commands":commands,"paths":paths,"repository":{"path":str(args.repo.resolve()),**git},
        "device_id":None,"device_id_note":"Record the BM1690 device id using the site's existing procedure before PCIe execution.",
    }
    if commands["lspci"]:
        result["pci_inventory"] = query([commands["lspci"],"-nn"])
    if commands["tpu-smi"]:
        result["tpu_smi_help"] = query([commands["tpu-smi"],"--help"])
    with args.output.open("x") as output:
        json.dump(result,output,indent=2,sort_keys=True)
        output.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
