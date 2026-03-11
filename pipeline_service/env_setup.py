import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Avoid "CuBLAS deterministic" warning when torch.use_deterministic_algorithms is used
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

def _detect_cpu_limit() -> None:
    """Detect available CPU cores from cgroup or fallback to nproc."""
    if os.environ.get("XATLAS_NUM_THREADS"):
        return

    cpu_count = None

    # Try cgroup v2
    try:
        with open("/sys/fs/cgroup/cpu.max", "r") as f:
            parts = f.read().strip().split()
            if parts[0] != "max" and len(parts) >= 2:
                quota, period = int(parts[0]), int(parts[1])
                if period > 0:
                    cpu_count = quota // period
    except (FileNotFoundError, ValueError, IndexError):
        pass

    # Try cgroup v1
    if not cpu_count:
        try:
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r") as f:
                quota = int(f.read().strip())
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r") as f:
                period = int(f.read().strip())
            if quota > 0 and period > 0:
                cpu_count = quota // period
        except (FileNotFoundError, ValueError):
            pass

    # Fallback to os.cpu_count()
    if not cpu_count or cpu_count <= 0:
        cpu_count = os.cpu_count() or 1

    # Cap threads for xatlas stability
    max_xatlas_threads = 13
    if cpu_count > max_xatlas_threads:
        print(f"XATLAS_NUM_THREADS: capping from {cpu_count} to {max_xatlas_threads} for stability")
        cpu_count = max_xatlas_threads

    os.environ["XATLAS_NUM_THREADS"] = str(cpu_count)
    print(f"XATLAS_NUM_THREADS set to: {cpu_count}")


_detect_cpu_limit()
