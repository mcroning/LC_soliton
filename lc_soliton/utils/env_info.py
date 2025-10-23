# lc_soliton/utils/env_info.py
import subprocess
import cupy

def cuda_versions():
    print("=== CUDA Environment Summary ===")

    # 1. From driver (nvidia-smi)
    try:
        out = subprocess.check_output(["nvidia-smi"], universal_newlines=True)
        for line in out.splitlines():
            if "CUDA Version" in line:
                print("Driver:", line.strip())
                break
    except Exception as e:
        print("Driver check failed:", e)

    # 2. From toolkit (nvcc)
    try:
        out = subprocess.check_output(["nvcc", "--version"], universal_newlines=True)
        for line in out.splitlines():
            if "release" in line:
                print("Toolkit:", line.strip())
                break
    except Exception as e:
        print("Toolkit check failed:", e)

    # 3. From CuPy runtime
    try:
        print("CuPy version:", cupy.__version__)
        rt = cupy.cuda.runtime.runtimeGetVersion()
        print(f"CuPy CUDA runtime: {rt/1000:.1f}")
    except Exception as e:
        print("CuPy check failed:", e)

if __name__ == "__main__":
    cuda_versions()
