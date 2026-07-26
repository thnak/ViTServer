"""
benchmark_trt.py
Loads TensorRT engines (FP32, FP16, INT8), runs warmup + timed iterations,
and reports latency/throughput statistics.
"""

import tensorrt as trt
import numpy as np
import time
import os
import sys


def load_engine(engine_path: str):
    """Deserialize a TensorRT engine from disk."""
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        data = f.read()
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(data)
    return engine


def get_dtype_size(dtype: trt.DataType) -> int:
    if dtype == trt.DataType.FLOAT:
        return 4
    elif dtype == trt.DataType.HALF:
        return 2
    elif dtype == trt.DataType.INT8:
        return 1
    else:
        return 4


def benchmark_engine(engine, label: str,
                     n_warmup: int = 100,
                     n_iter: int = 500):
    """Run benchmark on a single engine and return stats dict."""
    context = engine.create_execution_context()

    # Allocate device buffers for all bindings
    bindings = []
    for i in range(engine.num_bindings):
        shape = tuple(engine.get_binding_shape(i))
        dtype = engine.get_binding_dtype(i)
        volume = abs(trt.volume(shape))
        nbytes = volume * get_dtype_size(dtype)
        dev = trt.cuda_alloc(nbytes)
        bindings.append(int(dev))

    # Prepare random input (simulates normalized image [0,1])
    in_shape = tuple(engine.get_binding_shape(0))
    input_data = np.random.randn(*in_shape).astype(np.float32)
    input_data = np.clip(input_data, 0.0, 1.0)
    trt.cuda_memcpy(bindings[0], input_data.ctypes.data, input_data.nbytes,
                    trt.cudaMemcpyKind.cudaMemcpyHostToDevice)

    # Record output binding info for host copy
    out_info = []
    for i in range(1, engine.num_bindings):
        shape = tuple(engine.get_binding_shape(i))
        dtype = engine.get_binding_dtype(i)
        if dtype == trt.DataType.FLOAT:
            np_dtype = np.float32
        elif dtype == trt.DataType.HALF:
            np_dtype = np.float16
        else:
            np_dtype = np.float32
        out_info.append((i, shape, np_dtype))

    # Warmup
    for _ in range(n_warmup):
        context.execute_v2(bindings)

    # Timed iterations
    latencies = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        context.execute_v2(bindings)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    # Copy outputs back and validate
    print(f"  Output validation:")
    for idx, shape, np_dtype in out_info:
        vol = abs(trt.volume(shape))
        host = np.empty(vol, dtype=np_dtype)
        trt.cuda_memcpy(host.ctypes.data, bindings[idx], host.nbytes,
                        trt.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        host = host.reshape(shape)
        bname = engine.get_binding_name(idx)
        print(f"    {bname}: shape={shape}, dtype={np_dtype.__name__}, "
              f"min={host.min():.4f}, max={host.max():.4f}")
        # Sanity checks
        tol = 0.2  # relaxed for INT8
        if "box" in bname.lower():
            ok = bool((host >= -tol).all() and (host <= 1.0 + tol).all())
            print(f"      Box range check: {'PASS' if ok else 'FAIL'}")
        if "score" in bname.lower() or "cls" in bname.lower():
            ok = bool((host >= -tol).all() and (host <= 1.0 + tol).all())
            print(f"      Score range check: {'PASS' if ok else 'FAIL'}")

    latencies = np.array(latencies)
    mean_ms = float(np.mean(latencies))
    median_ms = float(np.median(latencies))
    p99_ms = float(np.percentile(latencies, 99))
    min_ms = float(np.min(latencies))
    max_ms = float(np.max(latencies))
    std_ms = float(np.std(latencies))
    throughput = 1000.0 / mean_ms

    print(f"\n  {label} Results:")
    print(f"    Mean latency:     {mean_ms:.4f} ms")
    print(f"    Median latency:   {median_ms:.4f} ms")
    print(f"    P99 latency:      {p99_ms:.4f} ms")
    print(f"    Min latency:      {min_ms:.4f} ms")
    print(f"    Max latency:      {max_ms:.4f} ms")
    print(f"    Std dev:          {std_ms:.4f} ms")
    print(f"    Throughput:       {throughput:.1f} inf/s")

    # Cleanup
    for mem in bindings:
        trt.cuda_free(mem)
    del context

    return {
        "name": label,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p99_ms": p99_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "std_ms": std_ms,
        "throughput": throughput,
    }


def get_engine_file_size(path: str) -> str:
    size = os.path.getsize(path)
    if size > 1048576:
        return f"{size / 1048576:.2f} MB"
    elif size > 1024:
        return f"{size / 1024:.2f} KB"
    return f"{size} B"


def main():
    base_dir = r"C:\Users\thanh\Git\ViTServer\training\runs\coco_3cls"

    configs = [
        ("FP32", "model_fp32.engine"),
        ("FP16", "model_fp16.engine"),
        ("INT8", "model_int8.engine"),
    ]

    results = []

    for label, fname in configs:
        path = os.path.join(base_dir, fname)
        if not os.path.isfile(path):
            print(f"[SKIP] {fname} not found")
            continue

        print(f"\n{'='*65}")
        print(f"Benchmarking {label}  ({get_engine_file_size(path)})")
        print(f"{'='*65}")

        engine = load_engine(path)
        print(f"  Bindings: {engine.num_bindings}")
        for i in range(engine.num_bindings):
            shape = tuple(engine.get_binding_shape(i))
            dtype = engine.get_binding_dtype(i)
            print(f"    [{i}] {engine.get_binding_name(i):20s}  "
                  f"shape={str(shape):20s}  dtype={dtype}")

        r = benchmark_engine(engine, label, n_warmup=100, n_iter=500)
        results.append(r)
        del engine

    # Summary table
    print("\n\n")
    print("=" * 72)
    print("  TENSORRT BENCHMARK SUMMARY")
    print("=" * 72)
    header = f"  {'Precision':<10} {'File Size':<14} {'Mean(ms)':<12} {'Median(ms)':<12} {'P99(ms)':<12} {'Inf/s':<10}"
    print(header)
    print("  " + "-" * 68)
    for r in results:
        fsize = get_engine_file_size(
            os.path.join(base_dir, f"model_{r['name'].lower()}.engine")
        )
        print(f"  {r['name']:<10} {fsize:<14} {r['mean_ms']:<12.4f} "
              f"{r['median_ms']:<12.4f} {r['p99_ms']:<12.4f} {r['throughput']:<10.1f}")

    # Speedup over FP32
    if len(results) >= 2:
        fp32_ms = results[0]["mean_ms"]
        print("\n  Speedup vs FP32:")
        for r in results:
            speedup = fp32_ms / r["mean_ms"]
            print(f"    {r['name']:>6s}: {speedup:.2f}x  "
                  f"({fp32_ms:.4f} ms -> {r['mean_ms']:.4f} ms)")


if __name__ == "__main__":
    main()
