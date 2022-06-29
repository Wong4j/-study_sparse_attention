import torch
import triton
from deepspeed.ops.sparse_attention import FixedSparsityConfig


import random
def make_layout(block, seq_len, sparsity=0.9):
    """create layout with specified (0~100%) sparsity"""
    layout_size = seq_len // block
    num_elems = layout_size * layout_size
    layout = torch.zeros((num_elems), dtype=torch.int)
    num_ones = int((1 - sparsity) * num_elems)
    index_arr = range(num_elems)
    rand_ones_idx = random.sample(index_arr, num_ones)
    layout[rand_ones_idx] = 1
    layout = layout.reshape(1, layout_size, layout_size)
    return layout

nt = {False: 'n', True: 't'}
square_confs = [
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],
        x_vals=[1024, 2048, 4096, 6144],
        line_arg='block',
        line_vals=[16, 32, 64, 128],
        line_names=['Block16', 'Block32', 'Block64', 'Block128'],
        ylabel='TFLOPS',
        plot_name=f'{op_mode}-square-{nt[AT]}{nt[BT]}-{dtype}',
        args={'op_mode': op_mode, 'AT': AT, 'BT': BT, 'dtype': dtype, 'provider': 'triton'}
    )
    for AT in [False] for BT in [False]
    for op_mode in ['sdd', 'dsd']
    for dtype in [torch.float16, torch.float32]
]


@triton.testing.perf_report(square_confs)
def bench_matmul(M, N, K, block, op_mode, AT, BT, dtype, provider, warmup=100, rep=100):
    Z, H = 1, 1
    # create layout
    shape = {'sdd': (M, N), 'dsd': (K, M) if AT else (M, K), 'dds': (N, K) if BT else (K, N)}[op_mode]
    #sparsity_config = FixedSparsityConfig(num_heads=1, block=block)
    #layout = sparsity_config.make_layout(seq_len=M)
    layout = make_layout(block, M, 0.9)
    # creat inputs
    a = torch.randn((Z, H, K, M) if AT else (Z, H, M, K), dtype=dtype, device='cuda')
    b = torch.randn((Z, H, N, K) if BT else (Z, H, K, N), dtype=dtype, device='cuda')
    # create op
    tflops = lambda ms: num_flops / ms * 1e3
    if provider == 'triton':
        op = triton.ops.blocksparse.matmul(layout, block, op_mode, device="cuda", trans_a=AT, trans_b=BT)
        # inputs
        a = triton.testing.sparsify_tensor(a, layout, block) if op_mode == 'dsd' else a
        b = triton.testing.sparsify_tensor(b, layout, block) if op_mode == 'dds' else b
        mean_ms, min_ms, max_ms = triton.testing.do_bench(lambda: op(a, b), warmup=warmup, rep=rep)
        num_flops = {
            'sdd': 2 * Z * K * float(layout.sum()) * block * block,
            'dsd': 2 * Z * N * float(layout.sum()) * block * block,
            'dds': 2 * Z * M * float(layout.sum()) * block * block
        }[op_mode] * 1e-12
        return tflops(mean_ms), tflops(min_ms), tflops(max_ms)

bench_matmul.run(print_data=True, show_plots=True)
