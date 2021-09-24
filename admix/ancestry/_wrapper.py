import numpy as np
import pandas as pd
import tempfile
from os.path import join
import subprocess
from ..io import read_digit_mat, write_digit_mat


def lampld_wrapper(
    ref_hap_list,
    sample_hap,
    snp_pos,
    window_size=300,
    n_proto=6,
    bin_dir="/u/project/pasaniuc/kangchen/admix-tools/cache/LAMPLD-v1.3/bin/",
):
    tmp = tempfile.TemporaryDirectory()
    tmp_dir = tmp.name

    assert len(ref_hap_list) in [2, 3], "Currently only support 2-way / 3-way ancestry"
    for ref_i in range(len(ref_hap_list)):
        write_digit_mat(join(tmp_dir, f"ref{ref_i}.hap"), ref_hap_list[ref_i])
    write_digit_mat(join(tmp_dir, "sample.hap"), sample_hap)
    np.savetxt(join(tmp_dir, "pos.txt"), snp_pos, fmt="%s")

    if len(ref_hap_list) == 2:
        bin_path = join(bin_dir, "haplanc2way")
    elif len(ref_hap_list) == 3:
        bin_path = join(bin_dir, "haplanc")
    else:
        raise NotImplementedError
    cmd = " ".join(
        [
            bin_path,
            str(window_size),
            str(n_proto),
            join(tmp_dir, "pos.txt"),
            *[join(tmp_dir, f"ref{ref_i}.hap") for ref_i in range(len(ref_hap_list))],
            join(tmp_dir, "sample.hap"),
            join(tmp_dir, "out.txt"),
            str(1),
        ]
    )
    print(cmd)
    subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    infer_lanc = read_digit_mat(join(tmp_dir, "out.txt"))
    tmp.cleanup()
    return infer_lanc
