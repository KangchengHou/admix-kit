import numpy as np
import pandas as pd
import tempfile
from os.path import join
import subprocess
from ..data import read_int_mat, write_int_mat


def lampld_wrapper(
    ref_hap_list,
    sample_hap,
    snp_pos,
    window_size=300,
    n_proto=6,
    bin_path="/u/project/pasaniuc/kangchen/admix-tools/cache/LAMPLD-v1.3/bin/haplanc",
):
    tmp = tempfile.TemporaryDirectory()
    tmp_dir = tmp.name

    assert len(ref_hap_list) == 3, "Currently only support 3-way ancestry"
    for ref_i in range(len(ref_hap_list)):
        write_int_mat(join(tmp_dir, f"ref{ref_i}.hap"), ref_hap_list[ref_i])
    write_int_mat(join(tmp_dir, "sample.hap"), sample_hap)
    np.savetxt(join(tmp_dir, "pos.txt"), snp_pos, fmt="%s")

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
    infer_lanc = read_int_mat(join(tmp_dir, "out.txt"))
    tmp.cleanup()
    return infer_lanc
