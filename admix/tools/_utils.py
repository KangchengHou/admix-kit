import subprocess
import tempfile
from os.path import join
from ..utils import get_cache_dir, cd
import urllib
import shutil
import os


def get_dependency(name, download=True):

    """Get path to an depenency
    Find the binary in the following locations:
    - $PATH
    - package installment directory admix-tools/.admix_cache/bin/<name>

    If not found in any of these locations, download the corresponding software package
    - plink: https://www.cog-genomics.org/plink/2.0/
    - gcta: https://github.com/gcta/gcta

    Parameters
    ----------
    download : bool
        whether to download plink if not found

    Returns
    -------
    Path to binary executable
    """
    # find in path
    if shutil.which(name):
        return shutil.which(name)

    # find in cache
    cache_dir = join(get_cache_dir(), "bin")
    os.makedirs(cache_dir, exist_ok=True)
    cache_bin_path = join(cache_dir, name)
    if shutil.which(cache_bin_path):
        return cache_bin_path
    else:
        # download
        if download:
            from sys import platform

            if name == "plink2":
                if platform == "darwin":
                    url = "https://s3.amazonaws.com/plink2-assets/alpha2/plink2_mac.zip"
                elif platform == "linux":
                    url = "https://s3.amazonaws.com/plink2-assets/alpha2/plink2_linux_x86_64.zip"
                else:
                    raise ValueError(f"Unsupported platform {platform}")

                with tempfile.TemporaryDirectory() as tmp_dir:
                    with cd(tmp_dir):
                        urllib.request.urlretrieve(
                            url,
                            "file.zip",
                        )
                        subprocess.check_call(f"unzip file.zip -d dir", shell=True)
                        subprocess.check_call(
                            f"mv dir/plink2 {cache_bin_path}", shell=True
                        )

            elif name == "plink":
                if platform == "darwin":
                    url = (
                        "https://s3.amazonaws.com/plink1-assets/plink_mac_20210606.zip"
                    )
                elif platform == "linux":
                    url = "https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20210606.zip"
                else:
                    raise ValueError(f"Unsupported platform {platform}")

                with tempfile.TemporaryDirectory() as tmp_dir:
                    with cd(tmp_dir):
                        urllib.request.urlretrieve(
                            url,
                            "file.zip",
                        )
                        subprocess.check_call(f"unzip file.zip -d dir", shell=True)
                        subprocess.check_call(
                            f"mv dir/plink {cache_bin_path}", shell=True
                        )

            elif name == "gcta64":
                if platform == "darwin":
                    platform_wildcard = "gcta_1.93.2beta_mac"
                elif platform == "linux":
                    platform_wildcard = "gcta_1.93.2beta"
                else:
                    raise ValueError(f"Unsupported platform {platform}")
                url = (
                    f"https://cnsgenomics.com/software/gcta/bin/{platform_wildcard}.zip"
                )
                with tempfile.TemporaryDirectory() as tmp_dir:
                    with cd(tmp_dir):
                        urllib.request.urlretrieve(
                            url,
                            "file.zip",
                        )
                        subprocess.check_call(f"unzip file.zip -d dir", shell=True)
                        subprocess.check_call(
                            f"mv dir/{platform_wildcard}/gcta64 {cache_bin_path}",
                            shell=True,
                        )
            elif name == "liftOver":
                if platform == "darwin":
                    url = "http://hgdownload.soe.ucsc.edu/admin/exe/macOSX.x86_64/liftOver"
                # NOTE: in case GLIBC version is not compatible, try
                # http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64.v369/liftOver
                elif platform == "linux":
                    url = (
                        "http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/liftOver"
                    )
                else:
                    raise ValueError(f"Unsupported platform {platform}")

                with tempfile.TemporaryDirectory() as tmp_dir:
                    with cd(tmp_dir):
                        urllib.request.urlretrieve(
                            url,
                            "liftOver",
                        )
                        subprocess.check_call(f"chmod +x liftOver", shell=True)
                        subprocess.check_call(
                            f"mv liftOver {cache_bin_path}", shell=True
                        )
            else:
                raise ValueError(f"Unsupported software {name}")

        else:
            raise ValueError(
                f"{name} not found in $PATH or {cache_dir}, set `download=True` to download from website"
            )

        return cache_bin_path
