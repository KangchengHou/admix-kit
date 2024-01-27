import subprocess
import tempfile
from os.path import join
from ..utils import get_cache_dir, cd
import urllib.request
import shutil
import os
import admix


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
    if os.path.exists(cache_bin_path):
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

            elif name == "hapgen2":
                if platform == "linux":
                    url = (
                        "http://mathgen.stats.ox.ac.uk/genetics_software/hapgen"
                        "/download/builds/x86_64/v2.2.0/hapgen2_x86_64.tar.gz"
                    )
                else:
                    raise ValueError(f"Unsupported platform {platform}")
                with tempfile.TemporaryDirectory() as tmp_dir:
                    with cd(tmp_dir):
                        urllib.request.urlretrieve(
                            url,
                            "file.tar.gz",
                        )
                        subprocess.check_call(f"tar -xvf file.tar.gz", shell=True)

                        subprocess.check_call(
                            f"mv hapgen2 {cache_bin_path}",
                            shell=True,
                        )

            elif name == "admix-simu":
                if platform == "linux":
                    url = "https://github.com/williamslab/admix-simu/archive/refs/heads/master.zip"
                else:
                    raise ValueError(f"Unsupported platform {platform}")
                with tempfile.TemporaryDirectory() as tmp_dir:
                    with cd(tmp_dir):
                        urllib.request.urlretrieve(
                            url,
                            "master.zip",
                        )
                        subprocess.check_call(f"unzip master.zip -d dir", shell=True)
                        # make
                        subprocess.check_call(
                            f"cd dir/admix-simu-master && make", shell=True
                        )
                        subprocess.check_call(
                            f"mv dir/admix-simu-master/ {cache_bin_path}",
                            shell=True,
                        )
            else:
                raise ValueError(f"Unsupported software {name}")

        else:
            raise ValueError(
                f"{name} not found in $PATH or {cache_dir}, set `download=True` to download from website"
            )

        return cache_bin_path


def get_cache_data(name: str, **kwargs) -> str:
    """
    Obtain the path to the cached data.

    Find the data in the following locations:
    - package installation directory admix-tools/.admix_cache/data/<name>

    If not found in any of these locations, download will start

    Parameters
    ----------
    name: str
        name of the data
        - genetic_map: used for HAPGEN2, download from
            https://alkesgroup.broadinstitute.org/Eagle/downloads/tables/
            kwargs["build"] = hg19 or hg38
        - hapmap3_snps: used to intersect with HM3 SNPs, download from
            https://ndownloader.figshare.com/files/25503788 (see https://privefl.github.io/bigsnpr/articles/LDpred2.html)

    Returns
    -------
    str: path to the cached file
    """

    # find in cache
    cache_dir = join(get_cache_dir(), "data", name)
    os.makedirs(cache_dir, exist_ok=True)

    if name == "genetic_map":
        assert kwargs["build"] in ["hg19", "hg38"]
        file_name = f"genetic_map_{kwargs['build']}_withX.txt.gz"
        cache_path = join(cache_dir, file_name)
        url = (
            "https://storage.googleapis.com/broad-alkesgroup-public/Eagle/downloads/tables/"
            + file_name
        )
    elif name == "hapmap3_snps":
        file_name = "hapmap3_snps.rds"
        cache_path = join(cache_dir, file_name)
        url = "https://ndownloader.figshare.com/files/25503788"
    else:
        raise ValueError(f"Unsupported data {name}")

    if not os.path.exists(cache_path):
        admix.logger.info(f"{name} not found at {cache_path}.")
        admix.logger.info(f"Downloading {name} from {url}.")
        admix.logger.info(
            f"If this gets stuck or fails, manually download {url} to {cache_path}."
        )

        try:
            urllib.request.urlretrieve(
                url,
                cache_path,
            )
        except Exception as e:
            admix.logger.info(
                f"Download failed. Please manually download {url} to {cache_path}."
            )
            raise e
    else:
        admix.logger.info(f"{name} found at {cache_path}.")
    return cache_path
