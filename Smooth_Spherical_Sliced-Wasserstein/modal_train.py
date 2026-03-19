"""
Modal cloud training script — just calls existing bash/python scripts.

Usage:
    modal run modal_train.py                        # run all
    modal run modal_train.py::run_density           # single
    modal run modal_train.py::run_swvi
    modal run modal_train.py::run_swae
    modal run modal_train.py::run_ssl

    modal volume get ssw-results / ./local_results/ # download results
"""

import modal

app = modal.App("smooth-ssw")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "numpy", "scipy", "pandas",
        "POT", "tqdm", "matplotlib", "geopandas", "geomloss",
        "trimesh", "scikit-learn", "igl",
    )
    .add_local_dir(".", remote_path="/root/ssw")
)

vol = modal.Volume.from_name("ssw-results", create_if_missing=True)


@app.function(image=image, gpu="A100", volumes={"/results": vol}, timeout=7200)
def run_density():
    import subprocess
    subprocess.run(["bash", "experiments.sh"],
                   cwd="/root/ssw/Density Estimation", check=True)
    subprocess.run("cp -r weights Results /results/density/",
                   shell=True, cwd="/root/ssw/Density Estimation")
    vol.commit()


@app.function(image=image, gpu="A100", volumes={"/results": vol}, timeout=7200)
def run_swvi():
    import subprocess
    subprocess.run(["python", "xp_swvi.py"],
                   cwd="/root/ssw/SWVI", check=True)
    subprocess.run("cp -r . /results/swvi/", shell=True, cwd="/root/ssw/SWVI")
    vol.commit()


@app.function(image=image, gpu="A100", volumes={"/results": vol}, timeout=7200)
def run_swae():
    import subprocess
    subprocess.run(["python", "xp_swae.py"],
                   cwd="/root/ssw/SWAE", check=True)
    subprocess.run("cp -r results /results/swae/",
                   shell=True, cwd="/root/ssw/SWAE")
    vol.commit()


@app.function(image=image, gpu="A100", volumes={"/results": vol}, timeout=7200)
def run_ssl():
    import subprocess
    # SSL/experiments.sh uses sbatch (SLURM) — call main.py directly instead
    for method, unif_w, num_proj in [("ssw", 6, 10), ("ssw", 6, 200),
                                     ("hypersphere", 1, 200), ("simclr", 1, 200)]:
        subprocess.run([
            "python", "main.py",
            "--method", method,
            "--unif_w", str(unif_w),
            "--num_projections", str(num_proj),
            "--feat_dim", "3",
            "--batch_size", "512",
            "--seed", "0",
            "--data_folder", "./data/",
            "--result_folder", "./results/",
        ], cwd="/root/ssw/SSL", check=True)
    subprocess.run("cp -r results /results/ssl/",
                   shell=True, cwd="/root/ssw/SSL")
    vol.commit()


@app.local_entrypoint()
def main():
    handles = [
        run_density.spawn(),
        run_swvi.spawn(),
        run_swae.spawn(),
        run_ssl.spawn(),
    ]
    for h in handles:
        h.get()
    print("All done. Download: modal volume get ssw-results / ./local_results/")
