import os, glob, subprocess

PASTA_H5 = os.environ.get("PASTA_H5", "dados_brutos")
PASTA_DERIV = os.environ.get("PASTA_DERIV", "derivados")
os.makedirs(PASTA_DERIV, exist_ok=True)
arquivos = sorted(glob.glob(os.path.join(PASTA_H5, "dataset_2comp_1leito_s6*.h5")))
print("level times for", len(arquivos), "files\n", flush=True)

for h5 in arquivos:
    base = os.path.basename(h5).replace(".h5", "")
    mask = os.path.join(PASTA_DERIV, "mascara_valid_%s.npz" % base)
    pre = os.path.join(PASTA_DERIV, "pre_%s.npz" % base)
    niv = os.path.join(PASTA_DERIV, "nivel_%s.npz" % base)
    pre_niv = os.path.join(PASTA_DERIV, "pre_nivel_%s.npz" % base)

    if not os.path.exists(niv):
        env = dict(os.environ)
        env["H5"] = h5
        env["SAIDA"] = niv
        print("=== level %s ===" % base, flush=True)
        r = subprocess.run(
            [
                "python3",
                "-u",
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "compute_level_times.py"
                ),
            ],
            env=env,
        )
        if r.returncode != 0:
            print("FAILED level", base)
            break

    if not os.path.exists(pre_niv):
        env = dict(os.environ)
        env["NIVEL"] = niv
        env["MASK"] = mask
        env["CACHE_IN"] = pre
        env["CACHE_OUT"] = pre_niv
        print("=== merge %s ===" % base, flush=True)
        r = subprocess.run(
            [
                "python3",
                "-u",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "merge_level_times.py"),
            ],
            env=env,
        )
        if r.returncode != 0:
            print("FAILED merge", base)
            break
    print("", flush=True)
print("ok")