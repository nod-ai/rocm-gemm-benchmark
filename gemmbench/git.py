import subprocess


def clone(remote, repo=None):
    cmd = [
        "git",
        "clone",
        "--recurse-submodules",
        str(remote),
    ]
    if repo is not None:
        cmd.append(str(repo))
    subprocess.run(cmd, check=True)


def short_hash(repo=None):
    p = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo,
        stdout=subprocess.PIPE,
        encoding="ascii",
        check=True,
    )
    return p.stdout.strip()
