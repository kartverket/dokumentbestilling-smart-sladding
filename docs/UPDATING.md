# Updating packages and images

How to bump a dependency, test it, and get it into production. Read
[What is pinned, and why](#what-is-pinned-and-why) before you touch CUDA,
torch or Python.

## Two images, two ways to production

| Image | Built from | Built by | Reaches production |
|---|---|---|---|
| Model API | `Dockerfile`, `requirements.txt`, `requirements-override.txt` | `./deploy.sh build` on `ribr07ml27` | When you run `./deploy.sh promote <tag>` |
| Skip job | `job/Dockerfile`, `job/requirements-skip-job.txt` | GitHub Actions, `docker-image.yml` | **On every push to main**, through GHCR and argocd |

A push to main deploys the skip job at once. Test a change to `job/` before it
reaches main, and watch the workflow after the push.

The model API changes only when someone builds and promotes it on the server.
Main can be ahead of the image that runs.

## What CI checks on a pull request

| Workflow | Runs when a PR changes | Checks |
|---|---|---|
| `ci.yml` | Anything | No valid fødselsnummer in tracked files, the VLM selftest, the unit tests in `tests/`, and the linters `ruff`, `actionlint` and `zizmor` |
| `check-model-packages.yml` | `Dockerfile`, `requirements.txt`, `requirements-override.txt` | The packages install together on Linux, and `pip-audit` finds no known vulnerabilities |
| `check-job-image.yml` | `job/` | The skip job image builds, and its modules import |

`ci.yml` also runs on every push to main. The other two run on pull requests
only, so a push straight to main skips them. Run the same checks locally
first. See [Test a bump locally](#test-a-bump-locally).

The unit tests run on CPU with numpy, PyMuPDF, Pillow and requests at the
versions the images use. They cover fødselsnummer recognition, box placement,
the OCR and profile rules, and the skip job against a fake database, API and
model. They make their own fødselsnummer at runtime from birth years before
the register, so no number is written in the code. `tests/requirements.txt`
pins pytest and the linters. Dependabot does not update that file, so bump it
by hand now and then.

`zizmor` reads `.github/zizmor.yml`, which lets the team's own reusable
workflows in `kartverket/*` follow a branch instead of a commit hash.

`check-model-packages.yml` ignores three vulnerability IDs that the pins below
hold back. When you move a pin, remove the IDs it fixes. When `pip-audit`
reports a new ID, fix it, or add it with a comment that says why it cannot be
fixed yet.

## What is pinned, and why

| Pin | Why |
|---|---|
| `nvidia/cuda:12.6.x` base image | Matches the cu126 wheels of torch and paddle. The server has a Tesla V100S (Volta, sm_70), and CUDA 13 dropped Volta |
| `numpy` in `requirements-override.txt` | paddlex declares numpy below 2.4, so pip will not resolve a newer one with `requirements.txt`. The model runs on newer, so the Dockerfile installs this file last, without dependency checks |
| `ultralytics==8.4.90` | A newer version changes the YOLO scores on the same page. 8.4.215 dropped a low-score box on the test document. Bump it only after a recall check on labelled data |
| `opencv-contrib-python==4.10.0.84` | The `cv2` the model runs. paddlex pins this exact version, and ultralytics pulls in `opencv-python`, which writes the same `cv2/` folder. The Dockerfile reinstalls the contrib package last, and the build stops when `cv2.__version__` is not 4.10.0. A bump of `opencv-python` or `opencv-python-headless` does not change what runs. Check `import cv2; cv2.__version__` in the image |
| `torch==2.12.1`, `torchvision==0.27.1` | The last builds for CUDA 12.6. The Dockerfile reinstalls them from the cu126 index, so the version in `requirements.txt` must match that index |
| `paddlepaddle-gpu` from the paddle cu126 index | Same reason. It is not on PyPI |
| Ubuntu 24.04, Python 3.12 | Pillow 13 and PyMuPDF 2 need Python 3.11 or newer |
| `setuptools` below 82 | Set by torch. A scanner may flag it, but you cannot raise it without a new torch |

`.github/dependabot.yml` ignores CUDA 12.7 and newer, torch, torchvision,
ultralytics and opencv-contrib-python, so Dependabot does not propose them. Moving off CUDA 12 means new
GPU hardware first.

The torch step in the Dockerfile reinstalls torch's dependencies from the
PyTorch index, and can replace a package `requirements.txt` pins. The build
checks every pin in `requirements.txt` and `requirements-override.txt` at the
end, and the version of `cv2` on import, and stops with a list of the packages
that differ. Test the version the
index installed, and pin it in `requirements-override.txt` if it works.

## Dependabot pull requests

When the checks above are green, the PR can be merged. Each PR bumps one
package, so the next PR may conflict after a merge. Dependabot rebases its
own PRs.

To take several bumps at once, apply them in one commit and let Dependabot
close its PRs. It does that on its next run, when main already has the
version.

Close by hand a PR for something you will not take. The ignore rules stop new
PRs, but do not close the ones already open.

## Test a bump locally

Run what applies to your change.

**The packages install together on Linux.** Run pip inside a Linux container,
not on a Mac. Some packages have rules that apply only on macOS, so a resolve
on a Mac can fail, or pass, where the server would not.

```bash
docker run --rm --platform linux/amd64 -v "$PWD":/w python:3.12-slim \
  pip install --dry-run --ignore-installed --report /w/report.json \
  --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
  -r /w/requirements.txt
```

This takes several minutes, since it reads the metadata of the CUDA wheels.
It leaves out `requirements-override.txt`, which pip cannot resolve together
with the rest. Test a bump there with a build on the server.
Compare the package list in `report.json` with the one before. Look for new
packages, not only new versions.

**No known vulnerabilities.** Write the resolved versions as `name==version`
lines to a file and scan it with the same ignores as CI:

```bash
pip-audit -r requirements-resolved.txt --no-deps --disable-pip \
  --ignore-vuln PYSEC-2025-194 --ignore-vuln PYSEC-2026-3447 \
  --ignore-vuln GHSA-xmj3-9gf7-h52w
```

`paddlepaddle-gpu` is not on PyPI, so `pip-audit` skips it.

**The selftests and unit tests pass.** The same as CI runs:

```bash
python utils/fnr_vakt.py --selftest
python utils/fnr_vakt.py --all
python -W error::FutureWarning utils/vlm_selftest.py
python -m pytest tests
```

A test marked `xfail` describes a known bug. When you fix the bug, the test
passes, pytest reports it as a failure, and you remove the marker.

**The linters pass, when you change Python files or workflows.**

```bash
pip install -r tests/requirements.txt
ruff check --select F --ignore F401,F541,F841 .
zizmor --offline .github/workflows
docker run --rm -v "$PWD":/repo -w /repo rhysd/actionlint:1.7.12
```

**Rendering is unchanged, when you bump PyMuPDF or Pillow.** The model sees
pages as images, so a new renderer can change what it finds. Render the same
PDF with the old and the new version and compare the pixels. Use a PDF you
make yourself, with rotated pages and mixed page sizes. Do not use real
documents.

**The job image builds and imports, when you change `job/`.**

```bash
docker build --platform linux/amd64 -t skip-job-test job/
docker run --rm --platform linux/amd64 --entrypoint python skip-job-test \
  -c "import skip_job, url_utils, pdf_utils, requests"
```

## Test on the server

Commit and push first. The server builds from its own checkout, so it needs
the commit on origin.

```bash
cd /home/smartsladding/dokumentbestilling-smart-sladding
git pull --ff-only
./deploy.sh build
./deploy.sh test <tag>
```

Compare test against prod on the test document. An empty diff means the same
boxes:

```bash
diff <(curl -s -X POST http://localhost:5071/model -H "Content-Type: application/pdf" --data-binary @utils/testdokument.pdf | jq -S .) \
     <(curl -s -X POST http://localhost:5072/model -H "Content-Type: application/pdf" --data-binary @utils/testdokument.pdf | jq -S .)
```

Check that both torch and paddle use the GPU. A wrong CUDA build does not
always crash. It can fall back to the CPU, or fail only on the first request:

```bash
docker exec smsl-test python -c "import torch, paddle; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability()); print(paddle.__version__, paddle.device.get_device())"
```

Expect `True (7, 0)` from torch and `gpu:0` from paddle. Then:

```bash
./deploy.sh promote <tag>
./deploy.sh stop test
```

`./deploy.sh rollback` goes back to the previous prod tag.

## Problems we have hit

| Symptom | Cause | Fix |
|---|---|---|
| `fatal: not a git repository` as root | Root's shell does not load `server.env`, so `$SLADD_REPO` is empty | Use the full path, or `source activate.sh` |
| Build hangs pulling `nvidia/cuda` | An expired Docker Hub token in the server's login | `docker logout`. Anonymous pulls work |
| `docker rmi` refuses: image is in use | A stopped `smsl-test` still uses it. `deploy.sh stop` does not remove the container | `docker rm smsl-test`, then `docker rmi` |
| The build stops with «Installed packages differ from their pins» | The torch step replaced a pinned package with the version on the PyTorch index | See [What is pinned, and why](#what-is-pinned-and-why) |
| The new image finds other boxes than expected | The server checkout was on another branch, or `server.env` pointed at another model | `git status` on the server, and check the model with `./deploy.sh versions` |
