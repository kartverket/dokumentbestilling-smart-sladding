# Isolating the VLM

The verifier sends crops of real documents to a model. The crops hold real
fødselsnumre, so the question is not only what the model can reach, but where
the prompt can end up. This is how the boundary is built and how you prove it
holds.

Read this together with the VLM verifier section in the README, which covers
what the verifier does. This file covers what it is allowed to touch.

---

## Two questions that get confused

**Can the model read files or reach the internet?** No, and not because we
configured it that way. A model is weights. Inference maps a prompt and an
image to tokens. It makes no syscalls, opens no files and holds no sockets.
The only way a model output turns into an action is if the calling program
gives it tools and then runs what comes back.

Ours does not. `app/vlm_client.py` builds this body and nothing else:

```python
body = {"model": model, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens}
```

No `tools`, no `tool_choice`, no `function_call`. The reply goes to
`parse_answer`, which reads it as JSON and keeps four strings. The verdict
is `ja` or `nei`, and `vlm_verifier.verify_page` uses it for one decision:
drop a box or keep it. Nothing from the model is executed,
written to a path, or used to build a path. This holds as long as no one adds
a `tools` field, so treat that as the line not to cross.

**Can the server process read files or reach the internet?** That depends
entirely on which server you run and how the unit is written. It is the real
question, and configuration alone is not an answer to it.

---

## Why llama-server

The server is the only part of this that can reach the network or the
filesystem, so what matters is what it does not have.

There is no registry client. A model is a path on the command line. Nothing
resolves a name, nothing looks anything up remotely, and a missing file fails
the load loudly instead of triggering a fetch.

There is no cloud backend and no remote inference path, so no setting exists
that could send a prompt somewhere else.

Request bodies are never written to disk. Ours are base64 PNGs of document
crops, so a server that dumped them for debugging would scatter fødselsnumre
into files nobody is watching.

The one path in llama.cpp that downloads anything is the `-hf` flag. Building
with `-DLLAMA_CURL=OFF` takes it out of the binary, so the property survives
someone writing that flag into the unit by mistake.

None of this is a promise about how we invoke the server. It is a property of
the build and of the unit, which is what makes it checkable, and the checks
are at the end of this file.

---

## The unit

`/etc/systemd/system/llama-server.service`. Bare-bones: one process, one
model, no web UI, no network route off the box, and a filesystem it cannot
write to.

```ini
[Unit]
Description=llama-server (VLM verifier)
After=local-fs.target

[Service]
ExecStart=/opt/llama.cpp/bin/llama-server \
    --model /data2/llama/qwen3.8-27b.gguf \
    --mmproj /data2/llama/qwen3.8-27b-mmproj.gguf \
    --host 127.0.0.1 --port 8080 \
    --ctx-size 9216 --parallel 3 \
    --n-gpu-layers 99 --flash-attn auto \
    --reasoning off --reasoning-budget 0 \
    --no-webui
User=llama
Group=llama
SupplementaryGroups=video
Restart=always

# No route off the box. systemd installs this as a cgroup BPF filter, so a
# firewall reload cannot drop it the way it drops an nftables rule.
IPAddressDeny=any
IPAddressAllow=localhost

ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
ProtectProc=invisible
ProcSubset=pid
ReadOnlyPaths=/data2/llama
NoNewPrivileges=yes
CapabilityBoundingSet=
AmbientCapabilities=
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectKernelLogs=yes
ProtectControlGroups=yes
ProtectClock=yes
RestrictNamespaces=yes
RestrictRealtime=yes
RestrictSUIDSGID=yes
LockPersonality=yes
RestrictAddressFamilies=AF_UNIX AF_INET
SystemCallArchitectures=native
SystemCallFilter=@system-service
UMask=0077

[Install]
WantedBy=multi-user.target
```

`ProtectSystem=strict` makes the whole filesystem read-only for this service.
Execution still works, so the binary runs as before, and with the model
directory read-only nothing can be written next to the weights.

Three directives are deliberately absent because they break the GPU.
`PrivateDevices=yes` hides `/dev/nvidia*`. `PrivateUsers=yes` usually breaks
device access the same way. `MemoryDenyWriteExecute=yes` breaks CUDA, which
generates code at runtime. Narrow the device list with `DevicePolicy=closed`
and a `DeviceAllow=` line per nvidia node if you want that tightened.

`User=llama` matters as much as the directives above it. A service running as
root can still be sandboxed, but then every one of those restrictions is the
only thing between a bug in the server and the rest of the machine. An
unprivileged user is the layer underneath them.

### If you want the stronger version later

`--host` binds a unix socket when the path ends in `.sock`. A unix socket is a
filesystem object, so it works across namespaces, and that lets you add
`PrivateNetwork=yes` and give the process a network namespace holding only
`lo`. Nothing to filter, nothing to allow. It needs a small transport change in
`vlm_client.call_model`, which speaks HTTP over TCP today, and a bind mount of
the socket into the prod container. Worth doing if outbound has to be provably
impossible rather than filtered.

---

## Five things that will bite

**Context is divided across slots.** `--ctx-size` is the total KV cache, and
each of the `--parallel` slots gets `ctx-size / parallel`. It is not a
per-request figure, which is the easy assumption to make. `--ctx-size 3072
--parallel 3` leaves 1024 tokens per slot, and the worst request measured here
is 2245. Multiply first: 3072 x 3 = 9216, and read `n_ctx_per_seq` from the
startup log rather than trusting the arithmetic.

**The model thinks unless told not to.** qwen3.8 emits `<think>` blocks by
default. `vlm_client` caps the answer at 150 tokens, so a monologue eats the
budget, the reply truncates, `parse_answer` cannot read it and every box falls
back to «ja». The verifier would then run and remove nothing, which looks like
success. `--reasoning off --reasoning-budget 0` turns it off at the
server, which is why the request body needs no thinking field of its own.
Measured after: 21 completion tokens, `finish_reason` stop, no `<think>` in
`content`.

**The GPU is a V100S, which is Volta, sm_70.** CUDA 13 dropped Volta, so the
toolkit has to be a 12.x and the build wants
`-DCMAKE_CUDA_ARCHITECTURES=70`. llama.cpp's fast flash-attention kernels want
Turing or newer, so `--flash-attn auto` is right here: forcing it on can be
slower than the fallback.

**Crops are smaller than Qwen-VL wants.** The loader warns that Qwen-VL needs
at least 1024 image tokens for grounding accuracy, and suggests
`--image-min-tokens 1024`. Our crops sit well under that. It is the most
likely reason a judgement here would differ from an earlier one for a reason
that has nothing to do with the model, so measure both ways on the same crop
set before reading anything into a comparison with run 38.

**Sizing, measured.** Model and projector loaded take 17.3 GB of the 32 GB,
leaving room for YOLO and Paddle. `--ctx-size 9216 --parallel 3` gives
`n_ctx_slot = 3072`, matching the worst measured request of 2245 tokens.
Raising `--image-min-tokens` pushes prompt tokens up and eats into both.

---

## Building llama-server

Rocky 9, CUDA toolkit 12.x, no `-hf` download path in the binary.

Check the CUDA situation first, because this host had a trap in it. Three
CUDA 13 trees were installed and `/usr/local/cuda` pointed at 13.3, so the
default `nvcc` could not build for the GPU in the machine:

```sh
readlink -f /usr/local/cuda
rpm -qa 'cuda*-13-*' | head
```

If 13.x is present, remove it. None of it can target sm_70, so it is pure
liability. Dry run first and confirm the transaction pulls in no `nvidia-*`
driver package:

```sh
dnf remove --assumeno 'cuda*-13-*'
dnf remove -y 'cuda*-13-*'
```

`alternatives` re-points `/usr/local/cuda` at the remaining 12.x by itself.
Then the toolkit, if it is missing. The package is `cuda-toolkit-12-x`, never
`cuda` or `nvidia-driver`: those pull a driver and can break the one already
running.

```sh
dnf install -y gcc-c++ cmake git
dnf install -y cuda-toolkit-12-4
/usr/local/cuda-12.4/bin/nvcc --list-gpu-arch | grep -w compute_70
```

The build. `git` needs the proxy; `dnf` does not, since the CUDA repo is
mirrored locally.

```sh
https_proxy=http://159.162.48.7:3128 \
  git clone --depth 1 https://github.com/ggml-org/llama.cpp /usr/local/src/llama.cpp
cmake -S /usr/local/src/llama.cpp -B /usr/local/src/llama.cpp/build \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_INSTALL_RPATH='$ORIGIN;/usr/local/cuda-12.4/targets/x86_64-linux/lib' \
    -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build /usr/local/src/llama.cpp/build -j "$(nproc)" \
    -t llama-server -t llama-mtmd-cli
```

Five of those flags are the point rather than tidiness. `LLAMA_CURL=OFF`
removes the `-hf` download path from the binary, so "it never fetches
anything" becomes a property of the build instead of a promise about how we
invoke it. `LLAMA_USE_PREBUILT_UI` defaults to ON and pulls a prebuilt web UI
from a HuggingFace bucket during the build. `CMAKE_CUDA_ARCHITECTURES=70`
stops it compiling for every architecture NVIDIA ever shipped. The two rpath
flags matter more than they look: without them the binaries keep a build-tree
rpath and resolve CUDA through the `alternatives` symlink, so `/opt/llama.cpp`
is not self-contained and a later CUDA 13 install silently repoints it at
libraries this card cannot run.

llama.cpp builds shared libraries, so the install copies the whole `bin`
directory rather than two files:

```sh
mkdir -p /opt/llama.cpp/bin
cp -a /usr/local/src/llama.cpp/build/bin/. /opt/llama.cpp/bin/
env -u LD_LIBRARY_PATH ldd /opt/llama.cpp/bin/llama-server | grep -E "not found|cuda"
```

Check that with `LD_LIBRARY_PATH` cleared, since the repo venv exports a long
list of PyTorch CUDA wheel paths and a service started by systemd will never
see them. Every `libggml*`, `libllama*` and `libmtmd*` should resolve inside
`/opt/llama.cpp/bin`, CUDA under `/usr/local/cuda-12.4`, and nothing should be
missing. At that point the source tree is deletable.

---

## Proving it

Claiming isolation is easy. These commands make it checkable. Run them after
any change to the unit and after any upgrade, since an upgrade can replace the
unit file.

The merged configuration, never the file you just wrote:

```bash
systemctl cat llama-server
```

systemd's own audit of what is and is not sandboxed:

```bash
systemd-analyze security llama-server.service
```

Not running as root:

```bash
systemctl show llama-server -p User -p MainPID
```

Outbound must fail. Two traps here. `IPAddressDeny` is a cgroup property of
the service, not of the user, so running curl as the `llama` user proves
nothing: that process sits outside the service's cgroup and reaches the
internet fine. And a blocked request on its own is not evidence either, since
this host has no direct egress at all and a request to an arbitrary address
times out whether or not the directive does anything.

So test the directive under a throwaway unit, against the proxy, which is the
only real way off the machine. `IPAddressDeny` drops packets silently, so the
failure shows up as a timeout rather than a refusal.

```bash
systemd-run --quiet --pipe --wait curl -m 8 -sS -o /dev/null \
  -w "http %{http_code}\n" -x http://159.162.48.7:3128 https://huggingface.co
systemd-run --quiet --pipe --wait --property=IPAddressDeny=any \
  --property=IPAddressAllow=localhost curl -m 8 -sS -o /dev/null \
  -w "http %{http_code}\n" -x http://159.162.48.7:3128 https://huggingface.co
```

The first must return `http 200` and the second must fail. Both returning 200
means the directive is not doing what this document claims.

The filesystem checks have the same trap. `ProtectSystem=strict` is a property
of the service's mount namespace, so `sudo -u llama touch` tests the wrong
thing. Enter the namespace instead. `/home` must look empty:

```bash
sudo nsenter -t "$(systemctl show -p MainPID --value llama-server)" -m ls /home
```

And the model directory must refuse a write even to root, because the mount
itself is read-only:

```bash
sudo nsenter -t "$(systemctl show -p MainPID --value llama-server)" -m touch /data2/llama/probe
```

Nothing that looks like a crop in the journal:

```bash
journalctl -u llama-server --since "1 hour ago" | grep -ci base64
```

And the pipeline can still reach it:

```bash
curl -s http://127.0.0.1:8080/v1/models
```

---

## What this does not cover

**Prompt injection.** A scanned document can contain text aimed at the model,
and the model reads the crop. It cannot do anything with an instruction: there
are no tools, and the reply is parsed down to three values. The worst case is a
flipped verdict on one box. In the dangerous direction, a wrongly removed
sladd, the fnr guard in `vlm_client.fnr_protects` sits behind it and reads the
line again with `find_fnr`. Treat injection as a recall risk, not an
exfiltration risk.

**A hostile model file.** Parsing an untrusted GGUF is code running on your GPU
host, and llama.cpp has had parser bugs like every other format reader. The
sandbox is what contains that, which is the argument for the unit hardening on
top of the smaller binary. Keep staging weights by hand from a known source.

**SELinux** is a third layer and already enforcing here. It is independent of
everything above, which is the point of having it. `/data2` is unlabeled as a
whole, but the staged weights came out `usr_t` and needed no `chcon`. Check
with `ls -lZ` rather than assuming either way: a service that cannot read its
model fails in a manner that looks nothing like a label problem.
