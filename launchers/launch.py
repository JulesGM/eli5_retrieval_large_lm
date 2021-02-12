import json
import pathlib
import os
import subprocess

from absl import flags
from absl import app
import pexpect
import shlex


_SCRIPT_DIRECTORY = pathlib.Path(__file__).resolve().parent
_ZONE_TPUV2 = "us-central1-f"
_ZONE_TPUV3 = "europe-west4-a"

# Args
_FLAG_NAME = flags.DEFINE_string(
        "instance_name",
        "jules",
        "",
)

_FLAG_TF_VERSION = flags.DEFINE_enum(
        "tf_version",
        "nightly",
        ["nightly"],
        ""
)

_FLAG_TPU_TYPE = flags.DEFINE_enum(
        "tpu_type",
        "v3",
        ["v2", "v3"],
        ""
)
_FLAG_TPU_QTY = flags.DEFINE_enum(
        "tpu_qty",
        "8",
        ["8"],
        "Size of the TPU group."
)


def flatten_once(collection):
    asd = []
    for x in collection:
        asd.extend(x)
    return asd


def h1(text):
    print("\n" + "#" * 80)
    print("# " + text)
    print("#" * 80)


def h2(text):
    print(text)


def main(argv):
    if len(argv) > 1:
        raise RuntimeError(argv)

    ###########################################################################
    # CTPU UP
    ###########################################################################
    # Prepare the flags
    if _FLAG_TPU_TYPE.value == "v2":
        zone = _ZONE_TPUV2
    elif _FLAG_TPU_TYPE.value == "v3":
        zone = _ZONE_TPUV3
    else:
        raise RuntimeError(_FLAG_TPU_TYPE.value)
    bin = "ctpu"
    positional_flags = ["up"]
    named_flags = {
        "-tf-version": _FLAG_TF_VERSION.value,
        "-zone": zone,
        "-tpu-size": f"{_FLAG_TPU_TYPE.value}-{_FLAG_TPU_QTY.value}"
    }
    cmd_flags = [bin] + positional_flags + [
            f"{k}={shlex.quote(v)}" for k, v
            in named_flags.items()
    ]
    line = ' '.join(cmd_flags)
    h1("Running `ctpu up`")
    print(f"Command:\n"
          f"\tbin:    '{bin}'\n"
          f"\tflags:  {json.dumps(named_flags, indent=4)}"
    )

    # Run the command
    ctpu = pexpect.spawn(line)
    print("[Spawned]")
    ctpu.expect("OK to create your Cloud TPU resources with the ")
    print("[Ok to create your Cloud TPU resources ...]")
    ctpu.sendline("y")
    ctpu.expect("About to ssh", timeout=5 * 60)
    print("[About to SSH ...]")
    ctpu.terminate(True)
    h2("Done with `ctpu up`")

    ###########################################################################
    # Copying setup.sh over
    ###########################################################################
    h1("Copying setup.sh")
    obj = subprocess.run([
            "gcloud", "compute", "scp",
            f"{_SCRIPT_DIRECTORY}/setup.sh",
            f"{_FLAG_NAME.value}@{_FLAG_NAME.value}:"
            f"/home/{_FLAG_NAME.value}/",
    ], stdout=subprocess.PIPE, check=True)
    h2("Done with sending setup.sh. Output:")
    if obj.stdout:
        print(obj.stdout)

    ###########################################################################
    # Running setup.sh
    ###########################################################################
    h1("Running setup.sh")
    subprocess.run([
            "gcloud", "compute", "ssh",
            f"{_FLAG_NAME.value}@{_FLAG_NAME.value}",
            f"--command=source /home/{_FLAG_NAME.value}/setup.sh"
    ], check=True)
    h2("Done running setup.sh. Output:")
    h1("All done.")


if __name__ == "__main__":
    app.run(main)
