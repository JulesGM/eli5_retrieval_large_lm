r"""
pytype launchers/launch.py -P . --check-variable-types \
  --check-container-types \
  --check-parameter-types --precise-return && \
  python check_flags.py launchers/launch.py && \
  FLAGS="$(python json_to_args.py configs/launcher_configs/query_cacher_tfrecord.json)" && \
  python launchers/launch.py $FLAGS
"""
import json
import pathlib
import operator
import os
import subprocess
import sys
import time

from absl import flags
from absl import app
import colored_traceback.auto  # pylint: disable=unused-import
import pexpect
import shlex


_SCRIPT_DIRECTORY = pathlib.Path(__file__).resolve().parent
sys.path.append(str(_SCRIPT_DIRECTORY.parent))
import utils

_PROJECT_NAME = "julesgm-research"
_ZONE_TPUV2 = "us-central1-f"
_ZONE_TPUV3 = "europe-west4-a"

_FLAG_IMAGE_FAMILY = flags.DEFINE_string(
  "image-family",
  "tf2-2-4-cpu",
  "See https://cloud.google.com/ai-platform/deep-learning-vm/docs/images"
)

_FLAG_SLEEP_TIME = flags.DEFINE_integer(
  "sleep_time",
  10,
  "How long to sleep between retries in seconds. "
  "Is also the duration of the sleep between major "
  "commands that take time."
)

# Args
_FLAG_NAME = flags.DEFINE_string(
  "instance-name",
  "jules",
  "",
)

_FLAG_BOOT_DISK_SIZE = flags.DEFINE_integer(
  "boot-disk-size",
  250,
  "Size of the boot disk, in gigabytes"
)

_FLAG_INSTANCE_TYPE = flags.DEFINE_string(
  "instance-type",
  None,
  "See https://cloud.google.com/compute/docs/machine-types for details."
)

_FLAG_TF_VERSION = flags.DEFINE_enum(
        "tf-version",
        "2.4.0",
        ["2.4.0"],
        "",
)

_FLAG_TPU_TYPE = flags.DEFINE_enum(
        "tpu-type",
        "v3",
        ["v2", "v3"],
        "",
)
_FLAG_TPU_QTY = flags.DEFINE_enum(
        "tpu-qty",
        "8",
        ["8"],
        "Size of the TPU group. This currently should always "
        "be 8.",
)
_FLAG_VM_ONLY = flags.DEFINE_boolean(
  "vm-only",
  False,
  "Whether to only create a VM and not reserve TPUs."
  "Great for running other tasks that don't require a TPU, "
  "but that still require a similar setup.",
)
_FLAG_PREEMPTIBLE_VM = flags.DEFINE_boolean(
  "preemptible-vm",
  False,
  "Whether or not we want the VM instance to be preemtible."
)
_FLAG_PREEMPTIBLE_TPU = flags.DEFINE_boolean(
  "preemptible-tpu",
  False,
  "Whether or not we want the TPU instance to be preemtible."
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


def try_command(command, title, sleep_time):
  h1(title)
  while True:
    try:
      subprocess.run(command, check=True)
      h2("Done with sending setup.sh.")
      break
    except subprocess.SubprocessError as err:
      print(f"Got error: `{err}`")
      print(f"Sleeping for {sleep_time} seconds.")
      time.sleep(sleep_time)
      h1(f"Retrying `{command}`")

def validate_instance_type_flag():
  # Validate the value:
  instance_tuple = _FLAG_INSTANCE_TYPE.value.strip().split("-")
  utils.check_equal(len(instance_tuple), 3)
  utils.check_equal(instance_tuple[0], "n1")
  utils.check_equal(instance_tuple[1], "standard")
  num_cpus = int(instance_tuple[2])
  # utils.check_operator(operator.le, num_cpus, 16)
  utils.check_operator(operator.ge, num_cpus, 0)

def start_using_ctpu():
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
  positional_flags = ["up", "-noconf"]
  named_flags = {
    "-tf-version": _FLAG_TF_VERSION.value,
    "-zone": zone,
    "-disk-size-gb": f"{_FLAG_BOOT_DISK_SIZE.value}GB",
    "-tpu-size": f"{_FLAG_TPU_TYPE.value}-{_FLAG_TPU_QTY.value}"
  }

  # Flags specific to this script's flags.
  if _FLAG_VM_ONLY.value:
    positional_flags.append("-vm-only")

  if _FLAG_INSTANCE_TYPE.value:
    validate_instance_type_flag()
    # Add it
    named_flags["-machine-type"] = _FLAG_INSTANCE_TYPE.value

  if _FLAG_PREEMPTIBLE_VM.value:
    positional_flags.append("-preemptible-vm")

  if _FLAG_PREEMPTIBLE_TPU.value:
    positional_flags.append("-preemptible")


  for key, value in named_flags.items():
    utils.check_isinstance(value, str)
    utils.check_isinstance(key, str)

  for key in named_flags:
    assert key.startswith("-"), key

  # Create the command
  cmd_flags = [bin] + positional_flags + [
      f"{k}={shlex.quote(v)}" for k, v
      in named_flags.items()
  ]

  h1("Running `ctpu up`")
  print(f"Command:\n"
      f"\tbin:    '{bin}'\n"
      f"\tflags:  {json.dumps(named_flags, indent=4)}"
  )

  # Run the command

  subprocess.run(cmd_flags)

def start_using_gcloud():
  if not _FLAG_VM_ONLY.value:
    raise ValueError(
      "Only -vm-only mode is currently supported "
      "with `gcloud` creation."
    )

  if not _FLAG_INSTANCE_TYPE.value:
    raise ValueError(
      "Using the full gcloud launcher is useless "
      "without an instance type."
    )

  validate_instance_type_flag()

  if _FLAG_TPU_TYPE.value == "v2":
    zone = _ZONE_TPUV2
  elif _FLAG_TPU_TYPE.value == "v3":
    zone = _ZONE_TPUV3
  else:
    raise RuntimeError(_FLAG_TPU_TYPE.value)

  positional = [
    "gcloud", "compute", "instances",
    "create", _FLAG_NAME.value,
  ]

  if _FLAG_PREEMPTIBLE_VM.value:
    positional.append("--preemptible")

  named_flags = {
    "--zone": zone,
    "--image-family":
      _FLAG_IMAGE_FAMILY.value,
    "--image-project":
      "deeplearning-platform-release",
    "--machine-type":
      _FLAG_INSTANCE_TYPE.value,
    "--boot-disk-size":
      f"{_FLAG_BOOT_DISK_SIZE.value}GB",
  }

  for key, value in named_flags.items():
    utils.check_isinstance(value, str)
    utils.check_isinstance(key, str)

  for key in named_flags:
    assert key.startswith("--"), key

  command = positional + [
    f"{k}={shlex.quote(v)}" for k, v
    in named_flags.items()
  ]

  print(f"Running gcloud command: {command}")
  subprocess.run(command)
  time.sleep(_FLAG_SLEEP_TIME.value)
  subprocess.run([
    "gcloud", "compute", "instances", "list"
  ])

  print("Starting the instance")
  subprocess.run([
    "gcloud", "compute", "instances", "start",
    _FLAG_NAME.value
  ])
  time.sleep(_FLAG_SLEEP_TIME.value)
  subprocess.run([
    "gcloud", "compute", "instances", "list"
  ])

def main(argv):
    if len(argv) > 1:
        raise RuntimeError(argv)

    if not subprocess.check_output(["which", "gcloud"]).strip():
      raise RuntimeError("`gcloud` is not in the path. `ctpu` won't work.")


    # start_using_ctpu()
    start_using_gcloud()

    ###########################################################################
    # Copying setup.sh over
    ###########################################################################
    h1("Copying setup.sh")
    try_command([
      "gcloud", "compute", "scp",
      f"{_SCRIPT_DIRECTORY}/setup.sh",
      f"{_FLAG_NAME.value}@{_FLAG_NAME.value}:"
      f"/home/{_FLAG_NAME.value}/",
    ],
      "Copying setup.sh", sleep_time=_FLAG_SLEEP_TIME.value
    )

    ###########################################################################
    # Running setup.sh
    ###########################################################################
    h1("Running setup.sh")
    try_command([
        "gcloud", "compute", "ssh",
        f"{_FLAG_NAME.value}@{_FLAG_NAME.value}",
        f"--command=source /home/{_FLAG_NAME.value}/setup.sh"
    ],
      "Running setup.sh", sleep_time=_FLAG_SLEEP_TIME.value
    )

    h1("All done.")


if __name__ == "__main__":
    app.run(main)
