r"""
pytype launchers/launch-instance.py -P . --check-variable-types \
  --check-container-types \
  --check-parameter-types --precise-return && \
  python check_flags.py launchers/launch-instance.py && \
  FLAGS="$(python json_to_args.py configs/launcher_configs/query_cacher_tfrecord.json)" && \
  python launchers/launch-instance.py $FLAGS
"""
import json
import pathlib
import operator
import os
from rich import print
import subprocess
import sys
import time

from absl import flags
from absl import app

import colored_traceback.auto  # pylint: disable=unused-import
import git
import pexpect
import shlex


_SCRIPT_DIRECTORY = pathlib.Path(__file__).resolve().parent
sys.path.append(str(_SCRIPT_DIRECTORY.parent))
import utils


_PROJECT_NAME = "julesgm-research"
_ZONE_TPUV2 = "us-central1-f"
_ZONE_TPUV3 = "europe-west4-a"

# gcloud create
# gcloud start
# gcloud tpu create
_FLAG_INSTANCE_NAME = flags.DEFINE_string(
  "instance-name",
  "jules",
  "Name of the VM and TPU instances.",
)
_FLAG_USER_NAME = flags.DEFINE_string(
  "username",
  "jules",
  "The gcloud username. "
)
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
_FLAG_USE_TPUS = flags.DEFINE_boolean(
  "use-tpus",
  False,
  "Whether to create a TPU."
)
_FLAG_TPU_ONLY = flags.DEFINE_boolean(
  "tpu-only",
  False,
  "",
)


def flatten_once(collection):
    asd = []
    for x in collection:
        asd.extend(x)
    return asd


def h1(text):
    print("\n" + "#" * utils.term_size())
    print("# " + "[green bold]" + text + "[/]")
    print("#" * utils.term_size())


def h2(text):
    print("[blue bold italic]" + text + "[/]")


def h3(text):
  print(text)


def try_command(command, title, sleep_time):
  while True:
    try:
      run_gcloud_command(command)
      print("")
      break
    except subprocess.SubprocessError as err:
      print("")
      print(f"Got error: `{err}`")
      print(f"Sleeping for {sleep_time} seconds.")
      time.sleep(sleep_time)
      print("")
      h2(f"Retrying {title}.")


def validate_instance_type_flag():
  # Validate the value:
  instance_tuple = _FLAG_INSTANCE_TYPE.value.strip().split("-")
  utils.check_equal(len(instance_tuple), 3)
  utils.check_equal(instance_tuple[0], "n1")
  utils.check_equal(instance_tuple[1], "standard")
  num_cpus = int(instance_tuple[2])
  # utils.check_operator(operator.le, num_cpus, 16)
  utils.check_operator(operator.ge, num_cpus, 0)


def run_gcloud_command(command):
  print(f"Running gcloud command:\n\t{command}")
  subprocess.run(command, check=True)


def start_using_gcloud():

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
    "gcloud", "compute", "instances", "create", _FLAG_INSTANCE_NAME.value,
  ]

  if _FLAG_PREEMPTIBLE_VM.value:
    positional.append("--preemptible")

  named_flags = {
    "--zone": zone,
    "--image-family": _FLAG_IMAGE_FAMILY.value,
    "--image-project": "deeplearning-platform-release",
    "--machine-type": _FLAG_INSTANCE_TYPE.value,
    "--boot-disk-size": f"{_FLAG_BOOT_DISK_SIZE.value}GB",
    "--scopes": "cloud-platform",
  }

  for key, value in named_flags.items():
    utils.check_isinstance(value, str)
    utils.check_isinstance(key, str)

  for key in named_flags:
    assert key.startswith("--"), key

  h2("Creating the VM instance.")
  command = positional + [
    f"{k}={shlex.quote(v)}" for k, v
    in named_flags.items()
  ]
  run_gcloud_command(command)
  print("")
  time.sleep(_FLAG_SLEEP_TIME.value)

  h2("Starting the instance.")
  command = [
    "gcloud", "compute", "instances", "start", _FLAG_INSTANCE_NAME.value
  ]
  run_gcloud_command(command)
  print("")
  time.sleep(_FLAG_SLEEP_TIME.value)

  h2("Listing the instances.")
  command = [
    "gcloud", "compute", "instances", "list"
  ]
  run_gcloud_command(command)
  print("")


def create_tpu_using_gcloud():
  utils.check_equal(_FLAG_TPU_TYPE.value, "v3")
  utils.check_equal(_FLAG_TPU_QTY.value, "8")
  assert not _FLAG_PREEMPTIBLE_TPU.value, _FLAG_PREEMPTIBLE_TPU.value

  positional_cmd = [
    "gcloud", "compute", "tpus", "create", _FLAG_INSTANCE_NAME.value
  ]

  if _FLAG_PREEMPTIBLE_TPU.value:
    positional_cmd += "--preemptible"

  named_arguments = {
    "--version": "2.4.1",
    "--accelerator-type": f"{_FLAG_TPU_TYPE.value}-{_FLAG_TPU_QTY.value}",
  }

  cmd = positional_cmd + [
    f"{k}={shlex.quote(v)}" for k, v in named_arguments.items()
  ]

  h2("Starting the TPUs.")
  run_gcloud_command(cmd)


def git_is_dirty(directory=_SCRIPT_DIRECTORY) -> bool:
  os.chdir(_SCRIPT_DIRECTORY)
  root = subprocess.check_output([
    "git", "rev-parse", "--show-toplevel",
    ]).decode().strip()
  return git.Repo(root).is_dirty(untracked_files=False)


def main(argv):
    if len(argv) > 1:
        raise RuntimeError(argv)

    if git_is_dirty():
      raise RuntimeError(
        "The git directory is dirty. Push the changes before running."
      )

    h1("Module args:")
    args = utils.get_module_args(argv[0])
    print(args)
    print("")

    if not subprocess.check_output(["which", "gcloud"]).strip():
      raise RuntimeError("`gcloud` is not in the path. `ctpu` won't work.")

    # if not _FLAG_TPU_ONLY.value:
    #   start_using_gcloud()
    #
    # if _FLAG_USE_TPUS.value and not _FLAG_VM_ONLY.value:
    #   create_tpu_using_gcloud()

    ###########################################################################
    # Copying setup.sh over
    ###########################################################################
    h1("Copying setup.sh")
    target_dir = f"/home/{_FLAG_USER_NAME.value}/"

    # try_command([
    #     "gcloud", "compute", "scp",
    #     f"{_SCRIPT_DIRECTORY}/setup.sh",
    #     f"{_FLAG_USER_NAME.value}@{_FLAG_INSTANCE_NAME.value}:{target_dir}",
    #   ], "Copying setup.sh", sleep_time=_FLAG_SLEEP_TIME.value
    # )

    ###########################################################################
    # Running setup.sh
    ###########################################################################
    h1("Running setup.sh")
    training_script_uri = (
      f"{target_dir}eli5_retrieval_large_lm/launchers/scripts/training.sh"
    )
    try_command([
        "gcloud", "compute", "ssh",
        f"{_FLAG_USER_NAME.value}@{_FLAG_INSTANCE_NAME.value}",
        f"--command=source {target_dir}setup.sh; "
        f"screen -S training -dm bash {training_script_uri}"
    ],
      "Running setup.sh", sleep_time=_FLAG_SLEEP_TIME.value
    )

    h1("All done.")


if __name__ == "__main__":
    app.run(main)
