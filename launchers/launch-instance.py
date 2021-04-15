r"""
This scripts assumes that we are running on Google Cloud Compute.

pytype launchers/launch-instance.py -P . --check-variable-types \
  --check-container-types \
  --check-parameter-types --precise-return && \
  python check_flags.py launchers/launch-instance.py && \
  FLAGS="$(python json_to_args.py configs/launcher_configs/query_cacher_tfrecord.json)" && \
  python launchers/launch-instance.py $FLAGS
"""
import colored_traceback.auto
import pathlib
import operator
import os
from rich import print
import shlex
import subprocess
import sys
import time
import yaml

from absl import flags
from absl import app
import git


_SCRIPT_DIRECTORY = pathlib.Path(__file__).resolve().parent
sys.path.append(str(_SCRIPT_DIRECTORY.parent))
import utils

_ONEVM_RUNTIME_VERSION = "v2-alpha"


_FLAG_ZONE = flags.DEFINE_string(
  "gcloud-zone",
  "europe-west4-a",
  "Which Google Cloud zone to use.",
)
_FLAG_RUN_SCRIPT = flags.DEFINE_boolean(
  "run-script",
  True,
  "Whether or not to run the training script at the end."
)
_FLAG_BOOT_DISK_SIZE = flags.DEFINE_integer(
  "boot-disk-size",
  250,
  "Size of the boot disk, in gigabytes"
)
_FLAG_IMAGE_FAMILY = flags.DEFINE_string(
  "image-family",
  "tf2-2-4-cpu",
  "See https://cloud.google.com/ai-platform/deep-learning-vm/docs/images"
)
_FLAG_INSTANCE_NAME = flags.DEFINE_string(
  "instance-name",
  "jules",
  "Name of the VM and TPU instances.",
)
_FLAG_INSTANCE_TYPE = flags.DEFINE_string(
  "instance-type",
  None,
  "See https://cloud.google.com/compute/docs/machine-types for details."
)
_FLAG_PREEMPTIBLE_TPU = flags.DEFINE_boolean(
  "preemptible-tpu",
  False,
  "Whether or not we want the TPU instance to be preemtible."
)
_FLAG_PREEMPTIBLE_VM = flags.DEFINE_boolean(
  "preemptible-vm",
  False,
  "Whether or not we want the VM instance to be preemtible."
)
_FLAG_SLEEP_TIME = flags.DEFINE_integer(
  "sleep-time",
  10,
  "How long to sleep between retries in seconds. "
  "Is also the duration of the sleep between major "
  "commands that take time."
)
_FLAG_TF_VERSION = flags.DEFINE_enum(
  "tf-version",
  "2.4.0",
  ["2.4.0"],
  "",
)
_FLAG_TPU_ONLY = flags.DEFINE_boolean(
  "tpu-only",
  False,
  "",
)
_FLAG_TPU_QTY = flags.DEFINE_enum(
  "tpu-qty",
  "8",
  ["8"],
  "Size of the TPU group. This currently should always "
  "be 8.",
)
_FLAG_TPU_TYPE = flags.DEFINE_enum(
  "tpu-type",
  "v3",
  ["v2", "v3"],
  "",
)
_FLAG_USE_TPUS = flags.DEFINE_boolean(
  "use-tpus",
  False,
  "Whether to create a TPU."
)
_FLAG_USER_NAME = flags.DEFINE_string(
  "username",
  "jules",
  "The gcloud username. "
)
_FLAG_VM_ONLY = flags.DEFINE_boolean(
  "vm-only",
  False,
  "Whether to only create a VM and not reserve TPUs."
  "Great for running other tasks that don't require a TPU, "
  "but that still require a similar setup.",
)

_FLAG_NGROK_CONFIG_PATH = flags.DEFINE_string(
  "ngrok-config-path",
  None,
  "Path of the user configuration file for ngrok."
)

_FLAG_USE_ONE_VM = flags.DEFINE_boolean(
  "use-one-vm",
  False,
  "Whether to use the 1VM setup, for IE jax."
)




def h1(text):
  print("\n" + "#" * utils.term_size())
  print("# " + "[green bold]" + text + "[/]")
  print("#" * utils.term_size())


def h2(text):
  print("[blue bold italic]" + text + "[/]")


def h3(text):
  print(text)


def try_command(command, title, sleep_time, shell=False):
  while True:
    try:
      run_gcloud_command(command, shell=shell)
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
  utils.check_contained(instance_tuple[0], {"n1", "n2"})

  utils.check_contained(instance_tuple[1], {"standard", "highmem"})
  num_cpus = int(instance_tuple[2])
  utils.check_operator(operator.le, num_cpus, 64)
  utils.check_operator(operator.ge, num_cpus, 0)


def run_gcloud_command(command, shell=False):
  print(f"Running gcloud command:\n\t{command}")
  subprocess.run(command, check=True, shell=shell)


def create_vm():
  if not _FLAG_INSTANCE_TYPE.value:
    raise ValueError(
      "Using the full gcloud launcher is useless "
      "without an instance type."
    )

  validate_instance_type_flag()

  positional = [
    "gcloud", "compute", "instances", "create", _FLAG_INSTANCE_NAME.value,
  ]

  if _FLAG_PREEMPTIBLE_VM.value:
    positional.append("--preemptible")

  named_flags = {
    "--zone": _FLAG_ZONE.value,
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


def create_one_vm_vm():
  runtime = _ONEVM_RUNTIME_VERSION

  if runtime == "v2-alpha":
    utils.check_equal(_FLAG_TPU_QTY.value, "8")

  command = ["gcloud", "alpha", "compute", "tpus",
             "tpu-vm", "create",
             f"{_FLAG_INSTANCE_NAME.value}",
             f"--zone={_FLAG_ZONE.value}",
             f"--accelerator-type={make_accelerator_type()}",
             f"--version={runtime}",
             ]

  run_gcloud_command(command)

def make_accelerator_type() -> str:
  utils.check_equal(_FLAG_TPU_TYPE.value, "v3")
  utils.check_equal(_FLAG_TPU_QTY.value, "8")
  assert not _FLAG_PREEMPTIBLE_TPU.value, _FLAG_PREEMPTIBLE_TPU.value
  return f"{_FLAG_TPU_TYPE.value}-{_FLAG_TPU_QTY.value}"


def create_tpu_using_gcloud():
  positional_cmd = [
    "gcloud", "compute", "tpus", "create", _FLAG_INSTANCE_NAME.value
  ]

  if _FLAG_PREEMPTIBLE_TPU.value:
    positional_cmd += "--preemptible"

  named_arguments = {
    "--version": "2.4.1",
    "--accelerator-type": make_accelerator_type(),
  }

  cmd = positional_cmd + [
    f"{k}={shlex.quote(v)}" for k, v in named_arguments.items()
  ]

  h2("Starting the TPUs.")
  run_gcloud_command(cmd)


def git_is_dirty(directory=_SCRIPT_DIRECTORY) -> bool:
  os.chdir(directory)
  root = subprocess.check_output([
    "git", "rev-parse", "--show-toplevel",
  ]).decode().strip()

  return git.Repo(root).is_dirty(untracked_files=False)


def git_is_pushed(directory=_SCRIPT_DIRECTORY) -> bool:
  os.chdir(directory)
  root = subprocess.check_output([
    "git", "rev-parse", "--show-toplevel",
  ]).decode().strip()
  repo = git.Repo(root)
  return "Your branch is up to date with" in repo.git.status()


def git_get_commit_id(directory=_SCRIPT_DIRECTORY) -> str:
  os.chdir(directory)
  commit_id = subprocess.check_output([
    "git", "rev-parse", "HEAD"
  ]).decode().strip()
  return commit_id


def send_file(input_file, target):
  if _FLAG_USE_ONE_VM.value:
    internal_command = shlex.quote(f"cat > {shlex.quote(target)}")
    command = "gcloud alpha compute tpus tpu-vms ssh "
    command += (f"{shlex.quote(_FLAG_USER_NAME.value)}@"
                f"{shlex.quote(_FLAG_INSTANCE_NAME.value)} "
                f"--command={internal_command}")
    command = f"cat {shlex.quote(input_file)} | " + command

    helper_text = f"Copying file `{input_file}`."
    try_command(
      command, helper_text, sleep_time=_FLAG_SLEEP_TIME.value
    )
  else:
    try_command(
        [
        "gcloud", "compute", "scp",
        input_file,
        f"{_FLAG_USER_NAME.value}@{_FLAG_INSTANCE_NAME.value}:{target}",
      ], f"Copying `{input_file}`", sleep_time=_FLAG_SLEEP_TIME.value
    )


def ssh_command(command: str, helper_text: str) -> None:
  if _FLAG_USE_ONE_VM.value:
    ssh_start = ["gcloud", "alpha", "compute", "tpus", "tpu-vms", "ssh"]
  else:
    ssh_start = ["gcloud", "compute", "ssh",]

  h1(helper_text)
  try_command(ssh_start + [
      f"{_FLAG_USER_NAME.value}@{_FLAG_INSTANCE_NAME.value}",
      f"--command={command}"
    ],
      helper_text, sleep_time=_FLAG_SLEEP_TIME.value
    )


def main(argv):
  if len(argv) > 1:
    raise RuntimeError(argv)

  if git_is_dirty() or not git_is_pushed():
    raise RuntimeError(
      "The git directory is dirty. Push the changes before running."
    )

  remote_home_dir = f"/home/{_FLAG_USER_NAME.value}/"

  h1("Module args:")
  args = utils.get_module_args(argv[0])
  print(args)
  print("")

  if not subprocess.check_output(["which", "gcloud"]).strip():
    raise RuntimeError("`gcloud` is not in the path. `ctpu` won't work.")

  if (_FLAG_USE_TPUS.value
      and not _FLAG_VM_ONLY.value
      and not _FLAG_USE_ONE_VM.value):
    create_tpu_using_gcloud()


  if _FLAG_TPU_ONLY.value:
    return

  ###########################################################################
  # Beginning of the VM-only stuff
  ###########################################################################
  if _FLAG_USE_ONE_VM.value:
    # create_one_vm_vm()
    pass
  else:
    create_vm()

  ###########################################################################
  # Copying files over
  ###########################################################################
  h1("Copying bashrc")
  send_file(
    f"{_SCRIPT_DIRECTORY}/bashrc",
    remote_home_dir,
  )

  h1("Copying setup.sh")
  send_file(
    f"{_SCRIPT_DIRECTORY}/setup.sh",
    remote_home_dir,
  )

  h1("Copying start_notebooks.sh")
  send_file(
    f"{_SCRIPT_DIRECTORY}/start_notebooks.sh",
    remote_home_dir,
  )

  ##############################################################################
  # Running setup.sh
  ##############################################################################

  # Build Screen Command
  project_dir = (
    f"{remote_home_dir}eli5_retrieval_large_lm/"
  )
  training_script_uri = (
    f"launchers/scripts/training.sh"
  )
  training_command = shlex.quote(
    f"cd {project_dir} && bash {training_script_uri}; exec bash"
  )
  setup_command_list = [
      f"source",
      f"{remote_home_dir}setup.sh",
      f"{git_get_commit_id()}",
  ]

  setup_command_list.append(_FLAG_USE_ONE_VM.value)
  with open(_FLAG_NGROK_CONFIG_PATH.value) as f_in:
    setup_command_list.append(
      yaml.load(f_in, Loader=yaml.Loader)["authtoken"]
    )

  # Build Setup Command
  setup_command = shlex.join(setup_command_list)
  ssh_command(setup_command, "Running setup.sh")

  if _FLAG_RUN_SCRIPT.value:
    screen_command = f"screen -S training -dm bash -c {training_command}"
    ssh_command(screen_command, "Running training")

  h1("All done.")


if __name__ == "__main__":
  app.run(main)
