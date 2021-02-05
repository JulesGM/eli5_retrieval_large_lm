import os
from absl import flags
from absl import app
import shlex

_ZONE_TPUV2 = "us-central1-f"
_ZONE_TPUV3 = "europe-west4-a"

# Args
_FLAG_TF_VERSION = flags.DEFINE_enum(
        "tf_version",
        "nightly",
        ["nightly"],
        ""
)

_FLAG_TPU_TYPE = flags.DEFINE_enum(
        "tpu_type",
        "v2",
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


def main(argv):
    if len(argv) > 1:
        raise RuntimeError(argv)

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
    cmd_flags = [bin] + positional_flags + [f"{k}={shlex.quote(v)}" for k, v
                                    in named_flags.items()]
    # cmd_flags = positional_flags + flatten_once(named_flags.items())

    print(f"Running command:\n"
          f"\tbin:    '{bin}'\n"
          f"\tflags:  {cmd_flags}"
    )
    print(f"The command would be:\n\t`{' '.join([bin, ] + cmd_flags)}`")
    os.execlp(bin, *cmd_flags)


if __name__ == "__main__":
    app.run(main)
