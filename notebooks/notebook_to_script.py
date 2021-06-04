import json
import os
import sys
DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
import utils

def main(in_target: utils.PathType, out_target: utils.PathType):
    utils.check_exists(in_target)
    parent_dir_out = os.path.dirname(os.path.abspath(out_target))
    utils.check_exists(parent_dir_out)
    utils.check_exists(out_target, inverse=True)

    all_code = ""
    with open(in_target) as fin:
        input_json = json.load(fin)
        cells = input_json["cells"]
        code_cells = lambda : (c for c in cells if c["cell_type"] == "code")
        for cell in code_cells():
            all_code += "".join(cell["source"]) + "\n\n"

    with open(out_target, "w") as fout:
        fout.write(all_code)

if __name__ == "__main__":
    assert 2 <= len(sys.argv) <= 3, len(sys.argv)

    if len(sys.argv) == 2:
        output_path = sys.argv[1] + ".py"
    elif len(sys.argv) == 3:
        output_path = sys.argv[2]
    else:
        raise RuntimeError()

    main(sys.argv[1].strip(), output_path)