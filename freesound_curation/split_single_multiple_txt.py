import argparse


def main(args):
    with open(args.fwhole, "r") as f:
        lines = f.readlines()

    with open(args.fsingle, "r") as f:
        lines_single = f.readlines()
    if args.fsplit is not None:
        with open(args.fsplit, "r") as f:
            lines_split = f.readlines()
    else:
        lines_split = []
    lines_excluded = lines_single + lines_split

    ids_excluded = []
    for line in lines_excluded:
        line = line.strip()
        if len(line.split()) > 1:
            aid, onset, offset, pad = line.split()
        else:
            aid = line
        if aid not in ids_excluded:
            ids_excluded.append(aid)

    with open(args.fout, "w") as f:
        for line in lines:
            line = line.strip()
            aid = line
            if aid in ids_excluded:
                continue
            else:
                f.write(line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fwhole", type=str, required=True)
    parser.add_argument("--fsingle", type=str, required=True)
    parser.add_argument("--fsplit", type=str, default=None)
    parser.add_argument("--fout", type=str, required=True)

    args = parser.parse_args()
    main(args)