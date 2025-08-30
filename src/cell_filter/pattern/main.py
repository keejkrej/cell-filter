from argparse import ArgumentParser
from cell_filter.pattern import Patterner


def main():
    p = ArgumentParser()
    p.add_argument("--patterns", default="data/20250806_patterns_after.nd2")
    p.add_argument("--cells", default="data/20250806_MDCK_timelapse_crop_fov0004.nd2")
    p.add_argument("--nuclei-channel", type=int, default=1)
    p.add_argument("--fov", type=int, default=0)
    p.add_argument("--fov-all", action="store_true")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    patterner = Patterner(patterns_path=args.patterns, cells_path=args.cells, nuclei_channel=args.nuclei_channel)
    if args.fov_all:
        for fov_idx in range(patterner.n_fovs):
            patterner.plot_view(fov_idx)
    else:
        patterner.plot_view(args.fov, args.output)
    patterner.close()


if __name__ == "__main__":
    main()


