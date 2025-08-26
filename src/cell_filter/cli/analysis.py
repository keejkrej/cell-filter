from argparse import ArgumentParser
from cell_filter.core.analyze import Analyzer


def main():
    p = ArgumentParser()
    p.add_argument("--patterns", default="data/20250806_patterns_after.nd2")
    p.add_argument("--cells", default="data/20250806_MDCK_timelapse_crop_fov0004.nd2")
    p.add_argument("--nuclei-channel", type=int, default=1)
    p.add_argument("--output", default="data/analysis/")
    p.add_argument("--n-cells", type=int, default=4)
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--range", default="0:1")
    args = p.parse_args()
    analyzer = Analyzer(
        patterns_path=args.patterns,
        cells_path=args.cells,
        output_folder=args.output,
        n_cells=args.n_cells,
        use_gpu=args.use_gpu,
        nuclei_channel=args.nuclei_channel,
    )
    if args.all:
        analyzer.process_views(0, analyzer.generator.n_views)
    else:
        view_range = list(map(int, args.range.split(":")))
        analyzer.process_views(view_range[0], view_range[1])


if __name__ == "__main__":
    main()


