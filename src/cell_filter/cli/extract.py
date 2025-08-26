from argparse import ArgumentParser
from cell_filter.core.extract import Extractor


def main():
    p = ArgumentParser()
    p.add_argument("--patterns", default="data/20250806_patterns_after.nd2")
    p.add_argument("--cells", default="data/20250806_MDCK_timelapse_crop_fov0004.nd2")
    p.add_argument("--nuclei-channel", type=int, default=1)
    p.add_argument("--time-series", default="data/analysis/")
    p.add_argument("--output", default="data/analysis/")
    p.add_argument("--min-frames", type=int, default=10)
    args = p.parse_args()
    extractor = Extractor(
        patterns_path=args.patterns,
        cells_path=args.cells,
        output_folder=args.output,
        nuclei_channel=args.nuclei_channel,
    )
    extractor.extract(time_series_dir=args.time_series, min_frames=args.min_frames)


if __name__ == "__main__":
    main()


