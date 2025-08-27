from argparse import ArgumentParser
from cell_filter.core.filter import Filterer
from cell_filter.utils.gpu_utils import validate_segmentation_requirements
import logging


def main():
    p = ArgumentParser()
    p.add_argument("--patterns", default="data/20250806_patterns_after.nd2")
    p.add_argument("--cells", default="data/20250806_MDCK_timelapse_crop_fov0004.nd2")
    p.add_argument("--nuclei-channel", type=int, default=1)
    p.add_argument("--output", default="data/analysis/")
    p.add_argument("--n-cells", type=int, default=4)
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    p.add_argument("--all", action="store_true")
    p.add_argument("--range", default="0:1")
    args = p.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(levelname)s - %(name)s - %(message)s"
    )

    # GPU validation (always required)
    try:
        validate_segmentation_requirements(enable_segmentation=True)
    except Exception as e:
        logging.error(f"GPU validation failed: {e}")
        return 1
    filter_processor = Filterer(
        patterns_path=args.patterns,
        cells_path=args.cells,
        output_folder=args.output,
        n_cells=args.n_cells,
        nuclei_channel=args.nuclei_channel,
    )
    if args.all:
        filter_processor.process_views(0, filter_processor.cropper.n_views)
    else:
        view_range = list(map(int, args.range.split(":")))
        filter_processor.process_views(view_range[0], view_range[1])


if __name__ == "__main__":
    main()
