# Command Examples

## Cell Counter Analysis
```bash
# Analyze cell patterns and generate results
python -m cell_counter.cli.analyze \
    --patterns /mnt/e/20220525_patterns_end.nd2 \
    --cells /mnt/e/20220525_MBAMB231.nd2 \
    --output ~/workspace/result/ \
    --all

# View information about patterns and cells
python -m cell_counter.cli.info \
    --patterns /mnt/e/20220525_patterns_end.nd2 \
    --cells /mnt/e/20220525_MBAMB231.nd2 \
    --view 0 \
    --debug
```

Note: Replace `src.main` with your actual main module path if different. 