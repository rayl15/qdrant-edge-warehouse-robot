# data/

`catalog/` ships with a small **real** sample (9 products, several views each)
from Amazon Berkeley Objects, CC BY 4.0. See `ATTRIBUTION.md`. Run it with
`python scripts/demo_real.py` (needs the `clip` extra).

To generate throwaway synthetic placeholders instead (e.g. for a quick
FakeEmbedder run of the CLI):

```bash
python scripts/make_synthetic_data.py
```

Expected layout:

```
data/
  catalog/
    SKU-1001_navy-crew-tshirt/     # one folder per SKU
      front.png                    # each image is a reference "view"
      folded.png
      crumpled.png
    SKU-1002_black-hoodie/
      ...
  bins/
    tote_01.png                    # a mixed-bin photo to plan picks against
```

Drop real product photos into the SKU folders and real tote photos into `bins/`
to run with `mixed-bin ... --real` (MobileCLIP2).
