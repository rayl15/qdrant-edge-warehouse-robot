# data/

Not committed with real imagery. Generate placeholders with:

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
