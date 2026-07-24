# Image attribution

Sample images in `data/catalog/` come from **Amazon Berkeley Objects (ABO)**, licensed
**CC BY 4.0**. Source: https://amazon-berkeley-objects.s3.amazonaws.com/index.html
Images are the downscaled ABO variant (<=256px). Each folder is one ABO listing
(`item_id`) with several catalog views.

ABO's permissively-licensed multi-view listings are footwear and accessories, so the
sample catalog is shoes, bags, hats, watches and sunglasses rather than garments. The
pipeline is identical for apparel; only the reference photos differ.

| Folder | ABO item_id | product_type | views |
|---|---|---|---|
| ABO-001_amazon-merk-vinden-dames | B06X9STHNG | SHOES | 4 |
| ABO-002_the-fix-amazon-brand | B01MTEI8M6 | SHOES | 4 |
| ABO-003_richter-kinderschuhe-bios-mules | B07J2BHRNW | SHOES | 4 |
| ABO-004_amazon-brand-the-fix | B078M7SX69 | HANDBAG | 4 |
| ABO-005_flavia-women-s-handbag | B07WRL3WYP | HANDBAG | 4 |
| ABO-006_amazonbasics-casual-sports-backpack | B07TPYLKQ6 | BACKPACK | 4 |
| ABO-007_jeff-aimy-warm-fleece | B0823VMWD1 | HAT | 3 |
| ABO-008_amazon-essentials-dress-watch | B07YQFCGXX | WATCH | 3 |
| ABO-009_peepers-anteojos-de-sol | B07PRN66LS | SUNGLASSES | 4 |

---

## Crumpled garments (`data/catalog_garments/`)

The garment catalog is 25 t-shirts, each photographed **wrinkled from the front
and the back**, used by `scripts/demo_crumpled.py` to test retrieval on genuinely
crumpled apparel (the footwear/accessories caveat above is why this second
catalog exists).

All photos are by Flickr user **ir0cko** and licensed **CC BY 2.0**
(<https://creativecommons.org/licenses/by/2.0/>). ir0cko publishes them as free
t-shirt mockup templates. No changes were made to the images other than saving
them at their source resolution.

| SKU | file | source (Flickr, CC BY 2.0) |
| --- | --- | --- |
| TEE-01-ARMY-GREEN | wrinkled-back.jpg | [Wrinkled Back- Army Green](https://www.flickr.com/photos/68525400@N00/2627596646) |
| TEE-01-ARMY-GREEN | wrinkled-front.jpg | [Wrinkled Front- Army Green](https://www.flickr.com/photos/68525400@N00/2626455469) |
| TEE-02-BANANA | wrinkled-back.jpg | [Wrinkled Back- Banana/Yellow](https://www.flickr.com/photos/68525400@N00/2627596464) |
| TEE-02-BANANA | wrinkled-front.jpg | [Wrinkled Front- Banana](https://www.flickr.com/photos/68525400@N00/2626437867) |
| TEE-03-BLACK | wrinkled-back.jpg | [Wrinkled Back- Black](https://www.flickr.com/photos/68525400@N00/2627596502) |
| TEE-03-BLACK | wrinkled-front.jpg | [Wrinkled Front- Black](https://www.flickr.com/photos/68525400@N00/2627226798) |
| TEE-04-BLUE-JEAN | wrinkled-back.jpg | [Wrinkled Back- Blue Jean](https://www.flickr.com/photos/68525400@N00/2626778831) |
| TEE-04-BLUE-JEAN | wrinkled-front.jpg | [Wrinkled Front- Blue Jean](https://www.flickr.com/photos/68525400@N00/2626475771) |
| TEE-05-BROWN | wrinkled-back.jpg | [Wrinkled Back- Brown](https://www.flickr.com/photos/68525400@N00/2626778861) |
| TEE-05-BROWN | wrinkled-front.jpg | [Wrinkled Front- Brown](https://www.flickr.com/photos/68525400@N00/2626455559) |
| TEE-06-DARK-GREY | wrinkled-back.jpg | [Wrinkled Back- Dark Grey](https://www.flickr.com/photos/68525400@N00/2626778671) |
| TEE-06-DARK-GREY | wrinkled-front.jpg | [Wrinkled Front- Dark Grey](https://www.flickr.com/photos/68525400@N00/2627226698) |
| TEE-07-FUSCHIA | wrinkled-back.jpg | [Wrinkled Back- Fuschia](https://www.flickr.com/photos/68525400@N00/2627597096) |
| TEE-07-FUSCHIA | wrinkled-front.jpg | [Wrinkled Front- Fuschia](https://www.flickr.com/photos/68525400@N00/2626475253) |
| TEE-08-GOLD | wrinkled-back.jpg | [Wrinkled Back- Gold](https://www.flickr.com/photos/68525400@N00/2626778705) |
| TEE-08-GOLD | wrinkled-front.jpg | [Wrinkled Front- Gold](https://www.flickr.com/photos/68525400@N00/2626437963) |
| TEE-09-HEATHER-GREY | wrinkled-back.jpg | [Wrinkled Back- Heather Grey](https://www.flickr.com/photos/68525400@N00/2626778739) |
| TEE-09-HEATHER-GREY | wrinkled-front.jpg | [Wrinkled Front- Heather Grey](https://www.flickr.com/photos/68525400@N00/2627226576) |
| TEE-10-KELLY-GREEN | wrinkled-back.jpg | [Wrinkled Back- Kelly Green](https://www.flickr.com/photos/68525400@N00/2626778539) |
| TEE-10-KELLY-GREEN | wrinkled-front.jpg | [Wrinkled Front- Kelly Green](https://www.flickr.com/photos/68525400@N00/2627256960) |
| TEE-11-LIGHT-BLUE | wrinkled-back.jpg | [Wrinkled Back- Light Blue](https://www.flickr.com/photos/68525400@N00/2627596260) |
| TEE-11-LIGHT-BLUE | wrinkled-front.jpg | [Wrinkled Front- Light Blue](https://www.flickr.com/photos/68525400@N00/2626475617) |
| TEE-12-LIGHT-PURPLE | wrinkled-back.jpg | [Wrinkled Back- Light Purple](https://www.flickr.com/photos/68525400@N00/2626778633) |
| TEE-12-LIGHT-PURPLE | wrinkled-front.jpg | [Wrinkled Front- Light Purple](https://www.flickr.com/photos/68525400@N00/2627274324) |
| TEE-13-MAROON | wrinkled-back.jpg | [Wrinkled Back- Maroon](https://www.flickr.com/photos/68525400@N00/2627596922) |
| TEE-13-MAROON | wrinkled-front.jpg | [Wrinkled Front- Maroon](https://www.flickr.com/photos/68525400@N00/2626455649) |
| TEE-14-NATURAL | wrinkled-back.jpg | [Wrinkled Back- Natural/Cream](https://www.flickr.com/photos/68525400@N00/2627596602) |
| TEE-14-NATURAL | wrinkled-front.jpg | [Wrinkled Front- Natural/Cream](https://www.flickr.com/photos/68525400@N00/2627233778) |
| TEE-15-NAVY-BLUE | wrinkled-back.jpg | [Wrinkled Back- Navy Blue](https://www.flickr.com/photos/68525400@N00/2626779225) |
| TEE-15-NAVY-BLUE | wrinkled-front.jpg | [Wrinkled Front- Navy Blue](https://www.flickr.com/photos/68525400@N00/2627304220) |
| TEE-16-OLIVE-GREEN | wrinkled-back.jpg | [Wrinkled Back- Olive Green](https://www.flickr.com/photos/68525400@N00/2627596992) |
| TEE-16-OLIVE-GREEN | wrinkled-front.jpg | [Wrinkled Front- Olive Green](https://www.flickr.com/photos/68525400@N00/2627256842) |
| TEE-17-ORANGE | wrinkled-back.jpg | [Wrinkled Back- Orange](https://www.flickr.com/photos/68525400@N00/2627596696) |
| TEE-17-ORANGE | wrinkled-front.jpg | [Wrinkled Front- Orange](https://www.flickr.com/photos/68525400@N00/2627256754) |
| TEE-18-PINK | wrinkled-back.jpg | [Wrinkled Back- Pink](https://www.flickr.com/photos/68525400@N00/2626779021) |
| TEE-18-PINK | wrinkled-front.jpg | [Wrinkled Front- Pink](https://www.flickr.com/photos/68525400@N00/2627274146) |
| TEE-19-PURPLE | wrinkled-back.jpg | [Wrinkled Back- Purple](https://www.flickr.com/photos/68525400@N00/2626779281) |
| TEE-19-PURPLE | wrinkled-front.jpg | [Wrinkled Front- Purple](https://www.flickr.com/photos/68525400@N00/2626475369) |
| TEE-20-RED | wrinkled-back.jpg | [Wrinkled Back- Red](https://www.flickr.com/photos/68525400@N00/2626779317) |
| TEE-20-RED | wrinkled-front.jpg | [Wrinkled Front- Red](https://www.flickr.com/photos/68525400@N00/2626475457) |
| TEE-21-ROYAL-BLUE | wrinkled-back.jpg | [Wrinkled Back- Royal Blue](https://www.flickr.com/photos/68525400@N00/2626779063) |
| TEE-21-ROYAL-BLUE | wrinkled-front.jpg | [Wrinkled Front- Royal Blue](https://www.flickr.com/photos/68525400@N00/2626486243) |
| TEE-22-SAND | wrinkled-back.jpg | [Wrinkled Back- Sand](https://www.flickr.com/photos/68525400@N00/2626779099) |
| TEE-22-SAND | wrinkled-front.jpg | [Wrinkled Front- Sand](https://www.flickr.com/photos/68525400@N00/2626408683) |
| TEE-23-SILVER | wrinkled-back.jpg | [Wrinkled Back- Silver](https://www.flickr.com/photos/68525400@N00/2627596852) |
| TEE-23-SILVER | wrinkled-front.jpg | [Wrinkled Front- Silver](https://www.flickr.com/photos/68525400@N00/2627226444) |
| TEE-24-TURQUOISE | wrinkled-back.jpg | [Wrinkled Back- Turquoise](https://www.flickr.com/photos/68525400@N00/2627596890) |
| TEE-24-TURQUOISE | wrinkled-front.jpg | [Wrinkled Front- Turquoise](https://www.flickr.com/photos/68525400@N00/2627303996) |
| TEE-25-WHITE | wrinkled-back.jpg | [Wrinkled Back- White](https://www.flickr.com/photos/68525400@N00/2627597142) |
| TEE-25-WHITE | wrinkled-front.jpg | [Wrinkled Front- White](https://www.flickr.com/photos/68525400@N00/2625883027) |
