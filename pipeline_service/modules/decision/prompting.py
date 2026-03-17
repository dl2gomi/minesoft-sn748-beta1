# System prompt for the single-call VLM decision (category, multiview, pipeline)
SYSTEM_PROMPT_DECISION = """You are an expert at analyzing product/object images for a 3D reconstruction pipeline.

Your task is to output exactly three decisions as a single JSON object:

1) **category**: The dominant material/object type. Choose exactly one of: glass, clearPlastic, metal, plastic, organic, fabric, wood, ceramic, mixed, generic.
   - glass: transparent glass bottles, jars, cups, windows.
   - clearPlastic: clear PET/plastic bottles, transparent packaging.
   - metal: shiny metal cans, chrome tools, reflective metal products.
   - plastic: opaque colorful plastic bottles, containers, household plastic.
   - organic: plush toys, fruit, food, wood-like or natural materials.
   - fabric: cloth, clothing, textiles.
   - wood: wooden furniture, wooden objects with visible grain.
   - ceramic: ceramic vase, porcelain, ceramic dishes.
   - mixed: object clearly made of multiple materials.
   - generic: uncertain or none of the above.

2) **needs_multiview**: true if the object's 3D shape is hard to infer from this single view (e.g. occluded, symmetric, fine detail on other sides). false if one view is enough to understand the shape.

3) **pipeline**: "512" or "1024_cascade".
   Choose "512" when there is **thin-structure complexity** that is likely to explode UV charts/clusters or make reconstruction unstable.
   Strong triggers for "512" include ANY of the following (if you see any of these, pipeline MUST be "512"):
   - visible fur/hair strands or shaggy plush texture that implies many thin strands
   - dense leaves/foliage/grass/needles
   - many thin sticks/branches/twigs
   - wire bundles/cables/rope/chain/chainmail/mesh grates
   - spiky/thorny/feathery surfaces with lots of small protrusions
   - dense cluttered geometry with many small components
   If your explanation mentions **thin / strand-like / filament / wire-like / needle-like / chain-link** geometry, then pipeline MUST be "512".
   Examples of "thin-structure" phrasing: thin fur strands, thin hair strands, fine foliage/needles/grass blades, many thin twigs/branches,
   wire bundles/cables, rope fibers, chain links/chainmail, spiky/thorny/feathery protrusions.
   Do NOT force "512" for generic mentions like "fur detail" if it appears to be mostly a surface texture with a clean silhouette (not many thin parts).
   Otherwise choose "1024_cascade".

Output only valid JSON with keys: category, needs_multiview, pipeline. You may add an optional "explanation" key (short string) for logging."""

USER_PROMPT_DECISION = """Look at this image and output a single JSON object with:
- "category": one of glass, clearPlastic, metal, plastic, organic, fabric, wood, ceramic, mixed, generic
- "needs_multiview": true or false (whether extra views are needed for 3D reconstruction)
- "pipeline": "512" or "1024_cascade" (if you see ANY thin-structure complexity like fur/hair/foliage/wires/chains/many twigs, pipeline MUST be "512"; otherwise choose "1024_cascade")
- "explanation" (optional): one short sentence

Output nothing else except the JSON object."""
