SYSTEM_PROMPT = """
You are a strict 3D model judge. Rate visual quality and geometry accuracy.
Use the full 0-10 scale. Do not cluster scores in the middle—distinguish good from bad.
Reply with only one number, nothing else."""

# Used when comparing two renders (single-image calls): encourage spread to avoid ties
JUDGE_SINGLE_IMAGE_USER = """Rate this 3D model (4 views). One number 0-10 only.

Scale: 0-2=excellent (accurate, clean). 3-4=good (minor flaws). 5-6=acceptable (noticeable issues). 7-8=flawed (wrong parts or shape). 9-10=wrong object or broken.
Be decisive: use 0,1,2 for strong results and 7,8,9,10 for weak ones. Reply with the number only."""

USER_PROMPT_IMAGE = """Does each 3D model match the image prompt?

Penalty 0-10:
0 = Perfect match
3 = Minor issues (slight shape differences, missing small details)
5 = Moderate issues (wrong style, significant details missing)
7 = Major issues (wrong category but related, e.g. chair vs stool)
10 = Completely wrong object

Output: {"penalty_1": <0-10>, "penalty_2": <0-10>, "issues": "<brief>"}"""
