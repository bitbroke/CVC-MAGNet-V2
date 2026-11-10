Recovery V3: original checkpoints, benchmark files and historical results remain unchanged.

Authoritative review provenance remains the V1 source. V2 batch_repair_benchmark.py
automatically asserts review outcomes and a panel identity, which do not establish
independent review. The hard-coded coordinate change is retained as a proposal only.
review_required_16.csv requires explicit recorded review; do not auto-attest it.

run_audit.py evaluates existing checkpoints only. Canonical native-grid segmentation
is reported separately from tip matching. The old pooled model-grid Dice is omitted.
For the offset checkpoint, compare quadratic decoding with offsets disabled against
learned-offset decoding of the SAME heatmap. Differences cannot alone prove gradient
interference. Tip results on mixed V2 labels remain exploratory; complete provisional
studies are reported separately. Neither is an external or clinically attested test.

contracts.py supplies explicit complete-instance supervision and exact continuous
offset targets for future runs. It does not change old checkpoint target conventions.
New target geometry must be separately versioned and tested against the actual image
resize, not merely against its own inverse. Shared-cell collisions are excluded from
offset supervision; unknown labels never become negative examples.

No 'perfect model' or clinical-readiness claim is justified. Remaining independent
requirements include trusted tip identities, external validation, anatomy labels and
pixel calibration before millimetre distance claims.
