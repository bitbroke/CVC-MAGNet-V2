import timm
from transformers import SegformerForSemanticSegmentation, SegformerConfig

print("=== TIMM HRNet models ===")
hrnet_models = timm.list_models("*hrnet*")
print(f"Found {len(hrnet_models)} HRNet models. Sample:", hrnet_models[:10])

small_hrnets = [m for m in hrnet_models if "w18" in m.lower() or "small" in m.lower()]
print("Small HRNet models:", small_hrnets)

print("\n=== Transformers SegFormer ===")
try:
    config = SegformerConfig.from_pretrained("nvidia/mit-b0", num_labels=1)
    segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", config=config, ignore_mismatched_sizes=True)
    print("Successfully instantiated SegformerForSemanticSegmentation from nvidia/mit-b0!")
except Exception as e:
    print("Transformers Segformer instantiation error:", e)

print("\n=== TIMM MIT / Segformer models ===")
mit_models = timm.list_models("*mit*")
print(f"MIT models in timm: {mit_models}")
