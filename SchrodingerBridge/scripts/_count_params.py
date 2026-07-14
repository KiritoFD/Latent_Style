import sys, torch, torch.nn as nn

# StyTR-2 params
sys.path.insert(0, "I:/StyTR2")
sys.path.insert(0, "I:/")
import models.StyTR as StyTR
import models.transformer as transformer_module

vgg = StyTR.vgg
vgg.load_state_dict(torch.load("I:/StyTR2/experiments/vgg_normalised.pth", map_location="cpu"))
vgg = nn.Sequential(*list(vgg.children())[:44])
decoder = StyTR.decoder
Trans = transformer_module.Transformer()
embedding = StyTR.PatchEmbed()

vgg_params = sum(p.numel() for p in vgg.parameters())
dec_params = sum(p.numel() for p in decoder.parameters())
trans_params = sum(p.numel() for p in Trans.parameters())
emb_params = sum(p.numel() for p in embedding.parameters())

total_stytr = vgg_params + dec_params + trans_params + emb_params
print("=== StyTR-2 ===")
print(f"VGG encoder: {vgg_params/1e6:.2f}M")
print(f"Decoder:     {dec_params/1e6:.2f}M")
print(f"Transformer: {trans_params/1e6:.2f}M")
print(f"Embedding:   {emb_params/1e6:.2f}M")
print(f"TOTAL:       {total_stytr/1e6:.2f}M")
print(f"Trainable (excl VGG): {(total_stytr-vgg_params)/1e6:.2f}M")

# AesPA-Net params
print()
print("=== AesPA-Net ===")
sys.path.insert(0, "I:/AesPA-Net")
try:
    # AesPA-Net model structure
    import models.decoder as aespa_decoder
    import models.transformer as aespa_transformer
    # Try different import patterns
    dec = aespa_decoder.Decoder()
    trans = aespa_transformer.Transformer()
    dec_params_a = sum(p.numel() for p in dec.parameters())
    trans_params_a = sum(p.numel() for p in trans.parameters())
    total_aespa = dec_params_a + trans_params_a + vgg_params  # VGG is shared
    print(f"VGG encoder: {vgg_params/1e6:.2f}M (shared)")
    print(f"Decoder:     {dec_params_a/1e6:.2f}M")
    print(f"Transformer: {trans_params_a/1e6:.2f}M")
    print(f"TOTAL:       {total_aespa/1e6:.2f}M")
    print(f"Trainable (excl VGG): {(total_aespa-vgg_params)/1e6:.2f}M")
except Exception as e:
    print(f"AesPA import error: {e}")
    # Fallback: count from checkpoint files
    try:
        dec_ckpt = torch.load("I:/AesPA-Net/train_results/aespa/log/dec_model_.pth", map_location="cpu")
        trans_ckpt = torch.load("I:/AesPA-Net/train_results/aespa/log/transformer_model_.pth", map_location="cpu")
        dec_params_b = sum(v.numel() for v in dec_ckpt.values() if hasattr(v, 'numel'))
        trans_params_b = sum(v.numel() for v in trans_ckpt.values() if hasattr(v, 'numel'))
        total_aespa_b = dec_params_b + trans_params_b + vgg_params
        print(f"From checkpoints:")
        print(f"VGG encoder: {vgg_params/1e6:.2f}M (shared)")
        print(f"Decoder:     {dec_params_b/1e6:.2f}M")
        print(f"Transformer: {trans_params_b/1e6:.2f}M")
        print(f"TOTAL:       {total_aespa_b/1e6:.2f}M")
        print(f"Trainable (excl VGG): {(total_aespa_b-vgg_params)/1e6:.2f}M")
    except Exception as e2:
        print(f"Checkpoint count error: {e2}")
