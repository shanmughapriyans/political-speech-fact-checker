"""Quick test: verify HF token and model access without downloading the model."""
import sys
sys.path.insert(0, ".")
import config

# 1. Load token from .env
try:
    token = config.get_hf_token()
    masked = token[:8] + "..." + token[-4:]
    print(f"[OK] HF_TOKEN loaded from .env: {masked}")
except EnvironmentError as e:
    print(f"[FAIL] {e}")
    sys.exit(1)

# 2. Authenticate with HF Hub (no model download)
try:
    from huggingface_hub import HfApi
    api = HfApi()
    user = api.whoami(token=token)
    username = user["name"]
    print(f"[OK] HF authentication: logged in as {username!r}")
except Exception as e:
    print(f"[FAIL] HF authentication failed: {e}")
    sys.exit(1)

# 3. Check model access (metadata only, no weights downloaded)
for model_id in [config.DIARIZATION_MODEL, "pyannote/segmentation-3.0"]:
    try:
        from huggingface_hub import model_info
        info = model_info(model_id, token=token)
        print(f"[OK] Model access confirmed: {model_id}")
    except Exception as e:
        print(f"[FAIL] Cannot access {model_id}: {e}")
        print(f"       Accept terms at: https://huggingface.co/{model_id}")
        sys.exit(1)

print()
print("All checks passed. Token is valid and model access is confirmed.")
print("You can now run:")
print("  python server.py --input file --file your_audio.wav --diarize")
