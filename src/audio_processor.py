import torch
import torchaudio
from transformers import ASTFeatureExtractor, ASTForAudioClassification
import gc

class AudioEngine:
    def __init__(self):
        self.device = "cpu"
        self.model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
        print(f"Initializing AudioEngine with AST on {self.device}...")
        try:
            self.feature_extractor = ASTFeatureExtractor.from_pretrained(self.model_name)
            self.model = ASTForAudioClassification.from_pretrained(self.model_name).to(self.device)
            print("AudioEngine initialized successfully.")
        except Exception as e:
            print(f"[Warning] Failed to initialize AudioEngine models (requires internet/transformers): {e}")
            self.model = None

    def process_audio(self, audio_path: str) -> str:
        """Analyzes an audio file for sound event detection."""
        if self.model is None:
            return "Audio model not loaded (Initialization failed)."
        
        try:
            # Load audio using torchaudio
            waveform, sampling_rate = torchaudio.load(audio_path)
            
            # AST expects 16kHz sampling rate
            if sampling_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
                waveform = resampler(waveform)
            
            # If stereo, convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                
            # AST model accepts inputs created by the feature extractor
            inputs = self.feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            top_probs, top_indices = torch.topk(probs, 5)
            
            predictions = []
            for i in range(5):
                prob_val = top_probs[i].item()
                label = self.model.config.id2label[top_indices[i].item()]
                predictions.append(f"{label}: {prob_val:.2f}")
                
            return f"[AST Classification] Predicted audio events: {', '.join(predictions)}"
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[Error] OOM Error encountered in AudioEngine: {e}")
            else:
                print(f"[Error] Runtime error in AudioEngine: {e}")
            return "Error: Runtime/OOM in Audio processing."
        except Exception as e:
            print(f"[Error] Exception extracting/processing audio features: {e}")
            return "Error in audio feature processing."
        finally:
            # Mandatory cleanup to satisfy MX150 hardware constraints
            torch.cuda.empty_cache()
            gc.collect()
