import torch, torch.nn.functional as F, cv2
from PIL import Image
from torchvision import transforms
from pathlib import Path

try:
    from src.peta_model import PetaAttrModel
except ImportError:
    from peta_model import PetaAttrModel

class AttrModel:
    def __init__(self, weights_path, device=0):
        self.device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        self.model = PetaAttrModel().to(self.device)

        weights_path = Path(weights_path)
        if not weights_path.is_absolute():
            project_root = Path(__file__).resolve().parents[1]
            candidate_paths = [
                project_root / weights_path,
                project_root / "models" / "peta" / weights_path.name,
            ]
            for candidate in candidate_paths:
                if candidate.exists():
                    weights_path = candidate
                    break

        self.model.load_state_dict(
    torch.load(str(weights_path), map_location=self.device, weights_only=True)
)
        self.model.eval()
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    @torch.no_grad()
    def batch_predict(self, crops):
        if not crops:
            return []
        imgs = []
        for c in crops:
            pil = Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
            imgs.append(self.tf(pil))
        batch = torch.stack(imgs).to(self.device)
        with torch.amp.autocast('cuda'):
            bin_logits, color_logits = self.model(batch)
        bin_probs = torch.sigmoid(bin_logits)
        color_probs = [F.softmax(c, dim=1) for c in color_logits]
        out = []
        for i in range(len(crops)):
            out.append({
                "bin_probs": bin_probs[i].cpu().tolist(),
                "color_probs": [cp[i].cpu().tolist() for cp in color_probs]
            })
        return out