import os
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.yolo-config'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))

MODEL_PATH = Path(os.getenv('MODEL_PATH', ROOT / 'models' / 'best.pt'))
CLIP_MODEL_PATH = Path(
    os.getenv('CLIP_MODEL_PATH', ROOT / '.cache' / 'clip' / 'ViT-B-32.pt')
)
CONFIDENCE_FLOOR = float(os.getenv('YOLO_CONFIDENCE_FLOOR', '0.25'))
CANDIDATE_THRESHOLD = float(os.getenv('YOLO_CANDIDATE_THRESHOLD', '0.75'))
MAX_BBOX_AREA_RATIO = float(os.getenv('YOLO_MAX_BBOX_AREA_RATIO', '0.85'))
JELLYFISH_CANDIDATE_THRESHOLD = float(
    os.getenv('YOLO_JELLYFISH_CANDIDATE_THRESHOLD', '0.85')
)
JELLYFISH_SEMANTIC_THRESHOLD = float(
    os.getenv('JELLYFISH_SEMANTIC_THRESHOLD', '0.35')
)

_model = None
_semantic_model = None
_semantic_preprocess = None
_semantic_text_features = None


def model():
    global _model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    if _model is None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                'AI 패키지가 아직 설치되지 않았습니다. requirements-ai.txt를 설치하세요.'
            ) from error
        _model = YOLO(str(MODEL_PATH))
    return _model


def _unknown_result(best_match: dict, reason: str):
    return {
        'speciesId': 'unknown',
        'speciesName': '판별 불가',
        'confidence': best_match['confidence'],
        'dangerLevel': '확인 필요',
        'classificationStatus': 'unknown',
        'bbox': best_match['bbox'],
        'reviewReason': reason,
        'description': (
            '현재 모델만으로 위험 생물의 종류를 신뢰성 있게 판별하기 어렵습니다.'
        ),
        'guidance': (
            '생물을 만지지 말고 안전거리를 유지한 뒤 현장 안전요원에게 문의하세요.'
        ),
        'firstAid': [],
    }


def _semantic_jellyfish_score(image: Image.Image):
    """Return jellyfish and strongest non-jellyfish CLIP probabilities.

    This verifier is only used for a high-confidence YOLO jellyfish result.
    If the optional local model cannot be loaded, fail closed and keep the
    result unconfirmed.
    """

    global _semantic_model, _semantic_preprocess, _semantic_text_features
    if not CLIP_MODEL_PATH.exists():
        return None

    try:
        import clip
        import torch

        if _semantic_model is None:
            _semantic_model, _semantic_preprocess = clip.load(
                str(CLIP_MODEL_PATH),
                device='cpu',
            )
            _semantic_model.eval()
            prompts = [
                'a photo of a jellyfish',
                'a photo of a Nomura jellyfish',
                'a photo of an octopus',
                'a photo of a fish',
                'a photo of an empty ocean',
                'a photo of trash or beach',
            ]
            with torch.inference_mode():
                _semantic_text_features = _semantic_model.encode_text(
                    clip.tokenize(prompts)
                )
                _semantic_text_features /= _semantic_text_features.norm(
                    dim=-1,
                    keepdim=True,
                )

        with torch.inference_mode():
            image_tensor = _semantic_preprocess(image).unsqueeze(0)
            image_features = _semantic_model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            probabilities = (
                100 * image_features @ _semantic_text_features.T
            ).softmax(dim=-1)[0]
        jellyfish_score = float(probabilities[0] + probabilities[1])
        other_score = float(probabilities[2:].max())
        return jellyfish_score, other_score
    except Exception:
        return None


def _candidate_result(match: dict, item: dict, semantic_score: float | None = None):
    result = {
        'speciesId': match['rawName'],
        'speciesName': item['name'],
        'confidence': match['confidence'],
        'dangerLevel': item['dangerLevel'],
        'classificationStatus': 'candidate',
        'bbox': match['bbox'],
        'description': item['description'],
        'guidance': item['guidance'],
        'firstAid': item['firstAid'],
    }
    if semantic_score is not None:
        result['semanticScore'] = round(semantic_score, 3)
        result['verificationMethod'] = 'yolo_and_semantic_guard'
    return result


def detect(image: Image.Image, species: dict):
    image_area = max(1, image.width * image.height)
    result = model().predict(
        image,
        conf=CONFIDENCE_FLOOR,
        max_det=5,
        verbose=False,
    )[0]
    matches = []

    for box in result.boxes:
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        raw_name = result.names[class_id]
        x1, y1, x2, y2 = [
            round(float(value)) for value in box.xyxy[0].tolist()
        ]
        matches.append(
            {
                'rawName': raw_name,
                'confidence': round(confidence, 3),
                'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                'bboxAreaRatio': (
                    max(0, x2 - x1) * max(0, y2 - y1) / image_area
                ),
                'item': species.get(raw_name),
            }
        )

    if not matches:
        return []

    matches.sort(key=lambda match: match['confidence'], reverse=True)
    best_match = matches[0]

    strong_matches = [
        match
        for match in matches
        if match['item'] is not None
        and match['confidence'] >= CANDIDATE_THRESHOLD
    ]

    if not strong_matches:
        return [_unknown_result(best_match, 'low_confidence')]

    best_candidate = strong_matches[0]
    item = best_candidate['item']
    return [_candidate_result(best_candidate, item)]
